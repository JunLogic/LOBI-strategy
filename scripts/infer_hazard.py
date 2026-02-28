from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


FEATURE_SET_CHOICES = ("all", "scalefree")


@dataclass(frozen=True)
class InferConfig:
    model_dir: Path
    dataset_parquet: Path
    label: str
    feature_set: str
    use_z_features: int
    out_dir: Path


def _log(message: str) -> None:
    print(f"[info] {message}")


def _to_open_time_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000).astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return None


def _load_feature_columns(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(c, str) for c in payload):
        raise ValueError(f"Invalid feature column payload in: {path}")
    if not payload:
        raise ValueError(f"feature_columns.json is empty: {path}")
    return payload


def _artifact_file_names(feature_set: str) -> tuple[str, str]:
    if feature_set == "scalefree":
        return ("model_pipeline_scalefree.joblib", "feature_columns_scalefree.json")
    return ("model_pipeline.joblib", "feature_columns.json")


def infer_hazard(config: InferConfig) -> dict[str, Any]:
    if config.feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"--feature_set must be one of {FEATURE_SET_CHOICES}.")
    if config.use_z_features not in (0, 1):
        raise ValueError("--use_z_features must be 0 or 1.")

    model_file, feature_cols_file = _artifact_file_names(config.feature_set)
    model_path = config.model_dir / model_file
    feature_cols_path = config.model_dir / feature_cols_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found: {model_path}")
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {feature_cols_path}")

    pipe = joblib.load(model_path)
    if not isinstance(pipe, Pipeline):
        raise ValueError(f"Loaded object is not a sklearn Pipeline: {model_path}")
    feature_cols = _load_feature_columns(feature_cols_path)
    if config.use_z_features == 1:
        feature_cols = [c for c in feature_cols if c.startswith("z_")]
        if not feature_cols:
            raise ValueError(
                "--use_z_features=1 selected no z_ columns from feature_columns.json."
            )

    df = pd.read_parquet(config.dataset_parquet)
    if df.empty:
        raise ValueError(f"Dataset parquet has no rows: {config.dataset_parquet}")
    if "open_time" not in df.columns:
        raise ValueError("Dataset must contain 'open_time'.")
    if config.label not in df.columns:
        raise ValueError(f"Label column '{config.label}' not found in dataset.")

    df = df.copy()
    df["open_time"] = _to_open_time_ms(df["open_time"])
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64")

    y = pd.to_numeric(df[config.label], errors="coerce")
    valid_y = y.notna()
    df = df.loc[valid_y].copy()
    y = y.loc[valid_y].astype(int)
    if df.empty:
        raise ValueError("No rows remain after dropping NaN labels.")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    x = df.reindex(columns=feature_cols).copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_prob = pipe.predict_proba(x)[:, 1]

    config.out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = config.out_dir / "oos_probs.csv"
    metrics_path = config.out_dir / "oos_metrics.json"

    pd.DataFrame(
        {
            "open_time": df["open_time"].astype("int64"),
            "y_true": y.astype(int).to_numpy(),
            "y_prob": y_prob.astype(float),
        }
    ).to_csv(probs_path, index=False)

    y0_mask = y == 0
    y1_mask = y == 1
    mean_prob_y0 = float(np.mean(y_prob[y0_mask])) if bool(y0_mask.any()) else None
    mean_prob_y1 = float(np.mean(y_prob[y1_mask])) if bool(y1_mask.any()) else None

    metrics: dict[str, Any] = {
        "base_rate": float(y.mean()),
        "roc_auc": _safe_roc_auc(y, y_prob),
        "pr_auc": _safe_pr_auc(y, y_prob),
        "mean_prob_when_y0": mean_prob_y0,
        "mean_prob_when_y1": mean_prob_y1,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _log(f"Rows scored: {len(df)}")
    _log(f"Missing feature columns filled with 0: {len(missing_cols)}")
    _log(f"base_rate={metrics['base_rate']:.6f}")
    _log(f"roc_auc={metrics['roc_auc']}")
    _log(f"pr_auc={metrics['pr_auc']}")
    _log(f"mean_prob_when_y0={metrics['mean_prob_when_y0']}")
    _log(f"mean_prob_when_y1={metrics['mean_prob_when_y1']}")
    _log(f"Wrote probabilities to: {probs_path}")
    _log(f"Wrote metrics to: {metrics_path}")

    return {"out_dir": str(config.out_dir), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run out-of-sample hazard inference using a saved sklearn pipeline."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--dataset_parquet", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--feature_set", default="all", choices=FEATURE_SET_CHOICES
    )
    parser.add_argument("--use_z_features", type=int, default=0, choices=(0, 1))
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferConfig(
        model_dir=Path(args.model_dir),
        dataset_parquet=Path(args.dataset_parquet),
        label=args.label,
        feature_set=args.feature_set,
        use_z_features=args.use_z_features,
        out_dir=Path(args.out_dir),
    )
    infer_hazard(config)


if __name__ == "__main__":
    main()
