from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ALLOWED_SYMBOLS = ("PEPEUSDT", "FLOKIUSDT")
FEATURE_SET_CHOICES = ("all", "scalefree")
SCALE_FREE_FEATURES = (
    "ret_1m",
    "rv_5m",
    "rv_30m",
    "rv_4h",
    "spread_proxy",
    "imbalance",
    "buyer_aggressor_ratio",
    "vol_z_30m",
    "drawdown_4h",
)
PREDICTION_SAMPLE_ROWS = 5000


@dataclass(frozen=True)
class TrainConfig:
    symbol: str
    dataset_parquet: Path
    label: str
    feature_set: str
    use_z_features: int
    train_ratio: float
    out_dir: Path
    threshold: float
    seed: int


def _log(message: str) -> None:
    print(f"[info] {message}")


def _to_open_time_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000).astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _select_feature_columns(
    df: pd.DataFrame, feature_set: str, use_z_features: int
) -> list[str]:
    if use_z_features == 1:
        return [
            col
            for col in df.columns
            if col.startswith("z_") and pd.api.types.is_numeric_dtype(df[col])
        ]

    if feature_set == "scalefree":
        return [col for col in SCALE_FREE_FEATURES if col in df.columns]

    exclude_exact = {"symbol", "open_time"}
    exclude_prefixes = ("y_", "dd_min_", "future_min_close_")
    feature_cols: list[str] = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if col.startswith(exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def _artifact_file_names(feature_set: str) -> tuple[str, str]:
    if feature_set == "scalefree":
        return ("model_pipeline_scalefree.joblib", "feature_columns_scalefree.json")
    return ("model_pipeline.joblib", "feature_columns.json")


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


def _build_run_dir(config: TrainConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.out_dir / config.symbol / config.label / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_label_dir(config: TrainConfig) -> Path:
    label_dir = config.out_dir / config.symbol / config.label
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir


def _write_calibration_plot(y_test: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label="Model")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_prob_hist(y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_prob, bins=40, color="#1f77b4", alpha=0.85)
    ax.set_title("Predicted Probability Histogram")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_hazard_logistic(config: TrainConfig) -> dict[str, Any]:
    if not (0.0 < config.train_ratio <= 1.0):
        raise ValueError("--train_ratio must be in (0, 1].")
    if not (0.0 <= config.threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1].")
    if config.feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"--feature_set must be one of {FEATURE_SET_CHOICES}.")
    if config.use_z_features not in (0, 1):
        raise ValueError("--use_z_features must be 0 or 1.")

    df = pd.read_parquet(config.dataset_parquet)
    if df.empty:
        raise ValueError(f"Dataset parquet has no rows: {config.dataset_parquet}")
    if config.label not in df.columns:
        raise ValueError(f"Label column '{config.label}' not found in dataset.")
    if "open_time" not in df.columns:
        raise ValueError("Dataset must contain 'open_time'.")

    df = df.copy()
    df["open_time"] = _to_open_time_ms(df["open_time"])
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64")

    df = df.dropna(subset=[config.label]).copy()
    if df.empty:
        raise ValueError("No rows remain after dropping NaN labels.")

    df = df.sort_values("open_time", kind="mergesort").reset_index(drop=True)

    feature_cols = _select_feature_columns(df, config.feature_set, config.use_z_features)
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found after exclusions.")

    x = df[feature_cols].copy()
    y = pd.to_numeric(df[config.label], errors="coerce")
    valid_y = y.notna()
    x = x.loc[valid_y].copy()
    y = y.loc[valid_y].astype(int)
    df = df.loc[valid_y].copy()

    if len(df) < 2:
        raise ValueError("Need at least 2 labeled rows to split train/test.")

    inf_count = int(np.isinf(x.to_numpy()).sum())
    nan_before_fill = int(x.isna().sum().sum())
    x = x.replace([np.inf, -np.inf], np.nan)
    nan_after_inf_replace = int(x.isna().sum().sum())
    x = x.fillna(0.0)
    nan_after_fill = int(x.isna().sum().sum())

    full_fit = config.train_ratio == 1.0
    label_dir = _build_label_dir(config)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    if full_fit:
        x_train = x
        y_train = y
        x_test = None
        y_test = None
        test_open_time = None
        run_dir = label_dir
    else:
        split_idx = int(len(df) * config.train_ratio)
        split_idx = min(max(split_idx, 1), len(df) - 1)
        x_train = x.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        x_test = x.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        test_open_time = df["open_time"].iloc[split_idx:].reset_index(drop=True)
        run_dir = _build_run_dir(config)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training labels have a single class; LogisticRegression requires two classes."
        )

    pipe.fit(x_train, y_train)
    model_file, feature_cols_file = _artifact_file_names(config.feature_set)
    joblib.dump(pipe, label_dir / model_file)
    (label_dir / feature_cols_file).write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )

    clf = pipe.named_steps["logreg"]
    coef_series = pd.Series(clf.coef_.ravel(), index=feature_cols, name="coef")
    coef_df = (
        coef_series.rename_axis("feature")
        .reset_index()
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    coef_df["rank"] = np.arange(1, len(coef_df) + 1, dtype=int)
    coef_df = coef_df[["feature", "coef", "abs_coef", "rank"]]
    coef_df.to_csv(run_dir / "coefficients.csv", index=False)

    if full_fit:
        y_prob_full = pipe.predict_proba(x)[:, 1]
        pd.DataFrame(
            {
                "open_time": df["open_time"].astype("int64"),
                "y_true": y.astype(int).to_numpy(),
                "y_prob": y_prob_full.astype(float),
            }
        ).to_csv(run_dir / "full_fit_probs.csv", index=False)
        base_rate = float(y.mean())
        metrics: dict[str, Any] = {
            "symbol": config.symbol,
            "label": config.label,
            "train_ratio": config.train_ratio,
            "threshold": config.threshold,
            "seed": config.seed,
            "feature_set": config.feature_set,
            "use_z_features": config.use_z_features,
            "full_fit": True,
            "n_rows_total_labeled": int(len(df)),
            "n_rows_train": int(len(x_train)),
            "n_rows_test": 0,
            "base_rate": base_rate,
            "feature_cleaning": {
                "inf_count_before": inf_count,
                "nan_count_before": nan_before_fill,
                "nan_count_after_inf_replace": nan_after_inf_replace,
                "nan_count_after_fill": nan_after_fill,
            },
        }
    else:
        y_prob = pipe.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= config.threshold).astype(int)
        base_rate = float(y_test.mean())
        roc_auc = _safe_roc_auc(y_test, y_prob)
        pr_auc = _safe_pr_auc(y_test, y_prob)
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        _write_calibration_plot(
            y_test=y_test, y_prob=y_prob, out_path=run_dir / "calibration_plot.png"
        )
        _write_prob_hist(y_prob=y_prob, out_path=run_dir / "prob_hist.png")

        pred_sample = pd.DataFrame(
            {
                "open_time": test_open_time,
                "y_true": y_test.reset_index(drop=True).astype(int),
                "y_prob": y_prob.astype(float),
                "y_pred": y_pred.astype(int),
            }
        )
        pred_sample.tail(PREDICTION_SAMPLE_ROWS).to_csv(
            run_dir / "predictions_sample.csv", index=False
        )

        metrics = {
            "symbol": config.symbol,
            "label": config.label,
            "train_ratio": config.train_ratio,
            "threshold": config.threshold,
            "seed": config.seed,
            "feature_set": config.feature_set,
            "use_z_features": config.use_z_features,
            "full_fit": False,
            "n_rows_total_labeled": int(len(df)),
            "n_rows_train": int(len(x_train)),
            "n_rows_test": int(len(x_test)),
            "base_rate_test": base_rate,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_at_threshold": precision,
            "recall_at_threshold": recall,
            "f1_at_threshold": f1,
            "confusion_matrix_labels_0_1": cm,
            "feature_cleaning": {
                "inf_count_before": inf_count,
                "nan_count_before": nan_before_fill,
                "nan_count_after_inf_replace": nan_after_inf_replace,
                "nan_count_after_fill": nan_after_fill,
            },
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    meta: dict[str, Any] = {
        "dataset_parquet": str(config.dataset_parquet),
        "label": config.label,
        "feature_set": config.feature_set,
        "use_z_features": config.use_z_features,
        "train_ratio": config.train_ratio,
        "threshold": config.threshold,
        "seed": config.seed,
        "n_rows_train": int(len(x_train)),
        "n_rows_test": 0 if full_fit else int(len(x_test)),
        "feature_list": feature_cols,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    top_n = min(10, len(coef_df))
    top = coef_df.head(top_n)
    _log(
        f"Event base rate ({'full dataset' if full_fit else 'test'}): {base_rate:.6f}"
    )
    _log(f"Prediction threshold: {config.threshold:.4f}")
    _log(f"Top {top_n} absolute coefficients:")
    for row in top.itertuples(index=False):
        _log(f"  rank={int(row.rank):2d} feature={row.feature} coef={row.coef:.6f}")
    _log(f"Wrote outputs to: {run_dir}")

    return {"run_dir": str(run_dir), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline logistic hazard model."
    )
    parser.add_argument("--symbol", required=True, choices=ALLOWED_SYMBOLS)
    parser.add_argument("--dataset_parquet", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--feature_set", default="all", choices=FEATURE_SET_CHOICES)
    parser.add_argument("--use_z_features", type=int, default=0, choices=(0, 1))
    parser.add_argument("--train_ratio", type=float, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        symbol=args.symbol,
        dataset_parquet=Path(args.dataset_parquet),
        label=args.label,
        feature_set=args.feature_set,
        use_z_features=args.use_z_features,
        train_ratio=args.train_ratio,
        out_dir=Path(args.out_dir),
        threshold=args.threshold,
        seed=args.seed,
    )
    train_hazard_logistic(config)


if __name__ == "__main__":
    main()
