from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODE_CHOICES = ("expanding", "rolling")
TARGET_PREFIXES = ("y_", "dd_min_", "future_min_close_")


@dataclass(frozen=True)
class ZScoreConfig:
    in_parquet: Path
    out_parquet: Path
    mode: str
    rolling_window_mins: int
    min_periods: int
    eps: float


def _log(message: str) -> None:
    print(f"[info] {message}")


def _to_open_time_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000).astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _select_base_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols: list[str] = []
    for col in df.columns:
        if col == "open_time":
            continue
        if col.startswith(TARGET_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            base_cols.append(col)
    return base_cols


def _compute_causal_z(
    series: pd.Series,
    mode: str,
    rolling_window_mins: int,
    min_periods: int,
    eps: float,
) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype("float64")
    if mode == "expanding":
        mu = x.expanding(min_periods=min_periods).mean().shift(1)
        sd = x.expanding(min_periods=min_periods).std(ddof=0).shift(1)
    else:
        mu = x.rolling(window=rolling_window_mins, min_periods=min_periods).mean().shift(1)
        sd = x.rolling(window=rolling_window_mins, min_periods=min_periods).std(ddof=0).shift(1)

    z = (x - mu) / (sd + eps)
    z = z.where(~(mu.isna() | sd.isna()), 0.0)
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
    return z


def build_dataset_zscore_1m(config: ZScoreConfig) -> pd.DataFrame:
    if config.mode not in MODE_CHOICES:
        raise ValueError(f"--mode must be one of {MODE_CHOICES}.")
    if config.rolling_window_mins <= 0:
        raise ValueError("--rolling_window_mins must be > 0.")
    if config.min_periods <= 0:
        raise ValueError("--min_periods must be > 0.")
    if config.eps <= 0:
        raise ValueError("--eps must be > 0.")

    df = pd.read_parquet(config.in_parquet)
    if df.empty:
        raise ValueError(f"Input parquet has no rows: {config.in_parquet}")
    if "open_time" not in df.columns:
        raise ValueError("Input parquet must contain 'open_time'.")

    work = df.copy()
    work["open_time"] = _to_open_time_ms(work["open_time"])
    work = work.dropna(subset=["open_time"]).copy()
    work["open_time"] = work["open_time"].astype("int64")
    work = work.sort_values("open_time", kind="mergesort").reset_index(drop=True)

    base_feature_cols = _select_base_feature_columns(work)
    if not base_feature_cols:
        raise ValueError("No numeric base feature columns found for z-scoring.")

    z_feature_cols: list[str] = []
    z_df = pd.DataFrame(index=work.index)
    for col in base_feature_cols:
        z_col = f"z_{col}"
        z_df[z_col] = _compute_causal_z(
            series=work[col],
            mode=config.mode,
            rolling_window_mins=config.rolling_window_mins,
            min_periods=config.min_periods,
            eps=config.eps,
        )
        z_feature_cols.append(z_col)

    passthrough_cols: list[str] = ["open_time"]
    if "symbol" in work.columns:
        passthrough_cols.append("symbol")
    passthrough_cols.extend([c for c in work.columns if c.startswith(TARGET_PREFIXES)])

    out_df = pd.concat([work[passthrough_cols].copy(), z_df], axis=1)
    out_df["open_time"] = out_df["open_time"].astype("int64")

    config.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(config.out_parquet, index=False)

    meta_path = Path(f"{config.out_parquet}.meta.json")
    meta: dict[str, Any] = {
        "mode": config.mode,
        "rolling_window_mins": config.rolling_window_mins,
        "min_periods": config.min_periods,
        "eps": config.eps,
        "base_feature_columns": base_feature_cols,
        "z_feature_columns": z_feature_cols,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(
        f"DONE rows={len(out_df):,}, z_features={len(z_feature_cols)}, out_parquet={config.out_parquet}, meta_json={meta_path}"
    )
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal z-scored dataset from per-minute feature+label parquet."
    )
    parser.add_argument("--in_parquet", required=True)
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--mode", default="expanding", choices=MODE_CHOICES)
    parser.add_argument("--rolling_window_mins", type=int, default=1440)
    parser.add_argument("--min_periods", type=int, default=60)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ZScoreConfig(
        in_parquet=Path(args.in_parquet),
        out_parquet=Path(args.out_parquet),
        mode=args.mode,
        rolling_window_mins=args.rolling_window_mins,
        min_periods=args.min_periods,
        eps=args.eps,
    )
    build_dataset_zscore_1m(config)


if __name__ == "__main__":
    main()
