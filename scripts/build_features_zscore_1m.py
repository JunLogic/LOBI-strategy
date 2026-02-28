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
class ZScoreBuildConfig:
    in_parquet: Path
    out_parquet: Path
    start_date: str
    end_date: str
    mode: str
    rolling_window_mins: int
    min_periods: int
    eps: float


def _log(message: str) -> None:
    print(f"[info] {message}")


def _utc_day_bounds_ms(start_date: str, end_date: str) -> tuple[int, int]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(
        milliseconds=1
    )
    return int(start_ts.value // 1_000_000), int(end_ts.value // 1_000_000)


def _to_open_time_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000).astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _select_base_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_exact = {"symbol", "open_time"}
    base_cols: list[str] = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if col.startswith(TARGET_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            base_cols.append(col)
    return base_cols


def _causal_zscore_by_symbol(
    series: pd.Series,
    group_keys: pd.Series,
    mode: str,
    rolling_window_mins: int,
    min_periods: int,
    eps: float,
) -> tuple[pd.Series, int]:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    grouped = values.groupby(group_keys, sort=False)

    if mode == "expanding":
        mu = grouped.transform(
            lambda s: s.expanding(min_periods=min_periods).mean().shift(1)
        )
        sigma = grouped.transform(
            lambda s: s.expanding(min_periods=min_periods).std(ddof=0).shift(1)
        )
        history_count = grouped.cumcount()
    else:
        mu = grouped.transform(
            lambda s: s.rolling(window=rolling_window_mins, min_periods=min_periods)
            .mean()
            .shift(1)
        )
        sigma = grouped.transform(
            lambda s: s.rolling(window=rolling_window_mins, min_periods=min_periods)
            .std(ddof=0)
            .shift(1)
        )
        history_count = grouped.transform(
            lambda s: s.shift(1).rolling(window=rolling_window_mins, min_periods=1).count()
        )

    z = (values - mu) / (sigma + eps)
    default_mask = history_count < min_periods
    z = z.where(~default_mask, 0.0)
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
    return z, int(default_mask.sum())


def build_features_zscore_1m(config: ZScoreBuildConfig) -> pd.DataFrame:
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

    start_ms, end_ms = _utc_day_bounds_ms(config.start_date, config.end_date)
    work = work[(work["open_time"] >= start_ms) & (work["open_time"] <= end_ms)].copy()
    if work.empty:
        raise ValueError(
            f"No rows in date range [{config.start_date}, {config.end_date}] for {config.in_parquet}"
        )

    work = work.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    if "symbol" in work.columns:
        group_keys = work["symbol"].astype("string").fillna("__MISSING_SYMBOL__")
    else:
        group_keys = pd.Series(["__ALL__"] * len(work), index=work.index, dtype="string")

    base_feature_cols = _select_base_feature_columns(work)
    if not base_feature_cols:
        raise ValueError("No numeric base feature columns found for z-scoring.")

    z_columns: dict[str, pd.Series] = {}
    default_counts: dict[str, int] = {}
    for col in base_feature_cols:
        z_col = f"z_{col}"
        z_values, default_count = _causal_zscore_by_symbol(
            series=work[col],
            group_keys=group_keys,
            mode=config.mode,
            rolling_window_mins=config.rolling_window_mins,
            min_periods=config.min_periods,
            eps=config.eps,
        )
        z_columns[z_col] = z_values
        default_counts[z_col] = default_count
        _log(f"defaulted_rows feature={z_col} count={default_count}")

    passthrough_cols = [c for c in work.columns if c not in base_feature_cols]
    ordered_passthrough: list[str] = []
    if "symbol" in passthrough_cols:
        ordered_passthrough.append("symbol")
    ordered_passthrough.append("open_time")
    for col in passthrough_cols:
        if col not in {"symbol", "open_time"}:
            ordered_passthrough.append(col)

    z_df = pd.DataFrame(z_columns, index=work.index)
    out_df = pd.concat([work[ordered_passthrough].copy(), z_df], axis=1)

    config.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(config.out_parquet, index=False)

    meta_path = config.out_parquet.with_suffix(".meta.json")
    meta: dict[str, Any] = {
        "input_parquet": str(config.in_parquet),
        "output_parquet": str(config.out_parquet),
        "mode": config.mode,
        "rolling_window_mins": config.rolling_window_mins if config.mode == "rolling" else None,
        "min_periods": config.min_periods,
        "eps": config.eps,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "base_feature_columns": base_feature_cols,
        "z_feature_columns": list(z_columns.keys()),
        "defaulted_rows_per_z_feature": default_counts,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(
        f"DONE rows={len(out_df):,}, z_features={len(z_columns)}, out_parquet={config.out_parquet}, meta_json={meta_path}"
    )
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causally z-scored 1-minute feature parquet."
    )
    parser.add_argument("--in_parquet", required=True)
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--start_date", required=True, help="UTC date, e.g. 2024-10-01")
    parser.add_argument("--end_date", required=True, help="UTC date, e.g. 2025-06-30")
    parser.add_argument("--mode", default="expanding", choices=MODE_CHOICES)
    parser.add_argument("--rolling_window_mins", type=int, default=1440)
    parser.add_argument("--min_periods", type=int, default=60)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ZScoreBuildConfig(
        in_parquet=Path(args.in_parquet),
        out_parquet=Path(args.out_parquet),
        start_date=args.start_date,
        end_date=args.end_date,
        mode=args.mode,
        rolling_window_mins=args.rolling_window_mins,
        min_periods=args.min_periods,
        eps=args.eps,
    )
    build_features_zscore_1m(config)


if __name__ == "__main__":
    main()
