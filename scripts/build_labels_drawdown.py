from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ALLOWED_SYMBOLS = ("PEPEUSDT", "FLOKIUSDT")


@dataclass(frozen=True)
class LabelBuildConfig:
    symbol: str
    features_parquet: Path
    out_parquet: Path
    start_date: str
    end_date: str
    horizons_mins: list[int]
    drawdown_thresholds: list[float]


def _log(message: str) -> None:
    print(f"[info] {message}")


def _utc_day_bounds_ms(start_date: str, end_date: str) -> tuple[int, int]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(
        milliseconds=1
    )
    return int(start_ts.value // 1_000_000), int(end_ts.value // 1_000_000)


def _parse_int_list(raw: str) -> list[int]:
    out = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one integer value.")
    if any(v <= 0 for v in out):
        raise ValueError("All horizons must be positive integers.")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one float value.")
    if any(v <= 0 or v >= 1 for v in out):
        raise ValueError("Drawdown thresholds must be fractions in (0, 1).")
    return out


def _future_min_close_excluding_now(close: pd.Series, horizon: int) -> pd.Series:
    # Use reversed rolling min to compute min over (t, t+horizon].
    shifted = close.shift(-1)
    future_min = shifted.iloc[::-1].rolling(window=horizon, min_periods=horizon).min().iloc[
        ::-1
    ]
    return future_min.astype("float64")


def _label_col_name(threshold: float, horizon: int) -> str:
    pct = int(round(threshold * 100))
    return f"y_dd_{pct}p_H{horizon}m"


def build_labels_drawdown(config: LabelBuildConfig) -> pd.DataFrame:
    start_ms, end_ms = _utc_day_bounds_ms(config.start_date, config.end_date)

    df = pd.read_parquet(config.features_parquet)
    if df.empty:
        raise ValueError(f"No rows in features parquet: {config.features_parquet}")

    required = {"open_time", "close", "symbol"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Features parquet missing required columns: {missing}")

    work = df.copy()
    if pd.api.types.is_datetime64_any_dtype(work["open_time"]):
        work["open_time"] = (
            pd.to_datetime(work["open_time"], utc=True, errors="coerce").view("int64")
            // 1_000_000
        )
    work["open_time"] = pd.to_numeric(work["open_time"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["open_time"]).copy()
    work["open_time"] = work["open_time"].astype("int64")
    work["close"] = pd.to_numeric(work["close"], errors="coerce").astype("float64")
    work = work.dropna(subset=["close"]).copy()

    work = work[(work["open_time"] >= start_ms) & (work["open_time"] <= end_ms)].copy()
    work = work.sort_values("open_time", kind="mergesort").reset_index(drop=True)

    if work.empty:
        raise ValueError(
            f"No feature rows in date range [{config.start_date}, {config.end_date}]"
        )

    close = work["close"].astype("float64")
    cont_cols: list[str] = []
    label_cols: list[str] = []

    for horizon in config.horizons_mins:
        future_min_col = f"future_min_close_H{horizon}m"
        dd_col = f"dd_min_H{horizon}m"

        future_min = _future_min_close_excluding_now(close=close, horizon=horizon)
        drawdown = (future_min / close) - 1.0

        work[future_min_col] = future_min.astype("float64")
        work[dd_col] = drawdown.astype("float64")
        cont_cols.extend([future_min_col, dd_col])

        for threshold in config.drawdown_thresholds:
            label_col = _label_col_name(threshold=threshold, horizon=horizon)
            label = np.where(drawdown.isna(), np.nan, (drawdown <= -threshold).astype("float64"))
            work[label_col] = pd.Series(label, index=work.index, dtype="float64")
            label_cols.append(label_col)

    work["symbol"] = work["symbol"].astype("string").fillna(config.symbol)
    work["open_time"] = work["open_time"].astype("int64")

    config.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(work, preserve_index=False), config.out_parquet)

    meta_path = config.out_parquet.with_suffix(".meta.json")
    meta: dict[str, Any] = {
        "symbol": config.symbol,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "horizons_mins": config.horizons_mins,
        "drawdown_thresholds": config.drawdown_thresholds,
        "label_columns": label_cols,
        "continuous_target_columns": cont_cols,
        "tail_behavior": "Last H minutes per horizon are NaN due to insufficient future window.",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file_paths": {"features_parquet": str(config.features_parquet)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(
        f"DONE rows={len(work):,}, label_cols={len(label_cols)}, "
        f"continuous_cols={len(cont_cols)}, out_parquet={config.out_parquet}, meta_json={meta_path}"
    )
    return work


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build drawdown event labels from 1-minute feature parquet."
    )
    parser.add_argument("--symbol", required=True, choices=ALLOWED_SYMBOLS)
    parser.add_argument("--features_parquet", required=True)
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--start_date", required=True, help="UTC date, e.g. 2024-10-01")
    parser.add_argument("--end_date", required=True, help="UTC date, e.g. 2025-06-30")
    parser.add_argument(
        "--horizons_mins",
        required=True,
        help="Comma-separated horizons in minutes, e.g. 240,1440",
    )
    parser.add_argument(
        "--drawdown_thresholds",
        required=True,
        help="Comma-separated drawdown thresholds in fractions, e.g. 0.30,0.50",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabelBuildConfig(
        symbol=args.symbol,
        features_parquet=Path(args.features_parquet),
        out_parquet=Path(args.out_parquet),
        start_date=args.start_date,
        end_date=args.end_date,
        horizons_mins=_parse_int_list(args.horizons_mins),
        drawdown_thresholds=_parse_float_list(args.drawdown_thresholds),
    )
    build_labels_drawdown(config)


if __name__ == "__main__":
    main()
