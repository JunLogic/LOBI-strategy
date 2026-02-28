from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ALLOWED_SYMBOLS = ("PEPEUSDT", "FLOKIUSDT")
MINUTE_MS = 60_000
EPS = 1e-12

TRADE_AGG_COLUMNS = [
    "trade_vol",
    "trade_dollar_vol",
    "n_trades",
    "buy_vol",
    "sell_vol",
]

DERIVED_FEATURE_COLUMNS = [
    "ret_1m",
    "rv_5m",
    "rv_30m",
    "rv_4h",
    "spread_proxy",
    "imbalance",
    "buyer_aggressor_ratio",
    "vol_z_30m",
    "peak_close_4h",
    "drawdown_4h",
]

OUTPUT_COLUMNS = [
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    *TRADE_AGG_COLUMNS,
    *DERIVED_FEATURE_COLUMNS,
]


@dataclass(frozen=True)
class BuildConfig:
    symbol: str
    klines_parquet: Path
    trades_parquet: Path
    out_parquet: Path
    start_date: str
    end_date: str


def _log(message: str) -> None:
    print(f"[info] {message}")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _utc_day_bounds_ms(start_date: str, end_date: str) -> tuple[int, int]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(
        milliseconds=1
    )
    return int(start_ts.value // 1_000_000), int(end_ts.value // 1_000_000)


def _normalize_open_time_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000).astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def load_klines_1m(config: BuildConfig, start_ms: int, end_ms: int) -> pd.DataFrame:
    df = pd.read_parquet(config.klines_parquet)
    if df.empty:
        raise ValueError(f"No rows in kline parquet: {config.klines_parquet}")

    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Kline parquet missing required columns: {missing}")

    if "symbol" not in df.columns:
        df["symbol"] = config.symbol

    work = df[["open_time", "open", "high", "low", "close", "volume", "symbol"]].copy()
    work["open_time"] = _normalize_open_time_ms(work["open_time"])
    work = work.dropna(subset=["open_time"]).copy()
    work["open_time"] = work["open_time"].astype("int64")

    for col in ("open", "high", "low", "close", "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("float64")

    work["symbol"] = work["symbol"].astype("string").fillna(config.symbol)
    work = work[(work["open_time"] >= start_ms) & (work["open_time"] <= end_ms)].copy()
    work = work.sort_values("open_time", kind="mergesort")
    work = work.drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)

    if work.empty:
        raise ValueError(
            f"No kline rows in date range [{config.start_date}, {config.end_date}]"
        )

    diffs = work["open_time"].diff().iloc[1:]
    bad_diffs = diffs[diffs != MINUTE_MS]
    if not bad_diffs.empty:
        first_bad_idx = int(bad_diffs.index[0])
        before_ts = int(work.loc[first_bad_idx - 1, "open_time"])
        after_ts = int(work.loc[first_bad_idx, "open_time"])
        raise ValueError(
            "Kline open_time is not a strict 1-minute grid after filtering; "
            f"first violation between {before_ts} and {after_ts}"
        )

    return work


def _aggregate_trade_batch(batch: pa.RecordBatch) -> pd.DataFrame:
    if batch.num_rows == 0:
        return pd.DataFrame(
            columns=["minute", "trade_vol", "trade_dollar_vol", "n_trades", "buy_vol", "sell_vol"]
        )

    batch_df = pa.Table.from_batches([batch]).to_pandas(use_threads=False)
    if batch_df.empty:
        return pd.DataFrame(
            columns=["minute", "trade_vol", "trade_dollar_vol", "n_trades", "buy_vol", "sell_vol"]
        )

    batch_df["time"] = pd.to_numeric(batch_df["time"], errors="coerce")
    batch_df["price"] = pd.to_numeric(batch_df["price"], errors="coerce")
    batch_df["qty"] = pd.to_numeric(batch_df["qty"], errors="coerce")
    batch_df = batch_df.dropna(subset=["time", "price", "qty", "isBuyerMaker"]).copy()
    if batch_df.empty:
        return pd.DataFrame(
            columns=["minute", "trade_vol", "trade_dollar_vol", "n_trades", "buy_vol", "sell_vol"]
        )

    batch_df["time"] = batch_df["time"].astype("int64")
    batch_df["minute"] = (batch_df["time"] // MINUTE_MS) * MINUTE_MS
    batch_df["qty"] = batch_df["qty"].astype("float64")
    batch_df["dollar"] = batch_df["price"].astype("float64") * batch_df["qty"]
    buyer_maker = batch_df["isBuyerMaker"].astype(bool)
    batch_df["buy_qty"] = np.where(~buyer_maker, batch_df["qty"], 0.0)
    batch_df["sell_qty"] = np.where(buyer_maker, batch_df["qty"], 0.0)

    grouped = (
        batch_df.groupby("minute", sort=False, as_index=False)
        .agg(
            trade_vol=("qty", "sum"),
            trade_dollar_vol=("dollar", "sum"),
            n_trades=("minute", "size"),
            buy_vol=("buy_qty", "sum"),
            sell_vol=("sell_qty", "sum"),
        )
        .astype(
            {
                "minute": "int64",
                "trade_vol": "float64",
                "trade_dollar_vol": "float64",
                "n_trades": "int64",
                "buy_vol": "float64",
                "sell_vol": "float64",
            }
        )
    )
    return grouped


def _write_partial_aggregates(partial_df: pd.DataFrame, tmp_root: Path) -> int:
    if partial_df.empty:
        return 0

    write_df = partial_df.copy()
    write_df["date"] = pd.to_datetime(write_df["minute"], unit="ms", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    table = pa.Table.from_pandas(write_df, preserve_index=False)
    pq.write_to_dataset(table, root_path=str(tmp_root), partition_cols=["date"])
    return len(write_df)


def _reduce_partial_dataset(tmp_root: Path) -> pd.DataFrame:
    if not tmp_root.exists() or not any(tmp_root.rglob("*.parquet")):
        return pd.DataFrame(columns=["open_time", *TRADE_AGG_COLUMNS])

    partial_ds = ds.dataset(tmp_root, format="parquet", partitioning="hive")
    scanner = partial_ds.scanner(
        columns=["minute", "trade_vol", "trade_dollar_vol", "n_trades", "buy_vol", "sell_vol"],
        use_threads=True,
        batch_size=250_000,
    )

    reduced: dict[int, list[float]] = {}
    for batch in scanner.to_batches():
        batch_df = pa.Table.from_batches([batch]).to_pandas(use_threads=False)
        if batch_df.empty:
            continue

        grouped = (
            batch_df.groupby("minute", sort=False, as_index=False)
            .agg(
                trade_vol=("trade_vol", "sum"),
                trade_dollar_vol=("trade_dollar_vol", "sum"),
                n_trades=("n_trades", "sum"),
                buy_vol=("buy_vol", "sum"),
                sell_vol=("sell_vol", "sum"),
            )
            .astype(
                {
                    "minute": "int64",
                    "trade_vol": "float64",
                    "trade_dollar_vol": "float64",
                    "n_trades": "int64",
                    "buy_vol": "float64",
                    "sell_vol": "float64",
                }
            )
        )

        for row in grouped.itertuples(index=False):
            key = int(row.minute)
            current = reduced.get(key)
            if current is None:
                reduced[key] = [
                    float(row.trade_vol),
                    float(row.trade_dollar_vol),
                    int(row.n_trades),
                    float(row.buy_vol),
                    float(row.sell_vol),
                ]
            else:
                current[0] += float(row.trade_vol)
                current[1] += float(row.trade_dollar_vol)
                current[2] += int(row.n_trades)
                current[3] += float(row.buy_vol)
                current[4] += float(row.sell_vol)

    if not reduced:
        return pd.DataFrame(columns=["open_time", *TRADE_AGG_COLUMNS])

    rows = [
        (minute, values[0], values[1], values[2], values[3], values[4])
        for minute, values in reduced.items()
    ]
    out = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "trade_vol",
            "trade_dollar_vol",
            "n_trades",
            "buy_vol",
            "sell_vol",
        ],
    )
    out["open_time"] = out["open_time"].astype("int64")
    out["trade_vol"] = out["trade_vol"].astype("float64")
    out["trade_dollar_vol"] = out["trade_dollar_vol"].astype("float64")
    out["n_trades"] = out["n_trades"].astype("int64")
    out["buy_vol"] = out["buy_vol"].astype("float64")
    out["sell_vol"] = out["sell_vol"].astype("float64")
    return out


def aggregate_trades_to_1m(config: BuildConfig, start_ms: int, end_ms: int) -> pd.DataFrame:
    trades_ds = ds.dataset(config.trades_parquet, format="parquet")
    scanner = trades_ds.scanner(
        columns=["time", "price", "qty", "isBuyerMaker"],
        filter=(ds.field("time") >= start_ms) & (ds.field("time") <= end_ms),
        use_threads=True,
        batch_size=250_000,
    )

    started = time.monotonic()
    batches = 0
    rows_scanned = 0
    partial_rows_written = 0
    seen_minutes: set[int] = set()
    log_every = 20

    with tempfile.TemporaryDirectory(
        prefix=f"features_agg_{config.symbol}_",
        dir=str(config.out_parquet.parent),
    ) as tmp_dir:
        tmp_root = Path(tmp_dir) / "partials"
        tmp_root.mkdir(parents=True, exist_ok=True)

        for batch in scanner.to_batches():
            batches += 1
            rows_scanned += batch.num_rows

            partial = _aggregate_trade_batch(batch)
            if not partial.empty:
                seen_minutes.update(partial["minute"].tolist())
                partial_rows_written += _write_partial_aggregates(partial, tmp_root)

            if batches % log_every == 0:
                elapsed = time.monotonic() - started
                _log(
                    "Trades scan "
                    f"batches={batches:,}, rows_scanned={rows_scanned:,}, "
                    f"elapsed={_format_elapsed(elapsed)}, unique_minutes={len(seen_minutes):,}, "
                    f"flushed_partial_rows={partial_rows_written:,}"
                )

        elapsed = time.monotonic() - started
        _log(
            "Trades scan complete "
            f"batches={batches:,}, rows_scanned={rows_scanned:,}, "
            f"elapsed={_format_elapsed(elapsed)}, unique_minutes={len(seen_minutes):,}, "
            f"flushed_partial_rows={partial_rows_written:,}"
        )

        reduced = _reduce_partial_dataset(tmp_root)

    return reduced


def merge_and_build_features(klines_df: pd.DataFrame, trade_agg_df: pd.DataFrame) -> pd.DataFrame:
    merged = klines_df.merge(trade_agg_df, how="left", on="open_time", sort=False)
    merged = merged.sort_values("open_time", kind="mergesort").reset_index(drop=True)

    for col in ("trade_vol", "trade_dollar_vol", "buy_vol", "sell_vol"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype("float64")
    merged["n_trades"] = (
        pd.to_numeric(merged["n_trades"], errors="coerce").fillna(0).astype("int64")
    )

    merged["ret_1m"] = np.log(merged["close"]).diff()
    merged["rv_5m"] = merged["ret_1m"].rolling(window=5, min_periods=5).std(ddof=0)
    merged["rv_30m"] = merged["ret_1m"].rolling(window=30, min_periods=30).std(ddof=0)
    merged["rv_4h"] = merged["ret_1m"].rolling(window=240, min_periods=240).std(ddof=0)
    merged["spread_proxy"] = (merged["high"] - merged["low"]) / merged["close"]
    merged["imbalance"] = (merged["buy_vol"] - merged["sell_vol"]) / (
        merged["buy_vol"] + merged["sell_vol"] + EPS
    )
    merged["buyer_aggressor_ratio"] = merged["buy_vol"] / (merged["trade_vol"] + EPS)

    vol_mean_30 = merged["trade_dollar_vol"].rolling(window=30, min_periods=30).mean()
    vol_std_30 = merged["trade_dollar_vol"].rolling(window=30, min_periods=30).std(ddof=0)
    merged["vol_z_30m"] = (merged["trade_dollar_vol"] - vol_mean_30) / (vol_std_30 + EPS)

    merged["peak_close_4h"] = merged["close"].rolling(window=240, min_periods=1).max()
    merged["drawdown_4h"] = merged["close"] / merged["peak_close_4h"] - 1.0

    merged["symbol"] = merged["symbol"].astype("string")
    merged["open_time"] = merged["open_time"].astype("int64")
    merged["n_trades"] = merged["n_trades"].astype("int64")

    for col in merged.columns:
        if col in {"symbol", "open_time", "n_trades"}:
            continue
        merged[col] = merged[col].astype("float64")

    return merged[OUTPUT_COLUMNS]


def _write_meta_json(config: BuildConfig, out_df: pd.DataFrame) -> Path:
    meta_path = config.out_parquet.with_suffix(".meta.json")
    payload: dict[str, Any] = {
        "symbol": config.symbol,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "windows_used": {
            "ret_1m_diff": 1,
            "rv_minutes": [5, 30, 240],
            "vol_z_30m": 30,
            "peak_close_4h": 240,
        },
        "feature_columns": [c for c in out_df.columns if c not in {"symbol", "open_time"}],
        "build_timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_file_paths": {
            "klines_parquet": str(config.klines_parquet),
            "trades_parquet": str(config.trades_parquet),
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def build_features_1m(config: BuildConfig) -> pd.DataFrame:
    start_ms, end_ms = _utc_day_bounds_ms(config.start_date, config.end_date)
    klines_df = load_klines_1m(config, start_ms=start_ms, end_ms=end_ms)
    trade_agg_df = aggregate_trades_to_1m(config, start_ms=start_ms, end_ms=end_ms)
    out_df = merge_and_build_features(klines_df, trade_agg_df)

    config.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    pq.write_table(table, config.out_parquet)
    meta_path = _write_meta_json(config, out_df)

    out_size_mb = config.out_parquet.stat().st_size / (1024**2)
    span_start = int(out_df["open_time"].iloc[0])
    span_end = int(out_df["open_time"].iloc[-1])
    _log(
        "DONE "
        f"rows={len(out_df):,}, open_time_span_ms=[{span_start}, {span_end}], "
        f"out_size_mb={out_size_mb:.3f}, out_parquet={config.out_parquet}, meta_json={meta_path}"
    )
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1-minute features from klines + trades.")
    parser.add_argument("--symbol", required=True, choices=ALLOWED_SYMBOLS)
    parser.add_argument("--klines_parquet", required=True)
    parser.add_argument("--trades_parquet", required=True)
    parser.add_argument("--out_parquet", required=True)
    parser.add_argument("--start_date", required=True, help="UTC date, e.g. 2024-10-01")
    parser.add_argument("--end_date", required=True, help="UTC date, e.g. 2025-06-30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        symbol=args.symbol,
        klines_parquet=Path(args.klines_parquet),
        trades_parquet=Path(args.trades_parquet),
        out_parquet=Path(args.out_parquet),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    build_features_1m(config)


if __name__ == "__main__":
    main()
