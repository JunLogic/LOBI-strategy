from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import MBarsConfig

BAR_MS = 5_000
BAR_SCHEMA = pa.schema(
    [
        ("bar_time_ms", pa.int64()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("dollar_volume", pa.float64()),
        ("trade_count", pa.int64()),
        ("buy_volume", pa.float64()),
        ("sell_volume", pa.float64()),
        ("return_5s", pa.float64()),
    ]
)


def _default_input_path(symbol: str) -> Path:
    return Path("data") / "processed" / f"{symbol}_trades.parquet"


def _default_output_path(symbol: str) -> Path:
    return Path("data") / "processed" / f"{symbol}_bars_5s.parquet"


def _new_buffer() -> dict[str, list[Any]]:
    return {name: [] for name in BAR_SCHEMA.names}


def _clear_buffer(buffer: dict[str, list[Any]]) -> None:
    for name in BAR_SCHEMA.names:
        buffer[name].clear()


def _append_bar_to_buffer(buffer: dict[str, list[Any]], bar: dict[str, Any]) -> None:
    open_px = float(bar["open"])
    return_5s = (float(bar["close"]) / open_px - 1.0) if open_px != 0.0 else 0.0
    buffer["bar_time_ms"].append(int(bar["bar_time_ms"]))
    buffer["open"].append(open_px)
    buffer["high"].append(float(bar["high"]))
    buffer["low"].append(float(bar["low"]))
    buffer["close"].append(float(bar["close"]))
    buffer["volume"].append(float(bar["volume"]))
    buffer["dollar_volume"].append(float(bar["dollar_volume"]))
    buffer["trade_count"].append(int(bar["trade_count"]))
    buffer["buy_volume"].append(float(bar["buy_volume"]))
    buffer["sell_volume"].append(float(bar["sell_volume"]))
    buffer["return_5s"].append(return_5s)


def _flush_bar_buffer(
    writer: pq.ParquetWriter,
    buffer: dict[str, list[Any]],
) -> int:
    row_count = len(buffer["bar_time_ms"])
    if row_count == 0:
        return 0

    table = pa.table(
        {
            "bar_time_ms": pa.array(buffer["bar_time_ms"], type=pa.int64()),
            "open": pa.array(buffer["open"], type=pa.float64()),
            "high": pa.array(buffer["high"], type=pa.float64()),
            "low": pa.array(buffer["low"], type=pa.float64()),
            "close": pa.array(buffer["close"], type=pa.float64()),
            "volume": pa.array(buffer["volume"], type=pa.float64()),
            "dollar_volume": pa.array(buffer["dollar_volume"], type=pa.float64()),
            "trade_count": pa.array(buffer["trade_count"], type=pa.int64()),
            "buy_volume": pa.array(buffer["buy_volume"], type=pa.float64()),
            "sell_volume": pa.array(buffer["sell_volume"], type=pa.float64()),
            "return_5s": pa.array(buffer["return_5s"], type=pa.float64()),
        },
        schema=BAR_SCHEMA,
    )
    writer.write_table(table)
    _clear_buffer(buffer)
    return row_count


def _start_bar(bar_time_ms: int, price: float, qty: float, is_buyer_maker: bool) -> dict[str, Any]:
    if is_buyer_maker:
        buy_volume = 0.0
        sell_volume = qty
    else:
        buy_volume = qty
        sell_volume = 0.0

    return {
        "bar_time_ms": bar_time_ms,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": qty,
        "dollar_volume": price * qty,
        "trade_count": 1,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
    }


def _update_bar(bar: dict[str, Any], price: float, qty: float, is_buyer_maker: bool) -> None:
    if price > bar["high"]:
        bar["high"] = price
    if price < bar["low"]:
        bar["low"] = price
    bar["close"] = price
    bar["volume"] += qty
    bar["dollar_volume"] += price * qty
    bar["trade_count"] += 1
    if is_buyer_maker:
        bar["sell_volume"] += qty
    else:
        bar["buy_volume"] += qty


def build_5s_bars_from_trades(
    symbol: str,
    in_path: str | Path | None = None,
    out_path: str | Path | None = None,
    batch_rows: int = 250_000,
    flush_bars: int = 200_000,
    log_every: int = 20,
) -> Path:
    """Build 5-second bars from trades parquet using a single-pass streaming accumulator."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    if flush_bars <= 0:
        raise ValueError("flush_bars must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    in_parquet = Path(in_path) if in_path is not None else _default_input_path(symbol)
    out_parquet = Path(out_path) if out_path is not None else _default_output_path(symbol)

    if not in_parquet.exists():
        raise FileNotFoundError(f"Input trades parquet not found: {in_parquet}")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    scan_columns = ["time", "price", "qty", "isBuyerMaker"]
    trades_ds = ds.dataset(in_parquet, format="parquet")
    missing = [name for name in scan_columns if name not in trades_ds.schema.names]
    if missing:
        raise ValueError(f"Trades parquet missing required columns: {missing}")

    start_ts = time.perf_counter()
    rows_scanned = 0
    batches_processed = 0
    bars_written = 0
    current_bar: dict[str, Any] | None = None
    bar_buffer = _new_buffer()

    print(f"[bars] symbol={symbol} input={in_parquet} output={out_parquet}")
    print(
        f"[bars] config: batch_rows={batch_rows} flush_bars={flush_bars} "
        f"log_every={log_every} bar_ms={BAR_MS}"
    )

    scanner = trades_ds.scanner(
        columns=scan_columns,
        batch_size=batch_rows,
        use_threads=True,
    )

    writer = pq.ParquetWriter(out_parquet, BAR_SCHEMA, compression="zstd")
    try:
        for batch in scanner.to_batches():
            batches_processed += 1
            rows_scanned += batch.num_rows
            if batch.num_rows == 0:
                continue

            times = batch.column(0).to_numpy(zero_copy_only=False)
            prices = batch.column(1).to_numpy(zero_copy_only=False)
            qtys = batch.column(2).to_numpy(zero_copy_only=False)
            makers = batch.column(3).to_numpy(zero_copy_only=False)

            for i in range(batch.num_rows):
                raw_time = times[i]
                raw_price = prices[i]
                raw_qty = qtys[i]
                raw_maker = makers[i]
                if (
                    raw_time is None
                    or raw_price is None
                    or raw_qty is None
                    or raw_maker is None
                ):
                    continue

                trade_time = int(raw_time)
                price = float(raw_price)
                qty = float(raw_qty)
                is_buyer_maker = bool(raw_maker)
                bar_time_ms = (trade_time // BAR_MS) * BAR_MS

                if current_bar is None:
                    current_bar = _start_bar(bar_time_ms, price, qty, is_buyer_maker)
                    continue

                if bar_time_ms != current_bar["bar_time_ms"]:
                    _append_bar_to_buffer(bar_buffer, current_bar)
                    if len(bar_buffer["bar_time_ms"]) >= flush_bars:
                        bars_written += _flush_bar_buffer(writer, bar_buffer)
                    current_bar = _start_bar(bar_time_ms, price, qty, is_buyer_maker)
                else:
                    _update_bar(current_bar, price, qty, is_buyer_maker)

            if batches_processed % log_every == 0:
                elapsed = time.perf_counter() - start_ts
                print(
                    f"[bars] progress batches={batches_processed} rows={rows_scanned} "
                    f"bars_written={bars_written} elapsed_s={elapsed:.2f}"
                )

        if current_bar is not None:
            _append_bar_to_buffer(bar_buffer, current_bar)
            current_bar = None

        bars_written += _flush_bar_buffer(writer, bar_buffer)
    finally:
        writer.close()

    elapsed = time.perf_counter() - start_ts
    print(
        f"[bars] done batches={batches_processed} rows={rows_scanned} "
        f"bars_written={bars_written} elapsed_s={elapsed:.2f}"
    )
    return out_parquet


def build_mbars(config: MBarsConfig) -> None:
    """Placeholder bar-builder entrypoint for the future MBARS module."""

    _ = config
