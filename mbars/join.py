from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

MINUTE_MS = 60_000

_TS_NAME_PRIORITY = (
    "minute_time_ms",
    "open_time",
    "timestamp",
    "time",
    "ts",
    "datetime",
    "date",
)

_PROB_NAME_PRIORITY = (
    "hazard_prob",
    "y_prob",
    "probability",
    "prob",
    "score",
    "p",
)


def _priority_rank(name: str, ordered_names: tuple[str, ...]) -> int:
    lowered = name.strip().lower()
    try:
        return ordered_names.index(lowered)
    except ValueError:
        return len(ordered_names) + 100


def _to_minute_ms(values: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray, str]:
    arr = values.combine_chunks()
    if arr.type == pa.null():
        return np.zeros(len(arr), dtype=np.int64), np.zeros(len(arr), dtype=bool), "null"

    if pa.types.is_timestamp(arr.type):
        ts_ms = pc.cast(arr, pa.timestamp("ms"))
        epoch_ms = pc.cast(ts_ms, pa.int64()).to_numpy(zero_copy_only=False)
        valid = arr.is_valid().to_numpy(zero_copy_only=False)
        minute_ms = (epoch_ms // MINUTE_MS) * MINUTE_MS
        return minute_ms.astype(np.int64, copy=False), valid.astype(bool, copy=False), "timestamp"

    if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type) or pa.types.is_decimal(arr.type):
        numeric_vals = pc.cast(arr, pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        valid = np.isfinite(numeric_vals)
        if not np.any(valid):
            return np.zeros(len(arr), dtype=np.int64), valid, "numeric-invalid"

        max_abs = float(np.max(np.abs(numeric_vals[valid])))
        if max_abs < 1.0e11:
            epoch_ms = np.floor(numeric_vals * 1000.0).astype(np.int64, copy=False)
            mode = "numeric-seconds"
        else:
            epoch_ms = np.floor(numeric_vals).astype(np.int64, copy=False)
            mode = "numeric-milliseconds"

        minute_ms = (epoch_ms // MINUTE_MS) * MINUTE_MS
        return minute_ms, valid, mode

    as_text = pd.Series(arr.to_pylist(), dtype="object")
    parsed = pd.to_datetime(as_text, utc=True, errors="coerce")
    valid = parsed.notna().to_numpy()
    if not np.any(valid):
        return np.zeros(len(arr), dtype=np.int64), valid, "string-invalid"

    epoch_ns = parsed.astype("int64", copy=False).to_numpy()
    epoch_ms = epoch_ns // 1_000_000
    minute_ms = (epoch_ms // MINUTE_MS) * MINUTE_MS
    return minute_ms.astype(np.int64, copy=False), valid.astype(bool, copy=False), "string-datetime"


def _to_float64(values: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    arr = values.combine_chunks()
    if arr.type == pa.null():
        valid = np.zeros(len(arr), dtype=bool)
        return np.full(len(arr), np.nan, dtype=np.float64), valid

    if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type) or pa.types.is_decimal(arr.type):
        as_float = pc.cast(arr, pa.float64(), safe=False).to_numpy(zero_copy_only=False)
        valid = np.isfinite(as_float)
        return as_float.astype(np.float64, copy=False), valid

    as_text = pd.Series(arr.to_pylist(), dtype="object")
    as_float = pd.to_numeric(as_text, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(as_float)
    return as_float, valid


def _detect_timestamp_column(hazard_table: pa.Table) -> tuple[str, np.ndarray, np.ndarray, str]:
    all_names = hazard_table.schema.names
    time_like_names = [
        name
        for name in all_names
        if any(token in name.lower() for token in ("time", "timestamp", "date", "datetime", "ts"))
    ]
    candidates = time_like_names if time_like_names else all_names

    best: tuple[int, int, str] | None = None
    best_payload: tuple[str, np.ndarray, np.ndarray, str] | None = None

    for name in candidates:
        minute_ms, valid, mode = _to_minute_ms(hazard_table[name])
        valid_count = int(valid.sum())
        if valid_count == 0:
            continue

        rank = _priority_rank(name, _TS_NAME_PRIORITY)
        score = (rank, -valid_count, name)
        if best is None or score < best:
            best = score
            best_payload = (name, minute_ms, valid, mode)

    if best_payload is None:
        raise ValueError(
            "Could not detect hazard timestamp column from CSV. "
            f"Available columns: {all_names}"
        )
    return best_payload


def _detect_probability_column(hazard_table: pa.Table, timestamp_col: str) -> tuple[str, np.ndarray, np.ndarray]:
    all_names = hazard_table.schema.names

    preferred = [
        name
        for name in all_names
        if name != timestamp_col and any(token in name.lower() for token in ("prob", "hazard"))
    ]
    fallback = [
        name
        for name in all_names
        if name != timestamp_col and not any(token in name.lower() for token in ("true", "label", "target"))
    ]
    candidates = preferred if preferred else fallback

    best: tuple[int, int, int, str] | None = None
    best_payload: tuple[str, np.ndarray, np.ndarray] | None = None

    for name in candidates:
        values, valid = _to_float64(hazard_table[name])
        valid_count = int(valid.sum())
        if valid_count == 0:
            continue

        valid_values = values[valid]
        in_unit_interval = int(np.logical_and(valid_values >= 0.0, valid_values <= 1.0).sum())
        rank = _priority_rank(name, _PROB_NAME_PRIORITY)

        score = (rank, -in_unit_interval, -valid_count, name)
        if best is None or score < best:
            best = score
            best_payload = (name, values, valid)

    if best_payload is None:
        raise ValueError(
            "Could not detect hazard probability column from CSV. "
            f"Available columns: {all_names}"
        )
    return best_payload


def _build_hazard_lookup(
    hazard_table: pa.Table,
    timestamp_col: str,
    minute_ms: np.ndarray,
    ts_valid: np.ndarray,
    prob_col: str,
    prob_vals: np.ndarray,
    prob_valid: np.ndarray,
) -> tuple[pa.Array, pa.Array, dict[str, Any]]:
    valid_rows = ts_valid & prob_valid
    valid_count = int(valid_rows.sum())
    if valid_count == 0:
        raise ValueError(
            "No valid hazard rows after parsing timestamp/probability columns: "
            f"timestamp_col={timestamp_col} prob_col={prob_col}"
        )

    minutes = minute_ms[valid_rows]
    probs = prob_vals[valid_rows]

    by_minute: dict[int, float] = {}
    for minute, prob in zip(minutes, probs):
        by_minute[int(minute)] = float(prob)

    sorted_minutes = np.fromiter(by_minute.keys(), dtype=np.int64)
    order = np.argsort(sorted_minutes)
    sorted_minutes = sorted_minutes[order]
    sorted_probs = np.array([by_minute[int(m)] for m in sorted_minutes], dtype=np.float64)

    lookup_minutes = pa.array(sorted_minutes, type=pa.int64())
    lookup_probs = pa.array(sorted_probs, type=pa.float64())

    info: dict[str, Any] = {
        "hazard_rows_total": hazard_table.num_rows,
        "hazard_rows_valid": valid_count,
        "hazard_unique_minutes": len(by_minute),
    }
    return lookup_minutes, lookup_probs, info


def join_hazard_to_5s(
    symbol: str,
    bars_5s_path: Path,
    hazard_csv_path: Path,
    out_path: Path,
    batch_rows: int = 500_000,
    log_every: int = 20,
) -> dict[str, Any]:
    if not symbol:
        raise ValueError("symbol must be non-empty.")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    bars_5s_path = Path(bars_5s_path)
    hazard_csv_path = Path(hazard_csv_path)
    out_path = Path(out_path)

    if not bars_5s_path.exists():
        raise FileNotFoundError(f"5s bars parquet not found: {bars_5s_path}")
    if not hazard_csv_path.exists():
        raise FileNotFoundError(f"Hazard probability CSV not found: {hazard_csv_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = time.perf_counter()
    print(f"[join] symbol={symbol} bars={bars_5s_path} hazard={hazard_csv_path}")
    print(f"[join] output={out_path} batch_rows={batch_rows} log_every={log_every}")

    hazard_table = pacsv.read_csv(hazard_csv_path)

    timestamp_col, minute_ms, ts_valid, ts_mode = _detect_timestamp_column(hazard_table)
    prob_col, prob_vals, prob_valid = _detect_probability_column(hazard_table, timestamp_col)

    print(
        f"[join] hazard columns detected: timestamp_col={timestamp_col} "
        f"prob_col={prob_col} timestamp_mode={ts_mode}"
    )

    hazard_minutes, hazard_probs, hazard_info = _build_hazard_lookup(
        hazard_table=hazard_table,
        timestamp_col=timestamp_col,
        minute_ms=minute_ms,
        ts_valid=ts_valid,
        prob_col=prob_col,
        prob_vals=prob_vals,
        prob_valid=prob_valid,
    )
    print(
        "[join] hazard lookup built: "
        f"rows_total={hazard_info['hazard_rows_total']} "
        f"rows_valid={hazard_info['hazard_rows_valid']} "
        f"unique_minutes={hazard_info['hazard_unique_minutes']}"
    )

    bars_ds = ds.dataset(bars_5s_path, format="parquet")
    if "bar_time_ms" not in bars_ds.schema.names:
        raise ValueError("Bars parquet missing required column: bar_time_ms")

    output_schema = bars_ds.schema.append(pa.field("minute_time_ms", pa.int64())).append(
        pa.field("hazard_prob", pa.float64())
    )

    scanner = bars_ds.scanner(
        columns=bars_ds.schema.names,
        batch_size=batch_rows,
        use_threads=True,
    )

    rows_scanned = 0
    rows_written = 0
    batches = 0
    missing_rows = 0
    missing_minutes: set[int] = set()
    bar_time_idx = bars_ds.schema.get_field_index("bar_time_ms")

    writer = pq.ParquetWriter(out_path, output_schema, compression="zstd")
    try:
        for batch in scanner.to_batches():
            batches += 1
            if batch.num_rows == 0:
                continue

            rows_scanned += batch.num_rows

            bar_time_col = pc.cast(batch.column(bar_time_idx), pa.int64(), safe=False)
            bar_time_vals = bar_time_col.to_numpy(zero_copy_only=False)
            minute_vals = (bar_time_vals // MINUTE_MS) * MINUTE_MS
            minute_time_ms = pa.array(minute_vals, type=pa.int64())

            lookup_idx = pc.index_in(minute_time_ms, value_set=hazard_minutes)
            hazard_prob = pc.take(hazard_probs, lookup_idx)
            hazard_prob = pc.cast(hazard_prob, pa.float64(), safe=False)

            missing_batch_rows = hazard_prob.null_count
            missing_rows += missing_batch_rows
            if missing_batch_rows > 0:
                missing_mask = pc.invert(pc.is_valid(hazard_prob))
                missing_minute_values = pc.unique(pc.filter(minute_time_ms, missing_mask))
                for value in missing_minute_values.to_pylist():
                    if value is not None:
                        missing_minutes.add(int(value))

            joined_batch = pa.RecordBatch.from_arrays(
                [batch.column(i) for i in range(batch.num_columns)]
                + [minute_time_ms, hazard_prob],
                names=output_schema.names,
            )
            writer.write_batch(joined_batch)
            rows_written += joined_batch.num_rows

            if batches % log_every == 0:
                elapsed = time.perf_counter() - start_ts
                print(
                    f"[join] progress batches={batches} rows_scanned={rows_scanned} "
                    f"rows_written={rows_written} missing_rows={missing_rows} elapsed_s={elapsed:.2f}"
                )
    finally:
        writer.close()

    elapsed = time.perf_counter() - start_ts
    missing_minutes_count = len(missing_minutes)

    print(
        f"[join] done batches={batches} rows_scanned={rows_scanned} rows_written={rows_written} "
        f"missing_rows={missing_rows} missing_minutes={missing_minutes_count} elapsed_s={elapsed:.2f}"
    )

    return {
        "symbol": symbol,
        "rows_scanned": rows_scanned,
        "rows_written": rows_written,
        "batches": batches,
        "missing_rows": missing_rows,
        "missing_minutes_count": missing_minutes_count,
        "elapsed_s": elapsed,
        "timestamp_col": timestamp_col,
        "prob_col": prob_col,
        "hazard_rows_total": hazard_info["hazard_rows_total"],
        "hazard_rows_valid": hazard_info["hazard_rows_valid"],
        "hazard_unique_minutes": hazard_info["hazard_unique_minutes"],
    }
