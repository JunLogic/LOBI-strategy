import re
import time
import zipfile
from collections import deque
from glob import glob
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SYMBOLS: List[str] = [
    "PEPEUSDT",
    "FLOKIUSDT",
]

TRADES_ONLY = True
TRADES_PROGRESS_EVERY_N_FILES = 5
TRADES_PROGRESS_ROLLING_WINDOW = 10

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

KLINE_OUTPUT_COLUMNS = [
    "open_time_raw",
    "open_time",
    "open_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_raw",
    "close_time",
    "close_time_utc",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

TRADE_OUTPUT_COLUMNS = [
    "time",
    "price",
    "qty",
    "isBuyerMaker",
    "trade_id",
]

TRADE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("time", pa.int64()),
        pa.field("price", pa.float64()),
        pa.field("qty", pa.float64()),
        pa.field("isBuyerMaker", pa.bool_()),
        pa.field("trade_id", pa.int64()),
    ]
)

_DATE_FROM_FILENAME_RE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")


def _log(message: str) -> None:
    print(f"[info] {message}")


def _warn(message: str) -> None:
    print(f"[warn] {message}")


def _format_hhmmss(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _file_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024**3)


def find_files(glob_patterns: Sequence[str]) -> List[Path]:
    seen = set()
    paths: List[Path] = []
    for pattern in glob_patterns:
        for raw_path in glob(pattern):
            path = Path(raw_path)
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths, key=lambda p: str(p))


def read_csv_or_zip(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_members = sorted(
                name for name in zf.namelist() if name.lower().endswith(".csv")
            )
            if not csv_members:
                raise ValueError(f"No CSV file found in ZIP archive: {path}")
            with zf.open(csv_members[0]) as handle:
                return pd.read_csv(handle, header=None, low_memory=False)
    if suffix == ".csv":
        return pd.read_csv(path, header=None, low_memory=False)
    raise ValueError(f"Unsupported file extension: {path}")


def _clean_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _as_bool_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "t": True,
            "f": False,
            "yes": True,
            "no": False,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_bool = numeric.map(lambda x: pd.NA if pd.isna(x) else bool(int(x)))
    out = mapped.where(mapped.notna(), numeric_bool)
    return out.astype("boolean")


def _normalize_epoch_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric.astype("Int64")

    abs_numeric = numeric.abs()
    ns_mask = abs_numeric >= 1e17
    us_mask = (abs_numeric >= 1e14) & (abs_numeric < 1e17)
    sec_mask = abs_numeric < 1e11

    out = numeric.copy()
    out.loc[ns_mask] = out.loc[ns_mask] // 1_000_000
    out.loc[us_mask] = out.loc[us_mask] // 1_000
    out.loc[sec_mask] = out.loc[sec_mask] * 1_000
    return out.round().astype("Int64")


def _header_like_first_row(df: pd.DataFrame, known_tokens: Iterable[str]) -> bool:
    if df.empty:
        return False
    first_row_clean = [_clean_name(v) for v in df.iloc[0].tolist()]
    return len(set(first_row_clean) & set(known_tokens)) >= 3


def normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KLINE_OUTPUT_COLUMNS)

    work = df.copy()
    kline_tokens = {
        "opentime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "closetime",
    }
    if _header_like_first_row(work, kline_tokens):
        work = work.iloc[1:].copy()

    col_count = min(work.shape[1], len(KLINE_COLUMNS))
    work = work.iloc[:, :col_count].copy()
    work.columns = KLINE_COLUMNS[:col_count]
    for missing_col in KLINE_COLUMNS[col_count:]:
        work[missing_col] = pd.NA

    open_raw = pd.to_numeric(work["open_time"], errors="coerce")
    close_raw = pd.to_numeric(work["close_time"], errors="coerce")
    work["open_time_raw"] = open_raw.astype("Int64")
    work["close_time_raw"] = close_raw.astype("Int64")

    work["open_time"] = _normalize_epoch_ms(work["open_time"])
    work["close_time"] = _normalize_epoch_ms(work["close_time"])
    work = work.dropna(subset=["open_time"]).copy()

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    int_cols = ["number_of_trades", "ignore"]
    for col in float_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in int_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")

    work["open_time_utc"] = pd.to_datetime(
        work["open_time"], unit="ms", utc=True, errors="coerce"
    )
    work["close_time_utc"] = pd.to_datetime(
        work["close_time"], unit="ms", utc=True, errors="coerce"
    )
    return work[KLINE_OUTPUT_COLUMNS]


def _default_trade_columns(col_count: int) -> List[str]:
    if col_count >= 7:
        base = [
            "tradeId",
            "price",
            "qty",
            "quoteQty",
            "time",
            "isBuyerMaker",
            "isBestMatch",
        ]
    elif col_count == 6:
        base = ["tradeId", "price", "qty", "time", "isBuyerMaker", "isBestMatch"]
    elif col_count == 5:
        base = ["tradeId", "price", "qty", "time", "isBuyerMaker"]
    elif col_count == 4:
        base = ["price", "qty", "time", "isBuyerMaker"]
    elif col_count == 3:
        base = ["price", "qty", "time"]
    else:
        base = [f"col_{i}" for i in range(col_count)]

    if col_count > len(base):
        extra = [f"extra_{i}" for i in range(col_count - len(base))]
        base.extend(extra)
    return base[:col_count]


def _pick_column(columns: Sequence[object], aliases: Sequence[str]) -> str | None:
    cleaned = {col: _clean_name(col) for col in columns}
    alias_list = [_clean_name(alias) for alias in aliases]
    for alias in alias_list:
        for col, clean_col in cleaned.items():
            if clean_col == alias:
                return str(col)
    return None


def _sort_trade_files(paths: Sequence[Path], default_day: int) -> List[Path]:
    def key(path: Path) -> tuple[int, int, int, str]:
        match = _DATE_FROM_FILENAME_RE.search(path.name)
        if match:
            day = int(match.group(3) or default_day)
            return (int(match.group(1)), int(match.group(2)), day, path.name.lower())
        return (9999, 12, 31, path.name.lower())

    return sorted(paths, key=key)


def _coerce_trade_chunk_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["time"] = (
        pd.to_numeric(df["time"], errors="coerce").astype("Int64")
        if "time" in df.columns
        else pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)
    )
    out["price"] = (
        pd.to_numeric(df["price"], errors="coerce").astype("float64")
        if "price" in df.columns
        else pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    )
    out["qty"] = (
        pd.to_numeric(df["qty"], errors="coerce").astype("float64")
        if "qty" in df.columns
        else pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")
    )
    out["isBuyerMaker"] = (
        _as_bool_series(df["isBuyerMaker"])
        if "isBuyerMaker" in df.columns
        else pd.Series(pd.array([pd.NA] * len(df), dtype="boolean"), index=df.index)
    )
    out["trade_id"] = (
        pd.to_numeric(df["trade_id"], errors="coerce").astype("Int64")
        if "trade_id" in df.columns
        else pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)
    )
    out = out.dropna(subset=["time"]).copy()
    return out[TRADE_OUTPUT_COLUMNS]


def normalize_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TRADE_OUTPUT_COLUMNS)

    work = df.copy()
    trade_tokens = {
        "id",
        "tradeid",
        "price",
        "qty",
        "quantity",
        "quoteqty",
        "time",
        "timestamp",
        "isbuyermaker",
    }
    if _header_like_first_row(work, trade_tokens):
        header = [str(v).strip() or f"col_{i}" for i, v in enumerate(work.iloc[0])]
        work = work.iloc[1:].copy()
        work.columns = header
    else:
        work.columns = _default_trade_columns(work.shape[1])

    if work.empty:
        return pd.DataFrame(columns=TRADE_OUTPUT_COLUMNS)

    id_col = _pick_column(work.columns, ["tradeId", "id", "a"])
    price_col = _pick_column(work.columns, ["price", "p"])
    qty_col = _pick_column(work.columns, ["qty", "quantity", "q"])
    time_col = _pick_column(work.columns, ["time", "timestamp", "transactTime", "t"])
    is_buyer_maker_col = _pick_column(work.columns, ["isBuyerMaker", "buyerMaker", "m"])

    out = pd.DataFrame(index=work.index)

    out["trade_id"] = (
        pd.to_numeric(work[id_col], errors="coerce").astype("Int64")
        if id_col is not None
        else pd.Series(pd.array([pd.NA] * len(work), dtype="Int64"), index=work.index)
    )
    out["price"] = (
        pd.to_numeric(work[price_col], errors="coerce")
        if price_col is not None
        else pd.Series([float("nan")] * len(work), index=work.index)
    )
    out["qty"] = (
        pd.to_numeric(work[qty_col], errors="coerce")
        if qty_col is not None
        else pd.Series([float("nan")] * len(work), index=work.index)
    )

    if time_col is not None:
        out["time"] = _normalize_epoch_ms(work[time_col])
    else:
        out["time"] = pd.Series(
            pd.array([pd.NA] * len(work), dtype="Int64"), index=work.index
        )

    out["isBuyerMaker"] = (
        _as_bool_series(work[is_buyer_maker_col])
        if is_buyer_maker_col is not None
        else pd.Series(pd.array([pd.NA] * len(work), dtype="boolean"), index=work.index)
    )
    return _coerce_trade_chunk_schema(out)


def _merge_symbol_klines(spot_root: Path, processed_root: Path, symbol: str) -> None:
    monthly_files = find_files(
        [str(spot_root / "spot" / "monthly" / "klines" / symbol / "1m" / "*.csv")]
    )
    daily_files = find_files(
        [str(spot_root / "spot" / "daily" / "klines" / symbol / "1m" / "*.csv")]
    )
    all_files = monthly_files + daily_files

    if not all_files:
        _warn(f"No kline files found for {symbol}.")
        return

    _log(
        f"{symbol} klines files: {len(monthly_files)} monthly + {len(daily_files)} daily"
    )
    frames: List[pd.DataFrame] = []
    for path in all_files:
        try:
            raw_df = read_csv_or_zip(path)
            _log(f"KLINES {symbol} loaded {len(raw_df):,} rows from {path.name}")
            frames.append(normalize_klines_df(raw_df))
        except Exception as exc:
            _warn(f"KLINES {symbol} failed to read {path}: {exc}")

    if not frames:
        _warn(f"No readable kline data for {symbol}.")
        return

    merged = pd.concat(frames, ignore_index=True)
    rows_pre = len(merged)
    merged = merged.drop_duplicates(subset=["open_time"], keep="first")
    merged = merged.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    rows_post = len(merged)

    output_path = processed_root / f"{symbol}_klines_1m.parquet"
    merged.to_parquet(output_path, index=False)
    _log(
        f"KLINES {symbol} pre-dedupe={rows_pre:,}, post-dedupe={rows_post:,}, wrote {output_path}"
    )


def _merge_symbol_trades(spot_root: Path, processed_root: Path, symbol: str) -> None:
    monthly_trade_files = _sort_trade_files(
        find_files(
            [
                str(spot_root / "spot" / "monthly" / "trades" / symbol / "*.csv"),
                str(spot_root / "spot" / "monthly" / "trades" / symbol / "*.zip"),
            ]
        ),
        default_day=1,
    )
    daily_trade_files = _sort_trade_files(
        find_files(
            [
                str(spot_root / "spot" / "daily" / "trades" / symbol / "*.csv"),
                str(spot_root / "spot" / "daily" / "trades" / symbol / "*.zip"),
            ]
        ),
        default_day=1,
    )

    if daily_trade_files:
        trade_files = daily_trade_files
        files_used_daily = len(daily_trade_files)
        files_used_monthly = 0
        if monthly_trade_files:
            _log(
                f"TRADES {symbol} daily files found, skipping {len(monthly_trade_files)} "
                "monthly files to avoid overlap."
            )
    else:
        trade_files = monthly_trade_files
        files_used_daily = 0
        files_used_monthly = len(monthly_trade_files)

    if not trade_files:
        _warn(f"No trade files found for {symbol}.")
        return

    _log(
        f"{symbol} trade files used: {files_used_monthly} monthly + "
        f"{files_used_daily} daily (total={len(trade_files)})"
    )

    output_path = processed_root / f"{symbol}_trades.parquet"
    tmp_output_path = processed_root / f"{symbol}_trades.parquet.tmp"
    if tmp_output_path.exists():
        _warn(f"TRADES {symbol} found stale temp output, deleting {tmp_output_path}")
        tmp_output_path.unlink()

    rows_written = 0
    files_failed = 0
    total_files = len(trade_files)
    progress_every = max(1, TRADES_PROGRESS_EVERY_N_FILES)
    rolling_window = max(1, TRADES_PROGRESS_ROLLING_WINDOW)
    files_processed = 0
    rolling_file_secs = deque(maxlen=rolling_window)
    merge_started = time.monotonic()

    try:
        writer = pq.ParquetWriter(tmp_output_path, TRADE_PARQUET_SCHEMA)
        try:
            for idx, path in enumerate(trade_files, start=1):
                file_started = time.monotonic()
                try:
                    raw_df = read_csv_or_zip(path)
                    chunk = normalize_trades_df(raw_df)

                    if chunk["trade_id"].notna().any():
                        chunk = chunk.drop_duplicates(subset=["trade_id"], keep="first")
                    else:
                        chunk = chunk.drop_duplicates(
                            subset=["time", "price", "qty", "isBuyerMaker"],
                            keep="first",
                        )
                    chunk = chunk.sort_values("time", kind="mergesort").reset_index(
                        drop=True
                    )

                    rows_after = len(chunk)
                    if rows_after > 0:
                        table = pa.Table.from_pandas(
                            chunk,
                            schema=TRADE_PARQUET_SCHEMA,
                            preserve_index=False,
                            safe=False,
                        )
                        writer.write_table(table)
                        rows_written += rows_after
                except Exception as exc:
                    files_failed += 1
                    _warn(f"TRADES {symbol} failed to process {path}: {exc}")
                finally:
                    files_processed = idx
                    file_elapsed = time.monotonic() - file_started
                    rolling_file_secs.append(file_elapsed)

                    if (
                        files_processed % progress_every == 0
                        or files_processed == total_files
                    ):
                        avg_sec_per_file = sum(rolling_file_secs) / len(rolling_file_secs)
                        remaining_files = total_files - files_processed
                        eta_seconds = remaining_files * avg_sec_per_file
                        elapsed = time.monotonic() - merge_started
                        current_size_gb = _file_size_gb(tmp_output_path)
                        _log(
                            "TRADES "
                            f"{symbol} progress={files_processed}/{total_files}, "
                            f"file={path.name}, rows_written_total={rows_written:,}, "
                            f"output_size_gb={current_size_gb:.4f}, "
                            f"elapsed={_format_hhmmss(elapsed)}, "
                            f"avg_sec_per_file_last_{rolling_window}={avg_sec_per_file:.2f}, "
                            f"eta={_format_hhmmss(eta_seconds)}"
                        )
        finally:
            writer.close()
    except Exception:
        if tmp_output_path.exists():
            _warn(f"TRADES {symbol} removing incomplete temp output {tmp_output_path}")
            tmp_output_path.unlink()
        raise

    tmp_output_path.replace(output_path)

    total_elapsed = time.monotonic() - merge_started
    final_size_gb = _file_size_gb(output_path)

    _log(
        f"TRADES {symbol} DONE files_used_daily={files_used_daily}, "
        f"files_used_monthly={files_used_monthly}, files_failed={files_failed}, "
        f"rows_written_total={rows_written:,}, total_elapsed={_format_hhmmss(total_elapsed)}, "
        f"final_size_gb={final_size_gb:.4f}, wrote {output_path}"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    spot_root = project_root / "data" / "binance" / "spot"
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    _log(f"Spot input root: {spot_root}")
    _log(f"Processed output root: {processed_root}")
    _log(f"Symbols: {SYMBOLS}")
    _log(f"TRADES_ONLY={TRADES_ONLY}")

    for symbol in SYMBOLS:
        _log(f"--- Processing {symbol} ---")
        if not TRADES_ONLY:
            _merge_symbol_klines(
                spot_root=spot_root, processed_root=processed_root, symbol=symbol
            )
        else:
            _log(f"Skipping klines merge for {symbol} (TRADES_ONLY=True).")
        _merge_symbol_trades(
            spot_root=spot_root, processed_root=processed_root, symbol=symbol
        )

    _log("Merge complete.")


if __name__ == "__main__":
    main()
