import calendar
import inspect
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from binance_historical_data import BinanceDataDumper


# ==========================
# USER CONFIGURATION
# ==========================

SYMBOLS_SPOT: List[str] = [
    "PEPEUSDT",
    "FLOKIUSDT",
]

DATE_START = "2024-10-01"
DATE_END = "2025-06-30"

# Use 1m for microstructure work
KLINES_INTERVAL = "1m"

# ==========================


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_month_windows(
    date_start: str, date_end: str
) -> Iterator[Tuple[str, str, str]]:
    start = _parse_date(date_start)
    end = _parse_date(date_end)
    if start > end:
        raise ValueError(f"Invalid range: {date_start} > {date_end}")

    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        month_start = date(year, month, 1)
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        window_start = max(month_start, start)
        window_end = min(month_end, end)

        if window_start <= window_end:
            month_key = f"{year:04d}-{month:02d}"
            yield month_key, window_start.isoformat(), window_end.isoformat()

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def _accepts_var_kwargs(func: Any) -> bool:
    signature = inspect.signature(func)
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _pick_kw_name(func: Any, *candidates: str) -> Optional[str]:
    signature = inspect.signature(func)
    for name in candidates:
        if name in signature.parameters:
            return name
    if _accepts_var_kwargs(func):
        return candidates[0]
    return None


def _filter_supported_kwargs(
    func: Any,
    kwargs: Dict[str, Any],
    call_name: str,
) -> Dict[str, Any]:
    if _accepts_var_kwargs(func):
        return dict(kwargs)

    signature = inspect.signature(func)
    filtered = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    dropped = [key for key in kwargs if key not in filtered]
    if dropped:
        print(f"[debug] Skipping unsupported kwargs for {call_name}: {dropped}")
    return filtered


def _build_dumper(
    out_root: Path,
    data_type: str,
    interval: Optional[str] = None,
) -> BinanceDataDumper:
    init_kwargs: Dict[str, Any] = {}

    path_key = _pick_kw_name(
        BinanceDataDumper,
        "path_dir_where_to_dump",
        "path_dir",
        "path",
        "output_dir",
    )
    if path_key is None:
        raise TypeError(
            "Could not find a supported output-path argument for BinanceDataDumper"
        )
    init_kwargs[path_key] = str(out_root)

    data_type_key = _pick_kw_name(BinanceDataDumper, "data_type")
    if data_type_key is not None:
        init_kwargs[data_type_key] = data_type

    if interval is not None:
        interval_key = _pick_kw_name(
            BinanceDataDumper,
            "data_frequency",
            "interval",
            "timeframe",
        )
        if interval_key is not None:
            init_kwargs[interval_key] = interval

    safe_init_kwargs = _filter_supported_kwargs(
        BinanceDataDumper,
        init_kwargs,
        "BinanceDataDumper.__init__",
    )
    return BinanceDataDumper(**safe_init_kwargs)


def _dump_data_compat(
    dumper: BinanceDataDumper,
    symbols: List[str],
    date_start: str,
    date_end: str,
    data_type: Optional[str] = None,
    interval: Optional[str] = None,
) -> None:
    kwargs: Dict[str, Any] = {}

    symbols_key = _pick_kw_name(dumper.dump_data, "tickers", "symbols", "symbol")
    if symbols_key is not None:
        if symbols_key == "symbol" and len(symbols) > 1:
            for symbol in symbols:
                _dump_data_compat(
                    dumper=dumper,
                    symbols=[symbol],
                    date_start=date_start,
                    date_end=date_end,
                    data_type=data_type,
                    interval=interval,
                )
            return
        if symbols_key == "symbol" and len(symbols) == 1:
            kwargs[symbols_key] = symbols[0]
        else:
            kwargs[symbols_key] = symbols

    start_key = _pick_kw_name(dumper.dump_data, "date_start", "start_date", "from_date")
    if start_key is not None:
        kwargs[start_key] = date_start

    end_key = _pick_kw_name(dumper.dump_data, "date_end", "end_date", "to_date")
    if end_key is not None:
        kwargs[end_key] = date_end

    if data_type is not None:
        data_type_key = _pick_kw_name(dumper.dump_data, "data_type")
        if data_type_key is not None:
            kwargs[data_type_key] = data_type

    if interval is not None:
        interval_key = _pick_kw_name(
            dumper.dump_data,
            "interval",
            "data_frequency",
            "timeframe",
        )
        if interval_key is not None:
            kwargs[interval_key] = interval

    safe_kwargs = _filter_supported_kwargs(dumper.dump_data, kwargs, "dump_data")
    try:
        dumper.dump_data(**safe_kwargs)
        return
    except TypeError as exc:
        start_key = _pick_kw_name(
            dumper.dump_data, "date_start", "start_date", "from_date"
        )
        end_key = _pick_kw_name(dumper.dump_data, "date_end", "end_date", "to_date")
        can_retry_with_dates = (
            start_key in safe_kwargs
            and end_key in safe_kwargs
            and isinstance(safe_kwargs[start_key], str)
            and isinstance(safe_kwargs[end_key], str)
        )
        if not can_retry_with_dates:
            raise

        retry_kwargs = dict(safe_kwargs)
        retry_kwargs[start_key] = _parse_date(retry_kwargs[start_key])
        retry_kwargs[end_key] = _parse_date(retry_kwargs[end_key])
        print("[debug] Retrying dump_data with datetime.date date_start/date_end.")
        try:
            dumper.dump_data(**retry_kwargs)
            return
        except Exception:
            raise exc


def _trades_month_exists(out_root: Path, symbol: str, month_key: str) -> bool:
    monthly_trades_dir = out_root / "spot" / "monthly" / "trades" / symbol
    base = f"{symbol}-trades-{month_key}"
    return (monthly_trades_dir / f"{base}.csv").exists() or (
        monthly_trades_dir / f"{base}.zip"
    ).exists()


def ensure_trades_downloaded(
    dumper: BinanceDataDumper,
    out_root: Path,
    symbols: List[str],
    date_start: str,
    date_end: str,
) -> None:
    month_windows = list(_iter_month_windows(date_start, date_end))
    expected_months = [month_key for month_key, _, _ in month_windows]
    existing_by_symbol: Dict[str, List[str]] = {}
    missing_by_symbol: Dict[str, List[str]] = {}

    print("[info] Ensuring monthly spot trades coverage...")
    for symbol in symbols:
        print(f"[info] Checking monthly trades for {symbol}...")
        for month_key, month_start, month_end in month_windows:
            if _trades_month_exists(out_root, symbol, month_key):
                continue

            print(
                "[info] Missing trades month "
                f"{symbol} {month_key}; downloading {month_start} -> {month_end}"
            )
            try:
                _dump_data_compat(
                    dumper=dumper,
                    symbols=[symbol],
                    date_start=month_start,
                    date_end=month_end,
                    data_type="trades",
                )
            except Exception as exc:
                print(f"[warn] Failed {symbol} {month_key}: {exc}")

        existing_months = [
            month_key
            for month_key in expected_months
            if _trades_month_exists(out_root, symbol, month_key)
        ]
        missing_months = [
            month_key
            for month_key in expected_months
            if month_key not in existing_months
        ]
        existing_by_symbol[symbol] = existing_months
        missing_by_symbol[symbol] = missing_months

    print("[summary] Trades monthly coverage")
    print(f"[summary] Expected months: {', '.join(expected_months)}")
    for symbol in symbols:
        existing = (
            ", ".join(existing_by_symbol[symbol])
            if existing_by_symbol[symbol]
            else "none"
        )
        missing = (
            ", ".join(missing_by_symbol[symbol])
            if missing_by_symbol[symbol]
            else "none"
        )
        print(f"[summary] {symbol} existing months: {existing}")
        print(f"[summary] {symbol} missing months:  {missing}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_root = project_root / "data" / "binance" / "spot"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Project root: {project_root}")
    print(f"[info] Writing into:  {out_root}")
    print(f"[info] Symbols:       {SYMBOLS_SPOT}")
    print(f"[info] Date range:    {DATE_START} -> {DATE_END}")
    print(f"[info] Interval:      {KLINES_INTERVAL}")

    # ---- Download klines ----
    print("[info] Downloading spot klines...")
    klines_dumper = _build_dumper(
        out_root=out_root,
        data_type="klines",
        interval=KLINES_INTERVAL,
    )
    _dump_data_compat(
        dumper=klines_dumper,
        symbols=SYMBOLS_SPOT,
        date_start=DATE_START,
        date_end=DATE_END,
        data_type="klines",
        interval=KLINES_INTERVAL,
    )

    # ---- Ensure trades month-by-month ----
    trades_dumper = _build_dumper(
        out_root=out_root,
        data_type="trades",
    )
    ensure_trades_downloaded(
        dumper=trades_dumper,
        out_root=out_root,
        symbols=SYMBOLS_SPOT,
        date_start=DATE_START,
        date_end=DATE_END,
    )

    print("[done] Spot downloads complete.")
    print("[done] Check: data/binance/spot")


if __name__ == "__main__":
    main()
