from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from .config import MBarsConfig
from .position_sizing import PositionContext, PositionSizingConfig, compute_position_size

ROLLING_BARS = 6
SCAN_BATCH_ROWS = 250_000


@dataclass
class _OpenShortPosition:
    position_id: int
    entry_time_ms: int
    entry_price: float
    size: float


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "none"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    if text == "":
        return "0"
    return text


def _make_run_tag(
    hazard_threshold: float,
    downside_return_threshold: float,
    stoploss_pct: float,
    takeprofit_pct: Optional[float],
    sizing_mode: str,
) -> str:
    return (
        f"ht{_fmt_float(hazard_threshold)}"
        f"_dt{_fmt_float(downside_return_threshold)}"
        f"_sl{_fmt_float(stoploss_pct)}"
        f"_tp{_fmt_float(takeprofit_pct)}"
        f"_sz{sizing_mode}"
    )


def _is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def _unrealized_pnl(open_positions: list[_OpenShortPosition], price: float) -> float:
    return sum((pos.entry_price - price) * pos.size for pos in open_positions)


def _close_position(
    *,
    position: _OpenShortPosition,
    exit_time_ms: int,
    exit_price: float,
    exit_reason: str,
    trades_writer: csv.DictWriter,
) -> tuple[float, int]:
    pnl = (position.entry_price - exit_price) * position.size
    bars_held = max(0, (exit_time_ms - position.entry_time_ms) // 5_000)
    ret_pct = ((position.entry_price - exit_price) / position.entry_price) if position.entry_price > 0 else 0.0

    trades_writer.writerow(
        {
            "position_id": position.position_id,
            "entry_time_ms": position.entry_time_ms,
            "exit_time_ms": exit_time_ms,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": pnl,
            "return_pct": ret_pct,
            "bars_held": bars_held,
            "exit_reason": exit_reason,
        }
    )
    return pnl, int(bars_held)


def run_momentum_backtest(
    symbol: str,
    bars_path: Path,
    out_dir: Path,
    hazard_threshold: float,
    downside_return_threshold: float,
    stoploss_pct: float,
    takeprofit_pct: Optional[float],
    initial_equity: float,
    sizing_config: PositionSizingConfig,
    log_every: int = 100_000,
) -> dict[str, Any]:
    if not symbol:
        raise ValueError("symbol must be non-empty.")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive.")
    if stoploss_pct <= 0:
        raise ValueError("stoploss_pct must be positive.")
    if takeprofit_pct is not None and takeprofit_pct <= 0:
        raise ValueError("takeprofit_pct must be positive when provided.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    bars_path = Path(bars_path)
    out_dir = Path(out_dir)

    if not bars_path.exists():
        raise FileNotFoundError(f"Bars parquet not found: {bars_path}")

    run_tag = _make_run_tag(
        hazard_threshold=hazard_threshold,
        downside_return_threshold=downside_return_threshold,
        stoploss_pct=stoploss_pct,
        takeprofit_pct=takeprofit_pct,
        sizing_mode=sizing_config.mode,
    )
    run_dir = out_dir / symbol / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.csv"
    summary_path = run_dir / "summary.json"

    bars_ds = ds.dataset(bars_path, format="parquet")
    required_columns = ["bar_time_ms", "close", "hazard_prob", "return_5s"]
    missing_columns = [c for c in required_columns if c not in bars_ds.schema.names]
    if missing_columns:
        raise ValueError(
            f"Bars parquet missing required columns: {missing_columns}. "
            "Expected a PHASE 2 hazard-joined bars parquet."
        )

    scanner = bars_ds.scanner(
        columns=required_columns,
        batch_size=SCAN_BATCH_ROWS,
        use_threads=True,
    )

    print(f"[bt] symbol={symbol} bars={bars_path}")
    print(
        f"[bt] params hazard_threshold={hazard_threshold} downside_threshold={downside_return_threshold} "
        f"stoploss_pct={stoploss_pct} takeprofit_pct={takeprofit_pct} initial_equity={initial_equity} "
        f"sizing_mode={sizing_config.mode}"
    )
    print(f"[bt] run_dir={run_dir}")

    start_ts = time.perf_counter()

    cash_equity = float(initial_equity)
    peak_equity = float(initial_equity)
    max_drawdown = 0.0

    rows_scanned = 0
    rows_written_equity = 0
    batches = 0
    skipped_rows = 0
    trade_count = 0
    win_count = 0
    loss_count = 0
    total_pnl = 0.0
    total_bars_held = 0
    opened_positions = 0

    position_id = 0
    open_positions: list[_OpenShortPosition] = []

    rolling_returns: deque[float] = deque(maxlen=ROLLING_BARS)
    rolling_sum = 0.0

    last_time_ms: Optional[int] = None
    last_price: Optional[float] = None

    with trades_path.open("w", newline="", encoding="utf-8") as trades_fp, equity_path.open(
        "w", newline="", encoding="utf-8"
    ) as equity_fp:
        trades_writer = csv.DictWriter(
            trades_fp,
            fieldnames=[
                "position_id",
                "entry_time_ms",
                "exit_time_ms",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "return_pct",
                "bars_held",
                "exit_reason",
            ],
        )
        trades_writer.writeheader()

        equity_writer = csv.DictWriter(
            equity_fp,
            fieldnames=[
                "bar_time_ms",
                "equity",
                "cash_equity",
                "open_positions",
                "hazard_prob",
                "rolling_return_30s",
            ],
        )
        equity_writer.writeheader()

        for batch in scanner.to_batches():
            batches += 1
            if batch.num_rows == 0:
                continue

            time_vals = pc.cast(batch.column(0), pa.int64()).to_numpy(zero_copy_only=False)
            price_vals = pc.cast(batch.column(1), pa.float64()).to_numpy(zero_copy_only=False)
            hazard_vals = pc.cast(batch.column(2), pa.float64()).to_numpy(zero_copy_only=False)
            return_vals = pc.cast(batch.column(3), pa.float64()).to_numpy(zero_copy_only=False)

            for i in range(batch.num_rows):
                rows_scanned += 1

                raw_time = time_vals[i]
                raw_price = price_vals[i]
                raw_hazard = hazard_vals[i]
                raw_ret = return_vals[i]

                if raw_time is None or raw_price is None:
                    skipped_rows += 1
                    continue

                bar_time_ms = int(raw_time)
                price = float(raw_price)

                if price <= 0.0 or not _is_finite(price):
                    skipped_rows += 1
                    continue

                hazard_prob = float(raw_hazard) if raw_hazard is not None else float("nan")
                ret_5s = float(raw_ret) if raw_ret is not None else float("nan")

                if not _is_finite(ret_5s):
                    if last_price is not None and last_price > 0.0:
                        ret_5s = (price / last_price) - 1.0
                    else:
                        ret_5s = 0.0

                if len(rolling_returns) == ROLLING_BARS:
                    rolling_sum -= rolling_returns[0]
                rolling_returns.append(ret_5s)
                rolling_sum += ret_5s

                rolling_ready = len(rolling_returns) == ROLLING_BARS
                rolling_return = rolling_sum if rolling_ready else 0.0

                trading_active = _is_finite(hazard_prob) and (hazard_prob > hazard_threshold)

                if not trading_active and open_positions:
                    still_open: list[_OpenShortPosition] = []
                    for pos in open_positions:
                        pnl, bars_held = _close_position(
                            position=pos,
                            exit_time_ms=bar_time_ms,
                            exit_price=price,
                            exit_reason="hazard_off",
                            trades_writer=trades_writer,
                        )
                        cash_equity += pnl
                        total_pnl += pnl
                        total_bars_held += bars_held
                        trade_count += 1
                        if pnl > 0:
                            win_count += 1
                        elif pnl < 0:
                            loss_count += 1
                    open_positions = still_open

                if trading_active and open_positions:
                    survivors: list[_OpenShortPosition] = []
                    stoploss_mult = 1.0 + stoploss_pct
                    takeprofit_mult = 1.0 - takeprofit_pct if takeprofit_pct is not None else None

                    for pos in open_positions:
                        stop_hit = price > (pos.entry_price * stoploss_mult)
                        tp_hit = (
                            takeprofit_mult is not None
                            and price < (pos.entry_price * takeprofit_mult)
                        )

                        if stop_hit:
                            pnl, bars_held = _close_position(
                                position=pos,
                                exit_time_ms=bar_time_ms,
                                exit_price=price,
                                exit_reason="stoploss",
                                trades_writer=trades_writer,
                            )
                            cash_equity += pnl
                            total_pnl += pnl
                            total_bars_held += bars_held
                            trade_count += 1
                            if pnl > 0:
                                win_count += 1
                            elif pnl < 0:
                                loss_count += 1
                        elif tp_hit:
                            pnl, bars_held = _close_position(
                                position=pos,
                                exit_time_ms=bar_time_ms,
                                exit_price=price,
                                exit_reason="takeprofit",
                                trades_writer=trades_writer,
                            )
                            cash_equity += pnl
                            total_pnl += pnl
                            total_bars_held += bars_held
                            trade_count += 1
                            if pnl > 0:
                                win_count += 1
                            elif pnl < 0:
                                loss_count += 1
                        else:
                            survivors.append(pos)
                    open_positions = survivors

                if trading_active and rolling_ready and (rolling_return < downside_return_threshold):
                    mtm_equity = cash_equity + _unrealized_pnl(open_positions, price)
                    context = PositionContext(
                        equity=mtm_equity,
                        price=price,
                        open_positions=len(open_positions),
                        hazard_prob=hazard_prob,
                    )
                    size = compute_position_size(context, sizing_config)
                    if _is_finite(size) and size > 0.0:
                        position_id += 1
                        opened_positions += 1
                        open_positions.append(
                            _OpenShortPosition(
                                position_id=position_id,
                                entry_time_ms=bar_time_ms,
                                entry_price=price,
                                size=float(size),
                            )
                        )

                mtm_equity = cash_equity + _unrealized_pnl(open_positions, price)
                if mtm_equity > peak_equity:
                    peak_equity = mtm_equity
                if peak_equity > 0.0:
                    drawdown = (peak_equity - mtm_equity) / peak_equity
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                equity_writer.writerow(
                    {
                        "bar_time_ms": bar_time_ms,
                        "equity": mtm_equity,
                        "cash_equity": cash_equity,
                        "open_positions": len(open_positions),
                        "hazard_prob": hazard_prob if _is_finite(hazard_prob) else "",
                        "rolling_return_30s": rolling_return,
                    }
                )
                rows_written_equity += 1

                last_time_ms = bar_time_ms
                last_price = price

                if rows_scanned % log_every == 0:
                    elapsed = time.perf_counter() - start_ts
                    print(
                        f"[bt] progress rows={rows_scanned} trades={trade_count} "
                        f"open_positions={len(open_positions)} equity={mtm_equity:.2f} "
                        f"max_dd={max_drawdown:.6f} elapsed_s={elapsed:.2f}"
                    )

        if open_positions and last_time_ms is not None and last_price is not None:
            for pos in open_positions:
                pnl, bars_held = _close_position(
                    position=pos,
                    exit_time_ms=last_time_ms,
                    exit_price=last_price,
                    exit_reason="end_of_data",
                    trades_writer=trades_writer,
                )
                cash_equity += pnl
                total_pnl += pnl
                total_bars_held += bars_held
                trade_count += 1
                if pnl > 0:
                    win_count += 1
                elif pnl < 0:
                    loss_count += 1
            open_positions = []

            if cash_equity > peak_equity:
                peak_equity = cash_equity
            if peak_equity > 0.0:
                drawdown = (peak_equity - cash_equity) / peak_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            equity_writer.writerow(
                {
                    "bar_time_ms": last_time_ms,
                    "equity": cash_equity,
                    "cash_equity": cash_equity,
                    "open_positions": 0,
                    "hazard_prob": "",
                    "rolling_return_30s": "",
                }
            )
            rows_written_equity += 1

    elapsed = time.perf_counter() - start_ts
    win_rate = (win_count / trade_count) if trade_count > 0 else 0.0
    avg_bars_held = (total_bars_held / trade_count) if trade_count > 0 else 0.0

    summary: dict[str, Any] = {
        "symbol": symbol,
        "bars_path": str(bars_path),
        "run_dir": str(run_dir),
        "rows_scanned": rows_scanned,
        "rows_written_equity": rows_written_equity,
        "batches": batches,
        "skipped_rows": skipped_rows,
        "opened_positions": opened_positions,
        "closed_trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "initial_equity": initial_equity,
        "final_equity": cash_equity,
        "max_drawdown": max_drawdown,
        "avg_bars_held": avg_bars_held,
        "elapsed_s": elapsed,
        "hazard_threshold": hazard_threshold,
        "downside_return_threshold": downside_return_threshold,
        "stoploss_pct": stoploss_pct,
        "takeprofit_pct": takeprofit_pct,
        "rolling_bars": ROLLING_BARS,
        "sizing": {
            "mode": sizing_config.mode,
            "fixed_size": sizing_config.fixed_size,
            "equity_fraction": sizing_config.equity_fraction,
            "max_notional": sizing_config.max_notional,
        },
        "files": {
            "trades_csv": str(trades_path),
            "equity_csv": str(equity_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[bt] done rows={rows_scanned} trades={trade_count} final_equity={cash_equity:.2f} "
        f"max_dd={max_drawdown:.6f} elapsed_s={elapsed:.2f}"
    )
    print(f"[bt] outputs trades={trades_path} equity={equity_path} summary={summary_path}")

    return summary


def run_mbars_backtest(config: MBarsConfig) -> None:
    """Compatibility placeholder entrypoint for previous scaffolding API."""

    _ = config
    raise NotImplementedError(
        "run_mbars_backtest(MBarsConfig) is not wired for PHASE 3. "
        "Use run_momentum_backtest(...) instead."
    )
