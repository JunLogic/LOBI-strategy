from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = Path(run_dir)
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.csv"
    summary_path = run_dir / "summary.json"

    if not trades_path.exists():
        raise FileNotFoundError(f"trades.csv not found: {trades_path}")
    if not equity_path.exists():
        raise FileNotFoundError(f"equity.csv not found: {equity_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    trades_df = pd.read_csv(trades_path)
    equity_df = pd.read_csv(equity_path)
    summary_dict = json.loads(summary_path.read_text(encoding="utf-8"))

    return trades_df, equity_df, summary_dict


def compute_drawdown(equity_df: pd.DataFrame) -> pd.DataFrame:
    if "equity" not in equity_df.columns:
        raise ValueError("equity_df must contain 'equity' column.")

    out = equity_df.copy()
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["equity"]).copy()

    if out.empty:
        out["drawdown"] = pd.Series(dtype="float64")
        return out

    out["equity_peak"] = out["equity"].cummax()
    out["drawdown"] = (out["equity"] / out["equity_peak"]) - 1.0
    return out


def _max_consecutive(signs: pd.Series, target: int) -> int:
    best = 0
    streak = 0
    for value in signs:
        if value == target:
            streak += 1
            if streak > best:
                best = streak
        else:
            streak = 0
    return int(best)


def summarize_trades(trades_df: pd.DataFrame) -> dict[str, Any]:
    if trades_df.empty:
        return {
            "count": 0,
            "win_rate": 0.0,
            "avg_pnl": None,
            "median_pnl": None,
            "total_pnl": 0.0,
            "avg_hold_seconds": None,
            "avg_bars_held": None,
            "pnl_quantiles": {
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
            },
            "max_consecutive_losses": 0,
            "max_consecutive_wins": 0,
        }

    work = trades_df.copy()
    if "pnl" not in work.columns:
        raise ValueError("trades_df must contain 'pnl' column.")

    work["pnl"] = pd.to_numeric(work["pnl"], errors="coerce")
    work = work.dropna(subset=["pnl"]).copy()

    if work.empty:
        return {
            "count": 0,
            "win_rate": 0.0,
            "avg_pnl": None,
            "median_pnl": None,
            "total_pnl": 0.0,
            "avg_hold_seconds": None,
            "avg_bars_held": None,
            "pnl_quantiles": {
                "p05": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p95": None,
            },
            "max_consecutive_losses": 0,
            "max_consecutive_wins": 0,
        }

    count = int(len(work))
    wins = int((work["pnl"] > 0.0).sum())

    avg_bars_held = None
    if "bars_held" in work.columns:
        work["bars_held"] = pd.to_numeric(work["bars_held"], errors="coerce")
        valid_bars = work["bars_held"].dropna()
        if not valid_bars.empty:
            avg_bars_held = float(valid_bars.mean())

    avg_hold_seconds = None
    if "entry_time_ms" in work.columns and "exit_time_ms" in work.columns:
        entry_ms = pd.to_numeric(work["entry_time_ms"], errors="coerce")
        exit_ms = pd.to_numeric(work["exit_time_ms"], errors="coerce")
        hold_secs = (exit_ms - entry_ms) / 1000.0
        hold_secs = hold_secs.dropna()
        if not hold_secs.empty:
            avg_hold_seconds = float(hold_secs.mean())
    elif avg_bars_held is not None:
        avg_hold_seconds = float(avg_bars_held * 5.0)

    pnl_q = work["pnl"].quantile([0.05, 0.25, 0.50, 0.75, 0.95])

    signs = work["pnl"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    return {
        "count": count,
        "win_rate": float(wins / count) if count > 0 else 0.0,
        "avg_pnl": float(work["pnl"].mean()),
        "median_pnl": float(work["pnl"].median()),
        "total_pnl": float(work["pnl"].sum()),
        "avg_hold_seconds": avg_hold_seconds,
        "avg_bars_held": avg_bars_held,
        "pnl_quantiles": {
            "p05": float(pnl_q.loc[0.05]),
            "p25": float(pnl_q.loc[0.25]),
            "p50": float(pnl_q.loc[0.50]),
            "p75": float(pnl_q.loc[0.75]),
            "p95": float(pnl_q.loc[0.95]),
        },
        "max_consecutive_losses": _max_consecutive(signs, -1),
        "max_consecutive_wins": _max_consecutive(signs, 1),
    }


def plot_equity(equity_df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    if equity_df.empty or "equity" not in equity_df.columns:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        y = pd.to_numeric(equity_df["equity"], errors="coerce")
        if "bar_time_ms" in equity_df.columns:
            x = pd.to_datetime(
                pd.to_numeric(equity_df["bar_time_ms"], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            )
            mask = x.notna() & y.notna()
            if mask.any():
                ax.plot(x[mask], y[mask], linewidth=1.0)
            else:
                ax.text(0.5, 0.5, "No valid equity data", ha="center", va="center", transform=ax.transAxes)
        else:
            y = y.dropna()
            if not y.empty:
                ax.plot(y.index, y.values, linewidth=1.0)
            else:
                ax.text(0.5, 0.5, "No valid equity data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title("Equity Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown(equity_df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    drawdown_df = compute_drawdown(equity_df)

    fig, ax = plt.subplots(figsize=(12, 4))
    if drawdown_df.empty:
        ax.text(0.5, 0.5, "No drawdown data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        y = pd.to_numeric(drawdown_df["drawdown"], errors="coerce")
        if "bar_time_ms" in drawdown_df.columns:
            x = pd.to_datetime(
                pd.to_numeric(drawdown_df["bar_time_ms"], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            )
            mask = x.notna() & y.notna()
            if mask.any():
                ax.plot(x[mask], y[mask], linewidth=1.0, color="tab:red")
                ax.fill_between(x[mask], y[mask], 0.0, color="tab:red", alpha=0.2)
            else:
                ax.text(
                    0.5, 0.5, "No valid drawdown data", ha="center", va="center", transform=ax.transAxes
                )
        else:
            valid = y.dropna()
            if not valid.empty:
                ax.plot(valid.index, valid.values, linewidth=1.0, color="tab:red")
                ax.fill_between(valid.index, valid.values, 0.0, color="tab:red", alpha=0.2)
            else:
                ax.text(
                    0.5, 0.5, "No valid drawdown data", ha="center", va="center", transform=ax.transAxes
                )

        ax.set_title("Drawdown")
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trade_pnl_hist(trades_df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    if trades_df.empty or "pnl" not in trades_df.columns:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").dropna()
        if pnl.empty:
            ax.text(0.5, 0.5, "No valid PnL values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        else:
            bins = min(50, max(10, int(len(pnl) ** 0.5)))
            ax.hist(pnl, bins=bins, edgecolor="black", linewidth=0.5)
            ax.set_title("Trade PnL Distribution")
            ax.set_xlabel("PnL")
            ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_diagnostics(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    trades_df, equity_df, summary_dict = load_run(run_dir)
    equity_dd = compute_drawdown(equity_df)
    trade_summary = summarize_trades(trades_df)

    equity_start = None
    equity_end = None
    max_drawdown = None
    if not equity_dd.empty:
        equity_start = float(equity_dd["equity"].iloc[0])
        equity_end = float(equity_dd["equity"].iloc[-1])
        max_drawdown = float(equity_dd["drawdown"].min())

    equity_plot = plots_dir / "equity.png"
    drawdown_plot = plots_dir / "drawdown.png"
    trade_hist_plot = plots_dir / "trade_pnl_hist.png"

    plot_equity(equity_dd, equity_plot)
    plot_drawdown(equity_dd, drawdown_plot)
    plot_trade_pnl_hist(trades_df, trade_hist_plot)

    diagnostics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "files": {
            "diagnostics_json": str(run_dir / "diagnostics.json"),
            "equity_plot": str(equity_plot),
            "drawdown_plot": str(drawdown_plot),
            "trade_pnl_hist_plot": str(trade_hist_plot),
        },
        "equity": {
            "rows": int(len(equity_dd)),
            "start_equity": equity_start,
            "final_equity": equity_end,
            "max_drawdown": max_drawdown,
        },
        "trades": trade_summary,
        "backtest_summary": summary_dict,
    }

    diagnostics_path = run_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return diagnostics
