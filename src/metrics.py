"""Performance metrics for the directional proxy backtest."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown from an equity curve."""
    running_max = equity.cummax()
    drawdown = equity - running_max
    return float(drawdown.min())


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown series from an equity curve."""
    running_max = equity.cummax()
    return (equity - running_max).rename("drawdown")


def compute_metrics(
    backtest_df: pd.DataFrame,
    annualization_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute core strategy metrics from backtest output."""
    pnl = backtest_df["pnl"]
    position = backtest_df["position"]
    trades = backtest_df["trade"]
    equity = backtest_df["equity"]

    total_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std(ddof=0))
    sharpe = 0.0 if std_pnl == 0.0 else mean_pnl / std_pnl
    if annualization_factor is not None and annualization_factor > 0:
        sharpe *= float(np.sqrt(annualization_factor))

    exposure_mask = position != 0
    exposure_count = int(exposure_mask.sum())
    win_rate = 0.0
    if exposure_count > 0:
        win_rate = float(((pnl > 0) & exposure_mask).sum() / exposure_count)

    num_trades = int(trades.sum())
    avg_trade_return = 0.0 if num_trades == 0 else total_pnl / num_trades

    return {
        "total_pnl": total_pnl,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe_per_step": float(sharpe),
        "max_drawdown": compute_max_drawdown(equity),
        "win_rate_exposed": win_rate,
        "avg_trade_return": float(avg_trade_return),
        "num_trades": num_trades,
    }
