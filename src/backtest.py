"""Backtest engine for directional proxy PnL."""

from __future__ import annotations

import pandas as pd

from src.strategy import label_to_direction


def realized_direction_from_labels(y_df: pd.DataFrame, label_col: str = "148") -> pd.Series:
    """Build realized direction series from FI-2010 labels."""
    assert label_col in y_df.columns, f"label_col={label_col} not found."
    return label_to_direction(y_df[label_col]).rename("realized_direction")


def run_directional_backtest(
    position: pd.Series,
    realized_direction: pd.Series,
    cost_per_trade: float = 0.01,
) -> pd.DataFrame:
    """
    Run proxy PnL backtest.

    pnl_t = position_t * realized_direction_t - cost * abs(position_t - position_{t-1})
    """
    assert len(position) == len(realized_direction), "Series lengths must match."

    position = position.astype(int).rename("position")
    realized_direction = realized_direction.astype(int).rename("realized_direction")
    assert position.isin([-1, 0, 1]).all(), "Position must be in {-1,0,1}."
    assert realized_direction.isin([-1, 0, 1]).all(), "Direction must be in {-1,0,1}."

    position_prev = position.shift(1).fillna(0).astype(int)
    trade_size = (position - position_prev).abs().rename("trade_size")
    trades = (trade_size > 0).astype(int).rename("trade")

    pnl = (position * realized_direction - cost_per_trade * trade_size).rename("pnl")
    equity = pnl.cumsum().rename("equity")

    out = pd.concat([position, realized_direction, trade_size, trades, pnl, equity], axis=1)
    assert out[["position", "realized_direction", "pnl", "equity"]].notna().all().all(), "NaNs found in key series."

    expected_trades = int(position.diff().fillna(position).ne(0).sum())
    assert int(trades.sum()) == expected_trades, "Trade count mismatch with position changes."
    return out

