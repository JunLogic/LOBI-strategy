from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MBarsConfig:
    """Placeholder config for the future hazard-gated momentum module."""

    symbol: str
    bars_parquet: Path
    hazard_probs_csv: Path
    out_dir: Path


@dataclass(frozen=True)
class RiskConfig:
    max_open_positions: int = 3
    max_notional_frac: float = 0.10
    min_position_size: float = 0.0
    max_hold_bars: int = 720
    equity_floor_frac: float = 0.2
    rebound_kill_pct: float = 0.02
    cooldown_bars: int = 120


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    bars_path: Path
    out_dir: Path
    hazard_threshold: float
    downside_return_threshold: float
    stoploss_pct: float
    takeprofit_pct: Optional[float]
    initial_equity: float
