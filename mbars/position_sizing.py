from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PositionSizingConfig:
    mode: str  # "fixed" | "equity_fraction" | "custom"
    fixed_size: float = 1.0
    equity_fraction: float = 0.02
    max_notional: Optional[float] = None


@dataclass(frozen=True)
class PositionContext:
    equity: float
    price: float
    open_positions: int
    hazard_prob: float


def custom_position_formula(context: PositionContext, config: PositionSizingConfig) -> float:
    """User-editable custom position formula placeholder."""
    _ = config
    return context.equity * 0.01 / context.price


def compute_position_size(context: PositionContext, config: PositionSizingConfig) -> float:
    if context.price <= 0.0:
        return 0.0

    if config.mode == "fixed":
        size = float(config.fixed_size)
    elif config.mode == "equity_fraction":
        notional = float(context.equity) * float(config.equity_fraction)
        size = notional / float(context.price)
    elif config.mode == "custom":
        size = float(custom_position_formula(context, config))
    else:
        raise ValueError(
            "Unsupported sizing mode. Expected one of: fixed, equity_fraction, custom"
        )

    if config.max_notional is not None:
        max_size = float(config.max_notional) / float(context.price)
        size = min(size, max_size)

    if size < 0.0:
        return 0.0
    return float(size)
