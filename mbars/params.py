from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_BACKTEST_PARAMS: dict[str, Any] = {
    "hazard_threshold": 0.6,
    "downside_threshold": -0.003,
    "stoploss": 0.008,
    "takeprofit": None,
    "rolling_bars": 6,
    "initial_equity": 100_000.0,
    "sizing": {
        "mode": "equity_fraction",
        "fixed_size": 1.0,
        "equity_fraction": 0.02,
        "max_notional": None,
    },
    "risk": {
        "max_open_positions": 3,
        "max_notional_frac": 0.10,
        "min_position_size": 0.0,
        "max_hold_bars": 720,
        "equity_floor_frac": 0.2,
        "rebound_kill_pct": 0.02,
        "cooldown_bars": 120,
    },
}


# Add per-symbol overrides here when needed.
SYMBOL_OVERRIDES: dict[str, dict[str, Any]] = {}


def _merge_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_symbol_params(symbol: str, spec_base: dict[str, Any]) -> dict[str, Any]:
    params = _merge_dict(DEFAULT_BACKTEST_PARAMS, spec_base)
    overrides = SYMBOL_OVERRIDES.get(symbol, {})
    if overrides:
        params = _merge_dict(params, overrides)
    return params
