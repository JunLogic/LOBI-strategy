"""Momentum-bars package scaffolding.

This package is intentionally minimal in this step. It reserves module
boundaries for a hazard-gated momentum implementation.
"""

from .backtest import run_mbars_backtest
from .bars import build_mbars
from .config import MBarsConfig

__all__ = ["MBarsConfig", "build_mbars", "run_mbars_backtest"]
