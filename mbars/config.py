from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MBarsConfig:
    """Placeholder config for the future hazard-gated momentum module."""

    symbol: str
    bars_parquet: Path
    hazard_probs_csv: Path
    out_dir: Path
