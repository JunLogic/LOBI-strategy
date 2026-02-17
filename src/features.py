"""Feature utilities for the FI-2010 prototype."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def select_feature_columns() -> List[str]:
    """Return the expected FI-2010 feature columns."""
    return [str(i) for i in range(144)]


def summarize_feature_stats(x_df: pd.DataFrame) -> Tuple[float, float]:
    """Return average absolute mean and average std over columns."""
    mean_abs = float(x_df.mean(axis=0).abs().mean())
    std_avg = float(x_df.std(axis=0, ddof=0).mean())
    return mean_abs, std_avg


def standardize_if_needed(
    x_df: pd.DataFrame,
    standardize: bool = False,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Standardize features if explicitly requested.

    By default this is a no-op, since FI-2010 features are typically
    already engineered/standardized.
    """
    if not standardize:
        return x_df.copy()

    col_mean = x_df.mean(axis=0)
    col_std = x_df.std(axis=0, ddof=0).replace(0.0, eps)
    x_std = (x_df - col_mean) / col_std
    return x_std

