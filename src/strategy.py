"""Signal and position generation for baseline strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

LABEL_TO_DIRECTION = {1.0: -1, 2.0: 0, 3.0: 1, 1: -1, 2: 0, 3: 1}


@dataclass(frozen=True)
class RuleModel:
    """Simple linear rule model defined by selected features and weights."""

    feature_names: List[str]
    weights: np.ndarray


def label_to_direction(label_series: pd.Series) -> pd.Series:
    """Map FI-2010 label classes {1,2,3} to {-1,0,1} direction."""
    direction = label_series.map(LABEL_TO_DIRECTION).astype(float)
    assert direction.notna().all(), "Unexpected label value encountered."
    return direction.astype(int)


def signal_to_position(signal: pd.Series) -> pd.Series:
    """Apply no-lookahead execution: position[t+1] = signal[t]."""
    shifted = signal.shift(1).fillna(0)
    position = shifted.astype(int)
    assert position.isin([-1, 0, 1]).all(), "Position must be in {-1,0,1}."
    return position


def oracle_label_strategy(y_df: pd.DataFrame, label_col: str = "148") -> Tuple[pd.Series, pd.Series]:
    """Oracle baseline using true label at time t as the signal."""
    assert label_col in y_df.columns, f"label_col={label_col} not found."
    signal = label_to_direction(y_df[label_col]).rename("signal")
    position = signal_to_position(signal).rename("position")
    return signal, position


def fit_rule_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    label_col: str = "148",
    top_k: int = 10,
) -> RuleModel:
    """Fit a simple linear rule using train-only absolute correlations."""
    assert label_col in y_train.columns, f"label_col={label_col} not found."
    target = label_to_direction(y_train[label_col]).astype(float)

    target_centered = target - target.mean()
    target_std = float(target_centered.std(ddof=0))
    if target_std == 0.0:
        corr = pd.Series(0.0, index=x_train.columns)
    else:
        x_centered = x_train - x_train.mean(axis=0)
        x_std = x_centered.std(axis=0, ddof=0)
        cov = x_centered.mul(target_centered, axis=0).mean(axis=0)
        denom = x_std * target_std
        corr = (cov / denom.replace(0.0, np.nan)).fillna(0.0)

    ranked = corr.abs().sort_values(ascending=False)
    selected = ranked.head(top_k).index.tolist()
    weights = corr.loc[selected].to_numpy(dtype=float)
    return RuleModel(feature_names=selected, weights=weights)


def rule_score(x_df: pd.DataFrame, model: RuleModel) -> pd.Series:
    """Compute the linear score from selected features and fixed weights."""
    x_selected = x_df[model.feature_names]
    scores = x_selected.to_numpy(dtype=float) @ model.weights
    return pd.Series(scores, index=x_df.index, name="score")


def rule_signal_from_score(score: pd.Series, threshold: float) -> pd.Series:
    """Threshold score into a directional signal in {-1,0,1}."""
    signal = pd.Series(0, index=score.index, dtype=int, name="signal")
    signal = signal.mask(score > threshold, 1)
    signal = signal.mask(score < -threshold, -1)
    assert signal.isin([-1, 0, 1]).all(), "Signal must be in {-1,0,1}."
    return signal


def rule_based_strategy(
    x_df: pd.DataFrame,
    model: RuleModel,
    threshold: float,
) -> Tuple[pd.Series, pd.Series]:
    """Generate rule-based signal and no-lookahead position."""
    score = rule_score(x_df, model)
    signal = rule_signal_from_score(score=score, threshold=threshold)
    position = signal_to_position(signal).rename("position")
    return signal.rename("signal"), position


def parse_threshold_grid(grid_text: str) -> Sequence[float]:
    """Parse comma-separated thresholds."""
    values = [item.strip() for item in grid_text.split(",") if item.strip()]
    thresholds = [float(item) for item in values]
    if not thresholds:
        raise ValueError("threshold_grid must contain at least one value.")
    return thresholds
