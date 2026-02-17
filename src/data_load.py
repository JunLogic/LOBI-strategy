"""Data loading utilities for the FI-2010 prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

FEATURE_COLS = [str(i) for i in range(144)]
LABEL_COLS = [str(i) for i in range(144, 149)]


def load_fi2010_csv(path: str) -> pd.DataFrame:
    """Load a FI-2010 CSV and normalize its column schema."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df.columns = [str(col) for col in df.columns]
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the FI-2010 dataframe into feature and label blocks."""
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    missing_labels = [col for col in LABEL_COLS if col not in df.columns]
    assert not missing_features, f"Missing feature columns: {missing_features[:5]}"
    assert not missing_labels, f"Missing label columns: {missing_labels[:5]}"

    x_df = df[FEATURE_COLS].copy()
    y_df = df[LABEL_COLS].copy()
    return x_df, y_df

