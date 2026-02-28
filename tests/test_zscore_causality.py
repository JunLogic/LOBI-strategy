from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_dataset_zscore_1m import ZScoreConfig, build_dataset_zscore_1m


def _write_dataset(path: Path, feature_values: list[float]) -> pd.DataFrame:
    start_ms = int(pd.Timestamp("2024-10-01T00:00:00Z").value // 1_000_000)
    rows = []
    for i, value in enumerate(feature_values):
        rows.append(
            {
                "open_time": start_ms + i * 60_000,
                "symbol": "PEPEUSDT",
                "imbalance": float(value),
                "y_dd_30p_H3m": 0.0,
                "dd_min_H3m": -0.1,
                "future_min_close_H3m": 100.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def test_zscore_causality_and_finite_values(tmp_path: Path) -> None:
    base_in = tmp_path / "base.parquet"
    perturbed_in = tmp_path / "perturbed.parquet"
    base_out = tmp_path / "base_z.parquet"
    perturbed_out = tmp_path / "perturbed_z.parquet"

    _write_dataset(base_in, feature_values=[1.0, 1.2, 1.4, 1.6, 2.0, 2.5])
    _write_dataset(perturbed_in, feature_values=[1.0, 1.2, 1.4, 1.6, 2.0, 250.0])

    df_base = build_dataset_zscore_1m(
        ZScoreConfig(
            in_parquet=base_in,
            out_parquet=base_out,
            mode="expanding",
            rolling_window_mins=1440,
            min_periods=2,
            eps=1e-12,
        )
    )
    df_perturbed = build_dataset_zscore_1m(
        ZScoreConfig(
            in_parquet=perturbed_in,
            out_parquet=perturbed_out,
            mode="expanding",
            rolling_window_mins=1440,
            min_periods=2,
            eps=1e-12,
        )
    )

    assert "z_imbalance" in df_base.columns

    cutoff = len(df_base) - 1
    left = df_base["z_imbalance"].iloc[:cutoff].to_numpy()
    right = df_perturbed["z_imbalance"].iloc[:cutoff].to_numpy()
    assert np.allclose(left, right, equal_nan=True)

    z_cols = [c for c in df_base.columns if c.startswith("z_")]
    assert z_cols
    assert np.isfinite(df_base[z_cols].to_numpy()).all()
    assert np.isfinite(df_perturbed[z_cols].to_numpy()).all()
    assert Path(f"{base_out}.meta.json").exists()
    assert Path(f"{perturbed_out}.meta.json").exists()
