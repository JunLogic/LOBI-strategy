from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_features_1m import (
    BuildConfig,
    DERIVED_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    TRADE_AGG_COLUMNS,
    build_features_1m,
)


def _write_klines(path: Path, symbol: str, closes: list[float]) -> pd.DataFrame:
    start_ms = int(pd.Timestamp("2024-10-01T00:00:00Z").value // 1_000_000)
    rows = []
    for i, close in enumerate(closes):
        open_time = start_ms + i * 60_000
        rows.append(
            {
                "open_time": open_time,
                "open": float(close) * 0.99,
                "high": float(close) * 1.01,
                "low": float(close) * 0.98,
                "close": float(close),
                "volume": 100.0 + i,
                "symbol": symbol,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def _write_trades(path: Path) -> pd.DataFrame:
    start_ms = int(pd.Timestamp("2024-10-01T00:00:00Z").value // 1_000_000)
    rows = [
        # Minute 0
        {"time": start_ms + 10_000, "price": 10.0, "qty": 2.0, "isBuyerMaker": False},
        {"time": start_ms + 20_000, "price": 10.2, "qty": 1.0, "isBuyerMaker": True},
        # Minute 2
        {"time": start_ms + 2 * 60_000 + 30_000, "price": 11.0, "qty": 3.0, "isBuyerMaker": False},
        # Minute 5
        {"time": start_ms + 5 * 60_000 + 45_000, "price": 12.0, "qty": 4.0, "isBuyerMaker": True},
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def test_build_features_smoke(tmp_path: Path) -> None:
    symbol = "PEPEUSDT"
    trades_path = tmp_path / "trades.parquet"
    _write_trades(trades_path)

    base_klines = tmp_path / "klines_base.parquet"
    future_perturbed_klines = tmp_path / "klines_future_perturbed.parquet"
    _write_klines(base_klines, symbol=symbol, closes=[100, 101, 102, 103, 104, 105])
    _write_klines(
        future_perturbed_klines,
        symbol=symbol,
        closes=[100, 101, 102, 103, 104, 10000],
    )

    out_base = tmp_path / "features_base.parquet"
    out_perturbed = tmp_path / "features_perturbed.parquet"

    df_base = build_features_1m(
        BuildConfig(
            symbol=symbol,
            klines_parquet=base_klines,
            trades_parquet=trades_path,
            out_parquet=out_base,
            start_date="2024-10-01",
            end_date="2024-10-01",
        )
    )
    df_perturbed = build_features_1m(
        BuildConfig(
            symbol=symbol,
            klines_parquet=future_perturbed_klines,
            trades_parquet=trades_path,
            out_parquet=out_perturbed,
            start_date="2024-10-01",
            end_date="2024-10-01",
        )
    )

    assert list(df_base.columns) == OUTPUT_COLUMNS
    assert df_base["open_time"].dtype == np.int64
    assert df_base["n_trades"].dtype == np.int64

    # Missing-trade minute (minute 1) should be zero-filled for trade aggregates.
    minute_1 = int(pd.Timestamp("2024-10-01T00:01:00Z").value // 1_000_000)
    row_1 = df_base.loc[df_base["open_time"] == minute_1].iloc[0]
    for col in TRADE_AGG_COLUMNS:
        assert pd.notna(row_1[col])
        assert row_1[col] == 0

    # Causality: perturbing the last close should not change derived features before the last row.
    cutoff = len(df_base) - 1
    for col in DERIVED_FEATURE_COLUMNS:
        left = df_base[col].iloc[:cutoff].to_numpy()
        right = df_perturbed[col].iloc[:cutoff].to_numpy()
        assert np.allclose(left, right, equal_nan=True), f"Non-causal feature detected: {col}"

    assert (tmp_path / "features_base.meta.json").exists()
    assert (tmp_path / "features_perturbed.meta.json").exists()
