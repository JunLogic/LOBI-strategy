from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_labels_drawdown import LabelBuildConfig, build_labels_drawdown


def _write_features(path: Path, symbol: str, closes: list[float]) -> pd.DataFrame:
    start_ms = int(pd.Timestamp("2024-10-01T00:00:00Z").value // 1_000_000)
    rows = []
    for i, close in enumerate(closes):
        open_time = start_ms + i * 60_000
        rows.append(
            {
                "symbol": symbol,
                "open_time": open_time,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
                "trade_vol": 0.0,
                "trade_dollar_vol": 0.0,
                "n_trades": 0,
                "buy_vol": 0.0,
                "sell_vol": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def test_build_labels_smoke(tmp_path: Path) -> None:
    symbol = "PEPEUSDT"
    features_path = tmp_path / "features.parquet"
    out_path = tmp_path / "dataset.parquet"

    # Known 30% drop from index 0 to index 2 within H=3:
    # close[0]=100, min(close[1..3])=70 => drawdown=-0.30 => label=1 for X=0.30
    _write_features(features_path, symbol=symbol, closes=[100.0, 98.0, 70.0, 95.0, 96.0, 97.0])

    out = build_labels_drawdown(
        LabelBuildConfig(
            symbol=symbol,
            features_parquet=features_path,
            out_parquet=out_path,
            start_date="2024-10-01",
            end_date="2024-10-01",
            horizons_mins=[3],
            drawdown_thresholds=[0.30, 0.50],
        )
    )

    label_30 = "y_dd_30p_H3m"
    label_50 = "y_dd_50p_H3m"
    dd_col = "dd_min_H3m"
    future_min_col = "future_min_close_H3m"

    assert label_30 in out.columns
    assert label_50 in out.columns
    assert dd_col in out.columns
    assert future_min_col in out.columns

    # Non-tail section exact expectations.
    assert out.loc[0, label_30] == 1.0
    assert out.loc[1, label_30] == 0.0
    assert out.loc[2, label_30] == 0.0
    assert out.loc[0, label_50] == 0.0
    assert out.loc[1, label_50] == 0.0
    assert out.loc[2, label_50] == 0.0

    # For H=3, last 3 rows must be NaN due to insufficient future window.
    tail = out.index[-3:]
    assert out.loc[tail, label_30].isna().all()
    assert out.loc[tail, label_50].isna().all()
    assert out.loc[tail, dd_col].isna().all()
    assert out.loc[tail, future_min_col].isna().all()

    assert (tmp_path / "dataset.meta.json").exists()
    saved = pd.read_parquet(out_path)
    assert np.isclose(saved.loc[0, dd_col], -0.30)
