# Project Map

## Directory Structure

```text
LOBI-Backtest/
  data/
    binance/spot/              # raw Binance dumps (csv/zip by symbol/date)
    processed/                 # merged parquet + engineered datasets
  docs/
    PROJECT_MAP.md
    MBARS_PLAN.md
  mbars/                       # momentum-bars module scaffold (new)
    __init__.py
    config.py
    bars.py
    backtest.py
  outputs/                     # experiment reports, model outputs, plots
  rugrisk/                     # hazard/risk modeling code (unchanged)
  scripts/
    __init__.py
    download_binance.py
    merge_binance_spot.py
    build_features_1m.py
    build_labels_drawdown.py
    build_dataset_zscore_1m.py
    train_hazard_logistic.py
    infer_hazard.py
    plot_hazard_diagnostics.py
  src/                         # FI-2010 prototype code
  tests/
    test_build_features_smoke.py
    test_build_labels_smoke.py
    test_zscore_causality.py
  pytest.ini
  requirements.txt
```

## Data Flow

1. Download raw Binance spot data:
   - `scripts/download_binance.py`
2. Merge and normalize to parquet:
   - `scripts/merge_binance_spot.py`
   - outputs:
     - `data/processed/<SYMBOL>_klines_1m.parquet`
     - `data/processed/<SYMBOL>_trades.parquet`
3. Build 1-minute features:
   - `scripts/build_features_1m.py`
   - output: `data/processed/<SYMBOL>_features_1m.parquet`
4. Build drawdown labels and dataset:
   - `scripts/build_labels_drawdown.py`
   - output: `data/processed/<SYMBOL>_dataset_1m.parquet`
5. Optional causal z-scored dataset:
   - `scripts/build_dataset_zscore_1m.py`
   - output: `data/processed/<SYMBOL>_dataset_1m_z.parquet`
6. Train hazard logistic model:
   - `scripts/train_hazard_logistic.py`
   - output under `outputs/hazard/<SYMBOL>/<LABEL>/`
7. Run out-of-sample inference:
   - `scripts/infer_hazard.py`
8. Diagnostics plots:
   - `scripts/plot_hazard_diagnostics.py`
9. Planned next stage (`mbars/`):
   - build 5s/10s bars
   - join hazard gate (1m -> short bars forward fill)
   - run hazard-gated momentum backtest

## Notes

- Existing hazard logic stays under `rugrisk/` and current scripts.
- `mbars/` is scaffolding only in this step; no trading logic moved yet.
