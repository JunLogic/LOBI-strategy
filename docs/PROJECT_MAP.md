# Project Map

## Directory Structure

```text
LOBI-Backtest/
  data/
    binance/spot/              # raw Binance dumps (csv/zip by symbol/date)
    processed/                 # merged parquet + engineered features
  docs/
    PROJECT_MAP.md
  outputs/                     # experiment reports/plots
  rugrisk/                     # hazard/risk modeling scaffolding
  scripts/
    __init__.py
    download_binance.py
    merge_binance_spot.py
    build_features_1m.py
  src/                         # FI-2010 backtest prototype pipeline
  tests/
    test_build_features_smoke.py
  pytest.ini
  requirements.txt
```

## Data Flow

1. Raw Binance files (`data/binance/spot/...`) are downloaded as CSV/ZIP.
2. `scripts/merge_binance_spot.py` normalizes and merges into processed parquet:
   - `data/processed/<SYMBOL>_klines_1m.parquet`
   - `data/processed/<SYMBOL>_trades.parquet`
3. `scripts/build_features_1m.py` builds per-minute feature parquet:
   - `data/processed/<SYMBOL>_features_1m.parquet`
   - `data/processed/<SYMBOL>_features_1m.meta.json`
4. Labels + hazard model are the next stage (not implemented yet in a dedicated training script).

## Key Scripts

- `scripts/download_binance.py`
  - Fetches Binance spot data archives.
- `scripts/merge_binance_spot.py`
  - Reads raw CSV/ZIP, normalizes schema, writes merged klines/trades parquet.
- `scripts/build_features_1m.py`
  - Uses kline grid + streaming trades aggregation (pyarrow dataset batches) to produce 1m causal features.
- `src/run_experiment.py`
  - Existing FI-2010 prototype runner (separate from Binance 1m feature pipeline).

## How To Run

Install deps:

```powershell
pip install -r requirements.txt
```

Merge Binance spot files to processed parquet:

```powershell
python scripts/merge_binance_spot.py
```

Build 1-minute features (example):

```powershell
python scripts/build_features_1m.py `
  --symbol PEPEUSDT `
  --klines_parquet data/processed/PEPEUSDT_klines_1m.parquet `
  --trades_parquet data/processed/PEPEUSDT_trades.parquet `
  --out_parquet data/processed/PEPEUSDT_features_1m.parquet `
  --start_date 2024-10-01 `
  --end_date 2025-06-30
```

Run tests from repo root:

```powershell
pytest -q
```

`pytest.ini` sets `pythonpath = .`, and `scripts/__init__.py` keeps `scripts` importable in tests.
