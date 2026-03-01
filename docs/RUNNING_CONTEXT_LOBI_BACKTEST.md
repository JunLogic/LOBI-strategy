# RUNNING_CONTEXT_LOBI_BACKTEST

## 1. Project Overview

### Purpose
LOBI-Backtest is a deterministic research and backtesting stack for hazard-gated momentum trading on Binance spot market data. The core objective is to combine:
- a 1-minute hazard probability model (regime detector), and
- a 5-second momentum execution engine.

### Architectural Separation
- **Hazard model layer**: training/inference pipeline under hazard/risk workflows (`scripts/train_hazard_logistic.py`, `scripts/infer_hazard.py`, outputs under `outputs/hazard/...`).
- **Momentum engine layer**: streaming bar construction, hazard join, backtest execution, diagnostics, and sweep orchestration under `mbars/` and `scripts/`.

Hazard model code and momentum engine are intentionally decoupled. Momentum consumes hazard probabilities as an input signal only.

### Deterministic Design Philosophy
- No stochastic behavior in single-run backtests.
- Same inputs + same parameters + same code => same outputs.
- No lookahead logic in signal generation or hazard alignment.

### Streaming Architecture
Large datasets are processed with `pyarrow.dataset` scanners in batches. This avoids full in-memory dataframe loads for primary market data paths.

### Memory Constraints
- Trades and 5s bars parquet can be large.
- Core path (bar building, hazard join, backtest) is streaming and bounded memory.
- Small artifacts (run outputs, diagnostics summaries) may use pandas safely.

---

## 2. Data Pipeline (PHASES 1-2)

### Source Data
- Raw Binance trade data and kline data are downloaded/merged via scripts in `scripts/`.
- Processed files are stored under `data/processed/`.

### 1m Hazard Dataset Path (Model Input/Output Context)
Typical hazard probability output used by momentum join:
- `outputs/hazard/<SYMBOL>/y_dd_30p_H1440m/full_fit_probs.csv`

Typical columns (auto-detected by join logic):
- timestamp column (commonly `open_time`)
- probability column (commonly `y_prob`)

### 5s Bar Construction (PHASE 1)
Input:
- `data/processed/<SYMBOL>_trades.parquet`

Output:
- `data/processed/<SYMBOL>_bars_5s.parquet`

Core bar schema:
- `bar_time_ms` (int64)
- `open`, `high`, `low`, `close` (float64)
- `volume`, `dollar_volume` (float64)
- `trade_count` (int64)
- `buy_volume`, `sell_volume` (float64)
- `return_5s` (float64)

### Hazard Join (PHASE 2)
Input:
- `data/processed/<SYMBOL>_bars_5s.parquet`
- `outputs/hazard/<SYMBOL>/y_dd_30p_H1440m/full_fit_probs.csv`

Output:
- `data/processed/<SYMBOL>_bars_5s_hazard.parquet`

Join rule (no lookahead):
- `minute_time_ms = (bar_time_ms // 60000) * 60000`
- `hazard_prob = hazard_prob_1m[minute_time_ms]`

Important constraints:
- Hazard probability is aligned to same-minute bucket only.
- Missing minute hazard => `hazard_prob = null`.
- No forward-fill across minute boundaries.

Joined schema = original 5s bar columns +
- `minute_time_ms` (int64)
- `hazard_prob` (float64)

### File Naming Convention
- Bars: `<SYMBOL>_bars_5s.parquet`
- Hazard-joined bars: `<SYMBOL>_bars_5s_hazard.parquet`

---

## 3. Momentum Engine (PHASE 3)

Implementation anchor:
- `mbars/backtest.py` -> `run_momentum_backtest(...)`
- `mbars/position_sizing.py` -> sizing abstraction

### Rolling Downside Trigger
- Rolling return over last `rolling_bars` of 5s returns (`return_5s`), default historically 6 (30s), now configurable.
- Trigger condition:
  - `rolling_return < downside_return_threshold`

### Entry Condition (Short-Only)
Entry considered when all are true:
- hazard gate active: `hazard_prob > hazard_threshold`
- rolling trigger active
- risk gates allow entry (Phase 5 controls)

### Exit Logic
Per-position exits include:
- stoploss
- takeprofit (optional)
- hazard deactivation (close active positions when gate off)
- end-of-data close
- risk exits (Phase 5): max-hold, kill-switch, equity floor

### Position Sizing Abstraction
`PositionSizingConfig` modes:
- `fixed`
- `equity_fraction`
- `custom` via editable `custom_position_formula(...)`

### Streaming Backtest Loop Structure
- Scan hazard-joined 5s bars in pyarrow batches.
- Iterate row-wise deterministically.
- Maintain rolling buffer and open positions in-memory (bounded).
- Write artifacts incrementally.

### Output Artifacts Per Run
`outputs/mbars/<SYMBOL>/<RUN_TAG>/`
- `trades.csv`
- `equity.csv`
- `summary.json`

---

## 4. Diagnostics Layer (PHASE 4)

Implementation anchor:
- `mbars/diagnostics.py`
- `scripts/plot_momentum_diagnostics.py`

### Diagnostics Components
- Equity curve plot (`equity.png`)
- Drawdown curve plot (`drawdown.png`)
- Trade PnL histogram (`trade_pnl_hist.png`)
- Trade summary stats (count, win rate, pnl stats, hold stats, streaks)

### Drawdown Formula
- `drawdown = (equity / equity.cummax()) - 1`

### Diagnostics Outputs
`outputs/mbars/<SYMBOL>/<RUN_TAG>/`
- `diagnostics.json`
- `plots/equity.png`
- `plots/drawdown.png`
- `plots/trade_pnl_hist.png`

`diagnostics.json` includes:
- file references
- equity summary (start/final/max drawdown)
- trade summary statistics
- embedded `backtest_summary`

---

## 5. Risk Layer (PHASE 5)

Implementation anchors:
- `mbars/config.py` -> `RiskConfig`
- `mbars/backtest.py` -> integrated risk state/counters

### Risk Rules

1. **max_open_positions** (default 3)
- If open positions already at cap, new entries are skipped.

2. **max_notional_frac** (default 0.10)
- Exposure cap per bar: `cap = equity * max_notional_frac`.
- Current exposure: `sum(abs(size) * price)` over open positions.
- New position is scaled down to fit remaining cap when possible.
- If scaled size `< min_position_size`, entry is skipped.

3. **max_hold_bars** (default 720)
- Position is force-closed at current price when held bars >= threshold.

4. **equity_floor_frac** (default 0.2)
- Floor: `initial_equity * equity_floor_frac`.
- If MTM equity <= floor:
  - close all open positions,
  - permanently disable new entries for rest of run,
  - record stop-trading flag/time.

5. **rebound_kill_pct + cooldown_bars** (defaults 0.02, 120)
- Track rolling low while hazard active or positions open.
- If current price rebounds above rolling low by kill threshold:
  - close all open positions immediately,
  - disable new entries for `cooldown_bars`.
- Uses only past/current values (no lookahead).

### Streaming Loop Interaction Order (High-Level)
1. Update rolling return and hazard gate.
2. Update/track rolling low.
3. Evaluate rebound kill-switch.
4. Handle hazard-off exits.
5. Evaluate per-position exits (max-hold, stoploss, takeprofit).
6. Evaluate equity floor stop-trading.
7. Attempt entry with max-pos and notional gates.
8. Write equity row and advance state.

### Risk Counters in `summary.json`
- `risk_stop_trading_triggered`
- `risk_stop_time_ms`
- `kill_switch_triggers`
- `entries_skipped_max_pos`
- `entries_skipped_notional`
- `entries_scaled_notional`
- `forced_exits_max_hold`
- `cooldown_total_bars`

---

## 6. Current Verified Behavior

### FLOKI Aggressive Run (validated)
Command profile:
- `hazard_threshold=0.0`
- `downside_threshold=-0.0005`
- `stoploss=0.02`

Observed summary:
- `closed_trades = 21698`
- `final_equity = 102719.15347675388`
- `max_drawdown = 0.06268559901777175` (~6.27%)
- `kill_switch_triggers = 2697` (> 0)
- `forced_exits_max_hold = 14795` (> 0)
- `entries_skipped_max_pos = 1032614` (large)

### PEPE Default Run (validated)
Command profile:
- default thresholds/risk

Observed summary:
- `closed_trades = 3`
- `final_equity = 100369.5756387727`
- `max_drawdown = 0.0007048848047196819` (small)
- hazard gating is selective/effective (very low trade count)

---

## 7. Output Directory Structure

```text
data/processed/
  <SYMBOL>_trades.parquet
  <SYMBOL>_bars_5s.parquet
  <SYMBOL>_bars_5s_hazard.parquet

outputs/mbars/<SYMBOL>/<RUN_TAG>/
  trades.csv
  equity.csv
  summary.json
  diagnostics.json
  plots/
    equity.png
    drawdown.png
    trade_pnl_hist.png

outputs/mbars/<SYMBOL>/sweeps/<SWEEP_ID>/
  results.csv
  results_top20.csv
  spec_used.yaml
  progress.log
```

### Run Tag Convention
Backtest run tags encode parameters in deterministic text form, e.g.:
- `ht0p6_dtm0p003_sl0p008_tpnone_szequity_fraction_rb6`

Components:
- `ht` hazard threshold
- `dt` downside threshold
- `sl` stoploss
- `tp` takeprofit (`none` if null)
- `sz` sizing mode
- `rb` rolling bars
- optional risk delta suffixes may be appended in sweep-generated tags when risk differs from defaults

---

## 8. Determinism + Reproducibility

- No random operations in single run backtests.
- Streaming scan over parquet input with fixed logic.
- No lookahead in hazard join or trading decisions.
- Sweep runner deterministic by default (ordered Cartesian product).
- Optional sweep shuffle is deterministic when seed is fixed.
- Sweep resume behavior: existing `summary.json` => run skipped.

---

## 9. Known Limitations

- No transaction cost model yet.
- No slippage model yet.
- No funding fee model.
- No cross-coin portfolio allocator.
- No parallel sweep execution engine yet.

---

## 10. Roadmap (Next Phases)

- **PHASE 6**: Large deterministic parameter sweep (implemented baseline orchestrator; can be expanded for scale).
- **PHASE 7**: Slippage + fee modeling.
- **PHASE 8**: Cross-symbol portfolio allocator.
- **PHASE 9**: Bridge into `LOBI-live-testnet`.
- **PHASE 10**: Online hazard gating in live environment.

---

## 11. Testnet Integration Plan

### Migration Target
Deploy selected sweep-approved configuration into `LOBI-live-testnet` with strict parity to backtest logic.

### Required Parity Components
- Hazard probability ingestion as runtime gate (`hazard_prob > threshold`).
- Same momentum trigger semantics (`rolling_bars`, downside threshold).
- Same risk controls (`RiskConfig` behavior and ordering).
- Same position sizing mode/formula.

### Integration Steps
1. Select winning config from `results.csv`/`results_top20.csv`.
2. Freeze parameters into testnet strategy config.
3. Port risk-state machine exactly (max-pos, exposure, max-hold, equity floor, rebound kill-switch, cooldown).
4. Validate event-by-event parity against offline replay for a fixed period.
5. Enable paper/testnet execution with identical gating and exits.

---

## 12. Minimal Command Reference

### Build 5s Bars
```powershell
python scripts\build_momentum_5s.py --symbol FLOKIUSDT
python scripts\build_momentum_5s.py --symbol PEPEUSDT
```

### Join Hazard Probabilities to 5s Bars
```powershell
python scripts\join_hazard_to_5s.py --symbol FLOKIUSDT
python scripts\join_hazard_to_5s.py --symbol PEPEUSDT
```

### Run Momentum Backtest
```powershell
python scripts\backtest_momentum.py --symbol FLOKIUSDT
python scripts\backtest_momentum.py --symbol PEPEUSDT
```

Aggressive FLOKI validation profile:
```powershell
python scripts\backtest_momentum.py --symbol FLOKIUSDT --hazard-threshold 0.0 --downside-threshold -0.0005 --stoploss 0.02
```

### Run Diagnostics
```powershell
python scripts\plot_momentum_diagnostics.py --run-dir outputs\mbars\FLOKIUSDT\<RUN_TAG>
python scripts\plot_momentum_diagnostics.py --run-dir outputs\mbars\PEPEUSDT\<RUN_TAG>
```

### Run Sweep
```powershell
python scripts\sweep_momentum.py --spec docs\sweeps\mbars_sweep_spec.yaml --symbol FLOKIUSDT
python scripts\sweep_momentum.py --spec docs\sweeps\mbars_sweep_spec.yaml --symbol PEPEUSDT
python scripts\sweep_momentum.py --spec docs\sweeps\mbars_sweep_spec.yaml --all-symbols
```

Smoke sweep example:
```powershell
python scripts\sweep_momentum.py --spec docs\sweeps\mbars_sweep_spec.yaml --symbol FLOKIUSDT --max-runs 10
```
