# MBARS Plan

## Objective

Prepare a hazard-gated momentum module (`mbars/`) that runs on short-horizon
bars and uses 1-minute hazard probabilities as a risk gate.

## Planned Pipeline

1. Build 5s/10s bars from raw trades using pyarrow streaming batches.
2. Load 1m hazard probabilities and join onto 5s/10s bars using forward fill
   from minute timestamps to bar timestamps.
3. Apply entry/exit logic with:
   - momentum-based entries
   - fixed stoploss
   - global kill-switch that halts new entries under risk-off conditions
4. Emit backtest outputs to disk.

## Join Details: 1m Hazard -> 5s Bars

- Hazard inputs are generated at 1-minute cadence (`open_time`, `y_prob`).
- MBARS bars are generated at 5-second or 10-second cadence.
- Use as-of style forward fill so each short bar gets the most recent known 1m
  hazard probability.
- No look-ahead: hazard value at time `t` may only use hazard timestamps
  `<= t`.

## Strategy Rules (Planned)

- Entry:
  - momentum trigger on 5s/10s bars
  - gated by hazard threshold (only enter when risk gate allows)
- Exit:
  - fixed stoploss
  - regular strategy exits
  - global kill-switch can flatten and block new entries

## Outputs

- `trades.csv`
- `equity_curve.csv`
- `summary.json`
