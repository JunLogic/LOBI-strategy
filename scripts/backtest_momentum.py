from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbars.backtest import run_momentum_backtest
from mbars.position_sizing import PositionSizingConfig


def _default_bars_path(symbol: str) -> Path:
    return Path("data") / "processed" / f"{symbol}_bars_5s_hazard.parquet"


def _default_out_dir() -> Path:
    return Path("outputs") / "mbars"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic hazard-gated momentum backtest on 5s bars."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. FLOKIUSDT")
    parser.add_argument(
        "--bars",
        dest="bars_path",
        default=None,
        help="Input hazard-joined bars parquet. Defaults to data/processed/<SYMBOL>_bars_5s_hazard.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output root directory. Defaults to outputs/mbars",
    )
    parser.add_argument("--hazard-threshold", type=float, default=0.6)
    parser.add_argument("--downside-threshold", type=float, default=-0.003)
    parser.add_argument("--stoploss", type=float, default=0.008)
    parser.add_argument("--takeprofit", type=float, default=None)
    parser.add_argument("--initial-equity", type=float, default=100000.0)
    parser.add_argument(
        "--sizing-mode",
        choices=("fixed", "equity_fraction", "custom"),
        default="equity_fraction",
    )
    parser.add_argument("--equity-fraction", type=float, default=0.02)
    parser.add_argument("--fixed-size", type=float, default=1.0)
    parser.add_argument("--max-notional", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bars_path = Path(args.bars_path) if args.bars_path else _default_bars_path(args.symbol)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir()

    sizing_config = PositionSizingConfig(
        mode=args.sizing_mode,
        fixed_size=args.fixed_size,
        equity_fraction=args.equity_fraction,
        max_notional=args.max_notional,
    )

    summary = run_momentum_backtest(
        symbol=args.symbol,
        bars_path=bars_path,
        out_dir=out_dir,
        hazard_threshold=args.hazard_threshold,
        downside_return_threshold=args.downside_threshold,
        stoploss_pct=args.stoploss,
        takeprofit_pct=args.takeprofit,
        initial_equity=args.initial_equity,
        sizing_config=sizing_config,
        log_every=args.log_every,
    )

    print(
        f"[bt] summary={summary['files']['summary_json']} trades={summary['closed_trades']} "
        f"final_equity={summary['final_equity']:.2f}"
    )


if __name__ == "__main__":
    main()
