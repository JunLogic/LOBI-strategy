from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbars.join import join_hazard_to_5s


def _default_bars_path(symbol: str) -> Path:
    return Path("data") / "processed" / f"{symbol}_bars_5s.parquet"


def _default_hazard_path(symbol: str) -> Path:
    return Path("outputs") / "hazard" / symbol / "y_dd_30p_H1440m" / "full_fit_probs.csv"


def _default_out_path(symbol: str) -> Path:
    return Path("data") / "processed" / f"{symbol}_bars_5s_hazard.parquet"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join 1-minute hazard probabilities onto 5-second bars."
    )
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. FLOKIUSDT")
    parser.add_argument(
        "--bars",
        dest="bars_5s_path",
        default=None,
        help="5s bars parquet path. Defaults to data/processed/<SYMBOL>_bars_5s.parquet",
    )
    parser.add_argument(
        "--hazard",
        dest="hazard_csv_path",
        default=None,
        help="Hazard probabilities CSV path. Defaults to outputs/hazard/<SYMBOL>/y_dd_30p_H1440m/full_fit_probs.csv",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output parquet path. Defaults to data/processed/<SYMBOL>_bars_5s_hazard.parquet",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=500_000,
        help="Rows per scanner batch (default: 500000).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Log every N processed batches (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    bars_5s_path = Path(args.bars_5s_path) if args.bars_5s_path else _default_bars_path(args.symbol)
    hazard_csv_path = Path(args.hazard_csv_path) if args.hazard_csv_path else _default_hazard_path(args.symbol)
    out_path = Path(args.out_path) if args.out_path else _default_out_path(args.symbol)

    stats = join_hazard_to_5s(
        symbol=args.symbol,
        bars_5s_path=bars_5s_path,
        hazard_csv_path=hazard_csv_path,
        out_path=out_path,
        batch_rows=args.batch_rows,
        log_every=args.log_every,
    )

    print(
        f"[join] output={out_path} rows_written={stats['rows_written']} "
        f"missing_minutes={stats['missing_minutes_count']}"
    )


if __name__ == "__main__":
    main()
