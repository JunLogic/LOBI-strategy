from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbars.bars import build_5s_bars_from_trades


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5-second bars from trades parquet.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. PEPEUSDT")
    parser.add_argument(
        "--in",
        dest="in_path",
        default=None,
        help="Input trades parquet path. Defaults to data/processed/<SYMBOL>_trades.parquet",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output bars parquet path. Defaults to data/processed/<SYMBOL>_bars_5s.parquet",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=250_000,
        help="Rows per scanner batch (default: 250000).",
    )
    parser.add_argument(
        "--flush-bars",
        type=int,
        default=200_000,
        help="Bars buffered before write flush (default: 200000).",
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
    out_path = build_5s_bars_from_trades(
        symbol=args.symbol,
        in_path=args.in_path,
        out_path=args.out_path,
        batch_rows=args.batch_rows,
        flush_bars=args.flush_bars,
        log_every=args.log_every,
    )
    print(f"[bars] output={out_path}")


if __name__ == "__main__":
    main()
