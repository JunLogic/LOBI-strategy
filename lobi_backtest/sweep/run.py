from __future__ import annotations

import argparse
from pathlib import Path

from .core import run_sweep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic parameter sweeps for the momentum backtest engine."
    )
    parser.add_argument("--spec", required=True, help="Path to sweep YAML spec.")
    parser.add_argument("--symbol", default=None, help="Single symbol to run.")
    parser.add_argument("--all-symbols", action="store_true", help="Run all symbols in spec.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand the sweep and print planned runs without executing backtests.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional max runs for this invocation. Overrides engine.max_runs in spec.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore resume skips and run all planned run_tags.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_sweep(
        spec_path=Path(args.spec),
        dry_run=bool(args.dry_run),
        max_runs=args.max_runs,
        force_rerun=bool(args.force_rerun),
        symbol=args.symbol,
        all_symbols=bool(args.all_symbols),
        shuffle=bool(args.shuffle),
        seed=args.seed,
        log_every=args.log_every,
        skip_errors=bool(args.skip_errors),
    )
    print(f"[sweep] done output_root={result['output_root']} sweep_id={result['sweep_id']}")
    for symbol, meta in result["symbols"].items():
        print(
            f"[sweep] symbol={symbol} total={meta['total_grid_runs']} scheduled={meta['scheduled_runs']} "
            f"skipped={meta['skipped']} errors={meta.get('errors', 0)}"
        )


if __name__ == "__main__":
    main()
