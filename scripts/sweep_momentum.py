from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lobi_backtest.sweep.core import run_sweep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical sweep CLI for deterministic momentum backtest parameter sweeps."
    )
    parser.add_argument("--spec", required=True, help="Path to sweep YAML spec.")
    parser.add_argument("--symbol", default=None, help="Single symbol to run.")
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Run all symbols listed in spec.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Deterministically shuffle parameter combinations with --seed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffle seed override. Defaults to spec random_seed when present.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum runs after resume filtering.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100_000,
        help="Progress logging interval passed to run_momentum_backtest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing backtests.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Bypass resume skips and rerun all planned run tags.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="When resuming, also skip run tags that previously ended with status=error.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=None,
        help="Force resume on.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=None,
        help="Force resume off.",
    )
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        default=None,
        help="Stop sweep on first run failure.",
    )
    parser.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        default=None,
        help="Continue sweep after per-run failures.",
    )
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
        resume=args.resume,
        fail_fast=args.fail_fast,
        output_root=Path("outputs") / "mbars",
    )

    print(
        f"[sweep] done spec_hash={result['spec_hash']} output_root={result['output_root']} "
        f"sweep_id={result['sweep_id']}"
    )
    for symbol, meta in result["symbols"].items():
        print(
            f"[sweep] symbol={symbol} total={meta['total_grid_runs']} scheduled={meta['scheduled_runs']} "
            f"skipped={meta['skipped']} errors={meta.get('errors', 0)} sweep_dir={meta['sweep_dir']}"
        )


if __name__ == "__main__":
    main()

