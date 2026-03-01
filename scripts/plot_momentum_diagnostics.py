from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mbars.diagnostics import write_diagnostics


def _default_root() -> Path:
    return Path("outputs") / "mbars"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write diagnostics and plots for an existing momentum backtest run directory."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Full run directory path, e.g. outputs/mbars/FLOKIUSDT/<RUN_TAG>",
    )
    parser.add_argument("--symbol", default=None, help="Convenience symbol when using --run-tag")
    parser.add_argument("--run-tag", default=None, help="Convenience run tag when using --symbol")
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir)

    if args.symbol and args.run_tag:
        return _default_root() / args.symbol / args.run_tag

    raise ValueError("Provide either --run-dir OR both --symbol and --run-tag.")


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args)

    diagnostics = write_diagnostics(run_dir)

    equity_stats = diagnostics["equity"]
    trade_stats = diagnostics["trades"]

    print(f"[diag] run_dir={run_dir}")
    print(
        f"[diag] trades={trade_stats['count']} win_rate={trade_stats['win_rate']:.4f} "
        f"total_pnl={trade_stats['total_pnl']}"
    )
    print(
        f"[diag] final_equity={equity_stats['final_equity']} "
        f"max_drawdown={equity_stats['max_drawdown']}"
    )
    print(f"[diag] diagnostics_json={diagnostics['files']['diagnostics_json']}")
    print(f"[diag] equity_plot={diagnostics['files']['equity_plot']}")
    print(f"[diag] drawdown_plot={diagnostics['files']['drawdown_plot']}")
    print(f"[diag] trade_pnl_hist_plot={diagnostics['files']['trade_pnl_hist_plot']}")


if __name__ == "__main__":
    main()
