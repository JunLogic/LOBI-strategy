"""CLI entrypoint for FI-2010 directional backtest prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from src.backtest import realized_direction_from_labels, run_directional_backtest
from src.data_load import load_fi2010_csv, split_features_labels
from src.features import standardize_if_needed, summarize_feature_stats
from src.metrics import compute_metrics
from src.plots import (
    ensure_output_dir,
    plot_drawdown_curve,
    plot_equity_curve,
    plot_pnl_histogram,
    plot_threshold_sensitivity,
)
from src.strategy import RuleModel, fit_rule_model, oracle_label_strategy, parse_threshold_grid, rule_based_strategy


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Run FI-2010 directional backtest prototype.")
    parser.add_argument("--label_col", type=str, default="148", help='Target label column (default: "148").')
    parser.add_argument("--cost", type=float, default=0.01, help="Cost per trade unit.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["oracle", "rule", "both"],
        default="both",
        help="Which strategy to run.",
    )
    parser.add_argument(
        "--threshold_grid",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated thresholds for rule strategy.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of features for rule strategy.")
    parser.add_argument(
        "--annualization_factor",
        type=float,
        default=None,
        help="Optional annualization factor for Sharpe (e.g., steps per year).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Apply z-score standardization to features (default is no-op).",
    )
    return parser.parse_args()


def print_metric_block(name: str, metric_dict: Dict[str, Any]) -> None:
    """Print a concise metric block."""
    print(f"\n{name}")
    print("-" * len(name))
    for key, value in metric_dict.items():
        if key == "num_trades":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")


def evaluate_strategy(
    signal: pd.Series,
    position: pd.Series,
    y_df: pd.DataFrame,
    label_col: str,
    cost: float,
    annualization_factor: float | None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run backtest and metrics for a precomputed signal/position."""
    realized = realized_direction_from_labels(y_df=y_df, label_col=label_col)
    backtest_df = run_directional_backtest(
        position=position,
        realized_direction=realized,
        cost_per_trade=cost,
    )
    # Keep signal for extra diagnostics.
    backtest_df = pd.concat([signal.rename("signal"), backtest_df], axis=1)
    metric_dict = compute_metrics(backtest_df=backtest_df, annualization_factor=annualization_factor)
    return backtest_df, metric_dict


def save_standard_plots(backtest_df: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    """Persist equity, drawdown, and pnl histogram plots."""
    plot_equity_curve(
        equity=backtest_df["equity"],
        save_path=output_dir / f"{prefix}_equity.png",
        title=f"{prefix} Equity Curve",
    )
    plot_drawdown_curve(
        equity=backtest_df["equity"],
        save_path=output_dir / f"{prefix}_drawdown.png",
        title=f"{prefix} Drawdown Curve",
    )
    plot_pnl_histogram(
        pnl=backtest_df["pnl"],
        save_path=output_dir / f"{prefix}_pnl_hist.png",
        title=f"{prefix} Per-step PnL Histogram",
    )


def tune_rule_threshold(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    label_col: str,
    cost: float,
    thresholds: Sequence[float],
    top_k: int,
    annualization_factor: float | None,
) -> Tuple[float, RuleModel, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Tune threshold on train only and return best train artifacts."""
    model = fit_rule_model(x_train=x_train, y_train=y_train, label_col=label_col, top_k=top_k)

    rows = []
    best_threshold = None
    best_train_bt = None
    best_metrics = None
    best_score = float("-inf")

    for thr in thresholds:
        signal, position = rule_based_strategy(x_df=x_train, model=model, threshold=thr)
        bt_df, metric_dict = evaluate_strategy(
            signal=signal,
            position=position,
            y_df=y_train,
            label_col=label_col,
            cost=cost,
            annualization_factor=annualization_factor,
        )

        rows.append(
            {
                "threshold": float(thr),
                "sharpe_per_step": metric_dict["sharpe_per_step"],
                "total_pnl": metric_dict["total_pnl"],
                "num_trades": int(metric_dict["num_trades"]),
            }
        )
        score = metric_dict["sharpe_per_step"]
        if score > best_score:
            best_score = score
            best_threshold = float(thr)
            best_train_bt = bt_df
            best_metrics = metric_dict

    assert best_threshold is not None and best_train_bt is not None and best_metrics is not None
    sensitivity_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return best_threshold, model, sensitivity_df, best_train_bt, best_metrics


def main() -> None:
    """Run full train/test experiment."""
    args = parse_args()
    thresholds = parse_threshold_grid(args.threshold_grid)

    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "FI2010_train.csv"
    test_path = project_root / "data" / "FI2010_test.csv"
    output_dir = ensure_output_dir(project_root / "outputs")

    train_df = load_fi2010_csv(str(train_path))
    test_df = load_fi2010_csv(str(test_path))
    x_train, y_train = split_features_labels(train_df)
    x_test, y_test = split_features_labels(test_df)

    x_train = standardize_if_needed(x_train, standardize=args.standardize)
    x_test = standardize_if_needed(x_test, standardize=args.standardize)
    mean_abs, std_avg = summarize_feature_stats(x_train)

    print("FI-2010 Directional Backtest Prototype")
    print("=====================================")
    print(f"train rows: {len(x_train)} | test rows: {len(x_test)}")
    print(f"label_col: {args.label_col} | cost: {args.cost} | strategy: {args.strategy}")
    print(f"standardize enabled: {args.standardize}")
    print(f"train feature avg(abs(mean)): {mean_abs:.6f}, avg(std): {std_avg:.6f}")

    report: Dict[str, object] = {
        "config": {
            "label_col": args.label_col,
            "cost": args.cost,
            "strategy": args.strategy,
            "threshold_grid": list(thresholds),
            "top_k": args.top_k,
            "annualization_factor": args.annualization_factor,
            "standardize": args.standardize,
        }
    }

    if args.strategy in {"oracle", "both"}:
        signal_train, position_train = oracle_label_strategy(y_df=y_train, label_col=args.label_col)
        oracle_train_bt, oracle_train_metrics = evaluate_strategy(
            signal=signal_train,
            position=position_train,
            y_df=y_train,
            label_col=args.label_col,
            cost=args.cost,
            annualization_factor=args.annualization_factor,
        )
        signal_test, position_test = oracle_label_strategy(y_df=y_test, label_col=args.label_col)
        oracle_test_bt, oracle_test_metrics = evaluate_strategy(
            signal=signal_test,
            position=position_test,
            y_df=y_test,
            label_col=args.label_col,
            cost=args.cost,
            annualization_factor=args.annualization_factor,
        )

        print_metric_block("Oracle Train", oracle_train_metrics)
        print_metric_block("Oracle Test", oracle_test_metrics)
        save_standard_plots(oracle_train_bt, output_dir, "oracle_train")
        save_standard_plots(oracle_test_bt, output_dir, "oracle_test")

        report["oracle_train"] = oracle_train_metrics
        report["oracle_test"] = oracle_test_metrics

    if args.strategy in {"rule", "both"}:
        best_thr, fitted_model, sensitivity_df, rule_train_bt, rule_train_metrics = tune_rule_threshold(
            x_train=x_train,
            y_train=y_train,
            label_col=args.label_col,
            cost=args.cost,
            thresholds=thresholds,
            top_k=args.top_k,
            annualization_factor=args.annualization_factor,
        )
        rule_signal_test, rule_position_test = rule_based_strategy(x_df=x_test, model=fitted_model, threshold=best_thr)
        rule_test_bt, rule_test_metrics = evaluate_strategy(
            signal=rule_signal_test,
            position=rule_position_test,
            y_df=y_test,
            label_col=args.label_col,
            cost=args.cost,
            annualization_factor=args.annualization_factor,
        )

        print(f"\nRule strategy selected threshold (train-only): {best_thr:.6f}")
        print(f"Rule selected features (train-only): {fitted_model.feature_names}")
        print_metric_block("Rule Train", rule_train_metrics)
        print_metric_block("Rule Test", rule_test_metrics)

        save_standard_plots(rule_train_bt, output_dir, "rule_train")
        save_standard_plots(rule_test_bt, output_dir, "rule_test")
        plot_threshold_sensitivity(
            sensitivity_df=sensitivity_df,
            save_path=output_dir / "rule_threshold_sensitivity.png",
            title="Rule Threshold Sensitivity (Train Sharpe)",
        )

        report["rule"] = {
            "selected_threshold": best_thr,
            "selected_features": fitted_model.feature_names,
            "train_metrics": rule_train_metrics,
            "test_metrics": rule_test_metrics,
            "threshold_sensitivity": sensitivity_df.to_dict(orient="records"),
        }

    report_path = output_dir / "report.txt"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {report_path}")
    print(f"Saved plots in: {output_dir}")


if __name__ == "__main__":
    main()
