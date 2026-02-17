"""Plotting utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import compute_drawdown_series


def ensure_output_dir(output_dir: str = "outputs") -> Path:
    """Create output directory if needed."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_equity_curve(equity: pd.Series, save_path: Path, title: str) -> None:
    """Save equity curve plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Equity (proxy units)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_drawdown_curve(equity: pd.Series, save_path: Path, title: str) -> None:
    """Save drawdown curve plot."""
    drawdown = compute_drawdown_series(equity)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(drawdown.index, drawdown.values, color="tab:red", linewidth=1.2)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="tab:red", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_pnl_histogram(pnl: pd.Series, save_path: Path, title: str) -> None:
    """Save histogram of per-step PnL."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnl.values, bins=50, alpha=0.85, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Per-step PnL")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_threshold_sensitivity(sensitivity_df: pd.DataFrame, save_path: Path, title: str) -> None:
    """Save threshold vs train-sharpe plot for rule strategy."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sensitivity_df["threshold"], sensitivity_df["sharpe_per_step"], marker="o", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Sharpe (per-step)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)

