from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALLOWED_SYMBOLS = ("PEPEUSDT", "FLOKIUSDT")


@dataclass(frozen=True)
class PlotConfig:
    symbol: str
    features_parquet: Path
    hazard_probs_csv: Path
    out_dir: Path


def _log(message: str) -> None:
    print(f"[info] {message}")


def _to_open_time_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return (dt.view("int64") // 1_000_000).astype("Int64")
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Features parquet has no rows: {path}")
    required = {"open_time", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Features parquet missing required columns: {missing}")

    out = df.copy()
    out["open_time"] = _to_open_time_ms(out["open_time"])
    out = out.dropna(subset=["open_time"]).copy()
    out["open_time"] = out["open_time"].astype("int64")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["close"]).copy()
    out["close"] = out["close"].astype("float64")

    if "ret_1m" not in out.columns:
        out["ret_1m"] = np.log(out["close"]).diff()
    else:
        out["ret_1m"] = pd.to_numeric(out["ret_1m"], errors="coerce")

    out["ret_1m"] = out["ret_1m"].astype("float64")
    out = out.sort_values("open_time", kind="mergesort").drop_duplicates(
        subset=["open_time"], keep="last"
    )
    return out[["open_time", "close", "ret_1m"]].reset_index(drop=True)


def _load_probs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Hazard probs CSV has no rows: {path}")
    if "open_time" not in df.columns:
        raise ValueError(
            "hazard_probs_csv is missing 'open_time'. Row-aligned fallback is not supported; "
            "please include open_time so data can be merged safely."
        )
    required = {"y_prob", "y_true"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Hazard probs CSV missing required columns: {missing}")

    out = df.copy()
    out["open_time"] = _to_open_time_ms(out["open_time"])
    out = out.dropna(subset=["open_time"]).copy()
    out["open_time"] = out["open_time"].astype("int64")
    out["y_prob"] = pd.to_numeric(out["y_prob"], errors="coerce")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out = out.dropna(subset=["y_prob", "y_true"]).copy()
    out["y_prob"] = out["y_prob"].astype("float64")
    out["y_true"] = out["y_true"].astype("int64")
    out = out.sort_values("open_time", kind="mergesort").drop_duplicates(
        subset=["open_time"], keep="last"
    )
    return out[["open_time", "y_prob", "y_true"]].reset_index(drop=True)


def _plot_hazard_vs_price(df: pd.DataFrame, out_path: Path, symbol: str) -> None:
    x = np.arange(len(df), dtype=int)
    fig, ax_left = plt.subplots(figsize=(12, 6))
    line_left = ax_left.plot(x, df["close"], color="#1f77b4", linewidth=1.0, label="close")
    ax_left.set_xlabel("Time Index")
    ax_left.set_ylabel("Close", color="#1f77b4")
    ax_left.tick_params(axis="y", labelcolor="#1f77b4")

    ax_right = ax_left.twinx()
    line_right = ax_right.plot(
        x, df["y_prob"], color="#d62728", linewidth=1.0, alpha=0.9, label="y_prob"
    )
    ax_right.set_ylabel("Hazard Probability", color="#d62728")
    ax_right.tick_params(axis="y", labelcolor="#d62728")
    ax_right.set_ylim(0.0, 1.0)

    lines = line_left + line_right
    labels = [line.get_label() for line in lines]
    ax_left.legend(lines, labels, loc="upper left")
    ax_left.set_title(f"{symbol} Hazard Probability vs Price")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_hazard_vs_ret_scatter(df: pd.DataFrame, out_path: Path, symbol: str) -> None:
    work = df.dropna(subset=["ret_1m", "y_prob"]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(work["ret_1m"], work["y_prob"], s=8, alpha=0.25, color="#1f77b4")
    ax.set_title(f"{symbol} Hazard Probability vs 1m Return")
    ax.set_xlabel("ret_1m")
    ax.set_ylabel("y_prob")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_hazard_vs_cumret_30m(df: pd.DataFrame, out_path: Path, symbol: str) -> None:
    work = df.copy()
    work["cumret_30m"] = work["ret_1m"].rolling(window=30, min_periods=30).sum()
    work = work.dropna(subset=["cumret_30m", "y_prob"]).copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(work["cumret_30m"], work["y_prob"], s=8, alpha=0.25, color="#d62728")
    ax.set_title(f"{symbol} Hazard Probability vs 30m Cumulative Return")
    ax.set_xlabel("cumret_30m")
    ax.set_ylabel("y_prob")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hazard_diagnostics(config: PlotConfig) -> dict[str, str]:
    features_df = _load_features(config.features_parquet)
    probs_df = _load_probs(config.hazard_probs_csv)

    merged = features_df.merge(probs_df, on="open_time", how="inner", sort=True)
    if merged.empty:
        raise ValueError("Merged dataset is empty after inner join on open_time.")

    merged = merged.sort_values("open_time", kind="mergesort").reset_index(drop=True)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    plot_price = config.out_dir / f"{config.symbol}_hazard_vs_price.png"
    plot_ret = config.out_dir / f"{config.symbol}_hazard_vs_ret_scatter.png"
    plot_cumret = config.out_dir / f"{config.symbol}_hazard_vs_cumret_30m.png"

    _plot_hazard_vs_price(merged, plot_price, config.symbol)
    _plot_hazard_vs_ret_scatter(merged, plot_ret, config.symbol)
    _plot_hazard_vs_cumret_30m(merged, plot_cumret, config.symbol)

    _log(f"Merged rows: {len(merged):,}")
    _log(f"Wrote plot: {plot_price}")
    _log(f"Wrote plot: {plot_ret}")
    _log(f"Wrote plot: {plot_cumret}")

    return {
        "hazard_vs_price": str(plot_price),
        "hazard_vs_ret_scatter": str(plot_ret),
        "hazard_vs_cumret_30m": str(plot_cumret),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hazard diagnostics against price and returns."
    )
    parser.add_argument("--symbol", required=True, choices=ALLOWED_SYMBOLS)
    parser.add_argument("--features_parquet", required=True)
    parser.add_argument("--hazard_probs_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PlotConfig(
        symbol=args.symbol,
        features_parquet=Path(args.features_parquet),
        hazard_probs_csv=Path(args.hazard_probs_csv),
        out_dir=Path(args.out_dir),
    )
    plot_hazard_diagnostics(config)


if __name__ == "__main__":
    main()
