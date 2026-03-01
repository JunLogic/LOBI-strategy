from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

from lobi_backtest.sweep.core import build_human_run_tag, collect_resume_tags, run_sweep
from scripts import sweep_momentum


def test_script_wrapper_delegates_to_core(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_run_sweep(**kwargs):  # type: ignore[no-untyped-def]
        called.update(kwargs)
        return {
            "spec_hash": "abc",
            "output_root": "outputs/mbars",
            "sweep_id": "sweep_20260302_000000",
            "symbols": {"FLOKIUSDT": {"total_grid_runs": 1, "scheduled_runs": 1, "skipped": 0, "errors": 0, "sweep_dir": "x"}},
        }

    monkeypatch.setattr(sweep_momentum, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep_momentum.py", "--spec", "dummy.yaml", "--symbol", "FLOKIUSDT", "--dry-run"],
    )
    sweep_momentum.main()

    assert called["spec_path"] == Path("dummy.yaml")
    assert called["symbol"] == "FLOKIUSDT"
    assert called["output_root"] == Path("outputs") / "mbars"


def test_dry_run_uses_outputs_mbars_layout(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "symbols:",
                "  - FLOKIUSDT",
                "base: {}",
                "grids:",
                "  hazard_threshold: [0.6]",
            ]
        ),
        encoding="utf-8",
    )

    out_root = tmp_path / "outputs" / "mbars"
    result = run_sweep(
        spec_path=spec_path,
        dry_run=True,
        max_runs=None,
        force_rerun=False,
        symbol="FLOKIUSDT",
        all_symbols=False,
        output_root=out_root,
    )
    meta = result["symbols"]["FLOKIUSDT"]
    sweep_dir = Path(meta["sweep_dir"])
    assert str(sweep_dir).startswith(str(out_root / "FLOKIUSDT" / "sweeps"))
    assert "runs" not in {part.lower() for part in sweep_dir.parts}


def test_run_tag_is_human_readable() -> None:
    run_tag = build_human_run_tag(
        hazard_threshold=0.6,
        downside_threshold=-0.003,
        stoploss=0.008,
        takeprofit=None,
        sizing_mode="equity_fraction",
        rolling_bars=6,
        risk_values={},
    )
    assert "_" in run_tag
    assert re.fullmatch(r"[0-9a-f]{16}", run_tag) is None


def test_resume_parquet_first_and_summary_fallback(tmp_path: Path) -> None:
    symbol_root = tmp_path / "outputs" / "mbars" / "FLOKIUSDT"
    parquet_dir = symbol_root / "sweeps" / "sweep_old"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"run_tag": "tag_ok", "status": "ok"}]).to_parquet(parquet_dir / "results.parquet", index=False)

    summary_dir = symbol_root / "tag_from_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    ok_tags, _, source = collect_resume_tags(symbol_root=symbol_root, skip_errors=False)
    assert "tag_ok" in ok_tags
    assert "tag_from_summary" not in ok_tags
    assert source == "parquet"

    corrupted_root = tmp_path / "outputs" / "mbars" / "PEPEUSDT"
    corrupted_parquet_dir = corrupted_root / "sweeps" / "sweep_bad"
    corrupted_parquet_dir.mkdir(parents=True, exist_ok=True)
    (corrupted_parquet_dir / "results.parquet").write_text("not parquet", encoding="utf-8")
    fallback_summary_dir = corrupted_root / "summary_fallback_tag"
    fallback_summary_dir.mkdir(parents=True, exist_ok=True)
    (fallback_summary_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    ok_tags_fallback, _, fallback_source = collect_resume_tags(symbol_root=corrupted_root, skip_errors=False)
    assert "summary_fallback_tag" in ok_tags_fallback
    assert "summary_fallback" in fallback_source

