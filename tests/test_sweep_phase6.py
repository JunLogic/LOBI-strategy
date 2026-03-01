from __future__ import annotations

import pandas as pd

from lobi_backtest.sweep.core import (
    RunPlan,
    canonical_json_dumps,
    compute_run_tag,
    expand_grid,
    select_runs_for_execution,
)


def _plan(run_tag: str) -> RunPlan:
    return RunPlan(
        run_index=1,
        overrides={},
        effective_config={},
        canonical_run_config_json="{}",
        run_config_hash="h",
        dataset_id="d",
        run_tag=run_tag,
        flattened_params={},
    )


def test_canonical_json_is_stable_across_dict_insertion_order() -> None:
    first = {
        "b": {"z": 3.0, "a": [2, 1.5]},
        "a": 1,
    }
    second = {
        "a": 1,
        "b": {"a": [2, 1.5], "z": 3.0},
    }
    assert canonical_json_dumps(first) == canonical_json_dumps(second)


def test_grid_expansion_order_is_deterministic() -> None:
    grid = {
        "momentum.lookback_bars": [120, 240],
        "momentum.downside_trigger_bp": [40, 60],
    }
    expanded = expand_grid(grid)
    assert expanded == [
        {"momentum.lookback_bars": 120, "momentum.downside_trigger_bp": 40},
        {"momentum.lookback_bars": 120, "momentum.downside_trigger_bp": 60},
        {"momentum.lookback_bars": 240, "momentum.downside_trigger_bp": 40},
        {"momentum.lookback_bars": 240, "momentum.downside_trigger_bp": 60},
    ]


def test_run_tag_is_stable_for_same_inputs() -> None:
    canonical = canonical_json_dumps({"hazard": {"hazard_threshold": 0.65}, "momentum": {"lookback_bars": 240}})
    first = compute_run_tag(canonical, dataset_id="dataset-1", code_id="code-1")
    second = compute_run_tag(canonical, dataset_id="dataset-1", code_id="code-1")
    assert first == second
    assert len(first) == 16


def test_resume_skips_completed_ok_but_keeps_errors() -> None:
    plans = [_plan("a"), _plan("b"), _plan("c")]
    existing = pd.DataFrame(
        [
            {"run_tag": "a", "status": "ok"},
            {"run_tag": "b", "status": "error"},
        ]
    )
    selected = select_runs_for_execution(
        plans=plans,
        existing_results=existing,
        resume=True,
        force_rerun=False,
        max_runs=None,
    )
    assert [plan.run_tag for plan in selected] == ["b", "c"]

