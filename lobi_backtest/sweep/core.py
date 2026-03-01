from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from mbars.backtest import run_momentum_backtest
from mbars.config import RiskConfig
from mbars.params import build_symbol_params
from mbars.position_sizing import PositionSizingConfig

PREVIEW_DRY_RUN_COUNT = 5
DEFAULT_LOG_EVERY = 100_000


@dataclass(frozen=True)
class SweepSpec:
    path: Path
    raw_text: str
    spec_hash: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class RunPlan:
    run_index: int
    overrides: dict[str, Any]
    effective_config: dict[str, Any]
    canonical_run_config_json: str
    run_config_hash: str
    dataset_id: str
    run_tag: str
    flattened_params: dict[str, Any]
    run_tag_hash: str = ""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a mapping.")
    return value


def _validate_spec(payload: dict[str, Any]) -> None:
    required_top_level = [
        "version",
        "sweep_name",
        "dataset",
        "engine",
        "hazard",
        "base_params",
        "sweep",
        "output",
    ]
    missing = [k for k in required_top_level if k not in payload]
    if missing:
        raise ValueError(f"Sweep spec missing required top-level keys: {missing}")

    if not isinstance(payload["version"], int):
        raise ValueError("'version' must be an integer.")
    if not isinstance(payload["sweep_name"], str) or not payload["sweep_name"].strip():
        raise ValueError("'sweep_name' must be a non-empty string.")

    _ensure_mapping(payload["dataset"], "dataset")
    _ensure_mapping(payload["engine"], "engine")
    _ensure_mapping(payload["hazard"], "hazard")
    _ensure_mapping(payload["base_params"], "base_params")
    sweep_cfg = _ensure_mapping(payload["sweep"], "sweep")
    output_cfg = _ensure_mapping(payload["output"], "output")

    sweep_type = sweep_cfg.get("type")
    if sweep_type != "grid":
        raise ValueError("Only sweep.type='grid' is supported.")

    grid = _ensure_mapping(sweep_cfg.get("grid"), "sweep.grid")
    if not grid:
        raise ValueError("'sweep.grid' must contain at least one parameter.")

    for key, values in grid.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("All sweep.grid keys must be non-empty strings.")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"sweep.grid['{key}'] must be a non-empty list.")

    filters = sweep_cfg.get("filters", [])
    if filters is None:
        filters = []
    if not isinstance(filters, list) or any(not isinstance(v, str) for v in filters):
        raise ValueError("'sweep.filters' must be a list of strings when provided.")

    if "root_dir" not in output_cfg:
        raise ValueError("'output.root_dir' is required.")

    results_table = str(output_cfg.get("results_table", "parquet")).lower()
    if results_table != "parquet":
        raise ValueError("Only 'output.results_table: parquet' is supported.")


def load_sweep_spec(spec_path: Path) -> SweepSpec:
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Sweep spec not found: {spec_path}")

    raw_text = spec_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a YAML mapping.")

    _validate_spec(payload)
    spec_hash = _sha256_hex(raw_text)
    return SweepSpec(path=spec_path, raw_text=raw_text, spec_hash=spec_hash, payload=payload)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Canonicalization does not support NaN or Infinity.")
    text = format(value, ".15g")
    if text == "-0":
        return "0"
    return text


def canonical_json_dumps(value: Any) -> str:
    """Produce canonical JSON with sorted keys and stable float formatting."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        items: list[str] = []
        for key in sorted(value.keys()):
            if not isinstance(key, str):
                raise TypeError("Canonical JSON only supports string dictionary keys.")
            item = (
                f"{json.dumps(key, ensure_ascii=False, separators=(',', ':'))}:"
                f"{canonical_json_dumps(value[key])}"
            )
            items.append(item)
        return "{" + ",".join(items) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        inner = ",".join(canonical_json_dumps(item) for item in value)
        return "[" + inner + "]"
    raise TypeError(f"Unsupported type for canonical JSON: {type(value)!r}")


def _set_dotted_key(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")

    cursor: dict[str, Any] = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            next_node: dict[str, Any] = {}
            cursor[part] = next_node
            cursor = next_node
            continue
        if not isinstance(existing, dict):
            raise ValueError(f"Dotted override conflicts with non-mapping key: {dotted_key!r}")
        cursor = existing
    cursor[parts[-1]] = deepcopy(value)


def _build_filter_context(overrides: Mapping[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        _set_dotted_key(nested, key, value)

    def _to_namespace(obj: Any) -> Any:
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
        return obj

    return {key: _to_namespace(value) for key, value in nested.items()}


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def _validate_filter_ast(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported expression element in filter: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Invalid filter variable.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Invalid filter attribute.")
    return tree


def _evaluate_filter(expression: str, overrides: Mapping[str, Any]) -> bool:
    tree = _validate_filter_ast(expression)
    context = _build_filter_context(overrides)
    result = eval(  # noqa: S307
        compile(tree, filename="<sweep-filter>", mode="eval"),
        {"__builtins__": {}},
        context,
    )
    return bool(result)


def expand_grid(grid: Mapping[str, list[Any]], filters: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Expand an ordered grid deterministically and apply optional post-expansion filters."""
    keys = list(grid.keys())
    value_lists = [grid[key] for key in keys]
    expressions = [expr for expr in (filters or []) if expr]

    combos: list[dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        override = {key: deepcopy(value) for key, value in zip(keys, values)}
        if expressions and not all(_evaluate_filter(expr, override) for expr in expressions):
            continue
        combos.append(override)
    return combos


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key in sorted(mapping.keys()):
        value = mapping[key]
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _merge_effective_config(spec: SweepSpec, overrides: Mapping[str, Any]) -> dict[str, Any]:
    payload = spec.payload
    effective: dict[str, Any] = {
        "dataset": deepcopy(payload["dataset"]),
        "hazard": deepcopy(payload["hazard"]),
    }
    for key, value in deepcopy(payload["base_params"]).items():
        effective[key] = value
    for dotted_key, value in overrides.items():
        _set_dotted_key(effective, dotted_key, value)
    return effective


def resolve_code_id(cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except Exception:  # noqa: BLE001
        return "nogit"
    if completed.returncode != 0:
        return "nogit"
    head = completed.stdout.strip()
    return head if head else "nogit"


def compute_dataset_id(dataset: Mapping[str, Any]) -> str:
    fingerprint = dataset.get("fingerprint")
    if fingerprint is not None and str(fingerprint).strip():
        return str(fingerprint).strip()

    metadata: dict[str, Any] = {}
    for key in sorted(dataset.keys()):
        if key == "fingerprint":
            continue
        value = dataset[key]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and (key.endswith("_path") or key.endswith("_root")):
            path = Path(value)
            info: dict[str, Any] = {"path": value, "exists": path.exists()}
            if path.exists():
                stat = path.stat()
                info.update(
                    {
                        "is_dir": path.is_dir(),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
            metadata[key] = info
            continue
        metadata[key] = value
    return _sha256_hex(canonical_json_dumps(metadata))


def compute_run_tag(canonical_run_config_json: str, dataset_id: str, code_id: str) -> str:
    payload = f"{canonical_run_config_json}|{dataset_id}|{code_id}"
    return _sha256_hex(payload)[:16]


def _normalize_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _resolve_bars_path(dataset: Mapping[str, Any]) -> Path:
    symbol = str(dataset.get("symbol", "")).strip()
    if not symbol:
        raise ValueError("dataset.symbol is required.")

    bars_path_value = dataset.get("bars_path")
    if bars_path_value is not None:
        bars_path = _normalize_path(bars_path_value)
    else:
        bars_root_value = dataset.get("bars_root")
        if bars_root_value is None:
            raise ValueError("dataset must define either bars_path or bars_root.")
        bars_root = _normalize_path(bars_root_value)
        if bars_root.suffix.lower() == ".parquet":
            bars_path = bars_root
        else:
            bars_path = bars_root / f"{symbol}_bars_5s_hazard.parquet"

    if not bars_path.exists():
        raise FileNotFoundError(f"Bars parquet not found: {bars_path}")
    return bars_path


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid float for {field_name}: {value!r}") from exc


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid int for {field_name}: {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_engine_configs(
    effective_config: Mapping[str, Any],
) -> tuple[PositionSizingConfig, RiskConfig, dict[str, Any]]:
    momentum = effective_config.get("momentum", {})
    if not isinstance(momentum, Mapping):
        raise ValueError("Effective config field 'momentum' must be a mapping.")

    risk_values = effective_config.get("risk", {})
    if not isinstance(risk_values, Mapping):
        raise ValueError("Effective config field 'risk' must be a mapping.")

    sizing_values = effective_config.get("sizing", {})
    if not isinstance(sizing_values, Mapping):
        raise ValueError("Effective config field 'sizing' must be a mapping.")

    hazard_values = effective_config.get("hazard", {})
    if not isinstance(hazard_values, Mapping):
        raise ValueError("Effective config field 'hazard' must be a mapping.")

    lookback_default = effective_config.get("rolling_bars", 6)
    lookback_bars = _as_int(
        momentum.get("lookback_bars", momentum.get("rolling_bars", lookback_default)),
        "momentum.lookback_bars",
    )

    downside_return_threshold: float
    if "downside_return_threshold" in momentum:
        downside_return_threshold = _as_float(
            momentum["downside_return_threshold"], "momentum.downside_return_threshold"
        )
    elif "downside_trigger_bp" in momentum:
        downside_bp = _as_float(momentum["downside_trigger_bp"], "momentum.downside_trigger_bp")
        downside_return_threshold = -abs(downside_bp) / 10_000.0
    elif "downside_threshold" in effective_config:
        downside_return_threshold = _as_float(
            effective_config["downside_threshold"], "downside_threshold"
        )
    else:
        downside_return_threshold = -0.003

    stoploss_pct = _as_float(
        momentum.get(
            "stoploss_pct", effective_config.get("stoploss_pct", effective_config.get("stoploss", 0.008))
        ),
        "stoploss_pct",
    )
    takeprofit_pct = _optional_float(
        momentum.get("takeprofit_pct", effective_config.get("takeprofit_pct", effective_config.get("takeprofit")))
    )

    initial_equity = _as_float(
        effective_config.get("initial_equity", momentum.get("initial_equity", 100_000.0)),
        "initial_equity",
    )

    hazard_threshold = _as_float(
        hazard_values.get("hazard_threshold", effective_config.get("hazard_threshold", 0.6)),
        "hazard.hazard_threshold",
    )

    sizing_config = PositionSizingConfig(
        mode=str(sizing_values.get("mode", "equity_fraction")),
        fixed_size=_as_float(sizing_values.get("fixed_size", 1.0), "sizing.fixed_size"),
        equity_fraction=_as_float(sizing_values.get("equity_fraction", 0.02), "sizing.equity_fraction"),
        max_notional=_optional_float(sizing_values.get("max_notional")),
    )

    rebound_kill_enabled = risk_values.get("rebound_kill", None)
    if rebound_kill_enabled is False:
        rebound_kill_pct = 0.0
    elif "rebound_kill_pct" in risk_values:
        rebound_kill_pct = _as_float(risk_values["rebound_kill_pct"], "risk.rebound_kill_pct")
    elif rebound_kill_enabled is True and "entry_rebound_bp" in momentum:
        rebound_kill_pct = abs(_as_float(momentum["entry_rebound_bp"], "momentum.entry_rebound_bp")) / 10_000.0
    else:
        rebound_kill_pct = 0.02

    risk_config = RiskConfig(
        max_open_positions=_as_int(risk_values.get("max_open_positions", 3), "risk.max_open_positions"),
        max_notional_frac=_as_float(
            risk_values.get("max_notional_frac", risk_values.get("exposure_cap", 0.10)),
            "risk.max_notional_frac",
        ),
        min_position_size=_as_float(risk_values.get("min_position_size", 0.0), "risk.min_position_size"),
        max_hold_bars=_as_int(risk_values.get("max_hold_bars", 720), "risk.max_hold_bars"),
        equity_floor_frac=_as_float(risk_values.get("equity_floor_frac", 0.2), "risk.equity_floor_frac"),
        rebound_kill_pct=rebound_kill_pct,
        cooldown_bars=_as_int(risk_values.get("cooldown_bars", 120), "risk.cooldown_bars"),
    )

    engine_params = {
        "rolling_bars": lookback_bars,
        "downside_return_threshold": downside_return_threshold,
        "stoploss_pct": stoploss_pct,
        "takeprofit_pct": takeprofit_pct,
        "initial_equity": initial_equity,
        "hazard_threshold": hazard_threshold,
    }
    return sizing_config, risk_config, engine_params


def build_run_plans(spec: SweepSpec, code_id: str) -> list[RunPlan]:
    sweep_cfg = spec.payload["sweep"]
    grid = sweep_cfg["grid"]
    filters = sweep_cfg.get("filters", [])

    combos = expand_grid(grid=grid, filters=filters)
    plans: list[RunPlan] = []
    for run_index, overrides in enumerate(combos, start=1):
        effective_config = _merge_effective_config(spec, overrides)
        dataset_id = compute_dataset_id(effective_config["dataset"])
        canonical_run_config_json = canonical_json_dumps(effective_config)
        run_config_hash = _sha256_hex(canonical_run_config_json)
        run_tag = compute_run_tag(canonical_run_config_json, dataset_id, code_id)
        flattened = _flatten_mapping({k: v for k, v in effective_config.items() if k != "dataset"})
        plans.append(
            RunPlan(
                run_index=run_index,
                overrides=overrides,
                effective_config=effective_config,
                canonical_run_config_json=canonical_run_config_json,
                run_config_hash=run_config_hash,
                dataset_id=dataset_id,
                run_tag=run_tag,
                flattened_params=flattened,
            )
        )
    return plans


def load_results_table(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(results_path)


def _ok_run_tags(existing_results: pd.DataFrame) -> set[str]:
    if existing_results.empty:
        return set()
    if "run_tag" not in existing_results.columns or "status" not in existing_results.columns:
        return set()
    mask = existing_results["status"].astype(str) == "ok"
    return set(existing_results.loc[mask, "run_tag"].astype(str).tolist())


def select_runs_for_execution(
    plans: Sequence[RunPlan],
    existing_results: pd.DataFrame,
    *,
    resume: bool,
    force_rerun: bool,
    max_runs: int | None,
) -> list[RunPlan]:
    if force_rerun:
        selected = list(plans)
    elif resume:
        completed_ok = _ok_run_tags(existing_results)
        selected = [plan for plan in plans if plan.run_tag not in completed_ok]
    else:
        selected = list(plans)

    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive when provided.")
        selected = selected[:max_runs]
    return selected


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_results_row(results_path: Path, row: Mapping[str, Any]) -> None:
    if results_path.exists():
        df = pd.read_parquet(results_path)
    else:
        df = pd.DataFrame()

    next_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = results_path.with_suffix(f"{results_path.suffix}.tmp")
    next_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, results_path)


def _build_results_row(
    *,
    plan: RunPlan,
    status: str,
    code_id: str,
    spec_hash: str,
    started_utc: str,
    ended_utc: str,
    runtime_seconds: float,
    summary: Mapping[str, Any] | None,
    error_message: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_tag": plan.run_tag,
        "status": status,
        "runtime_seconds": float(runtime_seconds),
        "code_id": code_id,
        "dataset_id": plan.dataset_id,
        "spec_hash": spec_hash,
        "run_config_hash": plan.run_config_hash,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "error": error_message,
    }

    for key in sorted(plan.flattened_params.keys()):
        row[f"param_{key.replace('.', '__')}"] = plan.flattened_params[key]

    if summary is not None:
        initial_equity = summary.get("initial_equity")
        final_equity = summary.get("final_equity")
        total_pnl = summary.get("total_pnl")
        if total_pnl is None and initial_equity is not None and final_equity is not None:
            total_pnl = float(final_equity) - float(initial_equity)
        row.update(
            {
                "pnl_proxy_end": total_pnl,
                "final_equity": final_equity,
                "max_drawdown": summary.get("max_drawdown"),
                "trades": summary.get("closed_trades"),
                "win_rate": summary.get("win_rate"),
                "kill_switch_triggers": summary.get("kill_switch_triggers"),
                "forced_exits_max_hold": summary.get("forced_exits_max_hold"),
                "entries_skipped_max_pos": summary.get("entries_skipped_max_pos"),
                "entries_skipped_notional": summary.get("entries_skipped_notional"),
                "entries_scaled_notional": summary.get("entries_scaled_notional"),
                "cooldown_total_bars": summary.get("cooldown_total_bars"),
            }
        )
    else:
        row.update(
            {
                "pnl_proxy_end": None,
                "final_equity": None,
                "max_drawdown": None,
                "trades": None,
                "win_rate": None,
                "kill_switch_triggers": None,
                "forced_exits_max_hold": None,
                "entries_skipped_max_pos": None,
                "entries_skipped_notional": None,
                "entries_scaled_notional": None,
                "cooldown_total_bars": None,
            }
        )
    return row


def _runtime_seconds(started_utc: str, ended_utc: str) -> float:
    started = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    ended = datetime.fromisoformat(ended_utc.replace("Z", "+00:00"))
    return max(0.0, (ended - started).total_seconds())


def _run_single(*, plan: RunPlan, spec: SweepSpec, output_root: Path) -> dict[str, Any]:
    sweep_name = str(spec.payload["sweep_name"])
    sweep_root = output_root / sweep_name
    run_dir = sweep_root / plan.run_tag

    effective_config = plan.effective_config
    dataset = effective_config["dataset"]
    dataset_symbol = str(dataset.get("symbol", "")).strip()
    if not dataset_symbol:
        raise ValueError("dataset.symbol is required and must be non-empty.")

    bars_path = _resolve_bars_path(dataset)
    sizing_config, risk_config, engine_params = _build_engine_configs(effective_config)

    engine_cfg = spec.payload["engine"]
    log_every = _as_int(engine_cfg.get("log_every", DEFAULT_LOG_EVERY), "engine.log_every")

    summary = run_momentum_backtest(
        symbol=sweep_name,
        bars_path=bars_path,
        out_dir=output_root,
        hazard_threshold=float(engine_params["hazard_threshold"]),
        downside_return_threshold=float(engine_params["downside_return_threshold"]),
        stoploss_pct=float(engine_params["stoploss_pct"]),
        takeprofit_pct=_optional_float(engine_params["takeprofit_pct"]),
        initial_equity=float(engine_params["initial_equity"]),
        sizing_config=sizing_config,
        risk_config=risk_config,
        rolling_bars=int(engine_params["rolling_bars"]),
        run_tag=plan.run_tag,
        log_every=log_every,
    )

    summary["symbol"] = dataset_symbol
    summary["sweep_name"] = sweep_name
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_plots = bool(engine_cfg.get("write_plots", False))
    if write_plots:
        from mbars.diagnostics import write_diagnostics

        write_diagnostics(run_dir)

    write_timeseries = bool(engine_cfg.get("write_per_run_timeseries", True))
    if not write_timeseries:
        for name in ("trades.csv", "equity.csv"):
            path = run_dir / name
            if path.exists():
                path.unlink()

    return summary


def _write_run_artifacts(
    *,
    run_dir: Path,
    plan: RunPlan,
    code_id: str,
    spec: SweepSpec,
    started_utc: str,
    ended_utc: str | None,
    status: str,
) -> None:
    _write_json(run_dir / "config_effective.json", plan.effective_config)
    provenance: dict[str, Any] = {
        "run_tag": plan.run_tag,
        "run_index": plan.run_index,
        "status": status,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "code_id": code_id,
        "dataset_id": plan.dataset_id,
        "spec_hash": spec.spec_hash,
        "run_config_hash": plan.run_config_hash,
        "spec_path": str(spec.path),
        "sweep_name": str(spec.payload["sweep_name"]),
    }
    _write_json(run_dir / "provenance.json", provenance)


def _write_sweep_provenance(spec: SweepSpec, sweep_root: Path, code_id: str) -> None:
    provenance = {
        "sweep_name": str(spec.payload["sweep_name"]),
        "spec_path": str(spec.path),
        "spec_hash": spec.spec_hash,
        "code_id": code_id,
        "created_utc": _utc_now_iso(),
        "engine_version": None,
    }
    _write_json(sweep_root / "provenance.json", provenance)

    output_cfg = spec.payload["output"]
    if bool(output_cfg.get("save_spec_copy", False)):
        copied_path = sweep_root / spec.path.name
        shutil.copy2(spec.path, copied_path)


def _effective_max_runs(spec: SweepSpec, cli_max_runs: int | None) -> int | None:
    if cli_max_runs is not None:
        if cli_max_runs <= 0:
            raise ValueError("--max-runs must be positive when provided.")
        return cli_max_runs

    engine_cfg = spec.payload["engine"]
    value = engine_cfg.get("max_runs", None)
    if value is None:
        return None
    parsed = _as_int(value, "engine.max_runs")
    if parsed <= 0:
        raise ValueError("engine.max_runs must be positive when provided.")
    return parsed


def _print_dry_run(plans: Sequence[RunPlan], selected_plans: Sequence[RunPlan]) -> None:
    print(f"[sweep] total_grid_runs={len(plans)}")
    print(f"[sweep] scheduled_runs={len(selected_plans)}")
    preview = list(selected_plans[:PREVIEW_DRY_RUN_COUNT])
    for idx, plan in enumerate(preview, start=1):
        override_json = json.dumps(plan.overrides, separators=(",", ":"), sort_keys=False)
        effective_json = json.dumps(plan.flattened_params, separators=(",", ":"), sort_keys=True)
        print(
            f"[sweep] dry_run[{idx}] run_tag={plan.run_tag} overrides={override_json} "
            f"effective_params={effective_json}"
        )


def _load_repo_spec(spec_path: Path) -> SweepSpec:
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Sweep spec not found: {spec_path}")
    raw_text = spec_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Sweep spec must be a YAML mapping.")
    return SweepSpec(
        path=spec_path,
        raw_text=raw_text,
        spec_hash=_sha256_hex(raw_text),
        payload=payload,
    )


def _fmt_tag_float(value: float | int | None) -> str:
    if value is None:
        return "none"
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    text = text.replace("-", "m").replace(".", "p")
    return text if text else "0"


def _risk_delta_suffix(risk_values: Mapping[str, Any]) -> str:
    defaults = RiskConfig()
    fields = [
        ("max_open_positions", "mop", defaults.max_open_positions),
        ("max_notional_frac", "mnf", defaults.max_notional_frac),
        ("min_position_size", "mps", defaults.min_position_size),
        ("max_hold_bars", "mhb", defaults.max_hold_bars),
        ("equity_floor_frac", "eff", defaults.equity_floor_frac),
        ("rebound_kill_pct", "rkp", defaults.rebound_kill_pct),
        ("cooldown_bars", "cdb", defaults.cooldown_bars),
    ]
    parts: list[str] = []
    for key, alias, default in fields:
        value = risk_values.get(key, default)
        if value != default:
            parts.append(f"{alias}{_fmt_tag_float(value)}")
    return ("_" + "_".join(parts)) if parts else ""


def build_human_run_tag(
    *,
    hazard_threshold: float,
    downside_threshold: float,
    stoploss: float,
    takeprofit: float | None,
    sizing_mode: str,
    rolling_bars: int,
    risk_values: Mapping[str, Any],
) -> str:
    base = (
        f"ht{_fmt_tag_float(hazard_threshold)}"
        f"_dt{_fmt_tag_float(downside_threshold)}"
        f"_sl{_fmt_tag_float(stoploss)}"
        f"_tp{_fmt_tag_float(takeprofit)}"
        f"_sz{sizing_mode}"
        f"_rb{int(rolling_bars)}"
    )
    return base + _risk_delta_suffix(risk_values)


def _is_hex_16(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{16}", value))


def _normalize_legacy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "symbols" in payload and "base" in payload and "grids" in payload:
        return {
            "symbols": list(payload["symbols"]),
            "base": deepcopy(payload["base"]),
            "grids": deepcopy(payload["grids"]),
            "filters": deepcopy(payload.get("filters", [])),
            "engine": deepcopy(payload.get("engine", {})),
            "shuffle": payload.get("shuffle", False),
            "random_seed": payload.get("random_seed", 0),
            "max_runs": payload.get("max_runs"),
            "output": deepcopy(payload.get("output", {})),
            "dataset": deepcopy(payload.get("dataset", {})),
        }

    if {
        "version",
        "sweep_name",
        "dataset",
        "engine",
        "hazard",
        "base_params",
        "sweep",
        "output",
    }.issubset(set(payload.keys())):
        dataset = _ensure_mapping(payload["dataset"], "dataset")
        base_params = deepcopy(_ensure_mapping(payload["base_params"], "base_params"))
        hazard = deepcopy(_ensure_mapping(payload["hazard"], "hazard"))
        sweep = deepcopy(_ensure_mapping(payload["sweep"], "sweep"))
        grid = deepcopy(_ensure_mapping(sweep.get("grid"), "sweep.grid"))
        filters = deepcopy(sweep.get("filters", [])) if sweep.get("filters") is not None else []
        symbol = dataset.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("dataset.symbol must be a non-empty string.")

        if "hazard_threshold" in hazard and "hazard_threshold" not in base_params:
            base_params["hazard_threshold"] = hazard["hazard_threshold"]
        if "momentum" in base_params and isinstance(base_params["momentum"], dict):
            momentum = base_params["momentum"]
            if "rolling_bars" not in base_params and "lookback_bars" in momentum:
                base_params["rolling_bars"] = momentum["lookback_bars"]
            if "downside_threshold" not in base_params and "downside_trigger_bp" in momentum:
                base_params["downside_threshold"] = -abs(float(momentum["downside_trigger_bp"])) / 10_000.0
            if "stoploss" not in base_params and "stoploss_pct" in momentum:
                base_params["stoploss"] = momentum["stoploss_pct"]
            if "takeprofit" not in base_params and "takeprofit_pct" in momentum:
                base_params["takeprofit"] = momentum["takeprofit_pct"]
        if "risk" in base_params and isinstance(base_params["risk"], dict):
            if "exposure_cap" in base_params["risk"] and "max_notional_frac" not in base_params["risk"]:
                base_params["risk"]["max_notional_frac"] = base_params["risk"]["exposure_cap"]

        return {
            "symbols": [symbol],
            "base": base_params,
            "grids": grid,
            "filters": filters,
            "engine": deepcopy(payload.get("engine", {})),
            "shuffle": False,
            "random_seed": 0,
            "max_runs": _ensure_mapping(payload["engine"], "engine").get("max_runs"),
            "output": deepcopy(payload.get("output", {})),
            "dataset": dataset,
        }

    raise ValueError(
        "Unsupported spec format. Expected legacy keys (symbols/base/grids) "
        "or Phase 6 keys (version/sweep_name/dataset/engine/base_params/sweep/output)."
    )


def _resolve_symbols_from_payload(
    payload: Mapping[str, Any],
    *,
    symbol: str | None,
    all_symbols: bool,
) -> list[str]:
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or any(not isinstance(item, str) for item in symbols):
        raise ValueError("Spec 'symbols' must be a list of strings.")
    if all_symbols:
        return list(symbols)
    if symbol:
        return [symbol]
    if len(symbols) == 1:
        return [symbols[0]]
    raise ValueError("Provide --symbol or --all-symbols when spec contains multiple symbols.")


def _resolve_bars_path_for_symbol(
    *,
    symbol: str,
    base_params: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> Path:
    bars_path_value = base_params.get("bars_path")
    if bars_path_value is not None:
        return Path(str(bars_path_value))
    if "bars_path" in dataset:
        return Path(str(dataset["bars_path"]))
    bars_root_value = dataset.get("bars_root")
    if bars_root_value is not None:
        bars_root = Path(str(bars_root_value))
        if bars_root.suffix.lower() == ".parquet":
            return bars_root
        return bars_root / f"{symbol}_bars_5s_hazard.parquet"
    return Path("data") / "processed" / f"{symbol}_bars_5s_hazard.parquet"


def _effective_params_with_overrides(base_params: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    effective = deepcopy(base_params)
    for key, value in overrides.items():
        if "." in key:
            _set_dotted_key(effective, key, value)
        else:
            effective[key] = deepcopy(value)
    return effective


def _effective_mbars_config(
    *,
    symbol: str,
    bars_path: Path,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    momentum_raw = params.get("momentum", {})
    momentum_values = deepcopy(momentum_raw) if isinstance(momentum_raw, Mapping) else {}
    hazard_threshold = params.get("hazard_threshold", 0.6)
    downside_threshold = params.get("downside_threshold")
    if downside_threshold is None:
        if "downside_trigger_bp" in momentum_values:
            downside_threshold = -abs(float(momentum_values["downside_trigger_bp"])) / 10_000.0
        else:
            downside_threshold = -0.003
    stoploss = params.get("stoploss")
    if stoploss is None:
        stoploss = momentum_values.get("stoploss_pct", 0.008)
    takeprofit = params.get("takeprofit")
    if takeprofit is None:
        takeprofit = momentum_values.get("takeprofit_pct")
    rolling_bars = params.get("rolling_bars")
    if rolling_bars is None:
        rolling_bars = momentum_values.get("lookback_bars", 6)

    risk_values = deepcopy(params.get("risk", {})) if isinstance(params.get("risk", {}), Mapping) else {}
    if "exposure_cap" in risk_values and "max_notional_frac" not in risk_values:
        risk_values["max_notional_frac"] = risk_values["exposure_cap"]
    if bool(risk_values.get("rebound_kill", True)) and "rebound_kill_pct" not in risk_values:
        if "entry_rebound_bp" in momentum_values:
            risk_values["rebound_kill_pct"] = abs(float(momentum_values["entry_rebound_bp"])) / 10_000.0
    if risk_values.get("rebound_kill", True) is False:
        risk_values["rebound_kill_pct"] = 0.0

    momentum_values["downside_return_threshold"] = float(downside_threshold)
    momentum_values["lookback_bars"] = int(rolling_bars)
    momentum_values["stoploss_pct"] = float(stoploss)
    momentum_values["takeprofit_pct"] = None if takeprofit is None else float(takeprofit)

    effective = {
        "dataset": {
            "symbol": symbol,
            "bars_path": str(bars_path),
        },
        "hazard": {"hazard_threshold": float(hazard_threshold)},
        "momentum": momentum_values,
        "risk": risk_values,
        "sizing": deepcopy(params.get("sizing", {})) if isinstance(params.get("sizing", {}), Mapping) else {},
        "initial_equity": float(params.get("initial_equity", 100_000.0)),
        "raw_params": deepcopy(params),
    }
    return effective


def _build_repo_run_plans(
    *,
    symbol: str,
    base_params: Mapping[str, Any],
    grid: Mapping[str, list[Any]],
    filters: Sequence[str],
    bars_path: Path,
    code_id: str,
    shuffle: bool,
    seed: int,
) -> list[RunPlan]:
    combos = expand_grid(grid=grid, filters=filters)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(combos)

    plans: list[RunPlan] = []
    for run_index, overrides in enumerate(combos, start=1):
        params = _effective_params_with_overrides(base_params, overrides)
        effective_config = _effective_mbars_config(symbol=symbol, bars_path=bars_path, params=params)
        sizing_values = effective_config.get("sizing", {})
        risk_values = effective_config.get("risk", {})
        run_tag = build_human_run_tag(
            hazard_threshold=float(effective_config["hazard"]["hazard_threshold"]),
            downside_threshold=float(effective_config["momentum"]["downside_return_threshold"]),
            stoploss=float(effective_config["momentum"]["stoploss_pct"]),
            takeprofit=effective_config["momentum"]["takeprofit_pct"],
            sizing_mode=str(sizing_values.get("mode", "equity_fraction")),
            rolling_bars=int(effective_config["momentum"]["lookback_bars"]),
            risk_values=risk_values if isinstance(risk_values, Mapping) else {},
        )
        canonical = canonical_json_dumps(effective_config)
        dataset_id = compute_dataset_id(effective_config["dataset"])
        run_tag_hash = compute_run_tag(canonical, dataset_id, code_id)
        flattened = _flatten_mapping(params)
        plans.append(
            RunPlan(
                run_index=run_index,
                overrides=deepcopy(overrides),
                effective_config=effective_config,
                canonical_run_config_json=canonical,
                run_config_hash=_sha256_hex(canonical),
                dataset_id=dataset_id,
                run_tag=run_tag,
                flattened_params=flattened,
                run_tag_hash=run_tag_hash,
            )
        )
    return plans


def collect_resume_tags(
    *,
    symbol_root: Path,
    skip_errors: bool,
) -> tuple[set[str], set[str], str]:
    sweeps_root = symbol_root / "sweeps"
    ok_tags: set[str] = set()
    error_tags: set[str] = set()
    found_valid_parquet = False
    saw_corrupted_parquet = False

    parquet_paths = sorted(sweeps_root.glob("*/results.parquet")) if sweeps_root.exists() else []
    for parquet_path in parquet_paths:
        try:
            df = pd.read_parquet(parquet_path)
            found_valid_parquet = True
        except Exception:  # noqa: BLE001
            saw_corrupted_parquet = True
            continue
        if "run_tag" not in df.columns or "status" not in df.columns:
            saw_corrupted_parquet = True
            continue
        for _, row in df.iterrows():
            tag = str(row.get("run_tag", ""))
            status = str(row.get("status", ""))
            if not tag:
                continue
            if status == "ok":
                ok_tags.add(tag)
            elif status == "error":
                error_tags.add(tag)

    use_summary_fallback = (not found_valid_parquet) or saw_corrupted_parquet
    if use_summary_fallback and symbol_root.exists():
        for child in sorted(symbol_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "sweeps":
                continue
            if (child / "summary.json").exists():
                ok_tags.add(child.name)

    if found_valid_parquet and not saw_corrupted_parquet:
        source = "parquet"
    elif found_valid_parquet and saw_corrupted_parquet:
        source = "parquet+summary_fallback"
    elif saw_corrupted_parquet:
        source = "summary_fallback_corrupt_parquet"
    else:
        source = "summary_fallback_missing_parquet"

    if not skip_errors:
        error_tags = set()
    return ok_tags, error_tags, source


def _append_progress(progress_log_path: Path, line: str) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a", encoding="utf-8") as fp:
        fp.write(line.rstrip() + "\n")


def _write_repo_sweep_provenance(
    *,
    sweep_dir: Path,
    spec: SweepSpec,
    code_id: str,
    symbol: str,
    sweep_id: str,
    resume_source: str,
) -> None:
    payload = {
        "symbol": symbol,
        "sweep_id": sweep_id,
        "spec_path": str(spec.path),
        "spec_hash": spec.spec_hash,
        "code_id": code_id,
        "created_utc": _utc_now_iso(),
        "resume_source": resume_source,
    }
    _write_json(sweep_dir / "provenance.json", payload)


def run_sweep(
    *,
    spec_path: Path,
    dry_run: bool,
    max_runs: int | None,
    force_rerun: bool,
    symbol: str | None = None,
    all_symbols: bool = False,
    shuffle: bool | None = None,
    seed: int | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
    skip_errors: bool = False,
    resume: bool | None = None,
    fail_fast: bool | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    spec = _load_repo_spec(spec_path)
    payload = _normalize_legacy_payload(spec.payload)
    engine_cfg = payload.get("engine", {})
    output_cfg = payload.get("output", {})

    resolved_output_root = (
        _normalize_path(output_root)
        if output_root is not None
        else _normalize_path(output_cfg.get("root_dir", Path("outputs") / "mbars"))
    )
    code_id = resolve_code_id(cwd=spec.path.parent)

    selected_symbols = _resolve_symbols_from_payload(
        payload,
        symbol=symbol,
        all_symbols=all_symbols,
    )

    resolved_shuffle = bool(payload.get("shuffle", False)) if shuffle is None else bool(shuffle)
    resolved_seed = int(payload.get("random_seed", 0)) if seed is None else int(seed)
    resolved_resume = bool(engine_cfg.get("resume", True)) if resume is None else bool(resume)
    resolved_fail_fast = bool(engine_cfg.get("fail_fast", False)) if fail_fast is None else bool(fail_fast)
    spec_max_runs = payload.get("max_runs")
    if spec_max_runs is None:
        spec_max_runs = engine_cfg.get("max_runs")
    resolved_max_runs = max_runs if max_runs is not None else spec_max_runs
    if resolved_max_runs is not None:
        resolved_max_runs = int(resolved_max_runs)
        if resolved_max_runs <= 0:
            raise ValueError("--max-runs must be positive when provided.")

    sweep_id = datetime.now(timezone.utc).strftime("sweep_%Y%m%d_%H%M%S")
    result_by_symbol: dict[str, Any] = {}

    for active_symbol in selected_symbols:
        symbol_root = resolved_output_root / active_symbol
        sweep_dir = symbol_root / "sweeps" / sweep_id
        results_path = sweep_dir / "results.parquet"
        progress_log_path = sweep_dir / "progress.log"

        base_params = build_symbol_params(active_symbol, payload["base"])
        dataset_payload = payload.get("dataset", {})
        bars_path = _resolve_bars_path_for_symbol(
            symbol=active_symbol,
            base_params=base_params,
            dataset=dataset_payload if isinstance(dataset_payload, Mapping) else {},
        )
        grid = payload["grids"]
        filters = payload.get("filters", [])
        if not isinstance(filters, list):
            raise ValueError("filters must be a list when provided.")
        plans = _build_repo_run_plans(
            symbol=active_symbol,
            base_params=base_params,
            grid=grid,
            filters=filters,
            bars_path=bars_path,
            code_id=code_id,
            shuffle=resolved_shuffle,
            seed=resolved_seed,
        )

        if not force_rerun and resolved_resume:
            ok_tags, error_tags, resume_source = collect_resume_tags(
                symbol_root=symbol_root,
                skip_errors=skip_errors,
            )
            selected = [plan for plan in plans if plan.run_tag not in ok_tags and plan.run_tag not in error_tags]
        else:
            ok_tags = set()
            error_tags = set()
            resume_source = "none_force_rerun" if force_rerun else "none_resume_disabled"
            selected = list(plans)

        if resolved_max_runs is not None:
            selected = selected[:resolved_max_runs]

        skipped = len(plans) - len(selected)

        if dry_run:
            print(
                f"[sweep] symbol={active_symbol} total_grid_runs={len(plans)} "
                f"scheduled_runs={len(selected)} skipped={skipped}"
            )
            for idx, plan in enumerate(selected[:PREVIEW_DRY_RUN_COUNT], start=1):
                run_dir = resolved_output_root / active_symbol / plan.run_tag
                print(
                    f"[sweep] dry_run[{idx}] symbol={active_symbol} run_tag={plan.run_tag} "
                    f"run_dir={run_dir} sweep_results={results_path}"
                )
            result_by_symbol[active_symbol] = {
                "sweep_dir": str(sweep_dir),
                "results_path": str(results_path),
                "total_grid_runs": len(plans),
                "scheduled_runs": len(selected),
                "skipped": skipped,
                "dry_run": True,
            }
            continue

        sweep_dir.mkdir(parents=True, exist_ok=True)
        _write_repo_sweep_provenance(
            sweep_dir=sweep_dir,
            spec=spec,
            code_id=code_id,
            symbol=active_symbol,
            sweep_id=sweep_id,
            resume_source=resume_source,
        )
        if bool(output_cfg.get("save_spec_copy", True)):
            shutil.copy2(spec.path, sweep_dir / "spec_used.yaml")

        _append_progress(
            progress_log_path,
            (
                f"[sweep] symbol={active_symbol} sweep_id={sweep_id} total={len(plans)} "
                f"scheduled={len(selected)} skipped={skipped} resume={resolved_resume} "
                f"force_rerun={force_rerun} skip_errors={skip_errors} "
                f"shuffle={resolved_shuffle} seed={resolved_seed} "
                f"resume_ok_tags={len(ok_tags)} resume_error_tags={len(error_tags)}"
            ),
        )

        completed = 0
        errors = 0
        for idx, plan in enumerate(selected, start=1):
            run_dir = resolved_output_root / active_symbol / plan.run_tag
            started_utc = _utc_now_iso()
            summary: dict[str, Any] | None = None
            error_message: str | None = None
            status = "ok"
            try:
                if _is_hex_16(plan.run_tag):
                    raise ValueError("Run tag must be human-readable, not hash-only.")
                if not bars_path.exists():
                    raise FileNotFoundError(f"Bars parquet not found: {bars_path}")

                sizing_config, risk_config, engine_params = _build_engine_configs(plan.effective_config)
                summary = run_momentum_backtest(
                    symbol=active_symbol,
                    bars_path=bars_path,
                    out_dir=resolved_output_root,
                    hazard_threshold=float(engine_params["hazard_threshold"]),
                    downside_return_threshold=float(engine_params["downside_return_threshold"]),
                    stoploss_pct=float(engine_params["stoploss_pct"]),
                    takeprofit_pct=_optional_float(engine_params["takeprofit_pct"]),
                    initial_equity=float(engine_params["initial_equity"]),
                    sizing_config=sizing_config,
                    risk_config=risk_config,
                    rolling_bars=int(engine_params["rolling_bars"]),
                    run_tag=plan.run_tag,
                    log_every=int(log_every),
                )
                completed += 1
            except Exception as exc:  # noqa: BLE001
                status = "error"
                errors += 1
                error_message = f"{type(exc).__name__}: {exc}"
                err_payload = {
                    "run_tag": plan.run_tag,
                    "run_tag_hash": plan.run_tag_hash,
                    "started_utc": started_utc,
                    "ended_utc": _utc_now_iso(),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _write_json(run_dir / "error.json", err_payload)
                if resolved_fail_fast:
                    ended_utc = _utc_now_iso()
                    runtime_seconds = _runtime_seconds(started_utc, ended_utc)
                    row = _build_results_row(
                        plan=plan,
                        status=status,
                        code_id=code_id,
                        spec_hash=spec.spec_hash,
                        started_utc=started_utc,
                        ended_utc=ended_utc,
                        runtime_seconds=runtime_seconds,
                        summary=None,
                        error_message=error_message,
                    )
                    row["run_tag_hash"] = plan.run_tag_hash
                    _append_results_row(results_path, row)
                    _append_progress(
                        progress_log_path,
                        f"[error] symbol={active_symbol} idx={idx}/{len(selected)} run_tag={plan.run_tag} err={error_message}",
                    )
                    raise

            ended_utc = _utc_now_iso()
            runtime_seconds = _runtime_seconds(started_utc, ended_utc)
            row = _build_results_row(
                plan=plan,
                status=status,
                code_id=code_id,
                spec_hash=spec.spec_hash,
                started_utc=started_utc,
                ended_utc=ended_utc,
                runtime_seconds=runtime_seconds,
                summary=summary,
                error_message=error_message,
            )
            row["run_tag_hash"] = plan.run_tag_hash
            _append_results_row(results_path, row)

            _append_progress(
                progress_log_path,
                (
                    f"[{status}] symbol={active_symbol} idx={idx}/{len(selected)} "
                    f"run_tag={plan.run_tag} run_tag_hash={plan.run_tag_hash}"
                ),
            )
            print(
                f"[sweep] symbol={active_symbol} completed={completed} errors={errors} "
                f"skipped={skipped} last={plan.run_tag}"
            )

        if results_path.exists():
            results_df = pd.read_parquet(results_path)
            if not results_df.empty:
                top_df = results_df.copy()
                top_df["final_equity"] = pd.to_numeric(top_df.get("final_equity"), errors="coerce")
                top_df = top_df.sort_values("final_equity", ascending=False, kind="mergesort")
                top_df.head(20).to_csv(sweep_dir / "results_top20.csv", index=False)

        result_by_symbol[active_symbol] = {
            "sweep_dir": str(sweep_dir),
            "results_path": str(results_path),
            "total_grid_runs": len(plans),
            "scheduled_runs": len(selected),
            "completed": completed,
            "errors": errors,
            "skipped": skipped,
            "dry_run": False,
            "resume_source": resume_source,
        }

    return {
        "spec_path": str(spec.path),
        "spec_hash": spec.spec_hash,
        "code_id": code_id,
        "symbols": result_by_symbol,
        "sweep_id": sweep_id,
        "output_root": str(resolved_output_root),
    }
