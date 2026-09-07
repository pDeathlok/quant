"""Stable adapters between the DAG contract and existing production routines."""

from __future__ import annotations

from dataclasses import replace
import os
import subprocess
import sys
from typing import Any, Mapping

from quant.infrastructure.publication import publication_path
from quant.routine.checkpoint_store import canonical_fingerprint, fingerprint_artifact
from quant.routine.operation_contracts import NodeChanges, OperationContext, OperationResult


def result_from_payload(
    payload: Mapping[str, Any],
    node_ids: tuple[str, ...],
    context: OperationContext,
) -> OperationResult:
    payload = {**payload, "granted_workers": context.granted_workers}
    if payload.get("status") != "success":
        return OperationResult(
            status="failed",
            node_results={},
            error_category="operation",
            error=str(
                payload.get("stderr_tail")
                or payload.get("error")
                or payload.get("reason")
                or "operation failed"
            ),
        )
    explicit_changes = payload.get("node_changes", {})
    changes = {}
    for node_id in node_ids:
        if node_id in explicit_changes:
            value = explicit_changes[node_id]
            changes[node_id] = value if isinstance(value, NodeChanges) else NodeChanges(**value)
        elif "changed_partitions" in payload or "changed_keys" in payload:
            changes[node_id] = NodeChanges(
                partitions=tuple(payload.get("changed_partitions", ())),
                keys=tuple(payload.get("changed_keys", ())),
                full_rebuild=bool(payload.get("full_rebuild", False)),
            )
        else:
            # Legacy routines do not declare their exact impact window. Keep the
            # known dirty keys/dates, but never claim this is a complete journal.
            changes[node_id] = NodeChanges(
                partitions=context.dirty_partitions,
                keys=context.dirty_keys,
                full_rebuild=True,
            )
    fingerprints = dict(payload.get("output_fingerprints", {}))
    if context.project_root is not None:
        artifact_hashes: dict[str, str] = {}
        for node_id in node_ids:
            if node_id in fingerprints:
                continue
            paths = context.output_paths.get(node_id, ())
            try:
                for relative in paths:
                    canonical = (context.project_root / relative).absolute()
                    canonical.resolve().relative_to(context.project_root.resolve())
                    if relative not in artifact_hashes:
                        artifact_hashes[relative] = fingerprint_artifact(publication_path(canonical))
                if paths:
                    fingerprints[node_id] = canonical_fingerprint({
                        relative: artifact_hashes[relative] for relative in paths
                    })
            except (OSError, ValueError):
                # Manual caches remain usable; strict DAG execution rejects an
                # incomplete materialization instead of saving a false cache hit.
                continue
    return OperationResult(
        status="success",
        node_results={node_id: payload for node_id in node_ids},
        changed_partitions=tuple(sorted({p for change in changes.values() for p in change.partitions})),
        changed_keys=tuple(sorted({key for change in changes.values() for key in change.keys})),
        node_changes=changes,
        output_fingerprints=fingerprints,
        dataset_revisions=dict(payload.get("dataset_revisions", {})),
        input_fingerprints=dict(context.upstream_fingerprints),
        input_revisions=dict(context.upstream_revisions),
        metrics={
            "granted_workers": context.granted_workers,
            "output_identity_complete": all(node in fingerprints for node in node_ids),
        },
    )


_result = result_from_payload


def _refresh_start(context: OperationContext) -> str:
    if context.full_rebuild or (context.dirty_keys and not context.dirty_partitions):
        return "19900101"
    return min(context.dirty_partitions) if context.dirty_partitions else context.target_trade_date


def _path(context: OperationContext, relative: str) -> str:
    if context.project_root is None:
        raise ValueError("strict native adapter requires project_root")
    if relative in {"data/raw/daily_basic", "data/raw/top_list"}:
        from quant.data.market_snapshot import pinned_dataset_path

        pinned = pinned_dataset_path(relative.rsplit("/", 1)[-1])
        if pinned is not None:
            return str(pinned)
    return str(publication_path(context.project_root / relative))


def _strict_python(
    context: OperationContext, command: list[str], node_ids: tuple[str, ...], *,
    date_field: str, factor_schema: str | None = None,
) -> OperationResult:
    from quant.routine.pipeline import _extract_last_json_object
    from quant.data.market_snapshot import pinned_market_environment

    if context.project_root is None:
        raise ValueError("strict native adapter requires project_root")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(context.project_root / "src"), str(context.project_root / "scripts/research"))),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        **pinned_market_environment(),
    }
    if env.get("QUANT_PINNED_MARKET_MANIFEST"):
        from pathlib import Path

        # Only signal state validates every source row and its calculation
        # contract. Reuse its existing root; keep date-only base caches sealed.
        signal_root = Path(
            env.get("SIGNAL_FACTOR_ROOT") or env.get("DAILY_FACTOR_ROOT")
            or str(context.project_root / "data/features/daily_factor_layer")
        ).expanduser()
        if not signal_root.is_absolute():
            signal_root = context.project_root / signal_root
        env["SIGNAL_FACTOR_ROOT"] = str(signal_root)
        env["DAILY_FACTOR_ROOT"] = str(
            Path(env["QUANT_PINNED_MARKET_MANIFEST"]).parent.parent / "derived_factors"
        )
    if factor_schema is not None:
        env["PROJECT_FACTOR_COMPATIBILITY_MODE"] = factor_schema
    process = subprocess.run(
        [sys.executable, *command], cwd=context.project_root, env=env,
        capture_output=True, text=True, check=False,
    )
    payload = _extract_last_json_object(process.stdout)
    actual = str(payload.get(date_field) or "").replace("-", "")[:8]
    expected = context.target_trade_date.replace("-", "")[:8]
    success = process.returncode == 0 and payload.get("status") == "success" and actual == expected
    return result_from_payload({
        **payload,
        "status": "success" if success else "failed",
        "error": None if success else f"native operation failed or target mismatch: {actual} != {expected}",
        "stderr_tail": process.stderr[-4000:],
        "returncode": process.returncode,
        "node_changes": {node: NodeChanges(
            partitions=context.dirty_partitions or (context.target_trade_date,), keys=context.dirty_keys,
            full_rebuild=_refresh_start(context) == "19900101",
        ) for node in node_ids},
    }, node_ids, context)


def refresh_strategy_signals(context: OperationContext) -> OperationResult:
    if context.identity_required:
        from quant.features.project_factor_layer import LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION, resolve_project_factor_schema

        command = [
            "-m", "quant.research.strategy_signal_cache", "--workers", str(context.granted_workers),
            "--daily-dir", _path(context, "data/raw/daily"),
            "--incremental-start-date", _refresh_start(context),
            "--family-cache", _path(context, "data/features/b1/b1_family_rule_candidates.parquet"),
            "--extended-cache", _path(context, "data/features/z_skill_daily_candidates.parquet"),
            "--b1-gate-cache", _path(context, "data/features/b1/b1_gate_candidates.parquet"),
            "--b1-gate-manifest", _path(context, "data/features/b1/b1_gate_manifest.json"),
        ]
        if context.full_rebuild:
            command.extend(("--force-refresh", "--factor-mode", "legacy"))
        return _strict_python(context, command, ("feature.strategy_signals",), date_field="processed_through_date",
            factor_schema=resolve_project_factor_schema(os.getenv("ROUTINE_PRODUCTION_FACTOR_SCHEMA", LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION)))
    from quant.routine.pipeline import refresh_strategy_signal_cache

    payload = refresh_strategy_signal_cache(workers=context.granted_workers)
    return _result(payload, ("feature.strategy_signals",), context)


def refresh_active_project_features(
    context: OperationContext,
) -> OperationResult:
    if context.identity_required:
        from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
        from quant.application.selector_ranking import DEFAULT_SELECTOR_RANKING_CONFIG, SelectorRankingSource
        from quant.features.project_factor_layer import LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION, PROJECT_FACTOR_SCHEMA_VERSION, resolve_project_factor_schema

        default_schema = PROJECT_FACTOR_SCHEMA_VERSION if (
            DEFAULT_SELECTOR_RANKING_CONFIG.source == SelectorRankingSource.RIGHT_SIDE_UNIFIED
            and DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled
        ) else LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
        return _strict_python(context, [
            "scripts/research/refresh_b1_feature_cache.py", "--incremental-start-date", _refresh_start(context),
            "--workers", str(context.granted_workers), "--executor", "processes", "--no-adaptive-workers", "--live-only",
            "--daily-dir", _path(context, "data/raw/daily"),
            "--daily-basic-dir", _path(context, "data/raw/daily_basic"),
            "--gate-cache", _path(context, "data/features/b1/b1_gate_candidates.parquet"),
            "--gate-manifest", _path(context, "data/features/b1/b1_gate_manifest.json"),
            "--additional-gate-cache", _path(context, "data/features/z_skill_daily_candidates.parquet"),
            "--family-gate-cache", _path(context, "data/features/b1/b1_family_rule_candidates.parquet"),
            "--active-feature-out", _path(context, "data/features/b1/active_candidate_project_features.parquet"),
            "--active-feature-manifest", _path(context, "data/features/b1/active_candidate_project_features_manifest.json"),
        ], ("feature.project_daily",), date_field="source_latest_trade_date",
            factor_schema=resolve_project_factor_schema(os.getenv("ROUTINE_PRODUCTION_FACTOR_SCHEMA", default_schema)))
    from quant.routine.pipeline import build_features

    start_date = min(context.dirty_partitions) if context.dirty_partitions else None
    payload = build_features(
        incremental_start_date=start_date,
        workers=context.granted_workers,
    )
    return _result(payload, ("feature.project_daily",), context)


def run_right_side_unified(context: OperationContext) -> OperationResult:
    from quant.routine.right_side_unified_production import (
        run_right_side_unified_production,
    )

    if context.identity_required:
        from quant.application.selector_ranking import load_selector_ranking_config
        from quant.data.atomic_io import atomic_write_json
        from quant.routine.right_side_unified_production import (
            _production_input_snapshot,
            build_right_side_unified_production_features, score_right_side_unified_production,
            validate_right_side_unified_selector_adapter,
        )
        import pandas as pd

        if context.project_root is None:
            raise ValueError("strict right-side adapter requires project_root")
        shared_feature_paths = {
            "project_feature_path": publication_path(context.project_root / "data/features/b1/active_candidate_project_features.parquet"),
            "project_feature_manifest_path": publication_path(context.project_root / "data/features/b1/active_candidate_project_features_manifest.json"),
        }
        config = load_selector_ranking_config(context.project_root)
        config = replace(config, factor_workers=context.granted_workers, paths=replace(config.paths, **{
            field: publication_path(getattr(config.paths, field))
            for field in ("feature_output", "feature_manifest", "score_output", "score_manifest", "z_signal_cache", "family_signal_cache")
        }))
        fingerprint, source = _production_input_snapshot(config, pd.Timestamp(context.target_trade_date), project_root=context.project_root, **shared_feature_paths)
        feature = build_right_side_unified_production_features(context.target_trade_date, config=config, project_root=context.project_root, **shared_feature_paths)
        atomic_write_json({**feature, "source_input_fingerprint": fingerprint, "source_inputs": source}, config.paths.feature_manifest)
        score = score_right_side_unified_production(context.target_trade_date, config=config, project_root=context.project_root)
        payload = {**score, "status": "success", "checkpoint_reused": False,
                   "selector_adapter": validate_right_side_unified_selector_adapter(context.target_trade_date, config=config)}
    else:
        payload = run_right_side_unified_production(
            context.target_trade_date, factor_workers=context.granted_workers,
        )
    return _result(
        payload,
        (
            "feature.right_side_unified",
            "score.right_side_unified",
            "product.right_side_unified_adapter",
        ),
        context,
    )


def run_left_side_unified(context: OperationContext) -> OperationResult:
    from quant.routine.left_side_unified_production import run_left_side_production

    if context.identity_required:
        from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
        from quant.routine.left_side_unified_production import (
            build_left_side_production_features, score_left_side_production, validate_left_side_selector_adapter,
        )
        config = DEFAULT_LEFT_SIDE_RANKING_CONFIG
        config = replace(config, factor_workers=context.granted_workers, paths=replace(config.paths, **{
            field: publication_path(getattr(config.paths, field)) for field in (
                "feature_output", "feature_manifest", "score_output", "score_manifest", "project_feature_cache",
                "project_feature_manifest", "signal_cache", "b1_gate_cache", "family_signal_cache",
            )
        }))
        feature = build_left_side_production_features(context.target_trade_date, config=config)
        score = score_left_side_production(context.target_trade_date, config=config)
        payload = {"status": "success", "checkpoint_reused": False, "feature": feature, "score": score,
                   "adapter": validate_left_side_selector_adapter(context.target_trade_date, config=config)}
    else:
        payload = run_left_side_production(context.target_trade_date, factor_workers=context.granted_workers)
    return _result(
        payload,
        (
            "feature.left_side_unified",
            "score.left_side_unified",
            "product.left_side_unified_adapter",
        ),
        context,
    )


def refresh_chan_model_scores(context: OperationContext) -> OperationResult:
    if context.identity_required:
        return _strict_python(context, [
            "scripts/research/refresh_chan_model_live_scores.py", "--start", _refresh_start(context),
            "--end", context.target_trade_date, "--max-workers", str(context.granted_workers),
            "--executor", "processes", "--skip-backfill-snapshots", "--rebuild-candidates",
            "--candidate-start-date", "1990-01-01", "--daily-dir", _path(context, "data/raw/daily"),
            "--daily-basic-dir", _path(context, "data/raw/daily_basic"),
            "--top-list-dir", _path(context, "data/raw/top_list"),
            "--report-dir", _path(context, "reports/chan_daily"),
            "--scored-path", _path(context, "reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"),
            "--output-dir", _path(context, "reports/chan_daily/model_strategy"),
        ], ("feature.chan_live", "score.chan"), date_field="end")
    from quant.routine.pipeline import refresh_chan_model_scores as refresh

    payload = refresh(progress_callback=None, workers=context.granted_workers)
    return _result(payload, ("feature.chan_live", "score.chan"), context)


def shadow_only(context: OperationContext) -> OperationResult:
    raise RuntimeError(
        "operation has no cutover adapter; run with ROUTINE_DAG_EXECUTOR=shadow"
    )


__all__ = [
    "refresh_active_project_features",
    "refresh_chan_model_scores",
    "refresh_strategy_signals",
    "result_from_payload",
    "run_left_side_unified",
    "run_right_side_unified",
    "shadow_only",
]
