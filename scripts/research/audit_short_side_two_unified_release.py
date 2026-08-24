#!/usr/bin/env python
"""Strict postflight for the two-unified-ranker short-side release."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.application.selector_ranking import DEFAULT_SELECTOR_RANKING_CONFIG
from quant.data.atomic_io import atomic_write_json
from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    find_forbidden_aliases_in_payload,
)
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.routine.daily_dependency_runtime import resolve_model_contracts


REPORT_ROOT = (
    PROJECT_ROOT / "reports/production/short_side_two_unified_release_20260824"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected mapping JSON: {path}")
    return payload


def main() -> None:
    right = DEFAULT_SELECTOR_RANKING_CONFIG
    left = DEFAULT_LEFT_SIDE_RANKING_CONFIG
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    short_nodes = set(registry.required_node_ids("short"))
    right_bundle = joblib.load(right.paths.artifact)
    left_bundle = joblib.load(left.paths.artifact)
    right_manifest = _load_json(right.paths.artifact_manifest)
    left_manifest = _load_json(left.paths.artifact_manifest)
    right_score_manifest = _load_json(right.paths.score_manifest)
    left_score_manifest = _load_json(left.paths.score_manifest)
    b1_page_contract = _load_json(PROJECT_ROOT / "web/data/b1_daily_plan.json")
    configs = {
        "right": yaml.safe_load(right.config_path.read_text(encoding="utf-8")),
        "left": yaml.safe_load(
            (PROJECT_ROOT / "configs/strategies/left_side_unified.yaml").read_text(
                encoding="utf-8"
            )
        ),
        "b1": yaml.safe_load(
            (PROJECT_ROOT / "configs/strategies/b1_selected.yaml").read_text(
                encoding="utf-8"
            )
        ),
    }
    contracts, contract_audit = resolve_model_contracts(
        registry,
        PROJECT_ROOT,
        "short",
        cache_path=(
            PROJECT_ROOT
            / "data/contracts/daily_dependencies/two_unified_postflight_cache.json"
        ),
        strict=True,
    )
    required_union = sorted(
        {
            feature
            for contract in contracts.values()
            for feature in contract.required_feature_union
        }
    )
    effective_union = sorted(
        {
            feature
            for contract in contracts.values()
            for feature in contract.effective_feature_union
        }
    )
    payloads = {
        "right_bundle": right_bundle,
        "left_bundle": left_bundle,
        "right_manifest": right_manifest,
        "left_manifest": left_manifest,
        "right_score_manifest": right_score_manifest,
        "left_score_manifest": left_score_manifest,
        "right_score_columns": list(pd.read_parquet(right.paths.score_output).columns),
        "left_score_columns": list(pd.read_parquet(left.paths.score_output).columns),
        "b1_page_contract": b1_page_contract,
        "configs": configs,
        "daily_required_features": required_union,
        "daily_effective_features": effective_union,
    }
    forbidden_hits = {
        name: find_forbidden_aliases_in_payload(payload)
        for name, payload in payloads.items()
    }
    forbidden_hits = {name: hits for name, hits in forbidden_hits.items() if hits}
    registry_aliases = sorted(
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "compatibility_alias"
    )
    canonical_registry_features = [
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "feature"
    ]
    consumer_zero = {
        "legacy_b1_probability_models": {
            "active_daily_node": "score.b1" in short_nodes,
            "active_consumers": 0 if "score.b1" not in short_nodes else 1,
            "artifact_policy": "rollback_only",
            "page_consumer": "left_side_unified_rank_percentile",
        },
        "legacy_z_skill_models": {
            "active_daily_node": "score.z_skill" in short_nodes,
            "active_consumers": 0 if "score.z_skill" not in short_nodes else 1,
            "artifact_policy": "rollback_only",
            "page_consumer": "two_unified_rankers_only",
        },
        "forbidden_factor_aliases": {
            alias: 0 for alias in sorted(FORBIDDEN_COMPATIBILITY_ALIASES)
        },
    }
    rollback_paths = [
        PROJECT_ROOT / "models/production/right_side_unified/ranking.joblib",
        PROJECT_ROOT
        / "models/production/left_side_unified_canonical_v3_group4/ranking.joblib",
        *sorted((PROJECT_ROOT / "models/production/b1").glob("*.joblib")),
        *sorted((PROJECT_ROOT / "models/research/z_skill").glob("*.joblib")),
    ]
    acceptance = {
        "forbidden_alias_hits": forbidden_hits,
        "registry_compatibility_alias_count": len(registry_aliases),
        "registry_compatibility_aliases": registry_aliases,
        "legacy_active_consumer_count": sum(
            item["active_consumers"]
            for key, item in consumer_zero.items()
            if key != "forbidden_factor_aliases"
        ),
        "right_artifact_exists": right.paths.artifact.is_file(),
        "left_artifact_exists": left.paths.artifact.is_file(),
        "rollback_artifacts_missing": [
            str(path.relative_to(PROJECT_ROOT))
            for path in rollback_paths
            if not path.is_file()
        ],
        "daily_contract_status": contract_audit.get("status"),
    }
    passed = (
        not forbidden_hits
        and not registry_aliases
        and acceptance["legacy_active_consumer_count"] == 0
        and acceptance["right_artifact_exists"]
        and acceptance["left_artifact_exists"]
        and not acceptance["rollback_artifacts_missing"]
        and acceptance["daily_contract_status"] == "success"
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(consumer_zero, REPORT_ROOT / "consumer_zero_report.json")
    atomic_write_json(
        {
            "schema_version": "short-side-canonical-feature-list-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "registry_record_count": len(FACTOR_REGISTRY),
            "canonical_registry_feature_count": len(canonical_registry_features),
            "canonical_registry_features": canonical_registry_features,
            "right_scoring_input_count": len(right_bundle["features"]),
            "right_scoring_inputs": list(right_bundle["features"]),
            "right_model_feature_names_in": list(
                right_bundle["model"].feature_names_in_
            ),
            "left_scoring_input_count": len(left_bundle["features"]),
            "left_scoring_inputs": list(left_bundle["features"]),
            "left_model_feature_names_in": list(
                left_bundle["model"].feature_names_in_
            ),
            "daily_required_feature_union": required_union,
            "daily_effective_feature_union": effective_union,
        },
        REPORT_ROOT / "canonical_feature_list.json",
    )
    postflight = {
        "status": "passed" if passed else "failed",
        "schema_version": "short-side-two-unified-production-postflight-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "active_short_nodes": sorted(short_nodes),
        "acceptance": acceptance,
    }
    atomic_write_json(postflight, REPORT_ROOT / "postflight.json")
    print(json.dumps(postflight, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
