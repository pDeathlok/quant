#!/usr/bin/env python
"""Publish a validated canonical B1 bundle without mutating rollback artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_link_or_copy, atomic_write_csv, atomic_write_json
from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.right_side_factor_contract import factor_contract_sha256
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


MODEL_NAMES = ("up5_es", "up8_es", "up10_es", "down2_es", "down3_es")
RULE_TO_COMBO = {
    "current__b1_stable": "stable_up10_020_down3_040_fixed8_sl15_T5",
    "current__b1_aggressive": "aggressive_up8_070_down3_045_expiry_T9",
    "current__b1_baseline": "baseline_up8_055_trail5_dd2_sl2_T9",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=PROJECT_ROOT / "models/research/b1_canonical_v5",
    )
    parser.add_argument(
        "--production-model-dir",
        type=Path,
        default=PROJECT_ROOT / "models/production/b1_canonical_v5",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/b1/research/canonical_v5/evaluation",
    )
    parser.add_argument(
        "--release-report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/b1/releases/canonical_v5_20260824",
    )
    parser.add_argument("--release-id", default="b1-canonical-v5-20260824")
    parser.add_argument(
        "--enabled-rules",
        nargs="+",
        default=["current__b1_stable"],
    )
    return parser.parse_args()


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _parse_args()
    source_manifest_path = args.source_model_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("factor_schema_version") != PROJECT_FACTOR_SCHEMA_VERSION:
        raise RuntimeError("B1 source manifest is not canonical project-v5")
    if int(source_manifest.get("factor_count") or -1) != len(PROJECT_FACTOR_COLUMNS):
        raise RuntimeError("B1 source manifest factor count drifted")
    forbidden = find_forbidden_aliases_in_payload(source_manifest)
    if forbidden:
        raise RuntimeError(f"B1 source manifest contains forbidden aliases: {forbidden}")

    args.production_model_dir.mkdir(parents=True, exist_ok=True)
    model_items: dict[str, dict[str, object]] = {}
    artifact_digests: dict[str, str] = {}
    for name in MODEL_NAMES:
        source_path = args.source_model_dir / f"{name}.joblib"
        source_item = (source_manifest.get("models") or {}).get(name) or {}
        source_digest = _sha256(source_path)
        if source_digest != source_item.get("sha256"):
            raise RuntimeError(f"B1 research artifact checksum mismatch for {name}")
        model = joblib.load(source_path)
        features = tuple(str(value) for value in model.feature_names_in_)
        selected = tuple(str(value) for value in model.selected_features_)
        assert_no_forbidden_factor_names(features, context=f"B1 source {name}")
        assert_no_forbidden_factor_names(selected, context=f"B1 source selected {name}")
        if features != tuple(PROJECT_FACTOR_COLUMNS):
            raise RuntimeError(f"B1 {name} does not use the canonical 145-factor contract")
        if getattr(model, "factor_schema_version_", None) != PROJECT_FACTOR_SCHEMA_VERSION:
            raise RuntimeError(f"B1 {name} model schema is not canonical project-v5")

        target_path = args.production_model_dir / source_path.name
        atomic_link_or_copy(source_path, target_path)
        target_digest = _sha256(target_path)
        if target_digest != source_digest:
            raise RuntimeError(f"B1 production copy checksum mismatch for {name}")
        artifact_digests[name] = target_digest
        model_items[name] = {
            "path": _project_relative(target_path),
            "sha256": target_digest,
            "feature_count": len(features),
            "features": list(features),
            "selected_features": list(selected),
            "model_input_contract_sha256": factor_contract_sha256(
                features,
                schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
            ),
        }

    bundle_digest = hashlib.sha256(
        json.dumps(
            {"release_id": args.release_id, "artifacts": artifact_digests},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "b1-model-bundle-v2-canonical-alias-free",
        "release_id": args.release_id,
        "published_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": _project_relative(source_manifest_path),
        "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "factor_count": len(PROJECT_FACTOR_COLUMNS),
        "canonical_features": list(PROJECT_FACTOR_COLUMNS),
        "factor_contract_sha256": factor_contract_sha256(
            PROJECT_FACTOR_COLUMNS,
            schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
        ),
        "artifact_bundle_sha256": bundle_digest,
        "forbidden_aliases": [],
        "models": model_items,
    }
    forbidden = find_forbidden_aliases_in_payload(manifest)
    if forbidden:
        raise RuntimeError(f"B1 production manifest contains forbidden aliases: {forbidden}")
    atomic_write_json(manifest, args.production_model_dir / "manifest.json")

    source_summary = pd.read_csv(args.evaluation_dir / "current_threshold_periods.csv")
    unknown_rules = sorted(set(args.enabled_rules) - set(RULE_TO_COMBO))
    if unknown_rules:
        raise ValueError(f"Unknown B1 enabled rules: {unknown_rules}")
    summary = source_summary[source_summary["entry_rule"].isin(args.enabled_rules)].copy()
    if summary.empty:
        raise RuntimeError("B1 release summary has no enabled strategy rows")
    summary["combo"] = summary["entry_rule"].map(RULE_TO_COMBO)
    summary["release_id"] = args.release_id
    args.release_report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.release_report_dir / "summary.csv"
    atomic_write_csv(summary, summary_path, index=False)

    strategy_validation: dict[str, dict[str, object]] = {}
    for rule in args.enabled_rules:
        row = summary[
            (summary["entry_rule"] == rule) & (summary["period"] == "oot")
        ]
        if len(row) != 1:
            raise RuntimeError(f"B1 {rule} has no unique OOT evaluation row")
        item = row.iloc[0]
        passed = bool(
            int(item["trades"]) >= 30
            and float(item["avg_return_pct"]) > 0
            and float(item["profit_factor"]) > 1
        )
        strategy_validation[RULE_TO_COMBO[rule]] = {
            "trades": int(item["trades"]),
            "avg_return_pct": float(item["avg_return_pct"]),
            "profit_factor": float(item["profit_factor"]),
            "passed": passed,
        }
    audit = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "release_id": args.release_id,
        "model_dir": _project_relative(args.production_model_dir),
        "source_evaluation": _project_relative(
            args.evaluation_dir / "current_threshold_periods.csv"
        ),
        "selection_policy": "existing thresholds; OOT gate; enabled strategies only",
        "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "factor_count": len(PROJECT_FACTOR_COLUMNS),
        "forbidden_aliases": [],
        "status": (
            "valid"
            if all(item["passed"] for item in strategy_validation.values())
            else "incompatible"
        ),
        "strategy_validation": strategy_validation,
    }
    if audit["status"] != "valid":
        audit["reason"] = "At least one enabled B1 strategy failed the OOT return gate."
    atomic_write_json(audit, args.release_report_dir / "model_compatibility_audit.json")
    if audit["status"] != "valid":
        raise RuntimeError(str(audit["reason"]))

    print(
        json.dumps(
            {
                "status": "published",
                "release_id": args.release_id,
                "model_manifest": _project_relative(
                    args.production_model_dir / "manifest.json"
                ),
                "summary": _project_relative(summary_path),
                "compatibility_audit": _project_relative(
                    args.release_report_dir / "model_compatibility_audit.json"
                ),
                "artifact_bundle_sha256": bundle_digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
