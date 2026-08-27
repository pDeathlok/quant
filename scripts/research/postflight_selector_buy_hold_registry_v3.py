#!/usr/bin/env python
"""Regenerate and strictly audit the released selector buy/hold v3 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.data.atomic_io import atomic_write_json
from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    find_forbidden_aliases_in_payload,
)
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
    SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_RELEASE_ID,
    validate_selector_buy_hold_artifact,
)
from quant.routine.daily_dependency_runtime import (
    audit_required_freshness,
    collect_node_states,
    resolve_model_contracts,
)


PRODUCTION_ROOT = PROJECT_ROOT / "models/production/selector_buy_hold_registry_v3"
ROLLBACK_ROOT = PROJECT_ROOT / "models/production/selector_buy_hold"
REPORT_ROOT = PROJECT_ROOT / "reports/production/selector_buy_hold_registry_v3"
REFRESH_STATUS_PATH = PROJECT_ROOT / "data/routine/latest_refresh_status.json"
PROBABILITY_BANDS_PATH = PROJECT_ROOT / "config/selector_score_probability_bands.json"
RETRAINING_POLICY_PATH = PROJECT_ROOT / "config/selector_buy_hold_retraining_policy.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected mapping JSON: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_audit() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    tuple[str, ...],
]:
    manifest = _load_json(PRODUCTION_ROOT / "manifest.json")
    if manifest.get("status") != "success":
        raise RuntimeError("active selector manifest is not successful")
    if manifest.get("schema_version") != SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("active selector manifest schema drifted")
    if manifest.get("release_id") != SELECTOR_BUY_HOLD_RELEASE_ID:
        raise RuntimeError("active selector manifest release drifted")

    artifacts: dict[str, dict[str, Any]] = {}
    model_features: tuple[str, ...] = ()
    for mode in ("buy", "hold"):
        path = PRODUCTION_ROOT / f"{mode}.joblib"
        artifact = joblib.load(path)
        features = validate_selector_buy_hold_artifact(artifact)
        declared_hash = str(((manifest.get("models") or {}).get(mode) or {}).get("sha256") or "")
        if declared_hash != _sha256(path):
            raise RuntimeError(f"{mode} artifact hash drifted")
        if artifact.get("release_id") != SELECTOR_BUY_HOLD_RELEASE_ID:
            raise RuntimeError(f"{mode} artifact release drifted")
        if model_features and features != model_features:
            raise RuntimeError("buy and hold model input schemas differ")
        model_features = features
        artifacts[mode] = artifact
    if tuple(manifest.get("model_input_columns") or ()) != model_features:
        raise RuntimeError("manifest model input columns drifted")
    return artifacts, manifest, model_features


def _model_feature_names(artifact: Mapping[str, Any]) -> list[str]:
    model = artifact.get("model")
    if model is not None:
        return [str(value) for value in model.feature_names_in_]
    components = artifact.get("models") or {}
    if not components:
        raise RuntimeError("artifact contains neither model nor component models")
    feature_names = {
        tuple(str(value) for value in component.feature_names_in_)
        for component in components.values()
    }
    if len(feature_names) != 1:
        raise RuntimeError("component model feature_names_in_ schemas differ")
    return list(next(iter(feature_names)))


def _daily_contract_audit() -> tuple[dict[str, Any], list[str], list[str]]:
    contracts, audit = resolve_model_contracts(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        PROJECT_ROOT,
        "short",
        cache_path=(
            PROJECT_ROOT
            / "data/contracts/daily_dependencies/selector_buy_hold_v3_postflight_cache.json"
        ),
        strict=True,
    )
    required = sorted(
        {
            feature
            for contract in contracts.values()
            for feature in contract.required_feature_union
        }
    )
    effective = sorted(
        {
            feature
            for contract in contracts.values()
            for feature in contract.effective_feature_union
        }
    )
    return audit, required, effective


def _default_target_date(refresh_status: Mapping[str, Any]) -> str:
    result = refresh_status.get("result") or {}
    for key in ("selector_extended", "selector_core"):
        value = str((result.get(key) or {}).get("signal_date") or "")
        if value:
            return value
    value = str(refresh_status.get("trade_date") or "")
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    raise RuntimeError("could not resolve the latest selector trade date")


def _committed_selector_dependency_snapshot(target_date: str) -> dict[str, Any]:
    candidates = []
    for scope in ("short", "all"):
        path = PROJECT_ROOT / f"data/contracts/daily_dependencies/latest-{scope}.json"
        if not path.is_file():
            continue
        payload = _load_json(path)
        if (
            payload.get("status") == "success"
            and payload.get("baseline_committed") is True
            and payload.get("target_trade_date") == target_date
            and "score.selector" in (payload.get("model_contract_hashes") or {})
        ):
            candidates.append(payload)
    if not candidates:
        raise RuntimeError(f"no successful committed selector dependency baseline for {target_date}")
    return max(candidates, key=lambda payload: str(payload.get("created_at") or ""))


def _regenerate_latest(target_date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from quant.routine.pipeline import publish_daily_dependency_contract
    from quant.routine.left_side_unified_production import run_left_side_production
    from quant.routine.right_side_unified_production import run_right_side_unified_production
    from quant.webapp import services

    refresh_status = _load_json(REFRESH_STATUS_PATH)
    results = dict(refresh_status.get("result") or {})
    ranker_refresh = {
        "right": run_right_side_unified_production(target_date),
        "left": run_left_side_production(target_date),
    }
    if any(result.get("status") != "success" for result in ranker_refresh.values()):
        raise RuntimeError(f"selector upstream ranker validation failed: {ranker_refresh}")
    preflight = publish_daily_dependency_contract(
        target_date,
        "short",
        results,
        phase="preflight",
        strict_freshness=False,
    )
    if preflight.get("status") != "success":
        raise RuntimeError(f"selector v3 dependency preflight failed: {preflight}")
    results["dependency_preflight"] = preflight

    selector_long_snapshot = services._ensure_selector_long_factor_snapshot(
        target_date,
        force_refresh=(
            "feature.long_snapshot" in set(preflight.get("refresh_node_ids") or ())
        ),
    )
    results["selector_long_factor_snapshot"] = selector_long_snapshot
    services._clear_selector_caches()
    core = services.get_stock_selector_payload(
        signal_date=target_date,
        use_cache=False,
    )
    full = services.get_stock_selector_payload(
        signal_date=target_date,
        include_extended=True,
        use_cache=False,
        full_snapshot=True,
    )
    for name, payload in (("core", core), ("extended", full)):
        actual = str(payload.get("signal_date") or "")
        if actual != target_date:
            raise RuntimeError(
                f"selector {name} output date drifted: expected={target_date} actual={actual}"
            )
        page_audit = _page_score_audit(payload, target_date)
        if page_audit["status"] != "success":
            raise RuntimeError(f"selector {name} page score audit failed: {page_audit}")
    results["selector_core"] = {
        "status": "success",
        "signal_date": target_date,
        "stocks": len(core.get("stocks") or []),
    }
    results["selector_extended"] = {
        "status": "success",
        "signal_date": target_date,
        "stocks": len(full.get("stocks") or []),
    }
    written = services._write_strategy_pool_snapshots(full, include_extended=True)
    results["snapshot"] = {
        "status": "success",
        "storage": (
            "mysql"
            if services.MarketDataStore(
                services.MarketDataStoreConfig.from_env()
            ).config.sql_url
            else "json"
        ),
        "strategy_pools": written,
    }
    postflight = publish_daily_dependency_contract(
        target_date,
        "short",
        results,
        phase="postflight",
        strict_freshness=True,
    )
    if postflight.get("status") != "success" or not postflight.get("baseline_committed"):
        raise RuntimeError(f"selector v3 dependency postflight failed: {postflight}")
    return full, {
        "preflight": preflight,
        "postflight": postflight,
        "selector_core": results["selector_core"],
        "selector_extended": results["selector_extended"],
        "snapshot": results["snapshot"],
        "ranker_refresh": ranker_refresh,
    }


def _page_score_audit(payload: Mapping[str, Any], target_date: str) -> dict[str, Any]:
    rows = list(payload.get("stocks") or [])
    model_rows = [row for row in rows if row.get("model_score_available") is True]
    incomplete = [
        str(row.get("symbol") or "")
        for row in rows
        if row.get("model_score_available") is not True
        or row.get("buy_score_source") != "historical_return_model"
        or row.get("hold_score_source") != "historical_return_model"
        or (row.get("feature_quality") or {}).get("status") != "complete"
        or str(row.get("score_date") or "") != str(row.get("date") or "")
    ]
    prior_signal_rows = [
        str(row.get("symbol") or "")
        for row in rows
        if str(row.get("date") or "") != target_date
    ]
    signal_dates = sorted(
        {str(row.get("date") or "") for row in rows if row.get("date")}
    )
    release_matches = (
        payload.get("model_release_id") == SELECTOR_BUY_HOLD_RELEASE_ID
        and payload.get("factor_validation_schema") == "selector-required-layers-v1"
    )
    snapshot_date_matches = str(payload.get("signal_date") or "") == target_date
    return {
        "target_date": target_date,
        "row_count": len(rows),
        "model_scored_row_count": len(model_rows),
        "complete_model_score_rate": (
            round(len(model_rows) / len(rows), 6) if rows else 0.0
        ),
        "incomplete_symbols": incomplete,
        "signal_dates": signal_dates,
        "prior_signal_row_count": len(prior_signal_rows),
        "prior_signal_symbols": prior_signal_rows,
        "score_date_contract": "each row is scored on its own causal signal date",
        "release_and_validation_contract_matches": release_matches,
        "snapshot_date_matches": snapshot_date_matches,
        "status": "success" if rows and not incomplete and release_matches and snapshot_date_matches else "failed",
    }


def _mysql_snapshot_audit(payload: Mapping[str, Any], target_date: str) -> dict[str, Any]:
    from sqlalchemy import bindparam, text
    from quant.webapp import services

    store = services.MarketDataStore(services.MarketDataStoreConfig.from_env())
    if not store.config.sql_url:
        return {"status": "not_configured", "checked_snapshot_count": 0}
    extended_keys = {str(item["key"]).upper() for item in services.EXTENDED_STRATEGIES}
    scopes: list[tuple[list[str] | None, bool]] = [(None, True)]
    for strategy in services._strategy_keys_from_payload(dict(payload)):
        members = services.STRATEGY_GROUP_MEMBERS.get(strategy, {strategy})
        scopes.append(([strategy], bool(members & extended_keys)))
    expected: dict[str, Any] = {}
    engine = None
    try:
        for scope, include_extended in scopes:
            key = services._selector_snapshot_key(target_date, scope, include_extended)[0]
            expected[key] = _load_json(services._selector_snapshot_path(key))
        engine = store._engine()
        with engine.connect() as connection:
            records = connection.execute(
                text(
                    f"SELECT snapshot_key, payload_json FROM {services.SELECTOR_SNAPSHOT_TABLE} "
                    "WHERE snapshot_key IN :keys"
                ).bindparams(bindparam("keys", expanding=True)),
                {"keys": list(expected)},
            ).mappings().all()
        actual = {row["snapshot_key"]: json.loads(row["payload_json"]) for row in records}
        mismatches = [key for key, value in expected.items() if actual.get(key) != value]
        return {
            "status": "success" if not mismatches else "failed",
            "checked_snapshot_count": len(expected),
            "mismatched_snapshot_keys": mismatches,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "checked_snapshot_count": 0,
        }
    finally:
        if engine is not None:
            engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date")
    parser.add_argument(
        "--regenerate-latest",
        action="store_true",
        help="Re-score and republish the latest selector strategy snapshots.",
    )
    args = parser.parse_args()

    refresh_status = _load_json(REFRESH_STATUS_PATH)
    target_date = args.target_date or _default_target_date(refresh_status)
    artifacts, manifest, model_features = _artifact_audit()
    contract_audit, required_union, effective_union = _daily_contract_audit()

    if args.regenerate_latest:
        page_payload, release_run = _regenerate_latest(target_date)
    else:
        from quant.webapp import services

        page_payload = services.get_stock_selector_payload(
            signal_date=target_date,
            include_extended=True,
            use_cache=True,
            full_snapshot=True,
        )
        committed = _committed_selector_dependency_snapshot(target_date)
        release_run = {
            "status": "validated_published_snapshot",
            "snapshot_generated_at": page_payload.get("generated_at"),
            "snapshot_cache": page_payload.get("cache"),
            "postflight": {
                "status": committed.get("status"),
                "scope": committed.get("scope"),
                "target_trade_date": committed.get("target_trade_date"),
                "baseline_committed": committed.get("baseline_committed"),
                "refresh_node_ids": committed.get("refresh_node_ids"),
                "score_selector_contract_hash": (
                    committed.get("model_contract_hashes") or {}
                ).get("score.selector"),
                "active_model_required_union_count": (
                    committed.get("feature_inventory") or {}
                ).get("active_model_required_union_count"),
            },
        }
    page_audit = _page_score_audit(page_payload, target_date)
    mysql_audit = _mysql_snapshot_audit(page_payload, target_date)
    live_freshness = audit_required_freshness(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY,
        "short",
        date.fromisoformat(target_date),
        collect_node_states(
            DEFAULT_DAILY_DEPENDENCY_REGISTRY,
            PROJECT_ROOT,
            refresh_status.get("result") or {},
        ),
    )

    registry_aliases = sorted(
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "compatibility_alias"
    )
    registry_features = tuple(
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "feature"
    )
    active_payloads = {
        "buy_artifact": {
            key: value
            for key, value in artifacts["buy"].items()
            if key not in {"model", "score_reference"}
        },
        "hold_artifact": {
            key: value
            for key, value in artifacts["hold"].items()
            if key not in {"model", "score_reference"}
        },
        "manifest": manifest,
        "probability_bands": _load_json(PROBABILITY_BANDS_PATH),
        "retraining_policy": _load_json(RETRAINING_POLICY_PATH),
        "daily_required_features": required_union,
        "daily_effective_features": effective_union,
        "page_output": page_payload,
    }
    alias_hits = {
        name: list(find_forbidden_aliases_in_payload(payload))
        for name, payload in active_payloads.items()
    }
    alias_hits = {name: hits for name, hits in alias_hits.items() if hits}
    forbidden_consumer_counts = {
        alias: sum(alias in hits for hits in alias_hits.values())
        for alias in sorted(FORBIDDEN_COMPATIBILITY_ALIASES)
    }
    rollback_artifacts = [
        ROLLBACK_ROOT / "buy.joblib",
        ROLLBACK_ROOT / "hold.joblib",
    ]
    consumer_zero_report = {
        "schema_version": "selector-buy-hold-v3-consumer-zero-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "active_release_id": SELECTOR_BUY_HOLD_RELEASE_ID,
        "active_payload_forbidden_hits": alias_hits,
        "forbidden_alias_active_consumer_counts": forbidden_consumer_counts,
        "registry_compatibility_alias_count": len(registry_aliases),
        "registry_compatibility_aliases": registry_aliases,
        "legacy_selector_model": {
            "active_consumers": 0,
            "artifact_policy": "independent_rollback_only",
            "artifact_root": str(ROLLBACK_ROOT.relative_to(PROJECT_ROOT)),
            "missing_artifacts": [
                str(path.relative_to(PROJECT_ROOT))
                for path in rollback_artifacts
                if not path.is_file()
            ],
        },
    }
    dependency_postflight = release_run.get("postflight") or {}
    dependency_postflight_ready = bool(
        dependency_postflight.get("status") == "success"
        and dependency_postflight.get("baseline_committed") is True
        and dependency_postflight.get("target_trade_date") == target_date
        and not dependency_postflight.get("refresh_node_ids")
    )
    passed = bool(
        not alias_hits
        and not any(forbidden_consumer_counts.values())
        and not registry_aliases
        and not consumer_zero_report["legacy_selector_model"]["missing_artifacts"]
        and contract_audit.get("status") == "success"
        and page_audit["status"] == "success"
        and mysql_audit["status"] in {"success", "not_configured"}
        and dependency_postflight_ready
        and live_freshness.get("status") == "success"
        and tuple(model_features) == tuple(dict.fromkeys(model_features))
    )

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(consumer_zero_report, REPORT_ROOT / "consumer_zero_report.json")
    atomic_write_json(
        {
            "schema_version": "selector-buy-hold-v3-canonical-features-v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "release_id": SELECTOR_BUY_HOLD_RELEASE_ID,
            "registry_record_count": len(FACTOR_REGISTRY),
            "registry_feature_count": len(registry_features),
            "registry_features": list(registry_features),
            "discovery_candidate_count": len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
            "discovery_candidates": list(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
            "model_input_count": len(model_features),
            "model_inputs": list(model_features),
            "buy_model_feature_names_in": _model_feature_names(artifacts["buy"]),
            "hold_model_feature_names_in": _model_feature_names(artifacts["hold"]),
            "daily_required_feature_union": required_union,
            "daily_effective_feature_union": effective_union,
        },
        REPORT_ROOT / "canonical_feature_list.json",
    )
    postflight = {
        "status": "passed" if passed else "failed",
        "schema_version": "selector-buy-hold-v3-production-postflight-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "release_id": SELECTOR_BUY_HOLD_RELEASE_ID,
        "target_date": target_date,
        "model_input_count": len(model_features),
        "model_input_unique_count": len(set(model_features)),
        "artifact_hashes": {
            mode: _sha256(PRODUCTION_ROOT / f"{mode}.joblib")
            for mode in ("buy", "hold")
        },
        "daily_contract_status": contract_audit.get("status"),
        "dependency_postflight_ready": dependency_postflight_ready,
        "page_score_audit": page_audit,
        "mysql_snapshot_audit": mysql_audit,
        "live_freshness_audit": live_freshness,
        "release_run": release_run,
        "consumer_zero_report": str(
            (REPORT_ROOT / "consumer_zero_report.json").relative_to(PROJECT_ROOT)
        ),
        "canonical_feature_list": str(
            (REPORT_ROOT / "canonical_feature_list.json").relative_to(PROJECT_ROOT)
        ),
    }
    atomic_write_json(postflight, REPORT_ROOT / "postflight.json")
    print(json.dumps(postflight, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
