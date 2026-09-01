"""Runtime adapters for the executable daily dependency registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant.application.daily_dependencies import (
    DEFAULT_DAILY_DEPENDENCY_REGISTRY,
    DependencyNode,
    DependencyRegistry,
    FeatureUsage,
    FreshnessMode,
    Layer,
    ModelContract,
    NodeState,
    build_dependency_plan,
    classify_feature_usage,
    state_is_current,
)
from quant.data.atomic_io import atomic_write_json
from quant.features.factor_registry import (
    FACTOR_REGISTRY,
    LONG_PRODUCTION_FACTOR_COLUMNS,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS,
)
from quant.features.left_side_factor_contract import LEFT_SIDE_SCORING_INPUT_COLUMNS
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


MODEL_CACHE_SCHEMA_VERSION = "daily_model_contract_cache_v1"
SNAPSHOT_SCHEMA_VERSION = "daily_dependency_snapshot_v3_factor_governance"
DEFAULT_CONTRACT_DIR = Path("data/contracts/daily_dependencies")


_VOLATILE_RESULT_KEYS = {
    "checkpoint_reused",
    "command",
    "elapsed_seconds",
    "error",
    "finished_at",
    "script_elapsed_seconds",
    "started_at",
    "stderr_tail",
    "stdout_tail",
    "updated_at",
}
_CONTENT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


# Legacy Z artifacts were pickled while their training script was __main__.
# This class is used only in the isolated extraction subprocess below.
class AucGapEarlyStopping:  # pragma: no cover - instantiated only by pickle
    pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_ready(item) for item in value]
    return value


def _safe_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dependency artifact escapes project root: {relative_path}") from exc
    return candidate


def _sha256(path: Path) -> str:
    signature = _file_signature(path)
    key = (str(path.resolve()), signature["size"], signature["mtime_ns"])
    cached = _CONTENT_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _CONTENT_HASH_CACHE[key] = value
    return value


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _semantic_result_payload(value: Any) -> Any:
    """Remove retry/log metadata before fingerprinting a result checkpoint."""

    if isinstance(value, Mapping):
        return {
            str(key): _semantic_result_payload(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_RESULT_KEYS
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_result_payload(item) for item in value]
    return _json_ready(value)


def _canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _manifest_hashes(project_root: Path, manifest_path: str | None) -> dict[str, str]:
    if not manifest_path:
        return {}
    payload = _read_json(_safe_path(project_root, manifest_path))
    hashes: dict[str, str] = {}
    for item in (payload.get("models") or {}).values():
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        digest = str(item.get("sha256") or "")
        if path and digest:
            hashes[path] = digest
    return hashes


def _extract_artifact(path: Path, extractor: str) -> dict[str, Any]:
    import joblib

    artifact = joblib.load(path)
    if extractor == "sklearn_feature_names":
        features = [str(value) for value in getattr(artifact, "feature_names_in_", ())]
        effective = [
            str(value)
            for value in (getattr(artifact, "selected_features_", None) or features)
        ]
    elif extractor == "bundle_features":
        if not isinstance(artifact, dict):
            raise TypeError(f"expected model bundle dict: {path}")
        features = [str(value) for value in artifact.get("features") or ()]
        effective = list(features)
        model = artifact.get("model")
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(features):
            effective = [
                feature
                for feature, importance in zip(features, importances)
                if float(importance) > 0.0
            ]
    else:
        raise ValueError(f"unknown model feature extractor: {extractor}")
    if not features:
        raise RuntimeError(f"model artifact declares no required features: {path}")
    schema_version = (
        artifact.get("schema_version")
        if isinstance(artifact, dict)
        else getattr(artifact, "schema_version", None)
    )
    return {
        "features": features,
        "effective_features": effective,
        "schema_version": str(schema_version) if schema_version else None,
    }


def _extract_artifacts_subprocess(
    project_root: Path,
    extractor: str,
    paths: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    path_list = [str(path) for path in paths]
    if not path_list:
        return {}
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    runtime_pythonpath = str(project_root / "src")
    if existing_pythonpath:
        runtime_pythonpath = runtime_pythonpath + os.pathsep + existing_pythonpath
    env = {**os.environ, "PYTHONPATH": runtime_pythonpath}
    command = [
        sys.executable,
        "-m",
        "quant.routine.daily_dependency_runtime",
        "--extractor",
        extractor,
        "--extract-artifacts",
        *path_list,
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "model feature contract extraction failed: "
            + (completed.stderr[-4000:] or completed.stdout[-4000:])
        )
    try:
        payload = json.loads(completed.stdout)
    except ValueError as exc:
        raise RuntimeError(
            f"model feature contract extractor returned invalid JSON: {completed.stdout[-2000:]}"
        ) from exc
    return {
        str(key): dict(value)
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def resolve_model_contracts(
    registry: DependencyRegistry,
    project_root: Path,
    scope: str,
    *,
    cache_path: Path | None = None,
    strict: bool = True,
) -> tuple[dict[str, ModelContract], dict[str, Any]]:
    """Compile active model feature contracts, reusing unchanged artifacts."""

    active = set(registry.required_node_ids(scope))
    model_nodes = [
        registry.nodes[node_id]
        for node_id in registry.topological_order(active)
        if registry.nodes[node_id].artifact is not None
    ]
    effective_cache_path = cache_path or project_root / DEFAULT_CONTRACT_DIR / "model_contract_cache.json"
    cache = _read_json(effective_cache_path)
    cached_entries = (
        dict(cache.get("artifacts") or {})
        if cache.get("schema_version") == MODEL_CACHE_SCHEMA_VERSION
        else {}
    )
    next_entries = dict(cached_entries)
    contracts: dict[str, ModelContract] = {}
    errors: list[str] = []
    loaded_count = 0
    cache_hit_count = 0

    for node in model_nodes:
        assert node.artifact is not None
        expected_hashes = _manifest_hashes(project_root, node.artifact.manifest_path)
        resolved: list[tuple[str, Path, dict[str, int], str]] = []
        changed: list[Path] = []
        for relative in node.artifact.artifact_paths:
            path = _safe_path(project_root, relative)
            if not path.is_file():
                errors.append(f"{node.node_id}: missing artifact {relative}")
                continue
            signature = _file_signature(path)
            cached = cached_entries.get(relative) if isinstance(cached_entries.get(relative), dict) else {}
            if cached.get("signature") == signature and cached.get("sha256"):
                digest = str(cached["sha256"])
            else:
                digest = _sha256(path)
            expected = expected_hashes.get(relative)
            if expected and digest != expected:
                errors.append(
                    f"{node.node_id}: checksum mismatch for {relative}: expected={expected} actual={digest}"
                )
            reusable = (
                cached.get("signature") == signature
                and cached.get("sha256") == digest
                and cached.get("extractor") == node.artifact.extractor
                and cached.get("features")
                and (
                    not node.artifact.expected_schema
                    or cached.get("schema_version")
                    == node.artifact.expected_schema
                )
            )
            if not reusable:
                changed.append(path)
            else:
                cache_hit_count += 1
            resolved.append((relative, path, signature, digest))
        if errors and strict:
            continue

        try:
            extracted = _extract_artifacts_subprocess(
                project_root,
                node.artifact.extractor,
                changed,
            )
        except Exception as exc:
            errors.append(f"{node.node_id}: {exc}")
            continue
        loaded_count += len(changed)

        features_by_artifact: dict[str, tuple[str, ...]] = {}
        effective_by_artifact: dict[str, tuple[str, ...]] = {}
        hashes: list[tuple[str, str]] = []
        for relative, path, signature, digest in resolved:
            cached = cached_entries.get(relative) if isinstance(cached_entries.get(relative), dict) else {}
            data = extracted.get(str(path)) or extracted.get(relative) or cached
            features = tuple(str(value) for value in data.get("features") or ())
            effective = tuple(
                str(value) for value in data.get("effective_features") or features
            )
            if not features:
                errors.append(f"{node.node_id}: empty feature contract for {relative}")
                continue
            actual_schema = str(data.get("schema_version") or "")
            expected_schema = str(node.artifact.expected_schema or "")
            if expected_schema and actual_schema != expected_schema:
                errors.append(
                    f"{node.node_id}: schema mismatch for {relative}: "
                    f"expected={expected_schema} actual={actual_schema or 'missing'}"
                )
                continue
            next_entries[relative] = {
                "signature": signature,
                "sha256": digest,
                "extractor": node.artifact.extractor,
                "features": list(features),
                "effective_features": list(effective),
                "schema_version": actual_schema or None,
            }
            features_by_artifact[relative] = features
            effective_by_artifact[relative] = effective
            hashes.append((relative, digest))

        required_union = tuple(
            sorted({value for values in features_by_artifact.values() for value in values})
        )
        effective_union = tuple(
            sorted({value for values in effective_by_artifact.values() for value in values})
        )
        if not required_union:
            continue
        combined = hashlib.sha256(
            json.dumps(hashes, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()
        contracts[node.node_id] = ModelContract(
            node_id=node.node_id,
            artifact_hashes=tuple(hashes),
            features_by_artifact=features_by_artifact,
            effective_features_by_artifact=effective_by_artifact,
            required_feature_union=required_union,
            effective_feature_union=effective_union,
            combined_hash=combined,
        )

    if errors and strict:
        raise RuntimeError("daily model contract compilation failed: " + " | ".join(errors))
    cache_payload = {
        "schema_version": MODEL_CACHE_SCHEMA_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": next_entries,
    }
    atomic_write_json(cache_payload, effective_cache_path)
    audit = {
        "status": "success" if not errors else "degraded",
        "errors": errors,
        "active_model_nodes": len(model_nodes),
        "artifact_cache_hits": cache_hit_count,
        "artifacts_loaded": loaded_count,
        "cache_path": str(effective_cache_path),
    }
    return contracts, audit


def _lookup(payload: Any, dotted: str) -> Any:
    current = payload
    for part in dotted.split(".") if dotted else ():
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def _parse_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _parse_datetime(value: Any) -> datetime | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    return timestamp.to_pydatetime().replace(tzinfo=None)


def _evidence_value(
    project_root: Path,
    results: Mapping[str, Any],
    adapter: str,
    locator: str,
    date_field: str | None,
    predicate_field: str | None = None,
    expected_value: Any = None,
) -> tuple[date | None, datetime | None, str | None] | None:
    if adapter == "long_factor_snapshot":
        from quant.features.long_factor_snapshot import read_long_factor_snapshot

        try:
            _, manifest = read_long_factor_snapshot(_safe_path(project_root, locator), latest=True)
        except RuntimeError:
            return None
        value = manifest["signal_date"]
        return _parse_date(value), _parse_datetime(value), manifest["data_sha256"]
    if adapter == "result":
        payload = _lookup(results, locator)
        if not isinstance(payload, Mapping):
            return None
        status = str(payload.get("status") or "success")
        if status not in {"success", "skipped", "degraded"}:
            return None
        if (
            predicate_field
            and _lookup(payload, predicate_field) != expected_value
        ):
            return None
        value = _lookup(payload, date_field) if date_field else None
        return (
            _parse_date(value),
            _parse_datetime(value),
            _canonical_fingerprint(_semantic_result_payload(payload)),
        )
    if adapter == "json":
        path = _safe_path(project_root, locator)
        payload = _read_json(path)
        if not payload:
            return None
        status = payload.get("status")
        if status is not None and str(status) not in {
            "success",
            "complete",
            "completed",
            "valid",
        }:
            return None
        if (
            predicate_field
            and _lookup(payload, predicate_field) != expected_value
        ):
            return None
        value = _lookup(payload, date_field or "")
        return _parse_date(value), _parse_datetime(value), _stat_fingerprint(path)
    if adapter == "parquet_max":
        path = _safe_path(project_root, locator)
        if not path.is_file() or not date_field:
            return None
        try:
            values = pd.read_parquet(path, columns=[date_field])[date_field]
        except Exception:
            return None
        maximum = pd.to_datetime(values, errors="coerce").max()
        return _parse_date(maximum), _parse_datetime(maximum), _stat_fingerprint(path)
    if adapter == "file_fingerprint":
        path = _safe_path(project_root, locator)
        if not path.is_file():
            return None
        return None, datetime.fromtimestamp(path.stat().st_mtime), _stat_fingerprint(path)
    if adapter in {"glob_parquet_max", "glob_json_latest"}:
        paths = sorted(project_root.glob(locator))
        values: list[datetime] = []
        fingerprints: list[str] = []
        for path in paths:
            if adapter == "glob_parquet_max":
                if not date_field:
                    continue
                try:
                    frame = pd.read_parquet(path)
                    column = date_field if date_field in frame.columns else "date"
                    maximum = pd.to_datetime(frame[column], errors="coerce").max()
                    parsed = _parse_datetime(maximum)
                except Exception:
                    continue
            else:
                parsed = _parse_datetime(_lookup(_read_json(path), date_field or ""))
            if parsed is not None:
                values.append(parsed)
                fingerprints.append(_stat_fingerprint(path))
        if not values:
            return None
        latest = max(values)
        return latest.date(), latest, hashlib.sha256("".join(sorted(fingerprints)).encode()).hexdigest()
    raise ValueError(f"unsupported dependency evidence adapter: {adapter}")


def _stat_fingerprint(path: Path) -> str:
    return f"sha256:{_sha256(path)}"


def _output_artifact_paths(
    project_root: Path,
    locator: str,
    watermark: date | None,
) -> tuple[Path, ...]:
    """Resolve a registry output to the concrete files relevant to this run."""

    if "YYYYMMDD" in locator:
        if watermark is None:
            matches = project_root.glob(locator.replace("YYYYMMDD", "*"))
            return tuple(sorted(path for path in matches if path.is_file()))
        locator = locator.replace("YYYYMMDD", watermark.strftime("%Y%m%d"))

    path = _safe_path(project_root, locator)
    if path.is_file():
        return (path,)
    if not path.is_dir():
        return ()

    # Canonical market storage is month-partitioned. Hash only the partition
    # that owns the observed watermark instead of walking the full history.
    if watermark is not None and path.name.endswith("_partitioned"):
        partition = (
            path
            / f"year_month={watermark.strftime('%Y%m')}"
            / "data.parquet"
        )
        return (partition,) if partition.is_file() else ()
    return tuple(sorted(item for item in path.rglob("*") if item.is_file()))


def _node_output_fingerprint(
    project_root: Path,
    node: DependencyNode,
    watermark: date | None,
) -> str | None:
    records: list[dict[str, Any]] = []
    root = project_root.resolve()
    for locator in node.outputs:
        for path in _output_artifact_paths(project_root, locator, watermark):
            try:
                relative = path.resolve().relative_to(root).as_posix()
                signature = _file_signature(path)
                digest = _sha256(path)
            except (FileNotFoundError, OSError, ValueError):
                continue
            records.append({"path": relative, "size": signature["size"], "sha256": digest})
    return _canonical_fingerprint(records) if records else None


def collect_node_states(
    registry: DependencyRegistry,
    project_root: Path,
    results: Mapping[str, Any] | None = None,
) -> dict[str, NodeState]:
    result_payload = dict(results or {})
    states: dict[str, NodeState] = {}
    for node_id, node in registry.nodes.items():
        if not node.freshness.evidence:
            continue
        observed: list[tuple[date | None, datetime | None, str | None]] = []
        required_missing = False
        for evidence in node.freshness.evidence:
            item = _evidence_value(
                project_root,
                result_payload,
                evidence.adapter,
                evidence.locator,
                evidence.date_field,
                evidence.predicate_field,
                evidence.expected_value,
            )
            if item is None:
                required_missing = required_missing or evidence.required
            else:
                observed.append(item)
        if required_missing or not observed:
            continue
        dates = [item[0] for item in observed if item[0] is not None]
        checked = [item[1] for item in observed if item[1] is not None]
        fingerprints = sorted(item[2] for item in observed if item[2])
        watermark = min(dates) if dates else None
        output_fingerprint = _node_output_fingerprint(
            project_root,
            node,
            watermark,
        )
        if output_fingerprint:
            fingerprints.append(output_fingerprint)
            fingerprints.sort()
        polled_through = watermark if node.freshness.mode == FreshnessMode.POLLED_THROUGH else None
        states[node_id] = NodeState(
            node_id=node_id,
            watermark=watermark,
            polled_through=polled_through,
            checked_at=max(checked) if checked else None,
            output_fingerprint=(
                hashlib.sha256("".join(fingerprints).encode()).hexdigest()
                if fingerprints
                else None
            ),
            contract_version=node.contract_version,
        )
    return states


def _feature_catalogs(
    contracts: Mapping[str, ModelContract],
) -> dict[str, tuple[str, ...]]:
    selector = contracts.get("score.selector")
    chan = contracts.get("score.chan")
    return {
        "feature.project_daily": tuple(PROJECT_FACTOR_COLUMNS),
        "feature.long_snapshot": tuple(LONG_PRODUCTION_FACTOR_COLUMNS),
        "feature.selector_live": selector.required_feature_union if selector else (),
        "feature.chan_live": chan.required_feature_union if chan else (),
        "feature.right_side_unified_shadow": tuple(
            RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS
        ),
        "feature.right_side_unified": tuple(
            RIGHT_SIDE_SHADOW_MODEL_INPUT_COLUMNS
        ),
        "feature.left_side_unified": tuple(LEFT_SIDE_SCORING_INPUT_COLUMNS),
    }


def _effective_feature_requirements(
    registry: DependencyRegistry,
    contracts: Mapping[str, ModelContract],
) -> dict[str, tuple[str, ...]]:
    requirements: dict[str, set[str]] = {}
    for node_id, contract in contracts.items():
        node = registry.nodes[node_id]
        if node.artifact is None:
            continue
        requirements.setdefault(node.artifact.feature_node_id, set()).update(
            contract.effective_feature_union
        )
    return {
        node_id: tuple(sorted(columns))
        for node_id, columns in requirements.items()
    }


def resolve_active_source_options(
    project_root: Path,
    scope: str,
    *,
    registry: DependencyRegistry = DEFAULT_DAILY_DEPENDENCY_REGISTRY,
) -> dict[str, Any]:
    """Resolve source switches from current model artifacts and product roots."""

    contracts, audit = resolve_model_contracts(
        registry,
        project_root,
        scope,
        cache_path=project_root / DEFAULT_CONTRACT_DIR / "model_contract_cache.json",
        strict=True,
    )
    requirements = _effective_feature_requirements(registry, contracts)
    active = set(
        registry.required_node_ids(
            scope,
            effective_feature_requirements=requirements,
        )
    )
    datasets: list[str] = []
    if "data.top_list" in active:
        datasets.append("top_list")
    if "data.long_research_external" in active:
        datasets.extend(("margin_detail", "moneyflow", "holder_trade_recent"))
    return {
        "scope": scope,
        "include_market_daily": "data.market_daily" in active,
        "include_daily_basic": "data.daily_basic" in active,
        "include_stock_basic": "data.stock_basic" in active,
        "include_index": "data.csi300_daily" in active,
        "include_market_regime": "feature.market_regime" in active,
        "include_tradability": "data.tradability" in active,
        "long_factor_datasets": tuple(datasets),
        "include_financials": "data.financial_pit" in active,
        "include_analyst": "data.analyst_pit" in active,
        "active_source_nodes": tuple(
            node_id
            for node_id in registry.topological_order(active)
            if registry.nodes[node_id].layer == Layer.DATA_SOURCE
        ),
        "effective_feature_requirements": requirements,
        "model_contract_audit": audit,
    }


def _changed_model_nodes(
    previous: Mapping[str, Any],
    contracts: Mapping[str, ModelContract],
) -> tuple[str, ...]:
    previous_models = previous.get("model_contracts") or {}
    previous_hashes = previous.get("model_contract_hashes") or {}
    return tuple(
        node_id
        for node_id, contract in contracts.items()
        if str(
            previous_hashes.get(node_id)
            or (previous_models.get(node_id) or {}).get("combined_hash")
            or ""
        )
        != contract.combined_hash
    )


def _node_contract_hashes(
    registry: DependencyRegistry,
    project_root: Path,
    active_node_ids: Iterable[str],
) -> dict[str, str]:
    """Fingerprint code/config files declared by active registry nodes."""

    file_hashes: dict[str, str] = {}
    node_hashes: dict[str, str] = {}
    missing: list[str] = []
    for node_id in registry.topological_order(active_node_ids):
        node = registry.nodes[node_id]
        components: list[tuple[str, str]] = [
            ("registry_definition", hashlib.sha256(
                json.dumps(
                    _json_ready(asdict(node)),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest())
        ]
        for relative in node.contract_sources:
            path = _safe_path(project_root, relative)
            if not path.is_file():
                missing.append(f"{node_id}: {relative}")
                continue
            digest = file_hashes.get(relative)
            if digest is None:
                digest = _sha256(path)
                file_hashes[relative] = digest
            components.append((relative, digest))
        node_hashes[node_id] = hashlib.sha256(
            json.dumps(components, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()
    if missing:
        raise RuntimeError(
            "daily dependency contract source is missing: " + " | ".join(missing)
        )
    return node_hashes


def _changed_node_contracts(
    previous: Mapping[str, Any],
    current: Mapping[str, str],
) -> tuple[str, ...]:
    previous_hashes = previous.get("node_contract_hashes") or {}
    return tuple(
        node_id
        for node_id, digest in current.items()
        if str(previous_hashes.get(node_id) or "") != digest
    )


def _node_state_payload(states: Mapping[str, NodeState]) -> dict[str, dict[str, Any]]:
    return {
        node_id: _json_ready(asdict(state))
        for node_id, state in sorted(states.items())
    }


def _node_state_fingerprints(states: Mapping[str, NodeState]) -> dict[str, str]:
    return {
        node_id: str(state.output_fingerprint)
        for node_id, state in sorted(states.items())
        if state.output_fingerprint
    }


def _previous_state_fingerprints(previous: Mapping[str, Any]) -> dict[str, str]:
    direct = previous.get("node_state_fingerprints")
    if isinstance(direct, Mapping):
        return {
            str(node_id): str(value)
            for node_id, value in direct.items()
            if value
        }
    states = previous.get("node_states") or {}
    return {
        str(node_id): str(value.get("output_fingerprint"))
        for node_id, value in states.items()
        if isinstance(value, Mapping) and value.get("output_fingerprint")
    }


def _changed_node_states(
    previous: Mapping[str, Any],
    states: Mapping[str, NodeState],
    active_node_ids: Iterable[str],
) -> tuple[str, ...]:
    previous_fingerprints = _previous_state_fingerprints(previous)
    return tuple(
        node_id
        for node_id in sorted(set(active_node_ids))
        if states.get(node_id) is not None
        and states[node_id].output_fingerprint
        and previous_fingerprints.get(node_id)
        != states[node_id].output_fingerprint
    )


def _postflight_result_backed_nodes(
    registry: DependencyRegistry,
    states: Mapping[str, NodeState],
    active_node_ids: Iterable[str],
    target_date: date,
) -> set[str]:
    """Identify fresh, result-backed nodes completed by the current workflow.

    Nodes without standalone output artifacts use the current run result as
    both freshness evidence and their fingerprint. A changed successful result
    is therefore proof of completion, not evidence that the node must run a
    second time during postflight.
    """

    completed: set[str] = set()
    for node_id in active_node_ids:
        node = registry.nodes[node_id]
        if node.outputs or not any(
            evidence.adapter == "result" for evidence in node.freshness.evidence
        ):
            continue
        current, _ = state_is_current(
            node,
            states.get(node_id),
            target_date,
            datetime.now(),
        )
        if current:
            completed.add(node_id)
    return completed


def _identity_is_complete(
    payload: Mapping[str, Any],
    scope: str,
    *,
    committed: bool,
) -> bool:
    return bool(
        payload.get("schema_version") == SNAPSHOT_SCHEMA_VERSION
        and payload.get("identity_complete") is True
        and payload.get("scope") == scope
        and isinstance(payload.get("node_contract_hashes"), Mapping)
        and isinstance(payload.get("model_contract_hashes"), Mapping)
        and isinstance(payload.get("node_state_fingerprints"), Mapping)
        and (not committed or payload.get("baseline_committed") is True)
    )


def _usage_payload(usage: FeatureUsage) -> dict[str, Any]:
    payload = _json_ready(asdict(usage))
    payload.update(
        {
            "produced_count": len(usage.produced),
            "required_count": len(usage.required),
            "effective_count": len(usage.effective),
            "skippable_count": len(usage.skippable),
            "contract_only_zero_importance_count": len(
                usage.contract_only_zero_importance
            ),
            "unknown_count": len(usage.unknown),
        }
    )
    return payload


def audit_required_freshness(
    registry: DependencyRegistry,
    scope: str,
    target_date: date,
    states: Mapping[str, NodeState],
    *,
    effective_feature_requirements: Mapping[str, Iterable[str]] | None = None,
    results: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked: list[str] = []
    current_time = now or datetime.now()
    for node_id in registry.required_node_ids(
        scope,
        effective_feature_requirements=effective_feature_requirements,
    ):
        node = registry.nodes[node_id]
        if not node.final_gate:
            continue
        checked.append(node_id)
        state = states.get(node_id)
        current, reason = state_is_current(node, state, target_date, current_time)
        actual = (
            state.polled_through.isoformat()
            if state and state.polled_through
            else state.watermark.isoformat()
            if state and state.watermark
            else state.checked_at.isoformat()
            if state and state.checked_at
            else state.output_fingerprint
            if state
            else None
        )
        if not current:
            failures.append(
                {
                    "node_id": node_id,
                    "mode": node.freshness.mode.value,
                    "expected": target_date.isoformat(),
                    "actual": actual,
                }
            )
        if node.quality.fail_closed:
            quality_payload = None
            for alias in node.result_aliases:
                candidate = _lookup(results or {}, alias)
                if isinstance(candidate, Mapping):
                    evidence = candidate.get(node.quality.evidence_field)
                    if isinstance(evidence, Mapping):
                        quality_payload = evidence
                        break
            quality_status = (
                str(quality_payload.get("status") or "")
                if quality_payload is not None
                else "missing"
            )
            unresolved = (
                int(quality_payload.get("unresolved_missing_symbols") or 0)
                if quality_payload is not None
                else None
            )
            if (
                quality_status != "success"
                or unresolved is None
                or unresolved > node.quality.maximum_unresolved_missing
            ):
                failures.append(
                    {
                        "node_id": node_id,
                        "mode": "data_quality",
                        "expected": {
                            "status": "success",
                            "maximum_unresolved_missing": (
                                node.quality.maximum_unresolved_missing
                            ),
                        },
                        "actual": (
                            dict(quality_payload)
                            if quality_payload is not None
                            else None
                        ),
                    }
                )
    return {
        "status": "success" if not failures else "failed",
        "checked_nodes": checked,
        "failures": failures,
    }


def publish_daily_dependency_snapshot(
    project_root: Path,
    target_date: str | date,
    *,
    scope: str = "all",
    results: Mapping[str, Any] | None = None,
    phase: str = "preflight",
    strict_models: bool = True,
    strict_freshness: bool = False,
    raise_on_failure: bool = True,
    registry: DependencyRegistry = DEFAULT_DAILY_DEPENDENCY_REGISTRY,
) -> dict[str, Any]:
    target = _parse_date(target_date)
    if target is None:
        raise ValueError(f"invalid daily dependency target date: {target_date}")
    output_dir = project_root / DEFAULT_CONTRACT_DIR
    output_path = output_dir / "latest.json"
    scope_state_path = output_dir / f"latest-{scope}.json"
    committed_previous = _read_json(scope_state_path)
    result_payload = dict(results or {})
    preflight_identity = result_payload.get("dependency_preflight") or {}
    preflight_identity_is_current = bool(
        phase == "postflight"
        and isinstance(preflight_identity, Mapping)
        and _identity_is_complete(preflight_identity, scope, committed=False)
        and _parse_date(preflight_identity.get("target_trade_date")) == target
    )
    committed_identity_is_current = _identity_is_complete(
        committed_previous,
        scope,
        committed=True,
    )
    if phase == "postflight":
        if preflight_identity_is_current:
            previous = preflight_identity
            comparison_identity = "preflight"
        else:
            # A committed snapshot is a baseline for planning the next run,
            # not proof of what this run observed before executing refreshes.
            # Postflight must compare against its own same-scope, same-target
            # preflight identity or fail closed without advancing the baseline.
            previous = {}
            comparison_identity = "missing"
    elif committed_identity_is_current:
        previous = committed_previous
        comparison_identity = "committed"
    else:
        previous = {}
        comparison_identity = "missing"
    contracts, contract_audit = resolve_model_contracts(
        registry,
        project_root,
        scope,
        cache_path=output_dir / "model_contract_cache.json",
        strict=strict_models,
    )
    states = collect_node_states(registry, project_root, result_payload)
    effective_requirements = _effective_feature_requirements(registry, contracts)
    active = set(
        registry.required_node_ids(
            scope,
            effective_feature_requirements=effective_requirements,
        )
    )
    node_contract_hashes = _node_contract_hashes(
        registry,
        project_root,
        active,
    )
    changed_models = _changed_model_nodes(previous, contracts)
    changed_contract_nodes = _changed_node_contracts(
        previous,
        node_contract_hashes,
    )
    changed_state_nodes = _changed_node_states(previous, states, active)
    expected_preflight_changes = (
        set(previous.get("refresh_node_ids") or ())
        if comparison_identity == "preflight"
        else set()
    )
    changed_models = tuple(
        node_id
        for node_id in changed_models
        if node_id not in expected_preflight_changes
    )
    changed_contract_nodes = tuple(
        node_id
        for node_id in changed_contract_nodes
        if node_id not in expected_preflight_changes
    )
    changed_state_nodes = tuple(
        node_id
        for node_id in changed_state_nodes
        if node_id not in expected_preflight_changes
    )
    if phase == "postflight":
        result_backed_nodes = _postflight_result_backed_nodes(
            registry,
            states,
            active,
            target,
        )
        changed_state_nodes = tuple(
            node_id
            for node_id in changed_state_nodes
            if node_id not in result_backed_nodes
        )
    identity_missing_nodes = set(active) if comparison_identity == "missing" else set()
    plan = build_dependency_plan(
        registry,
        scope,
        target,
        states,
        changed_nodes=(
            set(changed_models)
            | set(changed_contract_nodes)
            | set(changed_state_nodes)
            | identity_missing_nodes
        ),
        effective_feature_requirements=effective_requirements,
    )
    catalogs = _feature_catalogs(contracts)
    usages = classify_feature_usage(registry, scope, contracts, catalogs)
    live_model_features = sorted(
        {
            feature
            for contract in contracts.values()
            for feature in contract.required_feature_union
        }
    )
    registered = {definition.name for definition in FACTOR_REGISTRY}
    registered_canonical = {
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "feature"
    }
    registered_aliases = {
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "compatibility_alias"
    }
    production_registered = {
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.lifecycle.startswith("production")
        or definition.lifecycle == "strategy_identity"
        or (
            definition.lifecycle == "compatibility_alias"
            and definition.refresh_cadence == "trade_daily"
        )
    }
    research_registered = {
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.lifecycle == "research_candidate"
    }
    freshness_audit = audit_required_freshness(
        registry,
        scope,
        target,
        states,
        effective_feature_requirements=effective_requirements,
        results=results,
    )
    refresh_entries = [
        entry
        for entry in plan
        if entry.active and entry.action in {"refresh", "refresh_if_changed", "poll"}
    ]
    strict_gate_failed = bool(
        strict_freshness
        and (
            freshness_audit["status"] != "success"
            or refresh_entries
        )
    )
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "registry_schema_version": registry.schema_version,
        "identity_complete": True,
        "baseline_committed": False,
        "comparison_identity": comparison_identity,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_trade_date": target.isoformat(),
        "scope": scope,
        "phase": phase,
        "status": "failed" if strict_gate_failed else contract_audit["status"],
        "active_nodes": list(registry.topological_order(active)),
        "node_contract_hashes": node_contract_hashes,
        "changed_model_nodes": list(changed_models),
        "changed_contract_nodes": list(changed_contract_nodes),
        "changed_state_nodes": list(changed_state_nodes),
        "node_states": _node_state_payload(states),
        "node_state_fingerprints": _node_state_fingerprints(states),
        "inactive_nodes": [
            node_id
            for node_id in registry.topological_order()
            if node_id not in active
        ],
        "inactive_research_nodes": [
            node_id
            for node_id, node in registry.nodes.items()
            if node.lifecycle.value == "research_only"
        ],
        "layer_counts": {
            layer.value: sum(
                1 for node_id in active if registry.nodes[node_id].layer == layer
            )
            for layer in Layer
        },
        "incremental_plan": [_json_ready(asdict(entry)) for entry in plan],
        "refresh_node_ids": [entry.node_id for entry in refresh_entries],
        "refresh_nodes": [
            {
                "node_id": entry.node_id,
                "layer": entry.layer,
                "action": entry.action,
                "ui_step": registry.nodes[entry.node_id].ui_step,
            }
            for entry in refresh_entries
        ],
        "model_contract_audit": contract_audit,
        "model_contract_hashes": {
            node_id: contract.combined_hash
            for node_id, contract in sorted(contracts.items())
        },
        "model_contracts": {
            node_id: {
                **_json_ready(asdict(contract)),
                "artifact_count": len(contract.artifact_hashes),
                "required_feature_count": len(contract.required_feature_union),
                "effective_feature_count": len(contract.effective_feature_union),
                "location_status": (
                    "production_consumed_from_research_path"
                    if registry.nodes[node_id].artifact
                    and registry.nodes[node_id].artifact.approved_research_path
                    else "production"
                ),
            }
            for node_id, contract in sorted(contracts.items())
        },
        "feature_usage": {
            usage.feature_node_id: _usage_payload(usage)
            for usage in usages
        },
        "feature_inventory": {
            "registered_factor_count": len(registered),
            "registered_canonical_factor_count": len(registered_canonical),
            "registered_compatibility_alias_count": len(registered_aliases),
            "registered_production_record_count": len(production_registered),
            "registered_research_candidate_count": len(research_registered),
            "active_model_required_union_count": len(live_model_features),
            "active_model_features_not_in_factor_registry": sorted(
                set(live_model_features) - registered
            ),
            "registered_not_in_active_model_artifacts": sorted(
                registered - set(live_model_features)
            ),
            "registered_not_in_model_is_safe_to_stop": False,
            "reason": (
                "Rule products (Tea/v44 and other deterministic features) also consume "
                "registered factors; only nodes outside the active product closure are skippable."
            ),
        },
        "refresh_classes": {
            "exact_trade_date": [
                node_id
                for node_id in registry.topological_order(active)
                if registry.nodes[node_id].freshness.mode
                == FreshnessMode.EXACT_TRADE_DATE
            ],
            "poll_daily_only": [
                node_id
                for node_id in registry.topological_order(active)
                if registry.nodes[node_id].freshness.mode
                == FreshnessMode.POLLED_THROUGH
            ],
            "scheduled_or_static": [
                node_id
                for node_id in registry.topological_order(active)
                if registry.nodes[node_id].freshness.mode
                in {FreshnessMode.TTL, FreshnessMode.IMMUTABLE, FreshnessMode.AS_OF}
            ],
            "not_in_current_scope": [
                node_id
                for node_id in registry.topological_order()
                if node_id not in active
            ],
        },
        "freshness_audit": freshness_audit,
        "registry": registry.as_dict(),
    }
    commit_baseline = bool(
        phase == "postflight"
        and strict_freshness
        and payload["status"] == "success"
        and freshness_audit["status"] == "success"
        and not refresh_entries
    )
    payload["baseline_committed"] = commit_baseline
    atomic_write_json(_json_ready(payload), output_path)
    dated_path = output_dir / f"{target.isoformat()}-{scope}-{phase}.json"
    atomic_write_json(_json_ready(payload), dated_path)
    if commit_baseline:
        atomic_write_json(_json_ready(payload), scope_state_path)
    if strict_gate_failed and raise_on_failure:
        freshness_details = ", ".join(
            f"{item['node_id']}={item['actual']}"
            for item in freshness_audit["failures"]
        )
        unresolved = ",".join(entry.node_id for entry in refresh_entries)
        details = "; ".join(
            value
            for value in (
                freshness_details,
                f"unresolved={unresolved}" if unresolved else "",
            )
            if value
        )
        raise RuntimeError(
            f"daily dependency freshness gate failed for {target.isoformat()}: {details}"
        )
    return {
        "status": payload["status"],
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "identity_complete": True,
        "baseline_committed": commit_baseline,
        "comparison_identity": comparison_identity,
        "target_trade_date": target.isoformat(),
        "scope": scope,
        "phase": phase,
        "active_node_count": len(active),
        "inactive_node_count": len(registry.nodes) - len(active),
        "model_contract_count": len(contracts),
        "active_model_required_union_count": len(live_model_features),
        "changed_model_nodes": list(changed_models),
        "changed_contract_nodes": list(changed_contract_nodes),
        "changed_state_nodes": list(changed_state_nodes),
        "node_contract_hashes": node_contract_hashes,
        "model_contract_hashes": payload["model_contract_hashes"],
        "node_state_fingerprints": payload["node_state_fingerprints"],
        "refresh_node_ids": payload["refresh_node_ids"],
        "refresh_nodes": payload["refresh_nodes"],
        "freshness_audit": freshness_audit,
        "path": str(output_path),
        "committed_path": str(scope_state_path),
        "dated_path": str(dated_path),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Daily dependency runtime helper")
    parser.add_argument("--extractor", choices=("sklearn_feature_names", "bundle_features"))
    parser.add_argument("--extract-artifacts", nargs="*")
    args = parser.parse_args()
    if args.extract_artifacts is not None:
        if not args.extractor:
            parser.error("--extractor is required with --extract-artifacts")
        payload = {
            str(path): _extract_artifact(Path(path), args.extractor)
            for path in args.extract_artifacts
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        return 0
    parser.error("no runtime operation requested")
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(_cli())


__all__ = [
    "audit_required_freshness",
    "collect_node_states",
    "publish_daily_dependency_snapshot",
    "resolve_active_source_options",
    "resolve_model_contracts",
]
