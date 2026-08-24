"""Explicit selector ranking-source contract and production score adapter.

The unified model has one semantic output: ``ranking_score``.  It is never
aliased to the legacy up/down probability fields, and it does not select an
entry/exit playbook.  The default source remains the existing Z-skill path;
the unified path fails closed on config, manifest, checksum, date, or coverage
drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from quant.core.paths import PROJECT_ROOT
from quant.application.left_side_ranking import (
    DEFAULT_LEFT_SIDE_RANKING_CONFIG,
    LeftSideRankingConfig,
    load_left_side_ranking_scores,
)
from quant.features.canonical_factor_names import (
    find_forbidden_aliases_in_payload,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.research.right_side_unified_features import RIGHT_SIDE_SIGNALS
from quant.research.right_side_unified_features import RULE_FEATURE_SCHEMA_VERSION
from quant.research.short_side_groups import LEFT_GROUP_MEMBERS


DEFAULT_SELECTOR_RANKING_CONFIG_PATH = Path(
    "configs/strategies/right_side_ranking_selector.yaml"
)
RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION = (
    "right-side-unified-ranking-production-v2-canonical-alias-free"
)
RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION = (
    "right-side-unified-ranking-score-v2-canonical-alias-free"
)
RIGHT_SIDE_PRODUCTION_APPROVAL_SCHEMA_VERSION = (
    "right-side-production-rollout-approval-v1"
)
RANKING_SCORE_FIELD = "ranking_score"
NORMALIZED_RANKING_SCORE_FIELD = "ranking_score_normalized"
RANKING_NORMALIZATION_SCHEMA_VERSION = "frozen-oot-b-quantile-cdf-v1"
FORBIDDEN_RANKING_ALIASES: tuple[str, ...] = (
    "pred_up5",
    "pred_up8",
    "pred_down3",
)


class SelectorRankingSource(str, Enum):
    LEGACY_Z_SKILL = "legacy_z_skill"
    RIGHT_SIDE_UNIFIED = "right_side_unified"


@dataclass(frozen=True)
class SelectorRankingPaths:
    artifact: Path
    artifact_manifest: Path
    feature_output: Path
    feature_manifest: Path
    score_output: Path
    score_manifest: Path
    z_signal_cache: Path
    family_signal_cache: Path
    market_data_root: Path
    promotion_approval: Path


@dataclass(frozen=True)
class SelectorRankingConfig:
    source: SelectorRankingSource
    promotion_enabled: bool
    failure_policy: str
    supported_strategy_keys: tuple[str, ...]
    replaced_signals: tuple[str, ...]
    preserved_legacy_signals: tuple[str, ...]
    paths: SelectorRankingPaths
    history_years: int
    config_path: Path
    factor_workers: int
    probability_calibration: str
    score_normalization: str
    normalized_score_field: str
    production_threshold_mode: str
    selection_policy: str
    research_threshold_reference: float


def _safe_path(project_root: Path, value: object) -> Path:
    root = project_root.resolve()
    path = (root / str(value or "")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"selector ranking path escapes project root: {value}") from exc
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"selector ranking config field must be a mapping: {field}")
    return value


def load_selector_ranking_config(
    project_root: Path = PROJECT_ROOT,
    config_path: Path | str = DEFAULT_SELECTOR_RANKING_CONFIG_PATH,
) -> SelectorRankingConfig:
    """Load the source switch and reject ambiguous/unsafe combinations."""

    root = project_root.resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"selector ranking config is missing: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("selector ranking config must be a mapping")
    forbidden_factor_aliases = find_forbidden_aliases_in_payload(payload)
    if forbidden_factor_aliases:
        raise ValueError(
            "selector ranking config contains forbidden factor aliases: "
            f"{forbidden_factor_aliases}"
        )
    release = _mapping(payload.get("release"), field="release")
    selector = _mapping(payload.get("selector"), field="selector")
    unified = _mapping(payload.get("right_side_unified"), field="right_side_unified")
    promotion = _mapping(payload.get("promotion"), field="promotion")
    playbook = _mapping(payload.get("playbook"), field="playbook")

    if release.get("lifecycle") != "production":
        raise ValueError("selector ranking release.lifecycle must be production")
    expected_nodes = {
        "feature.right_side_unified",
        "score.right_side_unified",
        "product.right_side_unified_adapter",
        "score.selector",
    }
    if set(release.get("dependency_nodes") or ()) != expected_nodes:
        raise ValueError("selector ranking dependency_nodes drifted")
    try:
        source = SelectorRankingSource(str(selector.get("ranking_source") or ""))
    except ValueError as exc:
        raise ValueError("selector ranking source is unsupported") from exc
    allowed = {str(value) for value in (selector.get("allowed_sources") or ())}
    if allowed != {item.value for item in SelectorRankingSource}:
        raise ValueError("selector ranking allowed_sources must enumerate both sources")
    if selector.get("failure_policy") != "fail_closed":
        raise ValueError("selector ranking failure_policy must be fail_closed")
    if selector.get("score_field") != RANKING_SCORE_FIELD:
        raise ValueError("unified selector score field must remain ranking_score")
    if tuple(selector.get("forbidden_aliases") or ()) != FORBIDDEN_RANKING_ALIASES:
        raise ValueError("selector ranking forbidden aliases drifted")
    ranking_policy = _mapping(
        selector.get("ranking_policy"), field="selector.ranking_policy"
    )
    if ranking_policy.get("probability_calibration") != "embedded_event_level_platt":
        raise ValueError("selector ranking probability calibration drifted")
    if ranking_policy.get("normalization") != "frozen_oot_b_quantile_cdf_v1":
        raise ValueError("selector ranking normalization drifted")
    if ranking_policy.get("normalized_score_field") != NORMALIZED_RANKING_SCORE_FIELD:
        raise ValueError("selector normalized score field drifted")
    if ranking_policy.get("production_threshold_mode") != "none_rank_only":
        raise ValueError("selector production threshold must remain disabled for rank-only use")
    if ranking_policy.get("selection_policy") != "downstream_top_n_ordering":
        raise ValueError("selector ranking selection policy drifted")
    research_threshold_reference = float(
        ranking_policy.get("research_threshold_reference")
    )
    if not np.isfinite(research_threshold_reference) or not (
        0.0 < research_threshold_reference < 1.0
    ):
        raise ValueError("selector research threshold reference is invalid")
    supported = tuple(str(value) for value in (selector.get("supported_strategy_keys") or ()))
    if supported != tuple(RIGHT_SIDE_SIGNALS):
        raise ValueError("selector ranking supported strategies drifted from training contract")
    replaced = tuple(str(value) for value in (selector.get("replaced_signals") or ()))
    preserved = tuple(
        str(value) for value in (selector.get("preserved_legacy_signals") or ())
    )
    expected_preserved = ("DUICHEN_VA", "NANA", "YIDONG_DILIAN")
    if replaced != tuple(RIGHT_SIDE_SIGNALS):
        raise ValueError("selector ranking replaced_signals drifted from the 14-signal scope")
    if preserved != expected_preserved:
        raise ValueError("selector ranking preserved_legacy_signals drifted")
    if set(replaced) & set(preserved):
        raise ValueError("selector ranking signal scopes overlap")

    enabled = promotion.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("selector ranking promotion.enabled must be boolean")
    if source == SelectorRankingSource.RIGHT_SIDE_UNIFIED and not enabled:
        raise ValueError(
            "right_side_unified ranking source requires explicit promotion.enabled=true"
        )
    if source == SelectorRankingSource.LEGACY_Z_SKILL and enabled:
        raise ValueError(
            "promotion.enabled=true requires selector.ranking_source=right_side_unified"
        )
    if promotion.get("preserve_rollback_score_node") != "score.z_skill":
        raise ValueError("selector ranking rollback source must remain score.z_skill")
    if promotion.get("preserved_legacy_score_node") != "score.z_skill":
        raise ValueError("left-side legacy signals must remain on score.z_skill")
    if int(promotion.get("rollback_observation_cycles") or 0) < 1:
        raise ValueError("selector ranking needs at least one rollback observation cycle")
    if playbook.get("coupling") != "independent" or (
        playbook.get("selector_ranking_does_not_choose_entry_or_exit") is not True
    ):
        raise ValueError("selector ranking and playbook layers must remain independent")
    if promotion.get("approval_schema_version") != (
        RIGHT_SIDE_PRODUCTION_APPROVAL_SCHEMA_VERSION
    ):
        raise ValueError("selector promotion approval schema drifted")
    if promotion.get("approval_field") != "approve_online":
        raise ValueError("selector promotion approval field must remain approve_online")
    approval_value = promotion.get("approval_manifest")
    if not approval_value:
        raise ValueError("selector promotion approval_manifest is required")
    if unified.get("artifact_schema_version") != (
        RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("right-side production artifact schema drifted")
    if unified.get("score_schema_version") != RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION:
        raise ValueError("right-side production score schema drifted")
    if unified.get("feature_schema_version") != RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION:
        raise ValueError("right-side production feature schema drifted")
    if unified.get("project_factor_schema_version") != PROJECT_FACTOR_SCHEMA_VERSION:
        raise ValueError("right-side production project factor schema drifted")
    if unified.get("rule_factor_schema_version") != RULE_FEATURE_SCHEMA_VERSION:
        raise ValueError("right-side production rule factor schema drifted")
    if unified.get("factor_contract_sha256") != (
        RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256
    ):
        raise ValueError("right-side production factor contract drifted")
    history_years = int(unified.get("history_years") or 0)
    factor_workers = int(unified.get("factor_workers") or 1)
    if history_years < 6:
        raise ValueError("right-side production feature build needs >= 6 history years")
    if not 1 <= factor_workers <= 32:
        raise ValueError("right-side production factor_workers must be in [1, 32]")

    def resolve(key: str) -> Path:
        value = unified.get(key)
        if not value:
            raise ValueError(f"right-side production config missing path: {key}")
        return _safe_path(root, value)

    return SelectorRankingConfig(
        source=source,
        promotion_enabled=enabled,
        failure_policy="fail_closed",
        supported_strategy_keys=supported,
        replaced_signals=replaced,
        preserved_legacy_signals=preserved,
        paths=SelectorRankingPaths(
            artifact=resolve("artifact"),
            artifact_manifest=resolve("artifact_manifest"),
            feature_output=resolve("feature_output"),
            feature_manifest=resolve("feature_manifest"),
            score_output=resolve("score_output"),
            score_manifest=resolve("score_manifest"),
            z_signal_cache=resolve("z_signal_cache"),
            family_signal_cache=resolve("family_signal_cache"),
            market_data_root=resolve("market_data_root"),
            promotion_approval=_safe_path(root, approval_value),
        ),
        history_years=history_years,
        config_path=path.resolve(),
        factor_workers=factor_workers,
        probability_calibration="embedded_event_level_platt",
        score_normalization="frozen_oot_b_quantile_cdf_v1",
        normalized_score_field=NORMALIZED_RANKING_SCORE_FIELD,
        production_threshold_mode="none_rank_only",
        selection_policy="downstream_top_n_ordering",
        research_threshold_reference=research_threshold_reference,
    )


# One process uses one source contract.  A reviewed source change therefore
# takes effect only after restart, keeping the Web adapter and dependency DAG
# on the same immutable choice throughout a daily run.
DEFAULT_SELECTOR_RANKING_CONFIG = load_selector_ranking_config()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_score_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(f"right-side ranking score manifest is unavailable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("right-side ranking score manifest must be a mapping")
    return payload


def validate_selector_promotion_approval(
    config: SelectorRankingConfig,
) -> dict[str, Any]:
    """Require an explicit operator release approval linked to research evidence.

    Research decisions remain immutable evidence.  A maintainer may approve a
    reversible production rollout for operational reasons, but the approval
    must acknowledge failed research gates and pin the exact research decision
    and successful shadow run instead of rewriting either result.
    """

    if config.source != SelectorRankingSource.RIGHT_SIDE_UNIFIED:
        raise RuntimeError("selector promotion approval is only valid for unified source")
    if not config.promotion_enabled:
        raise RuntimeError("selector promotion is not enabled")
    approval = _load_score_manifest(config.paths.promotion_approval)
    if approval.get("schema_version") != RIGHT_SIDE_PRODUCTION_APPROVAL_SCHEMA_VERSION:
        raise RuntimeError("selector promotion approval schema mismatch")
    if approval.get("approve_online") is not True:
        raise RuntimeError("selector promotion has no explicit approve_online approval")
    if approval.get("deployment_mode") != "reversible_unified_ranking_cutover":
        raise RuntimeError("selector promotion deployment mode is unsupported")
    if approval.get("playbook_promoted") is not False:
        raise RuntimeError("selector promotion must keep the playbook layer disabled")
    if approval.get("rollback_ranking_source") != "legacy_z_skill":
        raise RuntimeError("selector promotion approval must preserve legacy rollback")
    research = approval.get("research_decision")
    if not isinstance(research, Mapping):
        raise RuntimeError("selector promotion approval lacks research decision evidence")
    research_path = Path(str(research.get("path") or ""))
    if not research_path.is_absolute():
        research_path = PROJECT_ROOT / research_path
    if not research_path.is_file() or research.get("sha256") != _sha256(research_path):
        raise RuntimeError("selector promotion research decision checksum mismatch")
    research_payload = _load_score_manifest(research_path)
    if research_payload.get("selected_research_candidate") != approval.get(
        "selected_research_candidate"
    ):
        raise RuntimeError("selector promotion candidate differs from research decision")
    if research_payload.get("replace_online") is not True:
        raise RuntimeError("selector promotion requires a passing ranking replacement result")
    acknowledgements = set(str(value) for value in (approval.get("acknowledged_risks") or ()))
    required_acknowledgements = {
        "canonical_alias_free_retrain_completed",
        "legacy_artifact_preserved_for_rollback",
    }
    if not required_acknowledgements <= acknowledgements:
        raise RuntimeError("selector promotion approval does not acknowledge research risks")
    shadow = approval.get("shadow_acceptance")
    if not isinstance(shadow, Mapping):
        raise RuntimeError("selector promotion approval lacks shadow acceptance")
    shadow_path = Path(str(shadow.get("path") or ""))
    if not shadow_path.is_absolute():
        shadow_path = PROJECT_ROOT / shadow_path
    if not shadow_path.is_file() or shadow.get("sha256") != _sha256(shadow_path):
        raise RuntimeError("selector promotion shadow acceptance checksum mismatch")
    shadow_payload = _load_score_manifest(shadow_path)
    if (
        shadow_payload.get("status") != "success"
        or shadow_payload.get("production_affected") is not False
        or shadow_payload.get("selector_published") is not False
    ):
        raise RuntimeError("selector promotion shadow acceptance is not successful")
    selected = str(approval.get("selected_research_candidate") or "")
    if selected != "unified_long_task_deep":
        raise RuntimeError("selector promotion approval has no supported candidate")
    return approval


def load_right_side_ranking_scores(
    signal_date: str,
    *,
    config: SelectorRankingConfig,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    """Load one exact-date, checksum-pinned production ranking snapshot."""

    approval = validate_selector_promotion_approval(config)
    expected_date = pd.to_datetime(signal_date, errors="raise").normalize()
    manifest = _load_score_manifest(config.paths.score_manifest)
    artifact_manifest = _load_score_manifest(config.paths.artifact_manifest)
    if artifact_manifest.get("schema_version") != (
        RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError("right-side ranking artifact manifest schema mismatch")
    if artifact_manifest.get("lifecycle") != "production":
        raise RuntimeError("right-side ranking artifact is not a production release")
    if artifact_manifest.get("score_field") != RANKING_SCORE_FIELD:
        raise RuntimeError("right-side ranking artifact has the wrong output semantics")
    if artifact_manifest.get("playbook_coupling") != "independent":
        raise RuntimeError("right-side ranking artifact is coupled to a playbook")
    if artifact_manifest.get("probability_calibration") != config.probability_calibration:
        raise RuntimeError("right-side ranking artifact calibration mismatch")
    if artifact_manifest.get("production_threshold_mode") != (
        config.production_threshold_mode
    ):
        raise RuntimeError("right-side ranking artifact threshold policy mismatch")
    artifact_normalization = artifact_manifest.get("score_normalization") or {}
    if artifact_normalization.get("schema_version") != (
        RANKING_NORMALIZATION_SCHEMA_VERSION
    ):
        raise RuntimeError("right-side ranking artifact normalization mismatch")
    if tuple(artifact_manifest.get("replaced_signals") or ()) != (
        config.replaced_signals
    ):
        raise RuntimeError("right-side ranking artifact replaced_signals mismatch")
    if tuple(artifact_manifest.get("preserved_legacy_signals") or ()) != (
        config.preserved_legacy_signals
    ):
        raise RuntimeError("right-side ranking artifact preserved_legacy_signals mismatch")
    if artifact_manifest.get("selected_candidate") != approval.get(
        "selected_research_candidate"
    ):
        raise RuntimeError("right-side ranking artifact differs from promotion approval")
    if artifact_manifest.get("promotion_approval_sha256") != _sha256(
        config.paths.promotion_approval
    ):
        raise RuntimeError("right-side ranking artifact approval checksum mismatch")
    model_item = (artifact_manifest.get("models") or {}).get("ranking") or {}
    if not config.paths.artifact.is_file():
        raise RuntimeError("right-side ranking production artifact is missing")
    artifact_digest = _sha256(config.paths.artifact)
    if model_item.get("sha256") != artifact_digest:
        raise RuntimeError("right-side ranking production artifact checksum mismatch")
    if manifest.get("artifact_sha256") != artifact_digest:
        raise RuntimeError("right-side score snapshot was not produced by active artifact")
    checks = {
        "status": "success",
        "schema_version": RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
        "target_date": expected_date.date().isoformat(),
        "score_field": RANKING_SCORE_FIELD,
        "artifact_schema_version": RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "playbook_coupling": "independent",
        "normalized_score_field": NORMALIZED_RANKING_SCORE_FIELD,
        "probability_calibration": config.probability_calibration,
        "production_threshold_mode": config.production_threshold_mode,
        "selection_policy": config.selection_policy,
        "replaced_signals": list(config.replaced_signals),
        "preserved_legacy_signals": list(config.preserved_legacy_signals),
        "promotion_approval_sha256": _sha256(config.paths.promotion_approval),
    }
    drift = {
        field: (manifest.get(field), expected)
        for field, expected in checks.items()
        if manifest.get(field) != expected
    }
    if drift:
        raise RuntimeError(f"right-side ranking score manifest drifted: {drift}")
    score_normalization = manifest.get("score_normalization") or {}
    if score_normalization.get("schema_version") != (
        RANKING_NORMALIZATION_SCHEMA_VERSION
    ):
        raise RuntimeError("right-side ranking score normalization mismatch")
    if not config.paths.score_output.is_file():
        raise RuntimeError("right-side ranking score parquet is missing")
    digest = _sha256(config.paths.score_output)
    if manifest.get("output_sha256") != digest:
        raise RuntimeError("right-side ranking score parquet checksum mismatch")
    frame = pd.read_parquet(config.paths.score_output)
    required = {
        "symbol",
        "date",
        RANKING_SCORE_FIELD,
        NORMALIZED_RANKING_SCORE_FIELD,
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"right-side ranking snapshot missing columns: {sorted(missing)}")
    forbidden = set(FORBIDDEN_RANKING_ALIASES) & set(frame.columns)
    if forbidden:
        raise RuntimeError(
            "right-side ranking snapshot must not impersonate legacy targets: "
            f"{sorted(forbidden)}"
        )
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any() or not dates.eq(expected_date).all():
        raise RuntimeError("right-side ranking snapshot is not exact-date")
    symbols = frame["symbol"].astype(str)
    if frame.assign(symbol=symbols, date=dates).duplicated(["symbol", "date"]).any():
        raise RuntimeError("right-side ranking snapshot has duplicate keys")
    scores = pd.to_numeric(frame[RANKING_SCORE_FIELD], errors="coerce").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise RuntimeError("right-side ranking scores must be finite values in [0, 1]")
    normalized_scores = pd.to_numeric(
        frame[NORMALIZED_RANKING_SCORE_FIELD], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(normalized_scores).all() or (
        (normalized_scores < 0.0) | (normalized_scores > 100.0)
    ).any():
        raise RuntimeError("normalized right-side ranking scores must be in [0, 100]")
    if len(scores) > 1:
        ordered = np.argsort(scores, kind="stable")
        if np.any(np.diff(normalized_scores[ordered]) < -1e-9):
            raise RuntimeError("right-side ranking normalization is not monotonic")
    if int(manifest.get("candidate_count") or -1) != len(frame):
        raise RuntimeError("right-side ranking candidate_count mismatches parquet")
    return dict(zip(symbols, zip(scores, normalized_scores))), manifest


def _row_uses_supported_strategy(
    row: Mapping[str, Any],
    supported: set[str],
) -> bool:
    for signal in row.get("signals") or ():
        if not isinstance(signal, Mapping):
            continue
        keys = {
            str(signal.get("strategy_key") or "").upper(),
            str(signal.get("strategy_family") or "").upper(),
        }
        if keys & supported:
            return True
    return False


def apply_selector_ranking_source(
    rows: list[dict[str, Any]],
    signal_date: str | None,
    *,
    config: SelectorRankingConfig | None = None,
    left_config: LeftSideRankingConfig | None = None,
    require_all_ranked_candidates: bool = False,
) -> list[dict[str, Any]]:
    """Apply the two unified rankers with deterministic right-side precedence."""

    active = config or DEFAULT_SELECTOR_RANKING_CONFIG
    active_left = (
        left_config
        if left_config is not None
        else DEFAULT_LEFT_SIDE_RANKING_CONFIG
        if config is None
        else None
    )
    right_eligible_symbols = {
        str(row.get("symbol") or "")
        for row in rows
        if active.source == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        and _row_uses_supported_strategy(row, set(active.supported_strategy_keys))
    }
    if right_eligible_symbols:
        if not signal_date:
            raise RuntimeError("right-side unified selector ranking requires signal_date")
        right_scores, right_manifest = load_right_side_ranking_scores(
            signal_date, config=active
        )
        missing = sorted(right_eligible_symbols - set(right_scores))
        if missing:
            raise RuntimeError(
                "right-side unified ranking coverage is incomplete for selector rows: "
                f"{missing[:20]}"
            )
        if require_all_ranked_candidates:
            unconsumed = sorted(set(right_scores) - right_eligible_symbols)
            if unconsumed:
                raise RuntimeError(
                    "selector did not materialize all right-side ranked candidates: "
                    f"{unconsumed[:20]}"
                )
        right_artifact_sha = str(right_manifest.get("artifact_sha256") or "")
    else:
        right_scores = {}
        right_artifact_sha = ""

    left_members = {
        member
        for members in LEFT_GROUP_MEMBERS.values()
        for member in members
    }
    left_eligible_symbols = {
        str(row.get("symbol") or "")
        for row in rows
        if active_left is not None
        and active_left.enabled
        and str(row.get("symbol") or "") not in right_eligible_symbols
        and _row_uses_supported_strategy(row, left_members)
    }
    if left_eligible_symbols:
        if not signal_date:
            raise RuntimeError("left-side unified selector ranking requires signal_date")
        left_scores, left_manifest = load_left_side_ranking_scores(
            signal_date, config=active_left
        )
        missing = sorted(left_eligible_symbols - set(left_scores))
        if missing:
            raise RuntimeError(
                "left-side unified ranking coverage is incomplete for selector rows: "
                f"{missing[:20]}"
            )
        if require_all_ranked_candidates:
            unconsumed = sorted(set(left_scores) - left_eligible_symbols)
            if unconsumed:
                raise RuntimeError(
                    "selector did not materialize all left-side ranked candidates: "
                    f"{unconsumed[:20]}"
                )
        left_artifact_sha = str(left_manifest.get("artifact_sha256") or "")
    else:
        left_scores = {}
        left_artifact_sha = ""

    for row in rows:
        symbol = str(row.get("symbol") or "")
        if symbol in right_eligible_symbols:
            raw_score, normalized_score = right_scores[symbol]
            ranking_source = SelectorRankingSource.RIGHT_SIDE_UNIFIED.value
            artifact_sha = right_artifact_sha
            normalization = active.score_normalization
            threshold_mode = active.production_threshold_mode
        elif symbol in left_eligible_symbols:
            raw_score, normalized_score = left_scores[symbol]
            ranking_source = "left_side_unified"
            artifact_sha = left_artifact_sha
            normalization = active_left.score_normalization
            threshold_mode = "none_rank_only"
        else:
            row["ranking_source"] = "unified_ranker_not_applicable"
            continue
        row[RANKING_SCORE_FIELD] = float(raw_score)
        normalized_score = float(normalized_score)
        row[NORMALIZED_RANKING_SCORE_FIELD] = normalized_score
        row["ranking_score_percent"] = round(normalized_score, 6)
        row["selector_score"] = row["ranking_score_percent"]
        row["ranking_source"] = ranking_source
        row["ranking_score_date"] = pd.Timestamp(signal_date).date().isoformat()
        row["ranking_model_artifact_sha256"] = artifact_sha
        row["ranking_score_target"] = "cross_candidate_ordering_only"
        row["ranking_normalization"] = normalization
        row["ranking_threshold_mode"] = threshold_mode
    return rows


__all__ = [
    "DEFAULT_SELECTOR_RANKING_CONFIG_PATH",
    "DEFAULT_SELECTOR_RANKING_CONFIG",
    "FORBIDDEN_RANKING_ALIASES",
    "NORMALIZED_RANKING_SCORE_FIELD",
    "RANKING_SCORE_FIELD",
    "RANKING_NORMALIZATION_SCHEMA_VERSION",
    "RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION",
    "RIGHT_SIDE_PRODUCTION_APPROVAL_SCHEMA_VERSION",
    "RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION",
    "SelectorRankingConfig",
    "SelectorRankingPaths",
    "SelectorRankingSource",
    "apply_selector_ranking_source",
    "load_right_side_ranking_scores",
    "load_selector_ranking_config",
    "validate_selector_promotion_approval",
]
