"""Executable dependency contracts for the production daily refresh.

The registry is deliberately free of filesystem and model-loading code.  It
describes *what* must be current and how changes propagate; runtime adapters in
``quant.routine.daily_dependency_runtime`` provide the evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from quant.application.selector_ranking import (
    DEFAULT_SELECTOR_RANKING_CONFIG,
    RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
    RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
    SelectorRankingSource,
)
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.features.left_side_factor_contract import (
    LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
    LEFT_SIDE_FACTOR_COLUMNS,
    LEFT_SIDE_FACTOR_CONTRACT_SHA256,
    LEFT_SIDE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_SCORE_SCHEMA_VERSION,
)
from quant.features.factor_registry import (
    LONG_PRODUCTION_FACTOR_COLUMNS,
    LONG_PRODUCTION_FACTOR_SCHEMA_VERSION,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION,
    RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_unified_features import (
    RULE_FEATURE_COLUMNS,
    RULE_FEATURE_SCHEMA_VERSION,
)


REGISTRY_SCHEMA_VERSION = (
    "daily_dependency_registry_v5_selector_buy_hold_materialized"
)
PRODUCTION_PROJECT_FACTOR_SCHEMA = PROJECT_FACTOR_SCHEMA_VERSION
RIGHT_SIDE_SHADOW_PROJECT_FACTOR_SCHEMA = PROJECT_FACTOR_SCHEMA_VERSION
RIGHT_SIDE_SHADOW_RULE_FACTOR_SCHEMA = RULE_FEATURE_SCHEMA_VERSION
RIGHT_SIDE_SHADOW_FEATURE_SCHEMA = RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION
RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA = RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA_VERSION
RIGHT_SIDE_SHADOW_FACTOR_COUNT = len(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS)
RIGHT_SIDE_SHADOW_RULE_FACTOR_COUNT = len(RULE_FEATURE_COLUMNS)
LEGACY_Z_SIGNAL_KEYS: tuple[str, ...] = (
    "B2",
    "BREATHING",
    "DUICHEN_VA",
    "GOLDEN_BOWL",
    "KEY_K",
    "NANA",
    "VIOLENCE_K",
    "YIDONG_DILIAN",
    "YUEYUE",
    "ZAIHOU",
)


class Layer(str, Enum):
    DATA_SOURCE = "data_source"
    FEATURE = "feature"
    MODEL_SCORE = "model_score"
    PRODUCT = "product"


class Lifecycle(str, Enum):
    PRODUCTION = "production"
    RESEARCH_ONLY = "research_only"
    RETIRED = "retired"


class Cadence(str, Enum):
    TRADE_DAILY = "trade_daily"
    EVENT_POLL_DAILY = "event_poll_daily"
    WEEKLY = "weekly"
    STATIC = "static"
    ON_DEMAND = "on_demand"


class FreshnessMode(str, Enum):
    EXACT_TRADE_DATE = "exact_trade_date"
    POLLED_THROUGH = "polled_through"
    AS_OF = "as_of"
    TTL = "ttl"
    IMMUTABLE = "immutable"


class ColumnMode(str, Enum):
    ALL = "all"
    EXPLICIT = "explicit"
    MODEL_ARTIFACT = "model_artifact"


@dataclass(frozen=True)
class DependencyEdge:
    upstream: str
    column_mode: ColumnMode = ColumnMode.ALL
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceSpec:
    """One runtime freshness probe, tried in declaration order."""

    adapter: str
    locator: str
    date_field: str | None = None
    required: bool = True
    predicate_field: str | None = None
    expected_value: Any = None


@dataclass(frozen=True)
class FreshnessPolicy:
    mode: FreshnessMode
    evidence: tuple[EvidenceSpec, ...] = ()
    max_lag_sessions: int = 0
    max_age_days: int | None = None
    allow_empty: bool = True


@dataclass(frozen=True)
class IncrementalPolicy:
    partition_key: str | None = None
    primary_keys: tuple[str, ...] = ()
    write_mode: str = "upsert_target_partition"
    context_lookback_sessions: int = 0
    context_lookback_calendar_days: int = 0
    context_lookback_years: int = 0
    poll_overlap_calendar_days: int = 0
    forward_impact_sessions: int = 0
    dirty_mapper: str = "target_partition"
    supports_column_projection: bool = False


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_paths: tuple[str, ...]
    extractor: str
    feature_node_id: str
    manifest_path: str | None = None
    expected_schema: str | None = None
    approved_research_path: bool = False


@dataclass(frozen=True)
class DependencyNode:
    node_id: str
    layer: Layer
    owner: str
    lifecycle: Lifecycle
    cadence: Cadence
    inputs: tuple[DependencyEdge, ...]
    freshness: FreshnessPolicy
    incremental: IncrementalPolicy
    operation_id: str
    contract_version: str = "1"
    contract_sources: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    result_aliases: tuple[str, ...] = ()
    ui_step: str | None = None
    ui_order: int = 0
    feature_catalog_provider: str | None = None
    artifact: ArtifactSpec | None = None
    final_gate: bool = False
    notes: str = ""


@dataclass(frozen=True)
class NodeState:
    node_id: str
    watermark: date | None = None
    polled_through: date | None = None
    checked_at: datetime | None = None
    output_fingerprint: str | None = None
    contract_version: str | None = None


@dataclass(frozen=True)
class ChangeSet:
    partitions: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    full_rebuild: bool = False
    changed: bool = False


@dataclass(frozen=True)
class PlanEntry:
    node_id: str
    layer: str
    action: str
    reason: str
    active: bool
    dirty: ChangeSet = field(default_factory=ChangeSet)


@dataclass(frozen=True)
class ModelContract:
    node_id: str
    artifact_hashes: tuple[tuple[str, str], ...]
    features_by_artifact: Mapping[str, tuple[str, ...]]
    effective_features_by_artifact: Mapping[str, tuple[str, ...]]
    required_feature_union: tuple[str, ...]
    effective_feature_union: tuple[str, ...]
    combined_hash: str


@dataclass(frozen=True)
class FeatureUsage:
    feature_node_id: str
    produced: tuple[str, ...]
    required: tuple[str, ...]
    effective: tuple[str, ...]
    skippable: tuple[str, ...]
    contract_only_zero_importance: tuple[str, ...]
    unknown: tuple[str, ...]
    consumers: Mapping[str, tuple[str, ...]]
    projection_safe: bool


class DependencyRegistry:
    def __init__(
        self,
        nodes: Iterable[DependencyNode],
        scope_roots: Mapping[str, Sequence[str]],
        *,
        schema_version: str = REGISTRY_SCHEMA_VERSION,
    ) -> None:
        materialized = tuple(nodes)
        self.nodes = {node.node_id: node for node in materialized}
        if len(self.nodes) != len(materialized):
            raise ValueError("daily dependency registry contains duplicate node ids")
        self.scope_roots = {
            str(scope): tuple(str(node_id) for node_id in roots)
            for scope, roots in scope_roots.items()
        }
        self.schema_version = schema_version
        self.validate()

    def validate(self) -> None:
        layer_rank = {
            Layer.DATA_SOURCE: 0,
            Layer.FEATURE: 1,
            Layer.MODEL_SCORE: 2,
            Layer.PRODUCT: 3,
        }
        for node in self.nodes.values():
            for edge in node.inputs:
                if edge.upstream not in self.nodes:
                    raise ValueError(
                        f"daily dependency {node.node_id} references missing {edge.upstream}"
                    )
                upstream = self.nodes[edge.upstream]
                if layer_rank[upstream.layer] > layer_rank[node.layer]:
                    raise ValueError(
                        f"daily dependency layer inversion: {edge.upstream} -> {node.node_id}"
                    )
                if (
                    node.lifecycle == Lifecycle.PRODUCTION
                    and upstream.lifecycle != Lifecycle.PRODUCTION
                ):
                    raise ValueError(
                        "production daily dependency cannot consume non-production node: "
                        f"{edge.upstream} -> {node.node_id}"
                    )
            if node.artifact is not None:
                if node.layer != Layer.MODEL_SCORE:
                    raise ValueError(f"artifact contract belongs on model-score node: {node.node_id}")
                if node.artifact.feature_node_id not in self.nodes:
                    raise ValueError(
                        f"model {node.node_id} references missing feature node "
                        f"{node.artifact.feature_node_id}"
                    )
        for scope, roots in self.scope_roots.items():
            if not roots:
                raise ValueError(f"daily dependency scope has no roots: {scope}")
            for root in roots:
                if root not in self.nodes:
                    raise ValueError(f"daily dependency scope {scope} references missing root {root}")
                if self.nodes[root].layer != Layer.PRODUCT:
                    raise ValueError(f"daily dependency scope root must be a product: {root}")
        self.topological_order()

    def topological_order(self, node_ids: Iterable[str] | None = None) -> tuple[str, ...]:
        selected = set(node_ids or self.nodes)
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise ValueError(f"daily dependency graph contains a cycle at {node_id}")
            visiting.add(node_id)
            for edge in self.nodes[node_id].inputs:
                if edge.upstream in selected:
                    visit(edge.upstream)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(node_id)

        for node_id in sorted(selected):
            visit(node_id)
        return tuple(ordered)

    def required_node_ids(
        self,
        scope: str,
        *,
        effective_feature_requirements: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[str, ...]:
        if scope not in self.scope_roots:
            raise ValueError(f"unknown daily dependency scope: {scope}")
        required: set[str] = set()
        feature_requirements = {
            node_id: set(str(column) for column in columns)
            for node_id, columns in (effective_feature_requirements or {}).items()
        }

        def include(node_id: str) -> None:
            if node_id in required:
                return
            required.add(node_id)
            node = self.nodes[node_id]
            for edge in node.inputs:
                # A source edge can declare exactly which feature columns it
                # supplies.  Once model contracts are compiled, zero-use
                # sources can be pruned without weakening the artifact's full
                # input-shape contract (the feature builder may still emit a
                # harmless constant column).
                if (
                    node.layer == Layer.FEATURE
                    and edge.columns
                    and node_id in feature_requirements
                    and not (set(edge.columns) & feature_requirements[node_id])
                ):
                    continue
                include(edge.upstream)

        for root in self.scope_roots[scope]:
            include(root)
        return self.topological_order(required)

    def progress_steps(self, scope: str) -> tuple[str, ...]:
        active = set(self.required_node_ids(scope))
        by_step: dict[str, int] = {}
        for node_id in active:
            node = self.nodes[node_id]
            if node.ui_step:
                by_step[node.ui_step] = min(
                    by_step.get(node.ui_step, node.ui_order),
                    node.ui_order,
                )
        return tuple(sorted(by_step, key=lambda key: (by_step[key], key)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_roots": self.scope_roots,
            "nodes": [
                _enum_json(asdict(self.nodes[node_id]))
                for node_id in self.topological_order()
            ],
        }


def _enum_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_enum_json(item) for item in value]
    return value


def state_is_current(
    node: DependencyNode,
    state: NodeState | None,
    target_date: date,
    now: datetime,
) -> tuple[bool, str]:
    if state is None:
        return False, "no validated checkpoint"
    if state.contract_version not in {None, node.contract_version}:
        return False, "contract version changed"
    mode = node.freshness.mode
    if mode == FreshnessMode.EXACT_TRADE_DATE:
        return (
            (state.watermark == target_date, f"watermark={state.watermark}" )
            if state.watermark is not None
            else (False, "missing exact-date watermark")
        )
    if mode == FreshnessMode.POLLED_THROUGH:
        return (
            (state.polled_through is not None and state.polled_through >= target_date),
            f"polled_through={state.polled_through}",
        )
    if mode == FreshnessMode.AS_OF:
        return (
            (state.watermark is not None and state.watermark <= target_date),
            f"as_of={state.watermark}",
        )
    if mode == FreshnessMode.TTL:
        if state.checked_at is None:
            return False, "missing TTL checkpoint"
        max_age = node.freshness.max_age_days or 0
        age_days = max(0.0, (now - state.checked_at).total_seconds() / 86_400)
        return age_days <= max_age, f"age_days={age_days:.2f} max={max_age}"
    if mode == FreshnessMode.IMMUTABLE:
        return bool(state.output_fingerprint), "artifact fingerprint present"
    return False, f"unsupported freshness mode {mode.value}"


def build_dependency_plan(
    registry: DependencyRegistry,
    scope: str,
    target_date: date,
    states: Mapping[str, NodeState] | None = None,
    *,
    changed_nodes: Iterable[str] = (),
    effective_feature_requirements: Mapping[str, Sequence[str]] | None = None,
    now: datetime | None = None,
    include_unused: bool = True,
) -> tuple[PlanEntry, ...]:
    """Build a conservative two-phase incremental plan.

    Event sources use ``poll`` and mark downstream work as
    ``refresh_if_changed``.  Exact-date missing partitions are deterministic
    changes, so their downstream nodes are immediately marked ``refresh``.
    """

    state_map = dict(states or {})
    active = set(
        registry.required_node_ids(
            scope,
            effective_feature_requirements=effective_feature_requirements,
        )
    )
    explicit_changed = set(changed_nodes)
    definite_dirty: set[str] = set()
    possible_dirty: set[str] = set()
    entries: dict[str, PlanEntry] = {}
    current = now or datetime.now()

    for node_id in registry.topological_order():
        node = registry.nodes[node_id]
        if node_id not in active:
            if include_unused:
                entries[node_id] = PlanEntry(
                    node_id=node_id,
                    layer=node.layer.value,
                    action="skip_unused",
                    reason="outside active production product closure",
                    active=False,
                )
            continue
        upstream_ids = {edge.upstream for edge in node.inputs}
        inherited_definite = bool(upstream_ids & definite_dirty)
        inherited_possible = bool(upstream_ids & possible_dirty)
        is_current, current_reason = state_is_current(
            node,
            state_map.get(node_id),
            target_date,
            current,
        )
        target_partition = (target_date.isoformat(),) if node.incremental.partition_key else ()

        if node_id in explicit_changed:
            action, reason = "refresh", "input/config/artifact fingerprint changed"
            definite_dirty.add(node_id)
        elif node.freshness.mode == FreshnessMode.POLLED_THROUGH and not is_current:
            action, reason = "poll", current_reason
            possible_dirty.add(node_id)
        elif inherited_definite:
            action, reason = "refresh", "required upstream partition changed"
            definite_dirty.add(node_id)
        elif inherited_possible:
            action, reason = "refresh_if_changed", "event upstream may produce changes"
            possible_dirty.add(node_id)
        elif not is_current:
            action, reason = "refresh", current_reason
            definite_dirty.add(node_id)
        else:
            action, reason = "reuse", current_reason

        entries[node_id] = PlanEntry(
            node_id=node_id,
            layer=node.layer.value,
            action=action,
            reason=reason,
            active=True,
            dirty=ChangeSet(
                partitions=target_partition if action in {"refresh", "refresh_if_changed"} else (),
                changed=action == "refresh",
            ),
        )

    return tuple(entries[node_id] for node_id in registry.topological_order() if node_id in entries)


def classify_feature_usage(
    registry: DependencyRegistry,
    scope: str,
    model_contracts: Mapping[str, ModelContract],
    catalogs: Mapping[str, Sequence[str]],
) -> tuple[FeatureUsage, ...]:
    active = set(registry.required_node_ids(scope))
    consumers: dict[str, dict[str, set[str]]] = {}
    required: dict[str, set[str]] = {}
    effective: dict[str, set[str]] = {}
    all_required: set[str] = set()

    for node_id in active:
        node = registry.nodes[node_id]
        for edge in node.inputs:
            feature_node = registry.nodes[edge.upstream]
            if feature_node.layer != Layer.FEATURE:
                continue
            catalog = set(str(value) for value in catalogs.get(edge.upstream, ()))
            if edge.column_mode == ColumnMode.ALL:
                columns = catalog
                all_required.add(edge.upstream)
            elif edge.column_mode == ColumnMode.EXPLICIT:
                columns = set(edge.columns)
            else:
                contract = model_contracts.get(node_id)
                columns = set(contract.required_feature_union if contract else ())
                if contract:
                    effective.setdefault(edge.upstream, set()).update(
                        contract.effective_feature_union
                    )
            required.setdefault(edge.upstream, set()).update(columns)
            consumers.setdefault(edge.upstream, {}).setdefault(node_id, set()).update(columns)

    usages: list[FeatureUsage] = []
    feature_ids = sorted(
        node_id
        for node_id in active
        if registry.nodes[node_id].layer == Layer.FEATURE
    )
    for feature_id in feature_ids:
        node = registry.nodes[feature_id]
        produced = set(str(value) for value in catalogs.get(feature_id, ()))
        wanted = required.get(feature_id, set())
        effective_columns = effective.get(feature_id, set()) or set(wanted)
        unknown = wanted - produced
        safe_to_project = (
            node.incremental.supports_column_projection
            and feature_id not in all_required
            and not unknown
        )
        skippable = produced - wanted if safe_to_project else set()
        usages.append(
            FeatureUsage(
                feature_node_id=feature_id,
                produced=tuple(sorted(produced)),
                required=tuple(sorted(wanted)),
                effective=tuple(sorted(effective_columns)),
                skippable=tuple(sorted(skippable)),
                contract_only_zero_importance=tuple(sorted(wanted - effective_columns)),
                unknown=tuple(sorted(unknown)),
                consumers={
                    key: tuple(sorted(values))
                    for key, values in sorted(consumers.get(feature_id, {}).items())
                },
                projection_safe=safe_to_project,
            )
        )
    return tuple(usages)


def _edge(upstream: str, mode: ColumnMode = ColumnMode.ALL) -> DependencyEdge:
    return DependencyEdge(upstream=upstream, column_mode=mode)


def _result(
    locator: str,
    date_field: str | None = None,
    *,
    required: bool = True,
) -> EvidenceSpec:
    return EvidenceSpec("result", locator, date_field, required)


def _json(
    locator: str,
    date_field: str,
    *,
    predicate_field: str | None = None,
    expected_value: Any = None,
) -> EvidenceSpec:
    return EvidenceSpec(
        "json",
        locator,
        date_field,
        predicate_field=predicate_field,
        expected_value=expected_value,
    )


def _parquet(locator: str, date_field: str = "date") -> EvidenceSpec:
    return EvidenceSpec("parquet_max", locator, date_field)


def _file(locator: str) -> EvidenceSpec:
    return EvidenceSpec("file_fingerprint", locator)


def _exact(*evidence: EvidenceSpec, allow_empty: bool = True) -> FreshnessPolicy:
    return FreshnessPolicy(FreshnessMode.EXACT_TRADE_DATE, tuple(evidence), allow_empty=allow_empty)


def _polled(*evidence: EvidenceSpec) -> FreshnessPolicy:
    return FreshnessPolicy(FreshnessMode.POLLED_THROUGH, tuple(evidence))


def _static() -> FreshnessPolicy:
    return FreshnessPolicy(FreshnessMode.IMMUTABLE)


def _daily_incremental(
    *,
    lookback: int = 0,
    calendar_days: int = 0,
    years: int = 0,
    keys: tuple[str, ...] = ("date",),
    projection: bool = False,
) -> IncrementalPolicy:
    return IncrementalPolicy(
        partition_key="date",
        primary_keys=keys,
        context_lookback_sessions=lookback,
        context_lookback_calendar_days=calendar_days,
        context_lookback_years=years,
        supports_column_projection=projection,
    )


def _z_artifacts(signals: Sequence[str] | None = None) -> tuple[str, ...]:
    active_signals = (
        tuple(LEGACY_Z_SIGNAL_KEYS) if signals is None else tuple(signals)
    )
    return tuple(
        f"models/research/z_skill/{signal}_{label}.joblib"
        for signal in active_signals
        for label in ("down3", "up5", "up8")
    )


def build_default_daily_dependency_registry(
    selector_ranking_source: SelectorRankingSource | str | None = None,
) -> DependencyRegistry:
    """Return the current production dependency graph.

    Research paths used by Z/Chan are explicitly approved transitional inputs;
    the nodes themselves are production because the Web refresh consumes them.
    The runtime snapshot flags that location debt until artifacts are promoted.
    """

    configured_source = (
        DEFAULT_SELECTOR_RANKING_CONFIG.source
        if selector_ranking_source is None
        else selector_ranking_source
        if isinstance(selector_ranking_source, SelectorRankingSource)
        else SelectorRankingSource(str(selector_ranking_source))
    )
    left_side_enabled = DEFAULT_LEFT_SIDE_RANKING_CONFIG.enabled
    two_unified_rankers_active = (
        configured_source == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        and left_side_enabled
    )
    legacy_z_active_signals = (
        ()
        if two_unified_rankers_active
        else
        DEFAULT_SELECTOR_RANKING_CONFIG.preserved_legacy_signals
        if configured_source == SelectorRankingSource.RIGHT_SIDE_UNIFIED
        else LEGACY_Z_SIGNAL_KEYS
    )
    legacy_z_artifacts = _z_artifacts(legacy_z_active_signals)

    nodes = [
        DependencyNode(
            "data.market_daily", Layer.DATA_SOURCE, "routine.data_refresh",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (),
            _exact(_result("refresh_data", "dataset_trade_date")),
            _daily_incremental(keys=("trade_date", "ts_code")),
            "refresh_data",
            contract_sources=(
                "src/quant/routine/data_refresh.py",
                "src/quant/data/market_data_store.py",
                "src/quant/data/source_merge.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            outputs=("data/raw/daily_partitioned",),
            result_aliases=("refresh_data",), ui_step="refresh_data", ui_order=10,
            final_gate=True,
            notes="Append/repair only the target trade-date partition.",
        ),
        DependencyNode(
            "data.daily_basic", Layer.DATA_SOURCE, "routine.daily_basic_refresh",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("data.market_daily"),),
            _exact(_result("refresh_daily_basic", "latest_trade_date")),
            IncrementalPolicy("trade_date", ("trade_date", "ts_code"),
                              context_lookback_sessions=20,
                              poll_overlap_calendar_days=45),
            "refresh_daily_basic",
            contract_sources=(
                "src/quant/routine/daily_basic_refresh.py",
                "src/quant/routine/data_refresh.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            outputs=("data/raw/daily_basic/YYYYMMDD.parquet",),
            result_aliases=("refresh_daily_basic",), ui_step="refresh_data", ui_order=11,
            final_gate=True,
        ),
        DependencyNode(
            "data.csi300_daily", Layer.DATA_SOURCE, "routine.reference_data_refresh",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (),
            _exact(_result("refresh_reference_inputs.steps.index_000300", "latest_trade_date")),
            IncrementalPolicy("trade_date", ("ts_code", "trade_date"),
                              poll_overlap_calendar_days=10),
            "refresh_index_000300",
            contract_sources=(
                "src/quant/routine/reference_data_refresh.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            outputs=("data/raw/index_000300.SH.parquet",),
            result_aliases=("refresh_reference_inputs.steps.index_000300",),
            ui_step="refresh_data", ui_order=12, final_gate=True,
        ),
        DependencyNode(
            "data.stock_basic", Layer.DATA_SOURCE, "routine.reference_data_refresh",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY, (),
            _polled(
                _result(
                    "refresh_reference_inputs.steps.stock_basic",
                    "polled_through",
                )
            ),
            IncrementalPolicy(primary_keys=("ts_code",), write_mode="replace_on_change",
                              poll_overlap_calendar_days=1, dirty_mapper="changed_symbols"),
            "refresh_stock_basic",
            contract_sources=(
                "src/quant/routine/reference_data_refresh.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            outputs=("data/raw/stock_basic.parquet",),
            result_aliases=("refresh_reference_inputs.steps.stock_basic",),
            ui_step="refresh_data", ui_order=13, final_gate=True,
        ),
        DependencyNode(
            "data.financial_pit", Layer.DATA_SOURCE, "routine.reference_data_refresh",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY, (),
            _polled(
                _result(
                    "refresh_reference_inputs.steps.financials",
                    "polled_through",
                )
            ),
            IncrementalPolicy(primary_keys=("ts_code", "ann_date", "end_date"),
                              write_mode="pit_upsert", poll_overlap_calendar_days=120,
                              dirty_mapper="announcement_symbols"),
            "refresh_financial_pit",
            contract_sources=(
                "src/quant/routine/reference_data_refresh.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            outputs=("data/raw/fina_indicator.parquet", "data/raw/income.parquet",
                     "data/raw/cashflow.parquet"),
            result_aliases=("refresh_reference_inputs.steps.financials",),
            ui_step="refresh_data", ui_order=14, final_gate=True,
        ),
        DependencyNode(
            "data.analyst_pit", Layer.DATA_SOURCE, "routine.pipeline",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY, (),
            _polled(
                _result(
                    "refresh_reference_inputs.steps.analyst_forecast_snapshot",
                    "polled_through",
                )
            ),
            IncrementalPolicy(primary_keys=("ts_code", "report_date", "source"),
                              write_mode="pit_upsert", poll_overlap_calendar_days=30,
                              dirty_mapper="report_symbols"),
            "refresh_analyst_forecasts",
            contract_sources=(
                "src/quant/routine/pipeline.py",
                "scripts/research/refresh_analyst_forecasts.py",
                "src/quant/data/source_merge.py",
            ),
            outputs=("data/raw/analyst_forecasts.parquet",),
            result_aliases=("refresh_reference_inputs.steps.analyst_forecast_snapshot",),
            ui_step="refresh_data", ui_order=15, final_gate=True,
        ),
        DependencyNode(
            "data.top_list", Layer.DATA_SOURCE, "routine.reference_data_refresh",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY, (),
            _polled(
                _result(
                    "refresh_reference_inputs.steps.long_factor_sources.datasets.top_list",
                    "polled_through",
                )
            ),
            IncrementalPolicy("trade_date", ("trade_date", "ts_code"),
                              write_mode="upsert_target_partition", poll_overlap_calendar_days=3),
            "refresh_top_list",
            contract_sources=(
                "src/quant/routine/reference_data_refresh.py",
                "src/quant/data/long_factor_backfill.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            outputs=("data/raw/top_list/tushare_top_list_YYYYMMDD.parquet",),
            result_aliases=("refresh_reference_inputs.steps.long_factor_sources.datasets.top_list",),
            ui_step="refresh_data", ui_order=16, final_gate=True,
        ),
        DependencyNode(
            "data.cb_daily", Layer.DATA_SOURCE, "application.workspaces.convertible_bonds",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (),
            _exact(_result("convertible_bond_plan", "trade_date")),
            _daily_incremental(keys=("trade_date", "ts_code")),
            "refresh_convertible_bond_daily",
            contract_sources=(
                "src/quant/routine/convertible_bond_grid_plan.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            ui_step="refresh_data", ui_order=17,
            final_gate=True,
        ),
        DependencyNode(
            "data.cb_reference", Layer.DATA_SOURCE, "application.workspaces.convertible_bonds",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY, (),
            _polled(
                _result(
                    "convertible_bond_plan.data_refresh.reference_poll",
                    "polled_through",
                )
            ),
            IncrementalPolicy(primary_keys=("ts_code",), write_mode="upsert_on_change",
                              poll_overlap_calendar_days=7),
            "refresh_convertible_bond_reference",
            contract_sources=(
                "src/quant/routine/convertible_bond_grid_plan.py",
                "src/quant/data/tushare_fetcher.py",
            ),
            ui_step="refresh_data", ui_order=18,
            final_gate=True,
        ),
        DependencyNode(
            "data.cb_allotment_events", Layer.DATA_SOURCE, "application.workspaces.convertible_bonds",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY, (),
            _polled(_result("convertible_bond_allotment", "event_polled_through")),
            IncrementalPolicy(primary_keys=("ts_code", "event_id"), write_mode="upsert_on_change",
                              poll_overlap_calendar_days=30),
            "refresh_convertible_bond_allotment_events",
            contract_sources=(
                "src/quant/routine/convertible_bond_allotment.py",
                "src/quant/data/tushare_fetcher.py",
                "configs/convertible_bond_allotment_watchlist.csv",
            ),
            ui_step="refresh_data", ui_order=19,
            final_gate=True,
        ),
        DependencyNode(
            "data.byd_intraday_training", Layer.DATA_SOURCE, "application.workspaces.byd",
            Lifecycle.PRODUCTION, Cadence.STATIC, (),
            FreshnessPolicy(
                FreshnessMode.TTL,
                (EvidenceSpec("glob_parquet_max", "data/cache/baostock_002594_5min_*_qfq.parquet", "datetime"),),
                max_age_days=60,
            ),
            IncrementalPolicy(primary_keys=("datetime",), write_mode="append_new_bars"),
            "refresh_byd_intraday_when_due",
            contract_sources=(
                "src/quant/application/workspaces/byd.py",
                "scripts/research/fetch_byd_intraday_baostock.py",
            ),
            final_gate=True,
            notes="Historical training input; not a same-day feature.",
        ),
        DependencyNode(
            "data.similar_watchlist", Layer.DATA_SOURCE, "webapp.services",
            Lifecycle.PRODUCTION, Cadence.ON_DEMAND, (),
            FreshnessPolicy(
                FreshnessMode.IMMUTABLE,
                (EvidenceSpec("file_fingerprint", "data/research/similar_patterns/watchlist.json"),),
            ),
            IncrementalPolicy(),
            "read_similar_watchlist",
            contract_sources=("src/quant/webapp/services.py",),
            final_gate=True,
        ),
        DependencyNode(
            "data.tradability", Layer.DATA_SOURCE, "routine.reference_data_refresh",
            Lifecycle.RESEARCH_ONLY, Cadence.ON_DEMAND, (_edge("data.stock_basic"),),
            _exact(), _daily_incremental(keys=("trade_date", "ts_code")),
            "refresh_tradability_for_research", outputs=("data/raw/tradability/YYYYMMDD.parquet",),
            notes="No active production consumer; refreshed only by research/backtest jobs.",
        ),
        DependencyNode(
            "data.long_research_external", Layer.DATA_SOURCE, "data.long_factor_backfill",
            Lifecycle.RESEARCH_ONLY, Cadence.ON_DEMAND, (), _polled(),
            IncrementalPolicy(write_mode="pit_upsert", poll_overlap_calendar_days=45),
            "refresh_long_research_external",
            outputs=("data/raw/margin_detail", "data/raw/moneyflow", "data/raw/holder_trade.parquet"),
            notes="Margin/moneyflow/holder-trade inputs are not consumed by current Tea/v44 pages.",
        ),
        DependencyNode(
            "data.selector_training_history", Layer.DATA_SOURCE, "research.selector",
            Lifecycle.RESEARCH_ONLY, Cadence.ON_DEMAND, (), _static(), IncrementalPolicy(),
            "build_selector_training_history",
            outputs=("data/research/selector_model_history_2020.parquet",),
            notes="Frozen training/reference history; live scoring must never fall back to its date.",
        ),
        DependencyNode(
            "feature.strategy_signals", Layer.FEATURE, "research.rebuild_strategy_signal_cache",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("data.market_daily"),),
            _exact(
                _file("data/features/b1/b1_gate_candidates.parquet"),
                _file("data/features/b1/b1_family_rule_candidates.parquet"),
                _file("data/features/z_skill_daily_candidates.parquet"),
                _json(
                    "data/features/b1/b1_gate_manifest.json",
                    "processed_through_date",
                ),
            ),
            _daily_incremental(calendar_days=600, keys=("date", "symbol")),
            "refresh_strategy_signal_cache",
            contract_sources=(
                "scripts/research/rebuild_strategy_signal_cache.py",
                "scripts/research/analyze_b1_family_rule_backtest.py",
                "scripts/research/analyze_z_skill_entry_exit_backtest.py",
                "src/quant/data/market_data_store.py",
                "src/quant/data/source_merge.py",
                "src/quant/data/factors/__init__.py",
                "src/quant/data/factors/alpha101.py",
                "src/quant/data/factors/alpha191.py",
                "src/quant/data/factors/base.py",
                "src/quant/data/factors/data_adapter.py",
                "src/quant/data/factors/momentum.py",
                "src/quant/data/factors/technical.py",
                "src/quant/features/b1_gate.py",
                "src/quant/features/daily_factor_layer.py",
                "src/quant/features/project_factor_layer.py",
                "src/quant/features/variable_library.py",
                "src/quant/strategies/custom/triple_volume_breakout.py",
                "src/quant/strategies/custom/vegas_tunnel.py",
                "configs/strategies/triple_volume_breakout.yaml",
            ),
            outputs=("data/features/b1/b1_gate_candidates.parquet",
                     "data/features/b1/b1_family_rule_candidates.parquet",
                     "data/features/z_skill_daily_candidates.parquet",
                     "data/features/b1/b1_gate_manifest.json"),
            result_aliases=("signal_cache", "refresh_strategy_signal_cache"),
            ui_step="signal_cache", ui_order=30, final_gate=True,
        ),
        DependencyNode(
            "feature.project_daily", Layer.FEATURE, "features.project_factor_layer",
            Lifecycle.RETIRED if two_unified_rankers_active else Lifecycle.PRODUCTION,
            Cadence.ON_DEMAND if two_unified_rankers_active else Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("data.daily_basic"),
             _edge("feature.strategy_signals")),
            _exact(
                _file("data/features/b1/active_candidate_project_features.parquet"),
                _json(
                    "data/features/b1/active_candidate_project_features_manifest.json",
                    "target_date",
                    predicate_field="candidate_coverage_status",
                    expected_value="complete",
                ),
                _json(
                    "data/features/b1/active_candidate_project_features_manifest.json",
                    "target_date",
                    predicate_field="factor_count",
                    expected_value=len(PROJECT_FACTOR_COLUMNS),
                ),
                _json(
                    "data/features/b1/active_candidate_project_features_manifest.json",
                    "target_date",
                    predicate_field="factor_schema_version",
                    expected_value=PRODUCTION_PROJECT_FACTOR_SCHEMA,
                ),
            ),
            _daily_incremental(years=6, keys=("date", "symbol"), projection=True),
            "refresh_active_project_features",
            contract_sources=(
                "src/quant/data/source_merge.py",
                "src/quant/data/factors/__init__.py",
                "src/quant/data/factors/alpha101.py",
                "src/quant/data/factors/alpha191.py",
                "src/quant/data/factors/base.py",
                "src/quant/data/factors/data_adapter.py",
                "src/quant/data/factors/momentum.py",
                "src/quant/data/factors/technical.py",
                "src/quant/features/daily_factor_layer.py",
                "src/quant/features/project_factor_layer.py",
                "src/quant/features/variable_library.py",
                "src/quant/features/factor_registry.py",
                "src/quant/features/factor_execution.py",
                "src/quant/features/factor_governance.py",
                "configs/factors/governance.json",
                "scripts/research/refresh_b1_feature_cache.py",
                "scripts/research/train_b1_tushare_models.py",
            ),
            outputs=("data/features/b1/active_candidate_project_features.parquet",
                     "data/features/b1/active_candidate_project_features_manifest.json"),
            result_aliases=("feature_cache", "build_features"),
            ui_step="feature_cache", ui_order=40,
            feature_catalog_provider="project_factor_columns", final_gate=True,
            notes=(
                "Compute once for the B1/Z active-candidate union; preserve the B1 gate "
                "cache semantics. The historical training table is not a daily output."
            ),
        ),
        DependencyNode(
            "feature.market_regime", Layer.FEATURE, "features.market_regime",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("data.csi300_daily")),
            _exact(_json("data/features/market_regime/latest.json", "as_of")),
            _daily_incremental(calendar_days=252), "refresh_market_regime_snapshot",
            contract_sources=("src/quant/features/market_regime.py",),
            outputs=("data/features/market_regime/latest.json",),
            result_aliases=("refresh_reference_inputs.steps.market_regime",),
            ui_step="refresh_data", ui_order=20, final_gate=True,
        ),
        DependencyNode(
            "feature.selector_live", Layer.FEATURE, "webapp.services.selector",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("data.daily_basic"),
             _edge("feature.strategy_signals"),
             _edge("feature.long_snapshot"),
             *((_edge("feature.right_side_unified"),) if configured_source == SelectorRankingSource.RIGHT_SIDE_UNIFIED else ()),
             *((_edge("feature.left_side_unified"),) if left_side_enabled else ())),
            _exact(_result("selector_extended", "signal_date")),
            _daily_incremental(calendar_days=130, keys=("date", "symbol"), projection=True),
            "build_selector_live_features", result_aliases=("selector_core", "selector_extended"),
            contract_sources=("src/quant/webapp/services.py",),
            feature_catalog_provider="selector_artifact_features",
        ),
        DependencyNode(
            "feature.chan_live", Layer.FEATURE, "research.chan_daily",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (
                _edge("data.market_daily"),
                _edge("data.daily_basic"),
                DependencyEdge(
                    "data.top_list",
                    ColumnMode.EXPLICIT,
                    ("top_list_count", "top_net_amount_ratio", "top_net_rate"),
                ),
            ),
            _exact(
                _file("reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"),
                _json(
                    "reports/chan_daily/model_filter/live_refresh_manifest.json",
                    "processed_through",
                    predicate_field="feature_coverage.status",
                    expected_value="valid",
                ),
            ),
            _daily_incremental(calendar_days=120, keys=("date", "symbol"), projection=True),
            "build_chan_live_features",
            contract_sources=(
                "scripts/research/refresh_chan_model_live_scores.py",
                "src/quant/strategies/custom/chan_daily.py",
                "src/quant/strategies/custom/chan_model.py",
            ),
            outputs=("reports/chan_daily/model_filter/chan_model_scored_candidates.parquet",),
            feature_catalog_provider="chan_artifact_features",
        ),
        DependencyNode(
            "feature.long_snapshot", Layer.FEATURE, "webapp.services.long",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("data.daily_basic"), _edge("data.csi300_daily"),
             _edge("data.stock_basic"), _edge("data.financial_pit"), _edge("data.analyst_pit"),
             _edge("feature.market_regime")),
            _exact(
                EvidenceSpec("long_factor_snapshot", "data/features/long", "signal_date"),
                _json(
                    "data/features/long/latest.json",
                    "signal_date",
                    predicate_field="factor_count",
                    expected_value=len(LONG_PRODUCTION_FACTOR_COLUMNS),
                ),
                _json(
                    "data/features/long/latest.json",
                    "signal_date",
                    predicate_field="factor_schema_version",
                    expected_value=LONG_PRODUCTION_FACTOR_SCHEMA_VERSION,
                ),
                _json(
                    "data/features/long/latest.json",
                    "signal_date",
                    predicate_field="coverage_status",
                    expected_value="complete",
                ),
                _result("long_stock_pool.variants.0", "signal_date", required=False),
            ),
            _daily_incremental(
                calendar_days=450,
                years=8,
                keys=("date", "ts_code"),
            ),
            "refresh_long_factor_snapshot",
            contract_sources=(
                "src/quant/webapp/services.py",
                "src/quant/features/long_factor_snapshot.py",
                "src/quant/features/factor_registry.py",
                "configs/strategies/tea_master_long.yaml",
                "configs/strategies/long_dividend_quality.yaml",
            ),
            outputs=("data/features/long/latest.parquet", "data/features/long/latest.json"),
            result_aliases=("long_stock_pool",), ui_step="long_stock_pool", ui_order=80,
            feature_catalog_provider="long_factor_columns", final_gate=True,
        ),
        DependencyNode(
            "feature.cb_grid", Layer.FEATURE, "routine.convertible_bond_grid_plan",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("data.cb_daily"), _edge("data.cb_reference"), _edge("data.market_daily")),
            _exact(_result("convertible_bond_plan", "trade_date")),
            _daily_incremental(lookback=252, keys=("date", "ts_code")),
            "build_convertible_bond_grid_features",
            contract_sources=(
                "src/quant/routine/convertible_bond_grid_plan.py",
                "configs/strategies/convertible_bond_rotation.yaml",
            ),
            ui_step="convertible_bond_plan", ui_order=90,
        ),
        DependencyNode(
            "feature.byd_daily", Layer.FEATURE, "application.workspaces.byd",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("data.market_daily"),),
            _exact(_result("byd_daily_plan", "signal_date")),
            _daily_incremental(lookback=65, keys=("date", "symbol")),
            "build_byd_daily_features",
            contract_sources=(
                "src/quant/application/workspaces/byd.py",
                "src/quant/research/byd_daily_t_plan.py",
            ),
            ui_step="byd_daily_plan", ui_order=94,
        ),
        DependencyNode(
            "feature.similar_reference", Layer.FEATURE, "research.similar_patterns",
            Lifecycle.PRODUCTION, Cadence.WEEKLY, (_edge("data.market_daily"),),
            FreshnessPolicy(
                FreshnessMode.TTL,
                (
                    EvidenceSpec(
                        "result",
                        "similar_patterns",
                        "reference_library_refreshed_at",
                    ),
                ),
                max_age_days=7,
            ),
            IncrementalPolicy(primary_keys=("symbol", "reference_date"),
                              write_mode="append_or_reuse_weekly", context_lookback_sessions=260),
            "refresh_similar_reference_vectors_when_due",
            contract_sources=(
                "src/quant/research/similar_patterns.py",
                "src/quant/webapp/services.py",
            ),
            outputs=("data/research/similar_patterns/vector_cache",),
            ui_step="similar_patterns", ui_order=96,
            final_gate=True,
            notes="Historical reference library; intentionally not an exact-date feature.",
        ),
        DependencyNode(
            "feature.similar_target", Layer.FEATURE, "research.similar_patterns",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("data.csi300_daily"),
             _edge("data.stock_basic"), _edge("data.similar_watchlist")),
            _exact(_result("similar_patterns", "target_date")),
            _daily_incremental(lookback=260, keys=("date", "symbol")),
            "build_similar_target_context",
            contract_sources=(
                "src/quant/research/similar_patterns.py",
                "src/quant/webapp/services.py",
            ),
            ui_step="similar_patterns", ui_order=97,
        ),
        DependencyNode(
            "score.b1", Layer.MODEL_SCORE, "routine.b1_daily_plan",
            Lifecycle.RETIRED if left_side_enabled else Lifecycle.PRODUCTION,
            Cadence.ON_DEMAND if left_side_enabled else Cadence.TRADE_DAILY,
            (_edge("feature.project_daily", ColumnMode.MODEL_ARTIFACT),),
            _exact(_json("web/data/b1_daily_plan.json", "signal_date")),
            _daily_incremental(keys=("date", "symbol")), "score_b1_release",
            contract_sources=("configs/strategies/b1_selected.yaml",),
            result_aliases=("generate_daily_plan",), ui_step="daily_plan", ui_order=50,
            artifact=ArtifactSpec(
                tuple(f"models/production/b1/{name}.joblib" for name in
                      ("up5_es", "up8_es", "up10_es", "down2_es", "down3_es")),
                "sklearn_feature_names", "feature.project_daily",
                manifest_path="models/production/b1/manifest.json",
            ), final_gate=True,
        ),
        DependencyNode(
            "score.z_skill", Layer.MODEL_SCORE, "research.z_skill",
            Lifecycle.RETIRED if two_unified_rankers_active else Lifecycle.PRODUCTION,
            Cadence.ON_DEMAND if two_unified_rankers_active else Cadence.TRADE_DAILY,
            (_edge("feature.project_daily", ColumnMode.MODEL_ARTIFACT),
             _edge("feature.strategy_signals")),
            _exact(
                _file("reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_scored_candidates.parquet"),
                _json(
                    "reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_scored_candidates_manifest.json",
                    "target_date",
                    predicate_field="feature_coverage.status",
                    expected_value="valid",
                ),
                _json(
                    "reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_scored_candidates_manifest.json",
                    "target_date",
                    predicate_field="factor_schema_version",
                    expected_value=PRODUCTION_PROJECT_FACTOR_SCHEMA,
                ),
                _json(
                    "reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_scored_candidates_manifest.json",
                    "target_date",
                    predicate_field="scored_signals",
                    expected_value=sorted(legacy_z_active_signals),
                ),
            ),
            _daily_incremental(keys=("date", "symbol")), "score_z_skill_release",
            contract_sources=(
                "scripts/research/score_latest_strategy_models.py",
                "configs/strategies/right_side_ranking_selector.yaml",
            ),
            result_aliases=("model_score", "score_latest_models"), ui_step="model_score", ui_order=60,
            artifact=ArtifactSpec(legacy_z_artifacts, "sklearn_feature_names", "feature.project_daily",
                                  approved_research_path=True), final_gate=True,
            notes=(
                "Production-consumed transitional artifacts. After unified ranking "
                "promotion only DUICHEN_VA/NANA/YIDONG_DILIAN remain active; the "
                "seven replaced right-side members stay on disk for rollback only."
            ),
        ),
        DependencyNode(
            "feature.right_side_unified",
            Layer.FEATURE,
            "routine.right_side_unified_production",
            Lifecycle.PRODUCTION,
            Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("feature.strategy_signals")),
            _exact(
                _file("data/features/right_side_unified/latest_features.parquet"),
                _json(
                    "data/features/right_side_unified/feature_manifest.json",
                    "target_date",
                    predicate_field="candidate_coverage_status",
                    expected_value="complete",
                ),
                _json(
                    "data/features/right_side_unified/feature_manifest.json",
                    "target_date",
                    predicate_field="feature_schema_version",
                    expected_value=RIGHT_SIDE_SHADOW_FEATURE_SCHEMA,
                ),
                _json(
                    "data/features/right_side_unified/feature_manifest.json",
                    "target_date",
                    predicate_field="factor_contract_sha256",
                    expected_value=RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                ),
            ),
            _daily_incremental(years=6, keys=("date", "symbol")),
            "build_right_side_unified_production_features",
            contract_sources=(
                "configs/strategies/right_side_ranking_selector.yaml",
                "src/quant/application/selector_ranking.py",
                "src/quant/features/right_side_factor_contract.py",
                "src/quant/features/project_factor_layer.py",
                "src/quant/research/right_side_unified_features.py",
                "src/quant/research/right_side_unified_signals.py",
                "src/quant/routine/right_side_unified_production.py",
            ),
            outputs=(
                "data/features/right_side_unified/latest_features.parquet",
                "data/features/right_side_unified/feature_manifest.json",
            ),
            result_aliases=("right_side_unified_features",),
            ui_step="right_side_unified_features",
            ui_order=41,
            feature_catalog_provider="right_side_unified_factor_columns",
            final_gate=True,
            notes=(
                "Dormant while selector ranking_source=legacy_z_skill; production "
                "v4+118 feature contract for an explicit unified-ranker promotion."
            ),
        ),
        DependencyNode(
            "score.right_side_unified",
            Layer.MODEL_SCORE,
            "routine.right_side_unified_production",
            Lifecycle.PRODUCTION,
            Cadence.TRADE_DAILY,
            (_edge("feature.right_side_unified", ColumnMode.MODEL_ARTIFACT),),
            _exact(
                _file("data/features/right_side_unified/latest_scores.parquet"),
                _json(
                    "data/features/right_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="schema_version",
                    expected_value=RIGHT_SIDE_PRODUCTION_SCORE_SCHEMA_VERSION,
                ),
                _json(
                    "data/features/right_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="score_field",
                    expected_value="ranking_score",
                ),
                _json(
                    "data/features/right_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="playbook_coupling",
                    expected_value="independent",
                ),
            ),
            _daily_incremental(keys=("date", "symbol")),
            "score_right_side_unified_production",
            contract_sources=(
                "configs/strategies/right_side_ranking_selector.yaml",
                "src/quant/application/selector_ranking.py",
                "src/quant/routine/right_side_unified_production.py",
            ),
            outputs=(
                "data/features/right_side_unified/latest_scores.parquet",
                "data/features/right_side_unified/score_manifest.json",
            ),
            result_aliases=("right_side_unified_scores",),
            ui_step="right_side_unified_score",
            ui_order=42,
            artifact=ArtifactSpec(
                ("models/production/right_side_unified_canonical_v2/ranking.joblib",),
                "bundle_features",
                "feature.right_side_unified",
                manifest_path=(
                    "models/production/right_side_unified_canonical_v2/manifest.json"
                ),
                expected_schema=RIGHT_SIDE_PRODUCTION_ARTIFACT_SCHEMA_VERSION,
            ),
            final_gate=True,
            notes=(
                "Publishes ranking_score only; never aliases the ranker as "
                "pred_up5/pred_up8/pred_down3 and never chooses a playbook."
            ),
        ),
        DependencyNode(
            "feature.left_side_unified",
            Layer.FEATURE,
            "routine.left_side_unified_production",
            Lifecycle.PRODUCTION if left_side_enabled else Lifecycle.RETIRED,
            Cadence.TRADE_DAILY if left_side_enabled else Cadence.ON_DEMAND,
            (_edge("data.market_daily"), _edge("feature.strategy_signals")),
            _exact(
                _file("data/features/left_side_unified/latest_features.parquet"),
                _json(
                    "data/features/left_side_unified/feature_manifest.json",
                    "target_date",
                    predicate_field="candidate_coverage_status",
                    expected_value="complete",
                ),
                _json(
                    "data/features/left_side_unified/feature_manifest.json",
                    "target_date",
                    predicate_field="factor_contract_sha256",
                    expected_value=LEFT_SIDE_FACTOR_CONTRACT_SHA256,
                ),
            ),
            _daily_incremental(years=6, keys=("date", "symbol")),
            "build_left_side_production_features",
            contract_sources=(
                "configs/strategies/left_side_unified.yaml",
                "configs/strategies/b1_selected.yaml",
                "src/quant/application/left_side_ranking.py",
                "src/quant/features/left_side_factor_contract.py",
                "src/quant/research/left_side_unified_features.py",
                "src/quant/routine/left_side_unified_production.py",
            ),
            outputs=(
                "data/features/left_side_unified/latest_features.parquet",
                "data/features/left_side_unified/feature_manifest.json",
            ),
            result_aliases=("left_side_unified_features",),
            ui_step="left_side_unified_features",
            ui_order=43,
            feature_catalog_provider="left_side_unified_factor_columns",
            final_gate=True,
            notes="Four stable left groups share one canonical production ranker.",
        ),
        DependencyNode(
            "score.left_side_unified",
            Layer.MODEL_SCORE,
            "routine.left_side_unified_production",
            Lifecycle.PRODUCTION if left_side_enabled else Lifecycle.RETIRED,
            Cadence.TRADE_DAILY if left_side_enabled else Cadence.ON_DEMAND,
            (_edge("feature.left_side_unified", ColumnMode.MODEL_ARTIFACT),),
            _exact(
                _file("data/features/left_side_unified/latest_scores.parquet"),
                _json(
                    "data/features/left_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="schema_version",
                    expected_value=LEFT_SIDE_SCORE_SCHEMA_VERSION,
                ),
                _json(
                    "data/features/left_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="selector_adapter_status",
                    expected_value="ready",
                ),
            ),
            _daily_incremental(keys=("date", "symbol")),
            "score_left_side_production",
            contract_sources=(
                "configs/strategies/left_side_unified.yaml",
                "configs/strategies/b1_selected.yaml",
                "src/quant/application/left_side_ranking.py",
                "src/quant/routine/left_side_unified_production.py",
            ),
            outputs=(
                "data/features/left_side_unified/latest_scores.parquet",
                "data/features/left_side_unified/score_manifest.json",
            ),
            result_aliases=("left_side_unified_scores",),
            ui_step="left_side_unified_score",
            ui_order=44,
            artifact=ArtifactSpec(
                (
                    "models/production/left_side_unified_canonical_v4_group4/"
                    "ranking.joblib",
                ),
                "bundle_features",
                "feature.left_side_unified",
                manifest_path=(
                    "models/production/left_side_unified_canonical_v4_group4/"
                    "manifest.json"
                ),
                expected_schema=LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
            ),
            final_gate=True,
            notes="Ranking-only output with daily cross-sectional percentile normalization.",
        ),
        DependencyNode(
            "score.selector", Layer.MODEL_SCORE, "webapp.services.selector",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("feature.selector_live", ColumnMode.MODEL_ARTIFACT),
             *((_edge("score.b1"),) if not left_side_enabled else ()),
             *((_edge("score.z_skill"),) if not two_unified_rankers_active else ()),
             *((_edge("score.right_side_unified"),) if configured_source == SelectorRankingSource.RIGHT_SIDE_UNIFIED else ()),
             *((_edge("score.left_side_unified"),) if left_side_enabled else ())),
            _exact(_result("selector_extended", "signal_date")),
            _daily_incremental(keys=("date", "symbol")), "score_selector_buy_hold",
            contract_sources=(
                "src/quant/webapp/services.py",
                "src/quant/application/selector_ranking.py",
                "src/quant/application/left_side_ranking.py",
                "configs/strategies/right_side_ranking_selector.yaml",
                "configs/strategies/left_side_unified.yaml",
            ),
            result_aliases=("selector_core", "selector_extended"), ui_step="selector_core", ui_order=70,
            artifact=ArtifactSpec(
                ("models/production/selector_buy_hold_registry_v3/buy.joblib",
                 "models/production/selector_buy_hold_registry_v3/hold.joblib"),
                "bundle_features", "feature.selector_live",
                manifest_path=(
                    "models/production/selector_buy_hold_registry_v3/manifest.json"
                ),
                expected_schema=SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
            ), final_gate=True,
        ),
        DependencyNode(
            "score.chan", Layer.MODEL_SCORE, "research.chan_daily",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("feature.chan_live", ColumnMode.MODEL_ARTIFACT),),
            _exact(
                _file("reports/chan_daily/model_filter/chan_model_scored_candidates.parquet"),
                _json(
                    "reports/chan_daily/model_filter/live_refresh_manifest.json",
                    "processed_through",
                    predicate_field="feature_coverage.status",
                    expected_value="valid",
                ),
            ),
            _daily_incremental(keys=("date", "symbol")), "score_chan_release",
            contract_sources=("scripts/research/refresh_chan_model_live_scores.py",),
            result_aliases=("refresh_chan_model_scores",), ui_step="chan_model_strategy", ui_order=75,
            artifact=ArtifactSpec(
                tuple(f"models/research/chan_daily/{name}.joblib" for name in
                      ("target_win10", "target_big10", "target_good")),
                "bundle_features", "feature.chan_live", approved_research_path=True,
            ), final_gate=True,
            notes="Production-consumed transitional artifacts; promote into models/production.",
        ),
        DependencyNode(
            "score.byd_runtime", Layer.MODEL_SCORE, "research.byd_daily_t_plan",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("feature.byd_daily"), _edge("data.byd_intraday_training")),
            _exact(_result("byd_daily_plan", "signal_date")),
            _daily_incremental(keys=("date", "symbol")), "fit_or_reuse_byd_runtime_model",
            contract_sources=("src/quant/research/byd_daily_t_plan.py",),
            ui_step="byd_daily_plan", ui_order=94, final_gate=True,
        ),
        DependencyNode(
            "score.similar", Layer.MODEL_SCORE, "research.similar_patterns",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("feature.similar_reference"), _edge("feature.similar_target")),
            _exact(_result("similar_patterns", "target_date")),
            _daily_incremental(keys=("date", "symbol")), "score_similar_patterns",
            contract_sources=("src/quant/research/similar_patterns.py",),
            ui_step="similar_patterns", ui_order=98, final_gate=True,
        ),
        DependencyNode(
            "feature.right_side_unified_shadow",
            Layer.FEATURE,
            "routine.right_side_unified_shadow",
            Lifecycle.RESEARCH_ONLY,
            Cadence.TRADE_DAILY,
            (_edge("data.market_daily"), _edge("feature.strategy_signals")),
            _exact(
                _file("data/features/right_side_unified_shadow/latest_features.parquet"),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="candidate_coverage_status",
                    expected_value="complete",
                ),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="feature_schema_version",
                    expected_value=RIGHT_SIDE_SHADOW_FEATURE_SCHEMA,
                ),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="project_factor_schema_version",
                    expected_value=RIGHT_SIDE_SHADOW_PROJECT_FACTOR_SCHEMA,
                ),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="rule_factor_schema_version",
                    expected_value=RIGHT_SIDE_SHADOW_RULE_FACTOR_SCHEMA,
                ),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="rule_factor_count",
                    expected_value=RIGHT_SIDE_SHADOW_RULE_FACTOR_COUNT,
                ),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="factor_count",
                    expected_value=RIGHT_SIDE_SHADOW_FACTOR_COUNT,
                ),
                _json(
                    "data/features/right_side_unified_shadow/feature_manifest.json",
                    "target_date",
                    predicate_field="factor_contract_sha256",
                    expected_value=RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                ),
            ),
            _daily_incremental(years=6, keys=("date", "symbol")),
            "build_right_side_shadow_features",
            contract_sources=(
                "configs/strategies/right_side_unified.yaml",
                "src/quant/features/right_side_factor_contract.py",
                "src/quant/features/project_factor_layer.py",
                "src/quant/research/right_side_unified_features.py",
                "src/quant/research/right_side_unified_signals.py",
                "src/quant/routine/right_side_unified_shadow.py",
            ),
            outputs=(
                "data/features/right_side_unified_shadow/latest_features.parquet",
                "data/features/right_side_unified_shadow/feature_manifest.json",
            ),
            result_aliases=("right_side_shadow_features",),
            ui_step="right_side_shadow_features",
            ui_order=110,
            feature_catalog_provider="right_side_shadow_factor_columns",
            final_gate=True,
            notes=(
                "Independent research shadow: canonical project-v5 plus the "
                "canonical right-side rule union; never consumed by score.selector."
            ),
        ),
        DependencyNode(
            "score.right_side_unified_shadow",
            Layer.MODEL_SCORE,
            "routine.right_side_unified_shadow",
            Lifecycle.RESEARCH_ONLY,
            Cadence.TRADE_DAILY,
            (_edge("feature.right_side_unified_shadow", ColumnMode.MODEL_ARTIFACT),),
            _exact(
                _file("data/features/right_side_unified_shadow/latest_scores.parquet"),
                _json(
                    "data/features/right_side_unified_shadow/score_manifest.json",
                    "target_date",
                    predicate_field="artifact_schema_version",
                    expected_value=RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA,
                ),
                _json(
                    "data/features/right_side_unified_shadow/score_manifest.json",
                    "target_date",
                    predicate_field="factor_contract_sha256",
                    expected_value=RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                ),
            ),
            _daily_incremental(keys=("date", "symbol")),
            "score_right_side_shadow",
            contract_sources=(
                "configs/strategies/right_side_unified.yaml",
                "src/quant/routine/right_side_unified_shadow.py",
            ),
            outputs=(
                "data/features/right_side_unified_shadow/latest_scores.parquet",
                "data/features/right_side_unified_shadow/score_manifest.json",
            ),
            result_aliases=("right_side_shadow_scores",),
            ui_step="right_side_shadow_score",
            ui_order=111,
            artifact=ArtifactSpec(
                (
                    "models/research/right_side_unified_canonical_v5_rule113/shadow/ranking.joblib",
                ),
                "bundle_features",
                "feature.right_side_unified_shadow",
                manifest_path=(
                    "models/research/right_side_unified_canonical_v5_rule113/shadow/manifest.json"
                ),
                expected_schema=RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA,
                approved_research_path=True,
            ),
            final_gate=True,
            notes=(
                "Research-only ranking score. Missing/mismatched artifact blocks "
                "only rightSideShadow and leaves the production selector unchanged."
            ),
        ),
        DependencyNode(
            "product.right_side_unified_shadow",
            Layer.PRODUCT,
            "routine.right_side_unified_shadow",
            Lifecycle.RESEARCH_ONLY,
            Cadence.TRADE_DAILY,
            (_edge("score.right_side_unified_shadow"),),
            _exact(
                _file(
                    "reports/research/right_side_unified_canonical_v5_rule113/shadow/latest_candidates.parquet"
                ),
                _json(
                    "reports/research/right_side_unified_canonical_v5_rule113/shadow/product_manifest.json",
                    "target_date",
                    predicate_field="candidate_coverage_status",
                    expected_value="complete",
                ),
                _json(
                    "reports/research/right_side_unified_canonical_v5_rule113/shadow/product_manifest.json",
                    "target_date",
                    predicate_field="consumer",
                    expected_value="research_shadow_only",
                ),
            ),
            _daily_incremental(keys=("date", "symbol")),
            "publish_right_side_shadow_product",
            contract_sources=(
                "configs/strategies/right_side_unified.yaml",
                "src/quant/routine/right_side_unified_shadow.py",
            ),
            outputs=(
                "reports/research/right_side_unified_canonical_v5_rule113/shadow/latest_candidates.parquet",
                "reports/research/right_side_unified_canonical_v5_rule113/shadow/product_manifest.json",
            ),
            result_aliases=("right_side_shadow_product",),
            ui_step="right_side_shadow_product",
            ui_order=112,
            final_gate=True,
            notes="Not an input to score.selector or any production product.",
        ),
        DependencyNode(
            "product.right_side_unified_adapter",
            Layer.PRODUCT,
            "webapp.services.selector",
            Lifecycle.PRODUCTION,
            Cadence.TRADE_DAILY,
            (_edge("score.right_side_unified"),),
            _exact(
                _json(
                    "data/features/right_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="selector_adapter_status",
                    expected_value="ready",
                ),
                _json(
                    "data/features/right_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="score_field",
                    expected_value="ranking_score",
                ),
            ),
            _daily_incremental(keys=("date", "symbol")),
            "validate_right_side_unified_selector_adapter",
            contract_sources=(
                "configs/strategies/right_side_ranking_selector.yaml",
                "src/quant/application/selector_ranking.py",
                "src/quant/routine/right_side_unified_production.py",
                "src/quant/webapp/services.py",
            ),
            outputs=("data/features/right_side_unified/score_manifest.json",),
            result_aliases=("right_side_unified_adapter",),
            ui_step="right_side_unified_adapter",
            ui_order=45,
            final_gate=True,
            notes=(
                "Production-readiness scope only while the default selector source "
                "remains legacy_z_skill."
            ),
        ),
        DependencyNode(
            "product.left_side_unified_adapter",
            Layer.PRODUCT,
            "webapp.services.selector",
            Lifecycle.PRODUCTION if left_side_enabled else Lifecycle.RETIRED,
            Cadence.TRADE_DAILY if left_side_enabled else Cadence.ON_DEMAND,
            (_edge("score.left_side_unified"),),
            _exact(
                _json(
                    "data/features/left_side_unified/score_manifest.json",
                    "target_date",
                    predicate_field="selector_adapter_status",
                    expected_value="ready",
                ),
            ),
            _daily_incremental(keys=("date", "symbol")),
            "validate_left_side_selector_adapter",
            contract_sources=(
                "configs/strategies/left_side_unified.yaml",
                "src/quant/application/left_side_ranking.py",
                "src/quant/routine/left_side_unified_production.py",
                "src/quant/webapp/services.py",
            ),
            outputs=("data/features/left_side_unified/score_manifest.json",),
            result_aliases=("left_side_unified_adapter",),
            ui_step="left_side_unified_adapter",
            ui_order=46,
            final_gate=True,
            notes="Production adapter for B1/SB1/SUPER_B1/LOW_PULLBACK.",
        ),
        DependencyNode(
            "product.b1_plan", Layer.PRODUCT, "routine.b1_daily_plan",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (
                _edge("score.left_side_unified")
                if left_side_enabled
                else _edge("score.b1")
            ,),
            _exact(_json("web/data/b1_daily_plan.json", "signal_date")),
            _daily_incremental(), "generate_daily_plan", outputs=("web/data/b1_daily_plan.json",),
            contract_sources=("configs/strategies/b1_selected.yaml",),
            result_aliases=("generate_daily_plan",), ui_step="daily_plan", ui_order=51,
            final_gate=True,
        ),
        DependencyNode(
            "product.selector_core", Layer.PRODUCT, "webapp.services.selector",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("score.selector"),),
            _exact(_result("selector_core", "signal_date")), _daily_incremental(),
            "build_selector_core", result_aliases=("selector_core",),
            contract_sources=("src/quant/webapp/services.py",),
            ui_step="selector_core", ui_order=71, final_gate=True,
        ),
        DependencyNode(
            "product.selector_extended", Layer.PRODUCT, "webapp.services.selector",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("score.selector"),),
            _exact(_result("selector_extended", "signal_date")), _daily_incremental(),
            "build_selector_extended", result_aliases=("selector_extended",),
            contract_sources=(
                "src/quant/webapp/services.py",
                "configs/strategies/triple_volume_breakout.yaml",
            ),
            ui_step="selector_extended", ui_order=72, final_gate=True,
        ),
        DependencyNode(
            "product.short_snapshot", Layer.PRODUCT, "webapp.services.selector",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("product.b1_plan"), _edge("product.selector_core"),
             _edge("product.selector_extended")),
            _exact(_result("selector_extended", "signal_date")), _daily_incremental(),
            "write_strategy_pool_snapshots", result_aliases=("snapshot",),
            contract_sources=("src/quant/webapp/services.py",),
            ui_step="snapshot", ui_order=99,
            final_gate=True,
        ),
        DependencyNode(
            "product.chan", Layer.PRODUCT, "webapp.services.chan",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("score.chan"),),
            _exact(_result("chan_model_strategy", "signal_date")), _daily_incremental(),
            "generate_chan_model_strategy", result_aliases=("chan_model_strategy",),
            contract_sources=("src/quant/strategies/custom/chan_model.py",),
            ui_step="chan_model_strategy", ui_order=76, final_gate=True,
        ),
        DependencyNode(
            "product.long_pools", Layer.PRODUCT, "webapp.services.long",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY,
            (_edge("feature.long_snapshot"),),
            _exact(_result("long_stock_pool.variants.0", "signal_date")), _daily_incremental(),
            "refresh_long_stock_pool_variants", result_aliases=("long_stock_pool",),
            contract_sources=(
                "configs/strategies/tea_master_long.yaml",
                "configs/strategies/long_dividend_quality.yaml",
            ),
            ui_step="long_stock_pool", ui_order=82, final_gate=True,
        ),
        DependencyNode(
            "product.cb_grid", Layer.PRODUCT, "application.workspaces.convertible_bonds",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("feature.cb_grid"),),
            _exact(_result("convertible_bond_plan", "trade_date")), _daily_incremental(),
            "build_convertible_bond_grid_workspace", result_aliases=("convertible_bond_plan",),
            contract_sources=("configs/strategies/convertible_bond_rotation.yaml",),
            ui_step="convertible_bond_plan", ui_order=91, final_gate=True,
        ),
        DependencyNode(
            "product.cb_allotment", Layer.PRODUCT, "application.workspaces.convertible_bonds",
            Lifecycle.PRODUCTION, Cadence.EVENT_POLL_DAILY,
            (_edge("data.cb_allotment_events"), _edge("data.daily_basic"), _edge("data.stock_basic")),
            _polled(_result("convertible_bond_allotment", "asof")),
            IncrementalPolicy("date", ("date", "ts_code"), write_mode="reuse_if_event_unchanged"),
            "build_convertible_bond_allotment_workspace",
            contract_sources=(
                "src/quant/routine/convertible_bond_allotment.py",
                "src/quant/application/workspaces/convertible_bonds.py",
            ),
            result_aliases=("convertible_bond_allotment", "convertible_bond_allotments"),
            ui_step="convertible_bond_allotment", ui_order=92, final_gate=True,
        ),
        DependencyNode(
            "product.byd", Layer.PRODUCT, "application.workspaces.byd",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("score.byd_runtime"),),
            _exact(_result("byd_daily_plan", "signal_date")), _daily_incremental(),
            "build_byd_daily_workspace", result_aliases=("byd_daily_plan",),
            contract_sources=("src/quant/application/workspaces/byd.py",),
            ui_step="byd_daily_plan", ui_order=94, final_gate=True,
        ),
        DependencyNode(
            "product.similar", Layer.PRODUCT, "webapp.services.similar",
            Lifecycle.PRODUCTION, Cadence.TRADE_DAILY, (_edge("score.similar"),),
            _exact(_result("similar_patterns", "target_date")), _daily_incremental(),
            "refresh_similar_pattern_analysis",
            contract_sources=("src/quant/webapp/services.py",),
            outputs=("data/research/similar_patterns/web_watchlist_analysis.json",),
            result_aliases=("similar_patterns",), ui_step="similar_patterns", ui_order=98,
            final_gate=True,
        ),
        DependencyNode(
            "product.long_entry_research", Layer.PRODUCT, "research.long_entry",
            Lifecycle.RESEARCH_ONLY, Cadence.ON_DEMAND,
            (_edge("data.long_research_external"), _edge("data.tradability")),
            _static(), IncrementalPolicy(), "train_long_entry_research_models",
            notes="Seven research-only v1/v2 artifacts have no production page consumer.",
        ),
    ]
    scopes = {
        "all": (
            "product.short_snapshot", "product.chan", "product.long_pools",
            "product.cb_grid", "product.cb_allotment", "product.byd", "product.similar",
        ),
        "short": ("product.short_snapshot",),
        "chan": ("product.chan",),
        "long": ("product.long_pools",),
        "cb": ("product.cb_grid",),
        "cbAllotment": ("product.cb_allotment",),
        "byd": ("product.byd",),
        "similar": ("product.similar",),
        "rightSideShadow": ("product.right_side_unified_shadow",),
        "rightSideRankingCandidate": ("product.right_side_unified_adapter",),
        "leftSideRanking": ("product.left_side_unified_adapter",),
    }
    return DependencyRegistry(nodes, scopes)


DEFAULT_DAILY_DEPENDENCY_REGISTRY = build_default_daily_dependency_registry()


def required_source_options(scope: str) -> dict[str, Any]:
    """Translate the active graph into the existing reference-refresh switches."""

    active = set(DEFAULT_DAILY_DEPENDENCY_REGISTRY.required_node_ids(scope))
    long_datasets: list[str] = []
    if "data.top_list" in active:
        long_datasets.append("top_list")
    if "data.long_research_external" in active:
        long_datasets.extend(("margin_detail", "moneyflow", "holder_trade_recent"))
    return {
        "include_tradability": "data.tradability" in active,
        "long_factor_datasets": tuple(dict.fromkeys(long_datasets)),
        "include_financials": "data.financial_pit" in active,
        "include_analyst": "data.analyst_pit" in active,
    }


__all__ = [
    "ArtifactSpec",
    "Cadence",
    "ChangeSet",
    "ColumnMode",
    "DEFAULT_DAILY_DEPENDENCY_REGISTRY",
    "DependencyEdge",
    "DependencyNode",
    "DependencyRegistry",
    "EvidenceSpec",
    "FeatureUsage",
    "FreshnessMode",
    "FreshnessPolicy",
    "IncrementalPolicy",
    "Layer",
    "Lifecycle",
    "ModelContract",
    "NodeState",
    "PlanEntry",
    "PRODUCTION_PROJECT_FACTOR_SCHEMA",
    "RIGHT_SIDE_SHADOW_ARTIFACT_SCHEMA",
    "RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256",
    "RIGHT_SIDE_SHADOW_FACTOR_COUNT",
    "RIGHT_SIDE_SHADOW_FEATURE_SCHEMA",
    "RIGHT_SIDE_SHADOW_PROJECT_FACTOR_SCHEMA",
    "RIGHT_SIDE_SHADOW_RULE_FACTOR_COUNT",
    "RIGHT_SIDE_SHADOW_RULE_FACTOR_SCHEMA",
    "build_default_daily_dependency_registry",
    "build_dependency_plan",
    "classify_feature_usage",
    "required_source_options",
    "state_is_current",
]
