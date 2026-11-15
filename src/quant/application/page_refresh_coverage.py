"""Auditable page coverage, including optional feeds outside the hard DAG.

This inventory does not schedule work or relax any freshness gate. Strategy
filters share the parent product's materialization, not separate daily jobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from quant.application.daily_dependencies import DependencyRegistry


SHORT_STRATEGY_MEMBERS = {
    "B1": ("B1",), "B2": ("B2",), "B3": ("B3",), "SB1": ("SB1",),
    "SUPER_B1": ("SUPER_B1",), "STRONG_K": ("KEY_K", "VIOLENCE_K"),
    "DOUBLE_YANG": ("PINGHANG", "DOUBLE_GUN"),
    "LOW_PULLBACK": ("YIDONG_DILIAN", "NANA", "DUICHEN_VA"),
    "SUPPORT_PULLBACK": ("GOLDEN_BOWL", "ZAIHOU"),
    "RHYTHM_PLATFORM": ("BREATHING", "YUEYUE"), "CHANGAN": ("CHANGAN",),
    "KENGQI": ("KENGQI",), "VEGAS": ("VEGAS",),
    "TRIPLE_VOLUME_BREAKOUT": ("TRIPLE_VOLUME_BREAKOUT",),
}


@dataclass(frozen=True)
class PageRefreshContract:
    tab: str
    policy: str
    products: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()


PAGE_REFRESH_CONTRACTS = (
    PageRefreshContract("short", "trade_daily", (
        "product.short_snapshot", "product.selector_core", "product.selector_extended",
        "product.b1_plan",
    ), tuple(SHORT_STRATEGY_MEMBERS), ("short_etf_flow",)),
    PageRefreshContract("chan", "trade_daily", ("product.chan",), (
        "chan_model_primary", "chan_model_expanded", "chan_model_high_conviction",
    )),
    PageRefreshContract("long", "trade_daily", ("product.long_pools",),
                        ("tea", "tea_safe", "v44", "blood_chip")),
    PageRefreshContract("cb", "trade_daily", ("product.cb_grid",), (
        "core_market_scaled_market_gate", "return_core_trend_rebound",
        "strict_dynamic_floor98_grid", "strict_dynamic_no_floor_grid",
        "floor98_expanded_ladder_block", "defensive_low_price_dynamic_grid",
    )),
    PageRefreshContract("cbAllotment", "event_poll_daily", ("product.cb_allotment",)),
    PageRefreshContract("byd", "trade_daily", ("product.transition",), ("return", "win")),
    PageRefreshContract("similar", "trade_daily", ("product.similar",), ("watchlist",)),
    PageRefreshContract("futures", "on_demand"),
    PageRefreshContract("plans", "user_crud"),
    PageRefreshContract("reviews", "user_crud"),
)

OPTIONAL_REFRESH_DEPENDENCIES = {
    "short_etf_flow": {
        "policy": "optional_degradable",
        "hard_gate": False,
        "operation": "refresh_short_etf_flow",
        "consumers": ("short.etf_flow_panel", "short_etf_rank.left", "short_etf_rank.right"),
        "snapshot": "data/features/short_etf_flow/YYYYMMDD.json",
        "snapshot_date_policy": "exact_signal_date",
        "source_max_lag_sessions": {"fd_share": 2, "unit_nav": 2, "adj_factor": 2, "amount": 0},
        "require_all_feature_values": False,
        "fallback": "unavailable_panel_and_base_ranking",
    },
}

LONG_VARIANT_EXPOSURE = {
    "visible_buttons": ("tea", "blood_chip"),
    "daily_compatibility_outputs": ("tea_safe", "v44"),
    "catalog_only_not_daily": ("v43", "v34"),
}


def page_refresh_coverage_report(
    registry: DependencyRegistry,
    scope: str,
    target_date: str,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """Report declaration coverage and observed refresh status, not source quality.

    Source dates are passed through: the producer validates exchange-session
    D-2 limits. No calendar-day approximation or all-values check belongs here.
    """
    active = set(registry.required_node_ids(scope))
    pages = []
    for contract in PAGE_REFRESH_CONTRACTS:
        pages.append({
            **asdict(contract),
            "in_scope": bool(set(contract.products) & active),
            "registered": all(node in registry.nodes for node in contract.products),
            "hard_gate_registered": bool(contract.products) and all(
                node in registry.nodes and registry.nodes[node].final_gate
                for node in contract.products
            ),
        })
    optional = {}
    for key, contract in OPTIONAL_REFRESH_DEPENDENCIES.items():
        raw = results.get(key)
        result = raw if isinstance(raw, Mapping) else {}
        in_scope = any(
            key in page.optional_dependencies and bool(set(page.products) & active)
            for page in PAGE_REFRESH_CONTRACTS
        )
        status = str(result.get("status") or "not_observed")
        if not in_scope:
            status = "out_of_scope"
        elif result and result.get("signal_date") != target_date:
            status = "stale_or_undated"
        optional[key] = {
            **contract,
            "in_scope": in_scope,
            "status": status,
            "signal_date": result.get("signal_date"),
            "source_dates": result.get("source_dates"),
            "same_date_snapshot_retained": result.get("same_date_snapshot_retained"),
        }
    return {
        "schema_version": "page-refresh-coverage-v1",
        "scope": scope,
        "target_trade_date": target_date,
        "pages": pages,
        "short_strategy_members": SHORT_STRATEGY_MEMBERS,
        "long_variant_exposure": LONG_VARIANT_EXPOSURE,
        "optional_dependencies": optional,
    }
