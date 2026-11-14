"""Auditable coverage of the published pages and their daily products.

This inventory does not schedule work or relax any freshness gate. Strategy
filters share the parent product's materialization, not separate daily jobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING

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


PAGE_REFRESH_CONTRACTS = (
    PageRefreshContract("short", "trade_daily", (
        "product.short_snapshot", "product.selector_core", "product.selector_extended",
        "product.b1_plan",
    ), tuple(SHORT_STRATEGY_MEMBERS)),
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
    PageRefreshContract("byd", "trade_daily", ("product.byd",)),
    PageRefreshContract("similar", "trade_daily", ("product.similar",), ("watchlist",)),
    PageRefreshContract("plans", "user_crud"),
)

LONG_VARIANT_EXPOSURE = {
    "visible_buttons": ("tea", "blood_chip"),
    "daily_compatibility_outputs": ("tea_safe", "v44"),
    "catalog_only_not_daily": ("v43", "v34"),
}


def page_refresh_coverage_report(
    registry: DependencyRegistry,
    scope: str,
    target_date: str,
) -> dict[str, Any]:
    """Report declaration coverage, not execution success or source quality."""
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
    return {
        "schema_version": "page-refresh-coverage-v1",
        "scope": scope,
        "target_trade_date": target_date,
        "pages": pages,
        "short_strategy_members": SHORT_STRATEGY_MEMBERS,
        "long_variant_exposure": LONG_VARIANT_EXPOSURE,
    }
