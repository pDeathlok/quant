"""Application use cases for convertible-bond workspaces."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd


WorkspacePayload = dict[str, Any]
SnapshotReader = Callable[..., WorkspacePayload | None]
SnapshotWriter = Callable[..., None]
PayloadBuilder = Callable[..., WorkspacePayload]


def _canonical_snapshot_date(value: Any) -> str:
    text_value = str(value or "latest").strip()
    if text_value == "latest":
        return text_value
    compact = text_value.replace("-", "")
    if len(compact) == 8 and compact.isdigit():
        return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
    return text_value


@dataclass(frozen=True)
class ConvertibleBondGridDependencies:
    """External collaborators required by the convertible-bond grid workspace."""

    read_snapshot: SnapshotReader
    read_legacy_snapshot: Callable[[], WorkspacePayload | None]
    promote_legacy_snapshot: Callable[[str, WorkspacePayload], None]
    write_snapshot: SnapshotWriter
    refresh_daily: PayloadBuilder
    build_plan: PayloadBuilder


def build_convertible_bond_grid_workspace(
    trade_date: str | None = None,
    limit: int = 18,
    refresh: bool = False,
    *,
    dependencies: ConvertibleBondGridDependencies,
) -> WorkspacePayload:
    """Read or build the convertible-bond grid plan with durable snapshots."""

    params = {"limit": int(limit)}
    if not refresh:
        cached = dependencies.read_snapshot(
            "convertible_bond_grid_plan",
            snapshot_date=trade_date,
            params=params,
            allow_sql=False,
        )
        if cached is not None:
            return cached
        legacy = dependencies.read_legacy_snapshot()
        if isinstance(legacy, dict) and legacy.get("trade_date"):
            cached_date = _canonical_snapshot_date(legacy.get("trade_date"))
            requested_date = _canonical_snapshot_date(trade_date) if trade_date else None
            if not requested_date or cached_date <= requested_date:
                legacy["cache"] = {
                    "hit": True,
                    "backend": "legacy_filesystem",
                    "workspace": "convertible_bond_grid_plan",
                    "snapshot_date": cached_date,
                    "requested_date": requested_date,
                    "stale": bool(requested_date and cached_date != requested_date),
                }
                dependencies.promote_legacy_snapshot(cached_date, legacy)
                return legacy

    refresh_result = None
    if refresh and trade_date:
        refresh_result = dependencies.refresh_daily(trade_date=trade_date)
    payload = dependencies.build_plan(trade_date=trade_date, limit=limit)
    if refresh_result is not None:
        payload["data_refresh"] = refresh_result
    dependencies.write_snapshot(
        "convertible_bond_grid_plan",
        payload.get("trade_date") or trade_date,
        payload,
        params=params,
        write_sql=refresh,
    )
    return payload


@dataclass(frozen=True)
class ConvertibleBondAllotmentDependencies:
    """External collaborators required by the allotment workspace."""

    read_snapshot: SnapshotReader
    read_daily_cache: Callable[[], WorkspacePayload | None]
    write_daily_cache: Callable[[WorkspacePayload], None]
    write_snapshot: SnapshotWriter
    build_payload: PayloadBuilder
    is_daily_current: Callable[..., bool]


def evaluate_convertible_bond_allotment_quality(
    payload: WorkspacePayload,
    *,
    expected_trade_date: str | None = None,
    minimum_coverage: float = 0.90,
) -> WorkspacePayload:
    """Evaluate freshness and indicator completeness for allotment records."""

    minimum_coverage = min(1.0, max(0.0, float(minimum_coverage)))
    records = [item for item in payload.get("records") or [] if isinstance(item, dict)]
    data_sources = payload.get("data_sources") or {}
    stock_daily = data_sources.get("stock_daily")
    daily_basic = data_sources.get("daily_basic")

    def numeric_present(value: Any) -> bool:
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def normalized_date(value: Any) -> str | None:
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else parsed.date().isoformat()

    def metric(count: int, total: int) -> WorkspacePayload:
        ratio = 1.0 if total <= 0 else count / total
        return {
            "count": int(count),
            "total": int(total),
            "ratio": round(float(ratio), 6),
            "passed": bool(ratio >= minimum_coverage),
        }

    if not isinstance(stock_daily, dict) or not isinstance(daily_basic, dict):
        return {
            "status": "failed",
            "expected_trade_date": normalized_date(expected_trade_date),
            "minimum_coverage": minimum_coverage,
            "metrics": {},
            "issues": ["缺少 stock_daily 或 daily_basic 数据源元信息"],
        }

    requested = max(0, int(stock_daily.get("requested") or 0))
    matched = max(0, int(stock_daily.get("matched") or 0))
    expected_date = normalized_date(expected_trade_date)
    if expected_date is None:
        price_dates = [
            value
            for item in records
            if (value := normalized_date(item.get("stock_price_date"))) is not None
        ]
        expected_date = max(price_dates, default=None)

    current_prices = sum(
        normalized_date(item.get("stock_price_date")) == expected_date
        for item in records
        if expected_date is not None
    )
    daily_basic_matched = max(0, int(daily_basic.get("matched") or 0))
    metrics = {
        "stock_daily_match": metric(matched, requested),
        "stock_price_freshness": metric(current_prices, matched),
        "daily_basic_match": metric(daily_basic_matched, requested),
        "kdj_daily_j": metric(
            sum(numeric_present(item.get("kdj_daily_j")) for item in records),
            matched,
        ),
        "kdj_weekly_j": metric(
            sum(numeric_present(item.get("kdj_weekly_j")) for item in records),
            matched,
        ),
        "kdj_monthly_j": metric(
            sum(numeric_present(item.get("kdj_monthly_j")) for item in records),
            matched,
        ),
    }
    labels = {
        "stock_daily_match": "行情匹配率",
        "stock_price_freshness": "行情日期新鲜度",
        "daily_basic_match": "daily_basic 匹配率",
        "kdj_daily_j": "日线 J 完整率",
        "kdj_weekly_j": "周线 J 完整率",
        "kdj_monthly_j": "月线 J 完整率",
    }
    issues = [
        (
            f"{labels[key]}不足: {value['count']}/{value['total']} "
            f"({value['ratio']:.1%} < {minimum_coverage:.1%})"
        )
        for key, value in metrics.items()
        if not value["passed"]
    ]
    if stock_daily.get("error"):
        issues.append(f"stock_daily 读取异常: {stock_daily['error']}")
    if daily_basic.get("error"):
        issues.append(f"daily_basic 读取异常: {daily_basic['error']}")
    return {
        "status": "success" if not issues else "failed",
        "expected_trade_date": expected_date,
        "minimum_coverage": minimum_coverage,
        "metrics": metrics,
        "issues": issues,
    }


def build_convertible_bond_allotment_workspace(
    limit: int = 80,
    include_listed_days: int = 90,
    refresh: bool = False,
    stage_scope: str = "pipeline",
    expected_trade_date: str | None = None,
    validate_quality: bool = False,
    *,
    dependencies: ConvertibleBondAllotmentDependencies,
    minimum_coverage: float = 0.90,
) -> WorkspacePayload:
    """Read or build the allotment workspace and enforce its quality gate."""

    params = {
        "limit": int(limit),
        "include_listed_days": int(include_listed_days),
        "stage_scope": str(stage_scope),
    }
    if not refresh:
        cached = dependencies.read_snapshot(
            "convertible_bond_allotments",
            params=params,
            allow_sql=False,
        )
        if cached is None:
            cached = dependencies.read_daily_cache()
        if cached is not None:
            cached.setdefault("cache", {})
            is_current = dependencies.is_daily_current(cached)
            cached["cache"].update({"hit": True, "stale": not is_current})
            cached_quality = cached.get("quality") or {}
            cached["quality"] = evaluate_convertible_bond_allotment_quality(
                cached,
                expected_trade_date=cached_quality.get("expected_trade_date"),
                minimum_coverage=minimum_coverage,
            )
            return cached

    payload = dependencies.build_payload(
        limit=limit,
        include_listed_days=include_listed_days,
        refresh=refresh,
        stage_scope=stage_scope,
    )
    payload["quality"] = evaluate_convertible_bond_allotment_quality(
        payload,
        expected_trade_date=expected_trade_date,
        minimum_coverage=minimum_coverage,
    )
    dependencies.write_daily_cache(payload)
    dependencies.write_snapshot(
        "convertible_bond_allotments",
        payload.get("asof") or payload.get("trade_date") or payload.get("generated_at"),
        payload,
        params=params,
        write_sql=refresh,
    )
    if validate_quality and payload["quality"]["status"] != "success":
        details = "；".join(payload["quality"].get("issues") or ["未知质量问题"])
        raise RuntimeError(f"配债股数据质量门禁失败: {details}")
    return payload
