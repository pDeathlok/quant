from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant.core.paths import PROJECT_ROOT
from quant.data.atomic_io import atomic_link_or_copy, atomic_write_json
from quant.data.source_merge import normalize_ts_code
from quant.routine.convertible_bond_allotment import (
    build_convertible_bond_allotment_payload,
)


SCHEMA_VERSION = 1
ELIGIBLE_ENTRY_STAGES = frozenset({"exchange_approved", "registered"})
ACTIVE_MANAGED_STAGES = frozenset({"exchange_approved", "registered", "issuing"})
TERMINAL_STAGES = frozenset({"listed", "terminated", "delisted"})
ALLOTMENT_NOTE_BEGIN = "【配债股任务】"
ALLOTMENT_NOTE_END = "【/配债股任务】"
ALLOTMENT_DAILY_J_REMINDER_ID = "cb-allotment-daily-j-lt-5"
ALLOTMENT_DAILY_J_CONDITION_ID = "cb-allotment-daily-j-lt-5-condition"
WATCHLIST_ALERT_MAX_REMINDERS = 20
SHANGHAI_TZ = timezone(timedelta(hours=8))

DEFAULT_WATCHLIST_PATH = (
    PROJECT_ROOT / "data/research/similar_patterns/watchlist.json"
)
DEFAULT_ALLOTMENT_SNAPSHOT_PATH = (
    PROJECT_ROOT / "data/routine/convertible_bond_allotments_latest.json"
)
DEFAULT_PLAN_PATH = (
    PROJECT_ROOT / "data/routine/convertible_bond_watchlist_review_latest.json"
)

AllotmentWorkspaceLoader = Callable[..., dict[str, Any]]
WatchlistDataSynchronizer = Callable[[], dict[str, Any]]


def _now_shanghai() -> datetime:
    return datetime.now(tz=SHANGHAI_TZ)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_symbol(raw_symbol: Any) -> str:
    symbol = normalize_ts_code(str(raw_symbol or "").strip().upper())
    return symbol if re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", symbol) else ""


def _note_content(raw_note: Any) -> str:
    if isinstance(raw_note, dict):
        return str(raw_note.get("content") or "")
    return str(raw_note or "")


def is_managed_allotment_note(raw_note: Any) -> bool:
    """Recognize notes owned by this task plus the seven legacy note shapes."""

    content = _note_content(raw_note).strip()
    if not content:
        return False
    if ALLOTMENT_NOTE_BEGIN in content:
        return True
    if "配债股" in content:
        return True
    if re.search(r"(?:^|\n)\s*配债\s*(?:$|\n)", content):
        return True
    return "一手股数" in content and "含权量" in content


def _parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _record_action(record: Mapping[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    symbol = _canonical_symbol(record.get("stock_code"))
    action = {
        "symbol": symbol,
        "name": str(record.get("stock_name") or symbol),
        "stage": str(record.get("stage") or ""),
        "status": str(record.get("status") or ""),
        "announcement_title": record.get("announcement_title"),
        "announce_date": record.get("announce_date"),
        "record_date": record.get("record_date"),
        "stock_price": record.get("stock_price"),
        "stock_price_date": record.get("stock_price_date"),
        "kdj_daily_j": record.get("kdj_daily_j"),
        "kdj_weekly_j": record.get("kdj_weekly_j"),
        "kdj_monthly_j": record.get("kdj_monthly_j"),
        "rights_per_share": record.get("rights_per_share"),
        "shares_for_one_lot": record.get("shares_for_one_lot"),
        "rights_value_pct": record.get("rights_value_pct"),
        "announcement_url": record.get("announcement_url"),
    }
    if reason:
        action["reason"] = reason
    return action


def _action_sort_key(action: Mapping[str, Any]) -> tuple[int, int, str]:
    stage_rank = {"registered": 0, "exchange_approved": 1, "issuing": 2}
    date_digits = re.sub(r"\D", "", str(action.get("announce_date") or ""))[:8]
    date_rank = -int(date_digits) if len(date_digits) == 8 else 0
    return (
        stage_rank.get(str(action.get("stage") or ""), 9),
        date_rank,
        str(action.get("symbol") or ""),
    )


def _normalized_source_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "stock_code",
        "stock_name",
        "stage",
        "status",
        "announcement_title",
        "announce_date",
        "record_date",
        "pay_date",
        "issue_date",
        "stock_price",
        "stock_price_date",
        "kdj_daily_j",
        "kdj_weekly_j",
        "kdj_monthly_j",
        "rights_per_share",
        "shares_for_one_lot",
        "rights_value_pct",
        "announcement_url",
    )
    normalized = []
    for record in records:
        symbol = _canonical_symbol(record.get("stock_code"))
        if not symbol:
            continue
        row = {field: record.get(field) for field in fields}
        row["stock_code"] = symbol
        normalized.append(row)
    return sorted(normalized, key=lambda item: str(item["stock_code"]))


def _eligible_entry_records(
    records: list[dict[str, Any]],
    *,
    asof: date,
) -> dict[str, dict[str, Any]]:
    eligible: dict[str, dict[str, Any]] = {}
    for record in records:
        symbol = _canonical_symbol(record.get("stock_code"))
        if not symbol or record.get("stage") not in ELIGIBLE_ENTRY_STAGES:
            continue
        record_date = _parse_iso_date(record.get("record_date"))
        if record_date is not None and record_date < asof:
            continue
        eligible.setdefault(symbol, record)
    return eligible


def audit_allotment_result_consistency(
    *,
    daily_records: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
    asof: date,
) -> dict[str, Any]:
    """Compare daily-page and review outputs for the shared entry-stage universe."""

    daily = _eligible_entry_records(daily_records, asof=asof)
    review = _eligible_entry_records(review_records, asof=asof)
    daily_symbols = set(daily)
    review_symbols = set(review)
    compared_fields = (
        "stage",
        "announcement_title",
        "announce_date",
        "announcement_url",
    )
    field_mismatches: list[dict[str, Any]] = []
    for symbol in sorted(daily_symbols & review_symbols):
        differences = {
            field: {
                "daily": daily[symbol].get(field),
                "review": review[symbol].get(field),
            }
            for field in compared_fields
            if daily[symbol].get(field) != review[symbol].get(field)
        }
        if differences:
            field_mismatches.append(
                {"symbol": symbol, "differences": differences}
            )
    daily_only = sorted(daily_symbols - review_symbols)
    review_only = sorted(review_symbols - daily_symbols)
    status = (
        "success"
        if not daily_only and not review_only and not field_mismatches
        else "failed"
    )
    return {
        "status": status,
        "daily_count": len(daily),
        "review_count": len(review),
        "daily_only": daily_only,
        "review_only": review_only,
        "field_mismatches": field_mismatches,
        "compared_fields": list(compared_fields),
    }


def _plan_identity_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": plan.get("schema_version"),
        "asof": plan.get("asof"),
        "watchlist_sha256": plan.get("watchlist_sha256"),
        "source_fingerprint": plan.get("source_fingerprint"),
        "additions": plan.get("additions") or [],
        "removals": plan.get("removals") or [],
        "mark_as_allotment": plan.get("mark_as_allotment") or [],
        "manual_review": plan.get("manual_review") or [],
        "unchanged": plan.get("unchanged") or [],
    }


def _derive_plan_id(plan: Mapping[str, Any]) -> str:
    return _sha256_json(_plan_identity_payload(plan))[:16]


def build_watchlist_review_plan(
    *,
    records: list[dict[str, Any]],
    watchlist_payload: dict[str, Any],
    watchlist_sha256: str,
    asof: date,
    source: dict[str, Any],
) -> dict[str, Any]:
    raw_symbols = watchlist_payload.get("symbols", [])
    if not isinstance(raw_symbols, list):
        raise ValueError("自选池 symbols 必须是列表")
    symbols = [symbol for item in raw_symbols if (symbol := _canonical_symbol(item))]
    symbol_set = set(symbols)
    raw_notes = watchlist_payload.get("notes", {})
    notes = raw_notes if isinstance(raw_notes, dict) else {}
    managed_symbols = {
        symbol
        for symbol in symbols
        if is_managed_allotment_note(notes.get(symbol))
    }

    by_symbol: dict[str, dict[str, Any]] = {}
    for record in records:
        symbol = _canonical_symbol(record.get("stock_code"))
        if symbol and symbol not in by_symbol:
            by_symbol[symbol] = record

    eligible = _eligible_entry_records(list(by_symbol.values()), asof=asof)

    additions = [
        _record_action(record)
        for symbol, record in eligible.items()
        if symbol not in symbol_set
    ]
    mark_as_allotment = [
        _record_action(record, reason="已在自选池，但缺少配债股任务标记")
        for symbol, record in eligible.items()
        if symbol in symbol_set and symbol not in managed_symbols
    ]

    removals: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for symbol in sorted(managed_symbols):
        record = by_symbol.get(symbol)
        if record is None:
            manual_review.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "reason": "最新配债源中缺失；不能仅因缺失就自动出池",
                }
            )
            continue
        record_date = _parse_iso_date(record.get("record_date"))
        stage = str(record.get("stage") or "")
        if record_date is not None and record_date < asof:
            removals.append(
                _record_action(
                    record,
                    reason=f"股权登记日 {record_date.isoformat()} 已过",
                )
            )
        elif stage in TERMINAL_STAGES:
            removals.append(
                _record_action(record, reason=f"发行流程已进入终态 {stage}")
            )
        elif stage in ACTIVE_MANAGED_STAGES:
            unchanged.append(_record_action(record))
        else:
            manual_review.append(
                _record_action(
                    record,
                    reason=f"配债阶段异常回退为 {stage or '空'}；不自动出池",
                )
            )

    additions.sort(key=_action_sort_key)
    mark_as_allotment.sort(key=_action_sort_key)
    removals.sort(key=lambda item: str(item.get("symbol") or ""))
    manual_review.sort(key=lambda item: str(item.get("symbol") or ""))
    unchanged.sort(key=_action_sort_key)
    sync_symbols = {
        str(item.get("symbol") or "")
        for item in [*additions, *mark_as_allotment, *unchanged]
        if item.get("symbol")
    }

    source_records = _normalized_source_records(records)
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_shanghai().isoformat(timespec="seconds"),
        "asof": asof.isoformat(),
        "watchlist_updated_at": watchlist_payload.get("updated_at"),
        "watchlist_sha256": watchlist_sha256,
        "source": source,
        "source_fingerprint": _sha256_json(source_records),
        "managed_symbols_before": sorted(managed_symbols),
        "additions": additions,
        "removals": removals,
        "mark_as_allotment": mark_as_allotment,
        "manual_review": manual_review,
        "unchanged": unchanged,
        "counts": {
            "watchlist": len(symbols),
            "managed_before": len(managed_symbols),
            "additions": len(additions),
            "removals": len(removals),
            "mark_as_allotment": len(mark_as_allotment),
            "manual_review": len(manual_review),
            "unchanged": len(unchanged),
            "information_sync": len(sync_symbols),
            "daily_j_reminder_ensure": len(sync_symbols),
        },
    }
    plan["plan_id"] = _derive_plan_id(plan)
    return plan


def generate_watchlist_review_plan(
    *,
    watchlist_path: Path = DEFAULT_WATCHLIST_PATH,
    snapshot_path: Path = DEFAULT_ALLOTMENT_SNAPSHOT_PATH,
    output_path: Path = DEFAULT_PLAN_PATH,
    refresh: bool = False,
    today: date | None = None,
    workspace_loader: AllotmentWorkspaceLoader | None = None,
) -> dict[str, Any]:
    asof = today or _now_shanghai().date()
    watchlist_path = Path(watchlist_path)
    snapshot_path = Path(snapshot_path)
    output_path = Path(output_path)
    watchlist_payload = _read_json_object(watchlist_path)

    if refresh:
        if workspace_loader is None:
            raise ValueError(
                "--refresh 必须由组装层注入每日更新的配债工作区入口"
            )
        freshness = workspace_loader(
            refresh=True,
            stage_scope="pipeline",
            validate_quality=True,
        )
        source_mode = "shared_daily_update_refresh"
    else:
        freshness = _read_json_object(snapshot_path)
        source_mode = "daily_snapshot_plus_cached_raw_sources"

    polled_through = str(freshness.get("event_polled_through") or "")
    poll_status = str(freshness.get("event_poll_status") or "")
    if poll_status != "success" or polled_through != asof.isoformat():
        raise ValueError(
            "配债事件源不是当日成功快照: "
            f"expected={asof.isoformat()} actual={polled_through or '空'} "
            f"status={poll_status or '空'}；请先运行每日更新或使用 --refresh"
        )

    # The shared daily-update refresh owns all network polling and canonical
    # persistence.  This second build is cache-only and keeps expired record
    # dates solely for safe exit review.
    allotment_payload = build_convertible_bond_allotment_payload(
        limit=500,
        refresh=False,
        stage_scope="pipeline",
        include_expired_record_dates=True,
        today=asof,
    )
    consistency = audit_allotment_result_consistency(
        daily_records=list(freshness.get("records") or []),
        review_records=list(allotment_payload.get("records") or []),
        asof=asof,
    )
    if consistency["status"] != "success":
        raise ValueError(
            "每日更新与配债复核结果不一致，拒绝生成待执行计划: "
            + json.dumps(consistency, ensure_ascii=False, sort_keys=True)
        )

    source = {
        "mode": source_mode,
        "snapshot_path": str(snapshot_path),
        "snapshot_generated_at": freshness.get("generated_at"),
        "event_poll_status": poll_status,
        "event_polled_through": polled_through,
        "event_poll_sources": freshness.get("event_poll_sources"),
        "records_with_expired_dates": len(allotment_payload.get("records") or []),
        "result_consistency": consistency,
    }
    plan = build_watchlist_review_plan(
        records=list(allotment_payload.get("records") or []),
        watchlist_payload=watchlist_payload,
        watchlist_sha256=_sha256_path(watchlist_path),
        asof=asof,
        source=source,
    )
    plan["plan_path"] = str(output_path.resolve())
    atomic_write_json(plan, output_path)
    return plan


def _allotment_note_block(action: Mapping[str, Any], asof: str) -> str:
    status = str(action.get("status") or action.get("stage") or "待核对")
    lines = [ALLOTMENT_NOTE_BEGIN, f"{asof[5:]} 配债股 · {status}"]
    shares = action.get("shares_for_one_lot")
    rights_value = action.get("rights_value_pct")
    metrics = []
    if shares is not None:
        metrics.append(f"一手股数 {int(float(shares))}股")
    if rights_value is not None:
        metrics.append(f"含权量 {float(rights_value):.2f}%")
    if metrics:
        lines.append(" · ".join(metrics))
    else:
        lines.append("配售参数待发行公告校准")
    record_date = str(action.get("record_date") or "").strip()
    if record_date:
        lines.append(f"股权登记日 {record_date}")
    stock_price = action.get("stock_price")
    stock_price_date = str(action.get("stock_price_date") or "").strip()
    if stock_price is not None:
        price_line = f"现价 {float(stock_price):.2f}"
        if stock_price_date:
            price_line += f"（{stock_price_date}）"
        lines.append(price_line)
    kdj_values = []
    for label, field in (
        ("日J", "kdj_daily_j"),
        ("周J", "kdj_weekly_j"),
        ("月J", "kdj_monthly_j"),
    ):
        value = action.get(field)
        if value is not None:
            kdj_values.append(f"{label} {float(value):.2f}")
    if kdj_values:
        lines.append("KDJ " + " · ".join(kdj_values))
    announcement_title = str(action.get("announcement_title") or "").strip()
    if announcement_title:
        lines.append(f"公告 {announcement_title}")
    announcement_url = str(action.get("announcement_url") or "").strip()
    if announcement_url:
        lines.append(f"阶段公告 {announcement_url}")
    lines.append(ALLOTMENT_NOTE_END)
    return "\n".join(lines)


def _merge_allotment_note(existing: str, block: str) -> str:
    content = str(existing or "").strip()
    pattern = re.compile(
        re.escape(ALLOTMENT_NOTE_BEGIN)
        + r".*?"
        + re.escape(ALLOTMENT_NOTE_END),
        flags=re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(block, content).strip()
    return f"{content}\n\n{block}".strip() if content else block


def _daily_j_lt_5_reminder() -> dict[str, Any]:
    return {
        "id": ALLOTMENT_DAILY_J_REMINDER_ID,
        "note": "配债股：日线J < 5",
        "conditions": [
            {
                "id": ALLOTMENT_DAILY_J_CONDITION_ID,
                "conjunction": "and",
                "kind": "indicator",
                "operator": "lt",
                "value": 5.0,
                "indicator": "kdj_daily_j",
            }
        ],
    }


def _is_independent_daily_j_lt_5_reminder(raw_reminder: Any) -> bool:
    if not isinstance(raw_reminder, dict):
        return False
    conditions = raw_reminder.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        return False
    condition = conditions[0]
    if not isinstance(condition, dict):
        return False
    try:
        threshold = float(condition.get("value"))
    except (TypeError, ValueError):
        return False
    return (
        condition.get("kind") == "indicator"
        and condition.get("indicator") == "kdj_daily_j"
        and condition.get("operator") == "lt"
        and threshold == 5.0
    )


def _ensure_daily_j_lt_5_alert(
    raw_config: Any,
    *,
    updated_at: str,
) -> tuple[dict[str, Any], bool]:
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    raw_reminders = config.get("reminders")
    reminders = list(raw_reminders) if isinstance(raw_reminders, list) else []
    canonical = _daily_j_lt_5_reminder()
    for index, reminder in enumerate(reminders):
        if isinstance(reminder, dict) and reminder.get("id") == ALLOTMENT_DAILY_J_REMINDER_ID:
            reminders[index] = canonical
            break
    else:
        if not any(_is_independent_daily_j_lt_5_reminder(item) for item in reminders):
            if len(reminders) >= WATCHLIST_ALERT_MAX_REMINDERS:
                raise ValueError("配债股提醒已达到每股 20 个上限，无法添加日线J < 5提醒")
            reminders.append(canonical)
            added = True
        else:
            added = False
        config.update(
            {"enabled": True, "reminders": reminders, "updated_at": updated_at}
        )
        return config, added
    config.update({"enabled": True, "reminders": reminders, "updated_at": updated_at})
    return config, False


def apply_watchlist_review_plan(
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    watchlist_path: Path = DEFAULT_WATCHLIST_PATH,
    confirm_plan_id: str,
    today: date | None = None,
    data_synchronizer: WatchlistDataSynchronizer | None = None,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    watchlist_path = Path(watchlist_path)
    plan = _read_json_object(plan_path)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("不支持的配债自选池计划版本")
    expected_plan_id = _derive_plan_id(plan)
    if plan.get("plan_id") != expected_plan_id:
        raise ValueError("计划内容与 plan_id 不匹配，拒绝执行")
    if confirm_plan_id != expected_plan_id:
        raise ValueError(
            f"确认码不匹配：请使用 --confirm {expected_plan_id}"
        )
    current_date = today or _now_shanghai().date()
    if plan.get("asof") != current_date.isoformat():
        raise ValueError("计划不是当日生成，请重新运行 plan 后再确认")
    current_sha256 = _sha256_path(watchlist_path)
    if current_sha256 != plan.get("watchlist_sha256"):
        raise ValueError("自选池在生成计划后已变更，请重新运行 plan")

    watchlist = _read_json_object(watchlist_path)
    raw_symbols = watchlist.get("symbols", [])
    if not isinstance(raw_symbols, list):
        raise ValueError("自选池 symbols 必须是列表")
    symbols = [symbol for item in raw_symbols if (symbol := _canonical_symbol(item))]
    pinned = [
        symbol
        for item in watchlist.get("pinned", [])
        if (symbol := _canonical_symbol(item))
    ]
    notes = dict(watchlist.get("notes") or {})
    alerts = dict(watchlist.get("alerts") or {})

    removed = []
    for action in plan.get("removals") or []:
        symbol = _canonical_symbol(action.get("symbol"))
        if not symbol or symbol not in symbols:
            raise ValueError(f"待出池股票已不在自选池: {symbol or action.get('symbol')}")
        symbols = [item for item in symbols if item != symbol]
        pinned = [item for item in pinned if item != symbol]
        notes.pop(symbol, None)
        alerts.pop(symbol, None)
        removed.append(symbol)

    added = []
    marked = []
    actions_to_mark = [
        *(plan.get("additions") or []),
        *(plan.get("mark_as_allotment") or []),
    ]
    actions_to_sync = [
        *actions_to_mark,
        *(plan.get("unchanged") or []),
    ]
    additions = {
        _canonical_symbol(action.get("symbol"))
        for action in plan.get("additions") or []
    }
    mark_symbols = {
        _canonical_symbol(action.get("symbol")) for action in actions_to_mark
    }
    synced = []
    reminders_added = []
    reminders_present = []
    processed_symbols: set[str] = set()
    updated_at = _now_shanghai().replace(tzinfo=None).isoformat(timespec="seconds")
    for action in actions_to_sync:
        symbol = _canonical_symbol(action.get("symbol"))
        if not symbol:
            raise ValueError(f"待入池股票代码无效: {action.get('symbol')}")
        if symbol in processed_symbols:
            continue
        processed_symbols.add(symbol)
        if symbol in additions and symbol in symbols:
            raise ValueError(f"待入池股票已存在: {symbol}")
        if symbol not in symbols:
            if symbol not in additions:
                raise ValueError(f"待同步配债股票已不在自选池: {symbol}")
            symbols.append(symbol)
            added.append(symbol)
        raw_note = notes.get(symbol, {})
        merged = _merge_allotment_note(
            _note_content(raw_note),
            _allotment_note_block(action, str(plan["asof"])),
        )
        notes[symbol] = {
            "content": merged,
            "updated_at": updated_at,
        }
        alerts[symbol], reminder_added = _ensure_daily_j_lt_5_alert(
            alerts.get(symbol),
            updated_at=updated_at,
        )
        synced.append(symbol)
        reminders_present.append(symbol)
        if reminder_added:
            reminders_added.append(symbol)
        if symbol in mark_symbols:
            marked.append(symbol)

    output = dict(watchlist)
    output.update(
        {
            "updated_at": _now_shanghai().replace(tzinfo=None).isoformat(timespec="seconds"),
            "symbols": symbols,
            "pinned": [symbol for symbol in symbols if symbol in set(pinned)],
            "notes": {
                symbol: notes[symbol]
                for symbol in symbols
                if symbol in notes
            },
            "alerts": {
                symbol: alerts[symbol]
                for symbol in symbols
                if symbol in alerts
            },
        }
    )

    stamp = _now_shanghai().strftime("%Y%m%dT%H%M%S")
    backup_path = watchlist_path.with_name(
        f"{watchlist_path.stem}.before_cb_review_{stamp}_{expected_plan_id}.json"
    )
    atomic_link_or_copy(watchlist_path, backup_path)
    atomic_write_json(output, watchlist_path)
    data_sync: dict[str, Any] = {"status": "not_requested"}
    if data_synchronizer is not None:
        try:
            data_sync = data_synchronizer()
        except Exception as exc:
            atomic_link_or_copy(backup_path, watchlist_path)
            raise RuntimeError(
                "自选池评分数据同步失败，已回滚自选池: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return {
        "status": "applied",
        "plan_id": expected_plan_id,
        "watchlist_path": str(watchlist_path.resolve()),
        "backup_path": str(backup_path.resolve()),
        "added": added,
        "removed": removed,
        "marked_as_allotment": marked,
        "information_synced": sorted(synced),
        "daily_j_reminders_added": sorted(reminders_added),
        "daily_j_reminders_present": sorted(reminders_present),
        "data_sync": data_sync,
        "watchlist_count_after": len(symbols),
    }


def _format_action(action: Mapping[str, Any]) -> str:
    symbol = str(action.get("symbol") or "-")
    name = str(action.get("name") or symbol)
    status = str(action.get("status") or action.get("stage") or "-")
    details = []
    if action.get("record_date"):
        details.append(f"登记日 {action['record_date']}")
    if action.get("shares_for_one_lot") is not None:
        details.append(f"约 {int(float(action['shares_for_one_lot']))}股/手债")
    if action.get("rights_value_pct") is not None:
        details.append(f"含权量 {float(action['rights_value_pct']):.2f}%")
    if action.get("reason"):
        details.append(str(action["reason"]))
    suffix = f" · {' · '.join(details)}" if details else ""
    return f"- {symbol} {name} · {status}{suffix}"


def format_review_plan(plan: Mapping[str, Any]) -> str:
    sections = [
        f"配债自选池计划 {plan.get('plan_id')}",
        f"数据截止 {plan.get('asof')} · 当前配债股 {plan.get('counts', {}).get('managed_before', 0)} 只",
        "",
        f"待出池（{len(plan.get('removals') or [])}）",
    ]
    removals = plan.get("removals") or []
    sections.extend(_format_action(item) for item in removals)
    if not removals:
        sections.append("- 无")
    sections.extend(["", f"待入池（{len(plan.get('additions') or [])}）"])
    additions = plan.get("additions") or []
    sections.extend(_format_action(item) for item in additions)
    if not additions:
        sections.append("- 无")
    mark = plan.get("mark_as_allotment") or []
    if mark:
        sections.extend(["", f"已在自选池、待补配债标记（{len(mark)}）"])
        sections.extend(_format_action(item) for item in mark)
    review = plan.get("manual_review") or []
    if review:
        sections.extend(["", f"需人工复核（不自动执行，{len(review)}）"])
        sections.extend(_format_action(item) for item in review)
    sync_count = int(plan.get("counts", {}).get("information_sync") or 0)
    sections.extend(
        [
            "",
            (
                f"确认执行时同步配债阶段、配售参数、价格、KDJ 和公告信息（{sync_count}只），"
                "并确保每只股票都有独立的“日线J < 5”提醒。"
            ),
        ]
    )
    plan_path = str(plan.get("plan_path") or DEFAULT_PLAN_PATH)
    sections.extend(
        [
            "",
            f"计划文件: {plan_path}",
            "确认后执行:",
            (
                "PYTHONPATH=src python scripts/review_convertible_bond_watchlist.py "
                f"apply --plan {plan_path} --confirm {plan.get('plan_id')}"
            ),
        ]
    )
    return "\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="手工生成配债股自选池出入池计划，确认后再执行。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="只生成待确认名单，不修改自选池")
    plan_parser.add_argument("--refresh", action="store_true", help="生成名单前刷新配债事件源")
    plan_parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST_PATH)
    plan_parser.add_argument("--snapshot", type=Path, default=DEFAULT_ALLOTMENT_SNAPSHOT_PATH)
    plan_parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)
    plan_parser.add_argument("--json", action="store_true", help="以 JSON 输出计划")

    apply_parser = subparsers.add_parser("apply", help="使用精确确认码执行已生成的计划")
    apply_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    apply_parser.add_argument("--watchlist", type=Path, default=DEFAULT_WATCHLIST_PATH)
    apply_parser.add_argument("--confirm", required=True, help="plan 输出的完整 plan_id")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    workspace_loader: AllotmentWorkspaceLoader | None = None,
    data_synchronizer: WatchlistDataSynchronizer | None = None,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        raw_argv = ["plan"]
    args = build_parser().parse_args(raw_argv)
    try:
        if args.command == "plan":
            plan = generate_watchlist_review_plan(
                watchlist_path=args.watchlist,
                snapshot_path=args.snapshot,
                output_path=args.output,
                refresh=args.refresh,
                workspace_loader=workspace_loader,
            )
            if args.json:
                print(json.dumps(plan, ensure_ascii=False, indent=2))
            else:
                print(format_review_plan(plan))
            return 0
        result = apply_watchlist_review_plan(
            plan_path=args.plan,
            watchlist_path=args.watchlist,
            confirm_plan_id=args.confirm,
            data_synchronizer=data_synchronizer,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
