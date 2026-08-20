#!/usr/bin/env python3
"""Sync dashboard-selected companies into the strategy workbench watchlist.

The operation is deliberately idempotent: it preserves existing watchlist order,
pins, notes, and alert reminders, while maintaining one tagged Deep-research note
section and one independent daily KDJ-J < 5 reminder per selected company.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from quant.webapp.services import (
    add_similar_pattern_watch_symbol,
    get_similar_pattern_watchlist,
    save_similar_pattern_watch_alerts,
    save_similar_pattern_watch_note,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = Path("/Users/didi/Downloads/好公司筛选结果_20260811.csv")
DEFAULT_EVALUATIONS = (
    PROJECT_ROOT
    / "reports/good_company_deep_20260809/company_evaluations_deep_final.json"
)
WATCHLIST_PATH = PROJECT_ROOT / "data/research/similar_patterns/watchlist.json"
AUDIT_PATH = (
    PROJECT_ROOT
    / "reports/good_company_deep_20260809/watchlist_sync_20260811.json"
)
NOTE_BEGIN = "【好公司研究台·2026-08-11】"
NOTE_END = "【/好公司研究台】"
ALERT_REMINDER_ID = "gc-daily-j-lt5-20260811"
ALERT_CONDITION_ID = "gc-daily-j-lt5-cond-20260811"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"代码", "公司", "个股MD"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"CSV缺少字段: {sorted(missing)}")
    symbols = [str(row["代码"]).strip().upper() for row in rows]
    if len(symbols) != len(set(symbols)):
        raise ValueError("CSV包含重复股票代码")
    if not all(re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", item) for item in symbols):
        raise ValueError("CSV包含无效股票代码")
    return rows


def _load_evaluations(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("最终Deep评估文件不是公司列表")
    return {
        str(item.get("identity", {}).get("ts_code") or "").upper(): item
        for item in payload
        if isinstance(item, dict)
    }


def _compact(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip("，；。,. ") + "…"


def _percent(value: Any) -> str:
    try:
        return f"{float(value):+.1%}"
    except (TypeError, ValueError):
        return "未量化"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "未量化"


def _build_reason(row: dict[str, str], item: dict[str, Any]) -> str:
    identity = item.get("identity", {})
    gqs = item.get("gqs", {})
    valuation = item.get("valuation", {})
    base = valuation.get("base", {}) if isinstance(valuation, dict) else {}
    market = item.get("market", {})
    research = item.get("research", {})
    pillars = research.get("thesis_pillars", []) or []
    pillar_text = "；".join(
        _compact(value, 92).rstrip("。；") for value in pillars[:2]
    )
    if not pillar_text:
        pillar_text = _compact(research.get("summary"), 180)
    classification = (
        gqs.get("classification_detail")
        or gqs.get("classification")
        or row.get("分类")
        or "待分类"
    )
    report_link = str(row.get("个股MD") or "").strip()
    return "\n".join(
        [
            NOTE_BEGIN,
            (
                f"自选理由：{identity.get('name') or row.get('公司')}为{classification}，"
                f"GQS-F {_number(gqs.get('gqs_f'))}。{pillar_text}"
            ),
            (
                f"中性估值：现价{_number(market.get('current_price'))}元，"
                f"目标{_number(base.get('target_price'))}元，"
                f"价格空间{_percent(base.get('price_upside'))}，"
                f"含息总回报{_percent(base.get('total_return'))}。"
            ),
            f"关键反证：{_compact(research.get('strongest_counterargument'), 180)}",
            f"Deep报告：{report_link}",
            NOTE_END,
        ]
    )


def _merge_note(existing: str, reason: str) -> str:
    content = str(existing or "").strip()
    pattern = re.compile(
        re.escape(NOTE_BEGIN) + r".*?" + re.escape(NOTE_END),
        flags=re.DOTALL,
    )
    if pattern.search(content):
        return pattern.sub(reason, content).strip()
    return f"{content}\n\n{reason}".strip() if content else reason


def _has_daily_j_lt5(alerts: dict[str, Any]) -> bool:
    for reminder in alerts.get("reminders", []) or []:
        for condition in reminder.get("conditions", []) or []:
            try:
                value = float(condition.get("value"))
            except (TypeError, ValueError):
                continue
            if (
                condition.get("kind") == "indicator"
                and condition.get("indicator") == "kdj_daily_j"
                and condition.get("operator") == "lt"
                and abs(value - 5.0) < 1e-9
            ):
                return True
    return False


def _with_daily_j_lt5(alerts: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(alerts or {})
    reminders = list(config.get("reminders", []) or [])
    if not _has_daily_j_lt5(config):
        reminders.append(
            {
                "id": ALERT_REMINDER_ID,
                "note": "好公司研究台统一提醒：日线J＜5",
                "conditions": [
                    {
                        "id": ALERT_CONDITION_ID,
                        "conjunction": "and",
                        "kind": "indicator",
                        "indicator": "kdj_daily_j",
                        "operator": "lt",
                        "value": 5.0,
                    }
                ],
            }
        )
    return {"enabled": True, "reminders": reminders}


def _profiles_by_symbol() -> dict[str, dict[str, Any]]:
    payload = get_similar_pattern_watchlist(include_scores=False)
    return {str(item["symbol"]): item for item in payload.get("stocks", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--evaluations", type=Path, default=DEFAULT_EVALUATIONS)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    rows = _load_csv(args.csv)
    evaluations = _load_evaluations(args.evaluations)
    missing = [row["代码"] for row in rows if row["代码"] not in evaluations]
    if missing:
        raise ValueError(f"下列股票缺少最终Deep评估: {missing}")

    before = _profiles_by_symbol()
    selected = [row["代码"] for row in rows]
    plan: list[dict[str, Any]] = []
    for row in rows:
        symbol = row["代码"]
        profile = before.get(symbol, {})
        current_alerts = profile.get("alerts", {}) or {}
        reason = _build_reason(row, evaluations[symbol])
        merged_note = _merge_note(str(profile.get("note") or ""), reason)
        plan.append(
            {
                "symbol": symbol,
                "name": row["公司"],
                "already_present": symbol in before,
                "note": merged_note,
                "alert_already_present": _has_daily_j_lt5(current_alerts),
                "alerts": _with_daily_j_lt5(current_alerts),
            }
        )

    audit: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry_run",
        "source_csv": str(args.csv),
        "source_evaluations": str(args.evaluations),
        "selected_count": len(selected),
        "existing_selected_count": sum(item["already_present"] for item in plan),
        "new_symbol_count": sum(not item["already_present"] for item in plan),
        "existing_exact_alert_count": sum(item["alert_already_present"] for item in plan),
        "new_alert_count": sum(not item["alert_already_present"] for item in plan),
        "symbols": [
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "already_present": item["already_present"],
                "alert_already_present": item["alert_already_present"],
            }
            for item in plan
        ],
    }
    if not args.apply:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = WATCHLIST_PATH.with_name(f"watchlist.before_good_company_{stamp}.json")
    if WATCHLIST_PATH.exists():
        shutil.copy2(WATCHLIST_PATH, backup_path)

    for item in plan:
        if not item["already_present"]:
            add_similar_pattern_watch_symbol(item["symbol"])
        save_similar_pattern_watch_note(item["symbol"], item["note"])
        if not item["alert_already_present"]:
            save_similar_pattern_watch_alerts(item["symbol"], item["alerts"])

    after = _profiles_by_symbol()
    if not set(before).issubset(after):
        raise RuntimeError("写入后发现原有自选股票丢失")
    failures = []
    for item in plan:
        profile = after.get(item["symbol"])
        if not profile:
            failures.append(f"{item['symbol']}:missing")
            continue
        if NOTE_BEGIN not in str(profile.get("note") or ""):
            failures.append(f"{item['symbol']}:note")
        if not _has_daily_j_lt5(profile.get("alerts", {}) or {}):
            failures.append(f"{item['symbol']}:alert")
    if failures:
        raise RuntimeError(f"写入后验证失败: {failures}")

    audit.update(
        {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "backup_path": str(backup_path) if WATCHLIST_PATH.exists() else None,
            "watchlist_count_before": len(before),
            "watchlist_count_after": len(after),
            "verification": "PASS",
        }
    )
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
