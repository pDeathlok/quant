from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quant.routine import convertible_bond_watchlist_review as review_module
from quant.routine.convertible_bond_watchlist_review import (
    ALLOTMENT_NOTE_BEGIN,
    _derive_plan_id,
    _sha256_path,
    apply_watchlist_review_plan,
    audit_allotment_result_consistency,
    build_watchlist_review_plan,
    generate_watchlist_review_plan,
)


ASOF = date(2026, 8, 26)


def _record(
    stock_code: str,
    name: str,
    stage: str,
    *,
    record_date: str | None = None,
) -> dict:
    statuses = {
        "registered": "同意注册",
        "exchange_approved": "上市委通过",
        "issuing": "发行公告",
    }
    return {
        "stock_code": stock_code,
        "stock_name": name,
        "stage": stage,
        "status": statuses.get(stage, stage),
        "announce_date": "2026-08-20",
        "record_date": record_date,
        "stock_price": 20.0,
        "stock_price_date": "2026-08-26",
        "kdj_daily_j": 4.0,
        "kdj_weekly_j": 12.0,
        "kdj_monthly_j": 20.0,
        "shares_for_one_lot": 500,
        "rights_value_pct": 10.0,
        "announcement_url": f"https://example.test/{stock_code}",
    }


def _watchlist_payload() -> dict:
    return {
        "updated_at": "2026-08-26T10:00:00",
        "symbols": ["300001.SZ", "300003.SZ", "300004.SZ", "600000.SH"],
        "pinned": ["300004.SZ", "600000.SH"],
        "notes": {
            "300001.SZ": {"content": "8.20 配债股 · 同意注册", "updated_at": ""},
            "300003.SZ": {"content": "配债", "updated_at": ""},
            "300004.SZ": {
                "content": "一手股数 500股 · 含权量 10.00%",
                "updated_at": "",
            },
            "600000.SH": {"content": "长线观察", "updated_at": ""},
        },
        "alerts": {
            "300001.SZ": {
                "enabled": True,
                "reminders": [
                    {
                        "id": "existing-daily-j",
                        "note": "已有日线J提醒",
                        "conditions": [
                            {
                                "id": "existing-daily-j-condition",
                                "conjunction": "and",
                                "kind": "indicator",
                                "operator": "lt",
                                "value": 5.0,
                                "indicator": "kdj_daily_j",
                            }
                        ],
                    }
                ],
            },
            "300004.SZ": {"enabled": True, "reminders": [{"id": "remove-me"}]},
            "600000.SH": {
                "enabled": True,
                "reminders": [
                    {
                        "id": "keep-me",
                        "note": "价格和J值同时满足",
                        "conditions": [
                            {
                                "id": "price-condition",
                                "conjunction": "and",
                                "kind": "price",
                                "operator": "lt",
                                "value": 18.0,
                            },
                            {
                                "id": "combined-daily-j-condition",
                                "conjunction": "and",
                                "kind": "indicator",
                                "operator": "lt",
                                "value": 5.0,
                                "indicator": "kdj_daily_j",
                            },
                        ],
                    }
                ],
            },
        },
    }


def _records() -> list[dict]:
    return [
        _record("300001", "保留股份", "registered"),
        _record("300002", "新增股份", "registered"),
        _record("300004", "过期股份", "issuing", record_date="2026-08-25"),
        _record("600000", "已在池股份", "exchange_approved"),
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_plan(watchlist_path: Path) -> dict:
    watchlist = json.loads(watchlist_path.read_text(encoding="utf-8"))
    return build_watchlist_review_plan(
        records=_records(),
        watchlist_payload=watchlist,
        watchlist_sha256=_sha256_path(watchlist_path),
        asof=ASOF,
        source={"event_poll_status": "success", "event_polled_through": "2026-08-26"},
    )


def test_review_plan_separates_add_remove_mark_and_manual_review(tmp_path: Path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    _write_json(watchlist_path, _watchlist_payload())

    plan = _build_plan(watchlist_path)

    assert [item["symbol"] for item in plan["additions"]] == ["300002.SZ"]
    assert [item["symbol"] for item in plan["removals"]] == ["300004.SZ"]
    assert plan["removals"][0]["reason"] == "股权登记日 2026-08-25 已过"
    assert [item["symbol"] for item in plan["mark_as_allotment"]] == ["600000.SH"]
    assert [item["symbol"] for item in plan["manual_review"]] == ["300003.SZ"]
    assert [item["symbol"] for item in plan["unchanged"]] == ["300001.SZ"]
    assert plan["plan_id"] == _derive_plan_id(plan)


def test_result_consistency_compares_daily_and_review_announcement_evidence() -> None:
    daily = _record("300002", "新增股份", "registered")
    daily["announcement_title"] = "关于向不特定对象发行可转换公司债券注册稿"
    review = dict(daily)
    review["announcement_title"] = "关于同意注册的批复"

    result = audit_allotment_result_consistency(
        daily_records=[daily],
        review_records=[review],
        asof=ASOF,
    )

    assert result["status"] == "failed"
    assert result["daily_count"] == result["review_count"] == 1
    assert result["field_mismatches"] == [
        {
            "symbol": "300002.SZ",
            "differences": {
                "announcement_title": {
                    "daily": "关于向不特定对象发行可转换公司债券注册稿",
                    "review": "关于同意注册的批复",
                }
            },
        }
    ]


def test_refresh_plan_reuses_daily_update_then_builds_review_from_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    output_path = tmp_path / "plan.json"
    _write_json(watchlist_path, {"symbols": [], "notes": {}})
    record = _record("300002", "新增股份", "registered")
    record["announcement_title"] = "关于向不特定对象发行可转换公司债券注册稿"
    workspace_calls: list[dict] = []
    raw_calls: list[dict] = []

    def workspace_loader(**kwargs):
        workspace_calls.append(kwargs)
        return {
            "generated_at": "2026-08-26T09:00:00",
            "event_poll_status": "success",
            "event_polled_through": "2026-08-26",
            "event_poll_sources": {"pipeline": "success"},
            "records": [record],
        }

    def cached_review_builder(**kwargs):
        raw_calls.append(kwargs)
        return {"records": [dict(record)]}

    monkeypatch.setattr(
        review_module,
        "build_convertible_bond_allotment_payload",
        cached_review_builder,
    )

    plan = generate_watchlist_review_plan(
        watchlist_path=watchlist_path,
        output_path=output_path,
        refresh=True,
        today=ASOF,
        workspace_loader=workspace_loader,
    )

    assert workspace_calls == [
        {
            "refresh": True,
            "stage_scope": "pipeline",
            "validate_quality": True,
        }
    ]
    assert raw_calls == [
        {
            "limit": 500,
            "refresh": False,
            "stage_scope": "pipeline",
            "include_expired_record_dates": True,
            "today": ASOF,
        }
    ]
    assert plan["source"]["mode"] == "shared_daily_update_refresh"
    assert plan["source"]["result_consistency"]["status"] == "success"
    assert [item["symbol"] for item in plan["additions"]] == ["300002.SZ"]
    assert output_path.is_file()


def test_refresh_plan_rejects_daily_and_review_result_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    output_path = tmp_path / "plan.json"
    _write_json(watchlist_path, {"symbols": [], "notes": {}})
    daily_record = _record("300002", "新增股份", "registered")

    monkeypatch.setattr(
        review_module,
        "build_convertible_bond_allotment_payload",
        lambda **kwargs: {"records": []},
    )

    with pytest.raises(ValueError, match="每日更新与配债复核结果不一致"):
        generate_watchlist_review_plan(
            watchlist_path=watchlist_path,
            output_path=output_path,
            refresh=True,
            today=ASOF,
            workspace_loader=lambda **kwargs: {
                "event_poll_status": "success",
                "event_polled_through": "2026-08-26",
                "records": [daily_record],
            },
        )

    assert not output_path.exists()


def test_apply_requires_exact_confirmation_and_unchanged_watchlist(tmp_path: Path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    plan_path = tmp_path / "plan.json"
    _write_json(watchlist_path, _watchlist_payload())
    plan = _build_plan(watchlist_path)
    _write_json(plan_path, plan)

    with pytest.raises(ValueError, match="确认码不匹配"):
        apply_watchlist_review_plan(
            plan_path=plan_path,
            watchlist_path=watchlist_path,
            confirm_plan_id="wrong",
            today=ASOF,
        )

    changed = _watchlist_payload()
    changed["symbols"].append("600001.SH")
    _write_json(watchlist_path, changed)
    with pytest.raises(ValueError, match="自选池在生成计划后已变更"):
        apply_watchlist_review_plan(
            plan_path=plan_path,
            watchlist_path=watchlist_path,
            confirm_plan_id=plan["plan_id"],
            today=ASOF,
        )


def test_apply_updates_only_confirmed_actions_and_creates_backup(tmp_path: Path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    plan_path = tmp_path / "plan.json"
    original = _watchlist_payload()
    _write_json(watchlist_path, original)
    plan = _build_plan(watchlist_path)
    _write_json(plan_path, plan)

    sync_calls = []

    def sync_watchlist_data():
        saved = json.loads(watchlist_path.read_text(encoding="utf-8"))
        sync_calls.append(saved["symbols"])
        return {"status": "success", "scored_count": len(saved["symbols"])}

    result = apply_watchlist_review_plan(
        plan_path=plan_path,
        watchlist_path=watchlist_path,
        confirm_plan_id=plan["plan_id"],
        today=ASOF,
        data_synchronizer=sync_watchlist_data,
    )

    saved = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert result["added"] == ["300002.SZ"]
    assert result["removed"] == ["300004.SZ"]
    assert "300002.SZ" in saved["symbols"]
    assert "300004.SZ" not in saved["symbols"]
    assert "300004.SZ" not in saved["pinned"]
    assert "300004.SZ" not in saved["notes"]
    assert "300004.SZ" not in saved["alerts"]
    assert saved["notes"]["600000.SH"]["content"].startswith("长线观察")
    assert ALLOTMENT_NOTE_BEGIN in saved["notes"]["600000.SH"]["content"]
    assert "现价 20.00（2026-08-26）" in saved["notes"]["600000.SH"]["content"]
    assert "KDJ 日J 4.00 · 周J 12.00 · 月J 20.00" in saved["notes"]["600000.SH"]["content"]
    assert ALLOTMENT_NOTE_BEGIN in saved["notes"]["300001.SZ"]["content"]
    assert len(saved["alerts"]["300001.SZ"]["reminders"]) == 1
    assert saved["alerts"]["300001.SZ"]["reminders"][0]["id"] == "existing-daily-j"
    assert (
        saved["alerts"]["600000.SH"]["reminders"][0]
        == original["alerts"]["600000.SH"]["reminders"][0]
    )
    assert saved["alerts"]["600000.SH"]["reminders"][1]["id"] == "cb-allotment-daily-j-lt-5"
    assert saved["alerts"]["300002.SZ"]["reminders"][0]["conditions"] == [
        {
            "id": "cb-allotment-daily-j-lt-5-condition",
            "conjunction": "and",
            "kind": "indicator",
            "operator": "lt",
            "value": 5.0,
            "indicator": "kdj_daily_j",
        }
    ]
    assert result["information_synced"] == ["300001.SZ", "300002.SZ", "600000.SH"]
    assert result["daily_j_reminders_added"] == ["300002.SZ", "600000.SH"]
    assert result["daily_j_reminders_present"] == [
        "300001.SZ",
        "300002.SZ",
        "600000.SH",
    ]
    assert result["data_sync"] == {"status": "success", "scored_count": 4}
    assert sync_calls == [saved["symbols"]]

    backup_path = Path(result["backup_path"])
    assert backup_path.is_file()
    backup = json.loads(backup_path.read_text(encoding="utf-8"))
    assert backup == original


def test_apply_rolls_back_watchlist_when_data_sync_fails(tmp_path: Path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    plan_path = tmp_path / "plan.json"
    original = _watchlist_payload()
    _write_json(watchlist_path, original)
    original_bytes = watchlist_path.read_bytes()
    plan = _build_plan(watchlist_path)
    _write_json(plan_path, plan)

    with pytest.raises(RuntimeError, match="已回滚自选池"):
        apply_watchlist_review_plan(
            plan_path=plan_path,
            watchlist_path=watchlist_path,
            confirm_plan_id=plan["plan_id"],
            today=ASOF,
            data_synchronizer=lambda: (_ for _ in ()).throw(
                RuntimeError("score refresh failed")
            ),
        )

    assert json.loads(watchlist_path.read_text(encoding="utf-8")) == original
    assert watchlist_path.read_bytes() == original_bytes
