from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant.research.a_share_history import (
    ResearchHistoryError,
    find_baseline_record,
    list_research_records,
    load_research_context,
    load_research_record,
    research_bundle_template,
    save_research_record,
)


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def initial_bundle() -> dict:
    bundle = research_bundle_template("600519.SH", "贵州茅台")
    bundle["analysis_cutoff"] = "2026-04-30T18:00:00+08:00"
    bundle["conclusion"] = {
        "stance": "neutral",
        "confidence": "medium",
        "summary": "需求稳定，但估值安全边际一般。",
    }
    bundle["thesis"]["pillars"] = [
        {
            "id": "P1",
            "statement": "品牌力支持长期现金流质量",
            "status": "active",
            "falsifier": "核心产品批价持续低于阈值",
        }
    ]
    bundle["report_markdown"] = "# 贵州茅台首次研究基线\n\n结论：中性。"
    return bundle


def revision_payload() -> dict:
    return {
        "trigger_summary": "2026年中报发布，收入增速低于原基线。",
        "new_facts": ["2026H1收入同比增速低于原中性情景"],
        "belief_changes": [
            {
                "pillar_id": "P1",
                "before": "品牌力足以维持双位数增长",
                "after": "品牌力仍在，但短期需求弹性低于预期",
                "reason": "新披露经营数据",
                "classification": "new_information",
            }
        ],
        "model_changes": ["下调FY2026收入增速假设"],
        "valuation_changes": ["中性情景盈利基数下调"],
        "mistakes_and_lessons": ["原基线高估了渠道去库存速度"],
        "next_checks": ["跟踪季度合同负债与批价"],
    }


def test_history_saves_immutable_record_and_restores_report(tmp_path: Path) -> None:
    bundle = initial_bundle()

    saved = save_research_record(
        bundle,
        root=tmp_path,
        created_at=datetime(2026, 5, 1, 9, 0, tzinfo=SHANGHAI_TZ),
    )
    loaded = load_research_record(tmp_path, "600519.SH", saved["record_id"])

    assert saved["status"] == "saved"
    assert loaded["report_markdown"].startswith("# 贵州茅台首次研究基线")
    assert loaded["content_sha256"]
    assert (tmp_path / "600519.SH" / "latest.json").is_file()


def test_repeated_save_is_idempotent(tmp_path: Path) -> None:
    bundle = initial_bundle()
    first = save_research_record(bundle, root=tmp_path)
    second = save_research_record(bundle, root=tmp_path)

    assert second["status"] == "existing"
    assert second["record_id"] == first["record_id"]
    assert len(list_research_records(tmp_path, "600519.SH")) == 1


def test_update_auto_attaches_prior_baseline_and_keeps_change_ledger(
    tmp_path: Path,
) -> None:
    first = save_research_record(initial_bundle(), root=tmp_path)
    update = deepcopy(initial_bundle())
    update["analysis_cutoff"] = "2026-08-31T18:00:00+08:00"
    update["mode"] = "financial_update"
    update["trigger"] = {
        "type": "financial_report",
        "summary": "2026年中报更新",
        "source_refs": ["2026H1"],
    }
    update["revision"] = revision_payload()
    update["report_markdown"] = "# 贵州茅台2026年中报认知更新"

    second = save_research_record(update, root=tmp_path)
    loaded = load_research_record(tmp_path, "600519.SH", second["record_id"])
    baseline = find_baseline_record(
        tmp_path,
        "600519.SH",
        "2026-08-31T18:00:00+08:00",
    )

    assert second["baseline_record_id"] == first["record_id"]
    assert loaded["baseline_record_id"] == first["record_id"]
    assert loaded["revision"]["belief_changes"][0]["pillar_id"] == "P1"
    assert baseline is not None
    assert baseline["record_id"] == first["record_id"]
    assert list_research_records(tmp_path, "600519.SH")[0]["record_id"] == second["record_id"]


def test_repeated_full_coverage_cannot_ignore_existing_baseline(
    tmp_path: Path,
) -> None:
    first = save_research_record(initial_bundle(), root=tmp_path)
    repeated = deepcopy(initial_bundle())
    repeated["analysis_cutoff"] = "2026-05-31T18:00:00+08:00"
    repeated["report_markdown"] = "# 贵州茅台再次完整分析"

    with pytest.raises(ResearchHistoryError, match="revision object"):
        save_research_record(repeated, root=tmp_path)

    repeated["revision"] = revision_payload()
    saved = save_research_record(repeated, root=tmp_path)
    loaded = load_research_record(tmp_path, "600519.SH", saved["record_id"])

    assert saved["baseline_record_id"] == first["record_id"]
    assert loaded["revision"]["belief_changes"][0]["before"]
    assert loaded["revision"]["belief_changes"][0]["after"]


def test_context_loads_full_coverage_anchor_and_all_follow_ups(
    tmp_path: Path,
) -> None:
    first = save_research_record(initial_bundle(), root=tmp_path)
    update = deepcopy(initial_bundle())
    update["analysis_cutoff"] = "2026-08-31T18:00:00+08:00"
    update["mode"] = "financial_update"
    update["trigger"] = {
        "type": "financial_report",
        "summary": "2026年中报更新",
        "source_refs": ["2026H1"],
    }
    update["revision"] = revision_payload()
    update["report_markdown"] = "# 贵州茅台2026年中报认知更新"
    second = save_research_record(update, root=tmp_path)

    follow_up = deepcopy(update)
    follow_up["analysis_cutoff"] = "2026-09-30T18:00:00+08:00"
    follow_up["trigger"] = {
        "type": "scheduled_review",
        "summary": "月度跟踪",
        "source_refs": [],
    }
    follow_up["report_markdown"] = "# 贵州茅台月度跟踪"
    third = save_research_record(follow_up, root=tmp_path)

    context = load_research_context(
        tmp_path,
        "600519.SH",
        "2026-10-31T18:00:00+08:00",
    )

    assert context["anchor_record_id"] == first["record_id"]
    assert context["anchor_is_full_coverage"] is True
    assert context["chain_complete"] is True
    assert context["chain_warnings"] == []
    assert context["baseline_record_id"] == third["record_id"]
    assert [record["record_id"] for record in context["records"]] == [
        first["record_id"],
        second["record_id"],
        third["record_id"],
    ]
    assert all(record["report_markdown"].startswith("#") for record in context["records"])


def test_context_is_strictly_point_in_time(tmp_path: Path) -> None:
    first = save_research_record(initial_bundle(), root=tmp_path)
    update = deepcopy(initial_bundle())
    update["analysis_cutoff"] = "2026-08-31T18:00:00+08:00"
    update["mode"] = "financial_update"
    update["trigger"] = {
        "type": "financial_report",
        "summary": "2026年中报更新",
        "source_refs": ["2026H1"],
    }
    update["revision"] = revision_payload()
    update["report_markdown"] = "# 贵州茅台2026年中报认知更新"
    save_research_record(update, root=tmp_path)

    context = load_research_context(
        tmp_path,
        "600519.SH",
        "2026-08-31T18:00:00+08:00",
    )

    assert context["record_count"] == 1
    assert context["baseline_record_id"] == first["record_id"]


def test_context_follows_explicit_baseline_chain_and_reports_missing_full_coverage(
    tmp_path: Path,
) -> None:
    first = initial_bundle()
    first["mode"] = "legacy_follow_up"
    first["trigger"]["summary"] = "迁移的窄主题记录"
    saved_first = save_research_record(first, root=tmp_path)

    second = deepcopy(first)
    second["analysis_cutoff"] = "2026-05-31T18:00:00+08:00"
    second["trigger"] = {
        "type": "scheduled_review",
        "summary": "后续窄主题跟踪",
        "source_refs": [],
    }
    second["revision"] = revision_payload()
    second["report_markdown"] = "# 后续窄主题跟踪"
    saved_second = save_research_record(second, root=tmp_path)

    context = load_research_context(
        tmp_path,
        "600519.SH",
        "2026-06-30T18:00:00+08:00",
    )

    assert context["anchor_record_id"] == saved_first["record_id"]
    assert context["baseline_record_id"] == saved_second["record_id"]
    assert context["anchor_is_full_coverage"] is False
    assert context["chain_complete"] is True


def test_updated_research_requires_explicit_revision_ledger(tmp_path: Path) -> None:
    update = initial_bundle()
    update["trigger"] = {
        "type": "company_event",
        "summary": "重大合同",
        "source_refs": [],
    }

    with pytest.raises(ResearchHistoryError, match="revision object"):
        save_research_record(update, root=tmp_path)


def test_update_rejects_unclassified_belief_change(tmp_path: Path) -> None:
    update = initial_bundle()
    update["analysis_cutoff"] = "2026-05-31T18:00:00+08:00"
    update["trigger"] = {
        "type": "company_event",
        "summary": "重大合同",
        "source_refs": [],
    }
    update["revision"] = revision_payload()
    update["revision"]["belief_changes"][0]["classification"] = "hindsight"

    with pytest.raises(ResearchHistoryError, match="classification"):
        save_research_record(update, root=tmp_path)


def test_update_validates_optional_legacy_pillar_id_mapping(tmp_path: Path) -> None:
    first = save_research_record(initial_bundle(), root=tmp_path)
    update = deepcopy(initial_bundle())
    update["analysis_cutoff"] = "2026-05-31T18:00:00+08:00"
    update["trigger"] = {
        "type": "scheduled_review",
        "summary": "迁移记录论点映射",
        "source_refs": [],
    }
    update["revision"] = revision_payload()
    update["revision"]["pillar_id_mappings"] = [
        {
            "old_record_id": first["record_id"],
            "old_pillar_id": "P1",
            "old_statement": "品牌力支持长期现金流质量",
            "new_pillar_id": "P1",
            "reason": "命题语义一致，沿用稳定 ID",
        }
    ]
    update["report_markdown"] = "# 迁移记录论点映射"

    saved = save_research_record(update, root=tmp_path)
    loaded = load_research_record(tmp_path, "600519.SH", saved["record_id"])

    assert loaded["revision"]["pillar_id_mappings"][0]["old_record_id"] == first["record_id"]

    invalid = deepcopy(update)
    invalid["analysis_cutoff"] = "2026-06-30T18:00:00+08:00"
    del invalid["revision"]["pillar_id_mappings"][0]["reason"]

    with pytest.raises(ResearchHistoryError, match="reason"):
        save_research_record(invalid, root=tmp_path)


def test_baseline_must_be_strictly_before_new_cutoff(tmp_path: Path) -> None:
    saved = save_research_record(initial_bundle(), root=tmp_path)
    invalid = initial_bundle()
    invalid["baseline_record_id"] = saved["record_id"]
    invalid["trigger"] = {
        "type": "scheduled_review",
        "summary": "同一时点重复复盘",
        "source_refs": [],
    }
    invalid["revision"] = revision_payload()
    invalid["report_markdown"] = "# 同时点复盘"

    with pytest.raises(ResearchHistoryError, match="before analysis_cutoff"):
        save_research_record(invalid, root=tmp_path, auto_baseline=False)
