import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CASESET_PATH = PROJECT_ROOT / "tests/fixtures/convertible_bond_announcement_cases.json"
ALLOWED_STAGES = {
    "accepted",
    "announced",
    "board_plan",
    "exchange_approved",
    "issuing",
    "listed",
    "registered",
    "shareholder_approved",
    "terminated",
}


def _load_caseset() -> dict[str, Any]:
    payload = json.loads(CASESET_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("配债公告测试集根节点必须是对象")
    return payload


ANNOUNCEMENT_CASESET = _load_caseset()
TITLE_CASES = ANNOUNCEMENT_CASESET["title_cases"]
TIMELINE_CASES = ANNOUNCEMENT_CASESET["timeline_cases"]


def test_convertible_bond_announcement_caseset_is_well_formed():
    assert ANNOUNCEMENT_CASESET["schema_version"] == 1
    assert TITLE_CASES
    assert TIMELINE_CASES

    case_ids = [case["id"] for case in [*TITLE_CASES, *TIMELINE_CASES]]
    assert len(case_ids) == len(set(case_ids))

    for case in TITLE_CASES:
        assert isinstance(case["title"], str)
        assert isinstance(case["is_pipeline"], bool)
        if case["is_pipeline"]:
            assert case["stage"] in ALLOWED_STAGES
            assert case["status"]
        else:
            assert "stage" not in case
            assert "status" not in case

    for case in TIMELINE_CASES:
        assert case["announcements"]
        expected = case["expected"]
        if expected is not None:
            assert expected["stage"] in ALLOWED_STAGES
            assert expected["status"]
            assert expected["announce_date"]
            assert expected["url"]


@pytest.mark.parametrize("case", TITLE_CASES, ids=lambda case: case["id"])
def test_convertible_bond_announcement_title_matches_caseset(case: dict[str, Any]):
    import quant.routine.convertible_bond_allotment as module

    assert module._is_pipeline_issuance_title(case["title"]) is case["is_pipeline"]
    if not case["is_pipeline"]:
        return

    assert module._stage_from_title(case["title"]) == (case["stage"], case["status"])


@pytest.mark.parametrize("case", TIMELINE_CASES, ids=lambda case: case["id"])
def test_convertible_bond_announcement_timeline_matches_caseset(case: dict[str, Any]):
    import quant.routine.convertible_bond_allotment as module

    announcements = pd.DataFrame(
        [
            {
                "代码": case["stock_code"],
                "简称": case["stock_name"],
                "公告标题": announcement["title"],
                "公告时间": announcement["announce_time"],
                "公告链接": announcement["url"],
            }
            for announcement in case["announcements"]
        ]
    )

    pipeline = module._pipeline_from_announcements(announcements)
    expected = case["expected"]

    if expected is None:
        assert pipeline.empty
        return

    assert len(pipeline) == 1
    actual = pipeline.iloc[0]
    assert actual["stock_code"] == case["stock_code"]
    assert actual["stock_name"] == case["stock_name"]
    assert actual["stage"] == expected["stage"]
    assert actual["status"] == expected["status"]
    assert actual["announce_date"] == expected["announce_date"]
    assert actual["announcement_url"] == expected["url"]
