from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.data.long_factor_backfill import (
    RequestPolicy,
    backfill_stock_universe,
    backfill_trade_date_partitions,
    fetch_complete_date_range,
    fetch_complete_pages,
    merge_deduplicated,
)


class FakePro:
    def trade_cal(self, **kwargs):
        return pd.DataFrame({"cal_date": ["20240102", "20240103"], "is_open": ["1", "1"]})

    def moneyflow(self, *, trade_date: str):
        return pd.DataFrame(
            {
                "trade_date": [trade_date],
                "ts_code": ["000001.SZ"],
                "net_mf_amount": [1.0],
            }
        )

    def stock_basic(self, *, list_status: str, **kwargs):
        if list_status not in {"L", "D"}:
            return pd.DataFrame(columns=["ts_code", "list_status"])
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ" if list_status == "L" else "000002.SZ"],
                "list_status": [list_status],
                "list_date": ["19910101"],
                "delist_date": [None if list_status == "L" else "20200101"],
            }
        )


def test_merge_deduplicated_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    frame = pd.DataFrame({"id": [1, 1], "date": ["20240101", "20240101"], "value": [1, 2]})

    first = merge_deduplicated(path, frame, dedupe=("id", "date"), sort=("date", "id"))
    second = merge_deduplicated(path, frame, dedupe=("id", "date"), sort=("date", "id"))

    assert len(first) == len(second) == 1
    assert second["value"].iloc[0] == 2


def test_trade_date_backfill_resumes_completed_partitions(tmp_path: Path) -> None:
    pro = FakePro()
    audit = tmp_path / "audit"
    audit.mkdir()

    first = backfill_trade_date_partitions(
        pro,
        "moneyflow",
        "20240101",
        "20240103",
        tmp_path / "raw",
        audit,
        policy=RequestPolicy(sleep_seconds=0),
    )
    second = backfill_trade_date_partitions(
        pro,
        "moneyflow",
        "20240101",
        "20240103",
        tmp_path / "raw",
        audit,
        policy=RequestPolicy(sleep_seconds=0),
    )

    assert first["success"] == 2
    assert second["requested"] == 0
    assert second["already_complete"] == 2


def test_date_range_fetch_splits_provider_limit() -> None:
    calls: list[tuple[str, str]] = []

    def operation(start: str, end: str) -> pd.DataFrame:
        calls.append((start, end))
        if start != end:
            return pd.DataFrame({"value": range(3)})
        return pd.DataFrame({"value": [1]})

    frame, audit = fetch_complete_date_range(
        operation,
        "20240101",
        "20240102",
        provider_limit=3,
        policy=RequestPolicy(sleep_seconds=0),
    )

    assert len(frame) == 2
    assert calls == [
        ("20240101", "20240102"),
        ("20240101", "20240101"),
        ("20240102", "20240102"),
    ]
    assert [row["status"] for row in audit] == ["split", "success", "success"]


def test_page_fetch_continues_past_provider_limit() -> None:
    calls: list[tuple[int, int]] = []

    def operation(offset: int, limit: int) -> pd.DataFrame:
        calls.append((offset, limit))
        remaining = max(0, 4 - offset)
        return pd.DataFrame({"value": range(offset, offset + min(limit, remaining))})

    frame, audit = fetch_complete_pages(
        operation,
        page_size=2,
        policy=RequestPolicy(sleep_seconds=0),
    )

    assert len(frame) == 4
    assert calls == [(0, 2), (2, 2), (4, 2)]
    assert [row["rows"] for row in audit] == [2, 2, 0]


def test_page_fetch_stops_on_provider_duplicate_tail() -> None:
    calls: list[int] = []
    repeated = pd.DataFrame({"value": [1, 2]})

    def operation(offset: int, limit: int) -> pd.DataFrame:
        calls.append(offset)
        return repeated.copy()

    frame, audit = fetch_complete_pages(
        operation,
        page_size=2,
        policy=RequestPolicy(sleep_seconds=0),
    )

    assert frame["value"].tolist() == [1, 2]
    assert calls == [0, 2]
    assert audit[-1]["status"] == "duplicate_tail"


def test_stock_universe_keeps_delisted_rows(tmp_path: Path) -> None:
    frame, result = backfill_stock_universe(
        FakePro(),
        tmp_path,
        policy=RequestPolicy(sleep_seconds=0),
    )

    assert result["status"] == "success"
    assert result["active"] == 1
    assert result["delisted"] == 1
    assert set(frame["list_status"]) == {"L", "D"}
