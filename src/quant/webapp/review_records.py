"""Durable, immutable entry snapshots with separately editable review notes."""

from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from quant.data.atomic_io import atomic_write_json
from quant.routine.paths import WEB_DATA_DIR

REVIEW_RECORDS_PATH = WEB_DATA_DIR / "review_records.json"


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


@contextmanager
def _locked_records():
    REVIEW_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_RECORDS_PATH.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def get_review_records() -> dict[str, Any]:
    if not REVIEW_RECORDS_PATH.exists():
        return {"records": []}
    # Fail visibly on corrupt storage instead of silently replacing saved history.
    payload = json.loads(REVIEW_RECORDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("复盘记录文件格式错误")
    return payload


def add_review_record(data: dict[str, Any]) -> dict[str, Any]:
    symbol = str(data["symbol"]).strip().upper()
    match = re.fullmatch(r"(\d{6})(?:\.(SH|SZ|BJ))?", symbol)
    if not match:
        raise ValueError("请输入六位股票代码，可带 .SH、.SZ 或 .BJ 后缀")
    code, market = match.groups()
    market = market or ("BJ" if code.startswith(("920", "4", "8")) else "SH" if code.startswith(("6", "9")) else "SZ")
    now = _now()
    record = {
        **data,
        "id": uuid4().hex,
        "symbol": f"{code}.{market}",
        "name": str(data.get("name") or "").strip(),
        "xueqiu_url": f"https://xueqiu.com/S/{market}{code}",
        "created_at": now,
        "updated_at": now,
    }
    with _locked_records():
        payload = get_review_records()
        payload["records"].insert(0, record)
        atomic_write_json(payload, REVIEW_RECORDS_PATH)
    return {"record": record}


def update_review_note(record_id: str, note: str) -> dict[str, Any]:
    with _locked_records():
        payload = get_review_records()
        record = next((row for row in payload["records"] if row["id"] == record_id), None)
        if record is None:
            raise KeyError(record_id)
        record["note"] = note
        record["updated_at"] = _now()
        atomic_write_json(payload, REVIEW_RECORDS_PATH)
    return {"record": record}


def delete_review_record(record_id: str) -> dict[str, bool]:
    with _locked_records():
        payload = get_review_records()
        remaining = [row for row in payload["records"] if row["id"] != record_id]
        if len(remaining) == len(payload["records"]):
            raise KeyError(record_id)
        payload["records"] = remaining
        atomic_write_json(payload, REVIEW_RECORDS_PATH)
    return {"deleted": True}
