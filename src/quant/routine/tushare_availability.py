from __future__ import annotations

import re
from datetime import datetime, time
from zoneinfo import ZoneInfo


MARKET_TIMEZONE = ZoneInfo("Asia/Shanghai")
RETRY_INTERVAL_SECONDS = 10 * 60.0
RETRY_CUTOFF = time(17, 20)


def market_now() -> datetime:
    return datetime.now(MARKET_TIMEZONE)


def tushare_retry_deadline(trade_date: str, now: datetime) -> datetime | None:
    """Only today's publication gets a time-based wait; history keeps its budget."""
    market_date = now.astimezone(MARKET_TIMEZONE).date()
    if trade_date.replace("-", "") != market_date.strftime("%Y%m%d"):
        return None
    return datetime.combine(market_date, RETRY_CUTOFF, tzinfo=MARKET_TIMEZONE)


def tushare_retry_delay(deadline: datetime, now: datetime) -> float | None:
    remaining = (deadline - now.astimezone(MARKET_TIMEZONE)).total_seconds()
    return min(RETRY_INTERVAL_SECONDS, remaining) if remaining > 0 else None


def is_tushare_data_missing(error: str) -> bool:
    text = error.lower()
    if "tushare" not in text:
        return False
    markers = (
        "daily_basic model feature coverage below threshold",
        "daily_basic missing model feature columns",
        "daily_basic missing required columns",
        "daily returned no market rows",
        "daily market coverage below",
        "daily market response missing required columns",
        "daily did not reach expected trade date",
        "stock_basic returned no symbols",
        "stock_basic returned no rows",
    )
    return any(marker in text for marker in markers) or bool(
        re.search(r"tushare daily_basic returned \d+ rows .*minimum is", text)
    )
