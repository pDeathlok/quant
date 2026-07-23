"""Build point-in-time OHLCV inputs for the local A-share analysis skill.

The adapter deliberately reads the project's canonical market-data layer
instead of creating a second Tushare client or configuration path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from quant.data.atomic_io import atomic_write_json
from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import normalize_ts_code
from quant.routine.paths import PROJECT_ROOT, load_project_env


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
MINIMUM_BARS = 60
DEFAULT_BAR_COUNT = 250
DAILY_AVAILABLE_TIME = time(16, 0)


class SkillMarketDataError(ValueError):
    """Raised when canonical market data cannot satisfy the skill contract."""


def parse_analysis_cutoff(value: str | datetime) -> datetime:
    """Parse an aware cutoff and normalize it to Asia/Shanghai."""

    if isinstance(value, datetime):
        cutoff = value
    else:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            cutoff = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise SkillMarketDataError(
                "analysis_cutoff must be an ISO 8601 timestamp with timezone"
            ) from exc
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise SkillMarketDataError("analysis_cutoff must include an explicit UTC offset")
    return cutoff.astimezone(SHANGHAI_TZ)


def normalize_a_share_ticker(value: str) -> str:
    """Normalize and validate a Shanghai, Shenzhen, or Beijing A-share code."""

    ticker = normalize_ts_code(value).upper()
    if (
        len(ticker) != 9
        or ticker[6] != "."
        or not ticker[:6].isdigit()
        or ticker[7:] not in {"SH", "SZ", "BJ"}
    ):
        raise SkillMarketDataError(
            "ticker must resolve to a six-digit A-share code such as "
            "600519.SH, 000001.SZ, or 920000.BJ"
        )
    return ticker


def configured_market_store() -> MarketDataStore:
    """Create the canonical store after loading the project's existing .env."""

    load_project_env()
    config = MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data/raw")
    return MarketDataStore(config)


def _available_at(trade_date: pd.Timestamp) -> datetime:
    return datetime.combine(trade_date.date(), DAILY_AVAILABLE_TIME, tzinfo=SHANGHAI_TZ)


def _normalize_daily_frame(
    frame: pd.DataFrame,
    *,
    ticker: str,
    cutoff: datetime,
    bar_count: int,
) -> tuple[pd.DataFrame, list[str]]:
    if frame is None or frame.empty:
        raise SkillMarketDataError(f"no canonical daily rows found for {ticker}")

    required = {"trade_date", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SkillMarketDataError(f"canonical daily rows are missing columns: {missing}")
    if "vol" not in frame.columns and "volume" not in frame.columns:
        raise SkillMarketDataError("canonical daily rows are missing both vol and volume")

    out = frame.copy()
    if "ts_code" in out.columns:
        out = out[out["ts_code"].astype(str).str.upper().eq(ticker)]
    out["trade_date"] = pd.to_datetime(
        out["trade_date"].astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d",
        errors="coerce",
    )
    out = out.dropna(subset=["trade_date"])
    out = (
        out.sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .reset_index(drop=True)
    )
    out["available_at"] = out["trade_date"].map(_available_at)
    out = out[out["available_at"].map(lambda value: value <= cutoff)].tail(bar_count).copy()

    if len(out) < MINIMUM_BARS:
        raise SkillMarketDataError(
            f"{ticker} has only {len(out)} complete bars available at "
            f"{cutoff.isoformat()}; at least {MINIMUM_BARS} are required"
        )

    out["skill_volume"] = out["vol"] if "vol" in out.columns else out["volume"]
    if "volume" in out.columns:
        out["skill_volume"] = out["skill_volume"].combine_first(out["volume"])

    numeric_columns = ["open", "high", "low", "close", "skill_volume"]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[numeric_columns].isna().any().any():
        bad_columns = [
            column for column in numeric_columns if bool(out[column].isna().any())
        ]
        raise SkillMarketDataError(
            f"canonical daily rows contain non-numeric or missing values: {bad_columns}"
        )

    invalid_price = (
        (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (out["high"] < out[["open", "close", "low"]].max(axis=1))
        | (out["low"] > out[["open", "close", "high"]].min(axis=1))
    )
    if bool(invalid_price.any()):
        dates = out.loc[invalid_price, "trade_date"].dt.strftime("%Y-%m-%d").tolist()
        raise SkillMarketDataError(f"canonical daily rows violate OHLC invariants: {dates[:5]}")
    if bool((out["skill_volume"] < 0).any()):
        raise SkillMarketDataError("canonical daily rows contain negative volume")

    warnings: list[str] = []
    if len(out) < 120:
        warnings.append("少于120根日线，不支持完整 MA120 长期结构。")
    zero_volume_dates = out.loc[
        out["skill_volume"].eq(0), "trade_date"
    ].dt.strftime("%Y-%m-%d").tolist()
    if zero_volume_dates:
        warnings.append(
            "序列包含零成交量日期，需结合停牌或交易状态核验："
            + ", ".join(zero_volume_dates[-5:])
        )
    return out, warnings


def _payload_from_frame(
    frame: pd.DataFrame,
    *,
    ticker: str,
    cutoff: datetime,
    source: str,
    configured_backend: str,
    mirror_parquet: bool,
    accessed_at: datetime,
    warnings: list[str],
) -> dict[str, Any]:
    last_available_at = frame.iloc[-1]["available_at"]
    bars = [
        {
            "date": row.trade_date.date().isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.skill_volume),
        }
        for row in frame.itertuples(index=False)
    ]
    for bar in bars:
        if not all(
            math.isfinite(float(bar[field]))
            for field in ("open", "high", "low", "close", "volume")
        ):
            raise SkillMarketDataError("output bars contain non-finite values")

    return {
        "ticker": ticker,
        "as_of_date": cutoff.date().isoformat(),
        "analysis_cutoff": cutoff.isoformat(timespec="seconds"),
        "last_bar_available_at": last_available_at.isoformat(timespec="seconds"),
        "source": source,
        "price_basis": "unadjusted",
        "volume_unit": "lots (手)",
        "bars": bars,
        "provenance": {
            "provider": "Tushare",
            "interface": "daily" if "index" not in source.lower() else "index_daily",
            "configured_storage_backend": configured_backend,
            "mirror_parquet": mirror_parquet,
            "accessed_at": accessed_at.isoformat(timespec="seconds"),
            "publication_rule": "完整日线按交易日 16:00:00+08:00 保守视为可得",
            "requested_bar_count": len(bars),
        },
        "data_warnings": warnings,
    }


def build_stock_price_volume_payload(
    ticker: str,
    analysis_cutoff: str | datetime,
    *,
    bar_count: int = DEFAULT_BAR_COUNT,
    store: MarketDataStore | None = None,
    accessed_at: datetime | None = None,
) -> dict[str, Any]:
    """Read canonical stock bars and build the skill's point-in-time JSON."""

    if bar_count < MINIMUM_BARS:
        raise SkillMarketDataError(f"bar_count must be at least {MINIMUM_BARS}")
    normalized_ticker = normalize_a_share_ticker(ticker)
    cutoff = parse_analysis_cutoff(analysis_cutoff)
    market_store = store or configured_market_store()
    start_date = (cutoff.date() - timedelta(days=max(540, bar_count * 3))).strftime(
        "%Y%m%d"
    )
    frame = market_store.read_market_range(
        "daily",
        start_date=start_date,
        end_date=cutoff.strftime("%Y%m%d"),
        symbols=[normalized_ticker],
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "volume",
        ],
    )
    normalized, warnings = _normalize_daily_frame(
        frame,
        ticker=normalized_ticker,
        cutoff=cutoff,
        bar_count=bar_count,
    )
    actual_accessed_at = parse_analysis_cutoff(
        accessed_at or datetime.now(tz=SHANGHAI_TZ)
    )
    return _payload_from_frame(
        normalized,
        ticker=normalized_ticker,
        cutoff=cutoff,
        source="Tushare daily via project canonical MarketDataStore",
        configured_backend=market_store.config.backend,
        mirror_parquet=market_store.config.mirror_parquet,
        accessed_at=actual_accessed_at,
        warnings=warnings,
    )


def build_index_price_volume_payload(
    ticker: str,
    analysis_cutoff: str | datetime,
    *,
    bar_count: int = DEFAULT_BAR_COUNT,
    path: Path | None = None,
    accessed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the same contract for a locally refreshed broad/industry index."""

    if bar_count < MINIMUM_BARS:
        raise SkillMarketDataError(f"bar_count must be at least {MINIMUM_BARS}")
    normalized_ticker = normalize_a_share_ticker(ticker)
    cutoff = parse_analysis_cutoff(analysis_cutoff)
    index_path = path or PROJECT_ROOT / "data/raw" / f"index_{normalized_ticker}.parquet"
    if not index_path.is_file():
        raise SkillMarketDataError(
            f"local index history is unavailable: {index_path}; disclose the benchmark gap"
        )
    frame = pd.read_parquet(index_path)
    normalized, warnings = _normalize_daily_frame(
        frame,
        ticker=normalized_ticker,
        cutoff=cutoff,
        bar_count=bar_count,
    )
    actual_accessed_at = parse_analysis_cutoff(
        accessed_at or datetime.now(tz=SHANGHAI_TZ)
    )
    return _payload_from_frame(
        normalized,
        ticker=normalized_ticker,
        cutoff=cutoff,
        source=f"Tushare index_daily via project local reference file {index_path.name}",
        configured_backend="parquet_reference",
        mirror_parquet=True,
        accessed_at=actual_accessed_at,
        warnings=warnings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export point-in-time stock or index OHLCV from the project's existing "
            "market-data configuration into the analyze-a-shares JSON contract."
        )
    )
    parser.add_argument("--ticker", required=True, help="A-share or index code.")
    parser.add_argument(
        "--cutoff",
        default=datetime.now(tz=SHANGHAI_TZ).isoformat(timespec="seconds"),
        help="Aware ISO 8601 analysis cutoff (default: now in Asia/Shanghai).",
    )
    parser.add_argument(
        "--kind",
        choices=("stock", "index"),
        default="stock",
        help="Read the canonical stock store or a local index reference file.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=DEFAULT_BAR_COUNT,
        help=f"Maximum complete bars to export (default: {DEFAULT_BAR_COUNT}).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        help="Override the local parquet path when --kind=index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON atomically to this path; omit to print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.kind == "stock":
            payload = build_stock_price_volume_payload(
                args.ticker,
                args.cutoff,
                bar_count=args.bars,
            )
        else:
            payload = build_index_price_volume_payload(
                args.ticker,
                args.cutoff,
                bar_count=args.bars,
                path=args.index_path,
            )
    except (OSError, SkillMarketDataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        atomic_write_json(payload, args.output)
        print(json.dumps({"status": "saved", "path": str(args.output)}, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
