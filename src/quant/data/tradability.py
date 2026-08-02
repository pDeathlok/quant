"""Point-in-time A-share tradability data built from canonical Tushare inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


TRADABILITY_COLUMNS = (
    "trade_date",
    "ts_code",
    "pre_close",
    "up_limit",
    "down_limit",
    "is_suspended",
    "is_st",
    "st_type",
    "list_date",
    "market",
)


def _validate_trade_date(trade_date: str) -> str:
    normalized = str(trade_date).replace("-", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError("trade_date must use YYYYMMDD format")
    return normalized


def _normalize_provider_frame(
    frame: pd.DataFrame | None,
    *,
    trade_date: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Tushare tradability response must be a pandas DataFrame")
    if "ts_code" not in frame.columns:
        raise ValueError("Tushare tradability response missing ts_code")

    normalized = frame.copy()
    normalized["ts_code"] = normalized["ts_code"].astype(str).str.strip()
    normalized = normalized.loc[normalized["ts_code"].ne("")]
    if "trade_date" in normalized.columns:
        dates = normalized["trade_date"].astype(str).str.replace("-", "", regex=False)
        normalized = normalized.loc[dates.eq(trade_date)].copy()
    return normalized.drop_duplicates("ts_code", keep="last").reset_index(drop=True)


def _optional_column(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def build_daily_tradability(
    *,
    trade_date: str,
    stock_basic: pd.DataFrame,
    limits: pd.DataFrame,
    suspensions: pd.DataFrame,
    st_stocks: pd.DataFrame,
    minimum_coverage_rate: float = 0.98,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate and join one point-in-time A-share trading universe."""

    normalized_date = _validate_trade_date(trade_date)
    if not 0 < minimum_coverage_rate <= 1:
        raise ValueError("minimum_coverage_rate must be in (0, 1]")
    if not isinstance(stock_basic, pd.DataFrame) or "ts_code" not in stock_basic.columns:
        raise ValueError("stock_basic must contain ts_code")

    universe = stock_basic.copy()
    universe["ts_code"] = universe["ts_code"].astype(str).str.strip()
    universe = (
        universe.loc[universe["ts_code"].ne("")]
        .drop_duplicates("ts_code", keep="last")
        .sort_values("ts_code")
        .reset_index(drop=True)
    )
    if universe.empty:
        raise ValueError("stock_basic contains no listed A-share symbols")
    universe["list_date"] = _optional_column(universe, "list_date", "").fillna("").astype(str)
    universe["market"] = _optional_column(universe, "market", "").fillna("").astype(str)
    list_dates = universe["list_date"].str.replace("-", "", regex=False)
    active = list_dates.eq("") | list_dates.le(normalized_date)
    if "delist_date" in universe.columns:
        delist_dates = universe["delist_date"].fillna("").astype(str).str.replace("-", "", regex=False)
        active &= delist_dates.eq("") | delist_dates.ge(normalized_date)
    universe = universe.loc[active].reset_index(drop=True)
    if universe.empty:
        raise ValueError("stock_basic contains no A-share symbols active on trade_date")

    normalized_limits = _normalize_provider_frame(limits, trade_date=normalized_date)
    required_limit_columns = {"up_limit", "down_limit"}
    missing_limit_columns = sorted(required_limit_columns - set(normalized_limits.columns))
    if missing_limit_columns:
        raise ValueError(f"Tushare stk_limit missing columns: {missing_limit_columns}")
    for column in ("pre_close", "up_limit", "down_limit"):
        if column not in normalized_limits.columns:
            normalized_limits[column] = pd.NA
        normalized_limits[column] = pd.to_numeric(
            normalized_limits[column], errors="coerce"
        )

    normalized_suspensions = _normalize_provider_frame(
        suspensions,
        trade_date=normalized_date,
    )
    if not normalized_suspensions.empty and "suspend_type" in normalized_suspensions.columns:
        normalized_suspensions = normalized_suspensions.loc[
            normalized_suspensions["suspend_type"].astype(str).str.upper().eq("S")
        ]
    suspended_codes = set(normalized_suspensions.get("ts_code", pd.Series(dtype=str)))

    normalized_st = _normalize_provider_frame(st_stocks, trade_date=normalized_date)
    st_codes = set(normalized_st.get("ts_code", pd.Series(dtype=str)))
    st_type_column = (
        "type_name"
        if "type_name" in normalized_st.columns
        else "type" if "type" in normalized_st.columns else None
    )
    st_types = (
        normalized_st.set_index("ts_code")[st_type_column].fillna("").astype(str).to_dict()
        if st_type_column is not None
        else {}
    )

    output = universe[["ts_code", "list_date", "market"]].merge(
        normalized_limits[["ts_code", "pre_close", "up_limit", "down_limit"]],
        on="ts_code",
        how="left",
        validate="one_to_one",
    )
    output.insert(0, "trade_date", normalized_date)
    output["is_suspended"] = output["ts_code"].isin(suspended_codes)
    output["is_st"] = output["ts_code"].isin(st_codes)
    output["st_type"] = output["ts_code"].map(st_types).fillna("")

    covered = (
        output["up_limit"].notna()
        & output["down_limit"].notna()
        & output["up_limit"].gt(0)
        & output["down_limit"].gt(0)
    )
    coverage_rate = float(covered.mean())
    if coverage_rate < minimum_coverage_rate:
        raise ValueError(
            "Tushare stk_limit coverage "
            f"{coverage_rate:.2%} is below required {minimum_coverage_rate:.2%}"
        )

    output = output.loc[:, TRADABILITY_COLUMNS].sort_values("ts_code").reset_index(drop=True)
    audit: dict[str, object] = {
        "trade_date": normalized_date,
        "universe_rows": len(output),
        "limit_rows": len(normalized_limits),
        "covered_rows": int(covered.sum()),
        "coverage_rate": coverage_rate,
        "suspended_rows": int(output["is_suspended"].sum()),
        "st_rows": int(output["is_st"].sum()),
    }
    return output, audit
