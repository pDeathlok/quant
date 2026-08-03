"""Reusable research primitives for the second long-entry model iteration."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _date_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").groupby(frame["date"]).rank(
        method="average", pct=True
    )


def add_entry_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Build multi-horizon, drawdown-aware and industry-relative labels.

    All inputs are forward outcomes and the resulting columns are label-only.
    A higher value is always better.  The industry component falls back to the
    market rank when fewer than five same-industry observations are available.
    """

    out = frame.copy()
    components = {
        "excess_return_13w": 0.10,
        "excess_return_26w": 0.25,
        "excess_return_52w": 0.45,
        "mae_26w": 0.20,
    }
    market_parts: list[pd.Series] = []
    industry_parts: list[pd.Series] = []
    industry_group = [out["date"], out["industry"].fillna("未知").astype(str)]
    industry_size = out.groupby(industry_group)["ts_code"].transform("count")
    for column, weight in components.items():
        values = pd.to_numeric(out[column], errors="coerce")
        market_rank = values.groupby(out["date"]).rank(method="average", pct=True)
        industry_rank = values.groupby(industry_group).rank(method="average", pct=True)
        industry_rank = industry_rank.where(industry_size >= 5, market_rank)
        market_parts.append(market_rank * weight)
        industry_parts.append(industry_rank * weight)
    out["label_entry_utility_market"] = sum(market_parts)
    out["label_entry_utility_industry"] = sum(industry_parts)
    out["label_entry_utility"] = (
        out["label_entry_utility_market"] * 0.55
        + out["label_entry_utility_industry"] * 0.45
    )
    medium_components = {
        "excess_return_13w": 0.25,
        "excess_return_26w": 0.45,
        "mae_13w": 0.15,
        "mae_26w": 0.15,
    }
    medium_market_parts: list[pd.Series] = []
    medium_industry_parts: list[pd.Series] = []
    for column, weight in medium_components.items():
        values = pd.to_numeric(out[column], errors="coerce")
        market_rank = values.groupby(out["date"]).rank(method="average", pct=True)
        industry_rank = values.groupby(industry_group).rank(method="average", pct=True)
        industry_rank = industry_rank.where(industry_size >= 5, market_rank)
        medium_market_parts.append(market_rank * weight)
        medium_industry_parts.append(industry_rank * weight)
    out["label_entry_utility_26w"] = (
        sum(medium_market_parts) * 0.55 + sum(medium_industry_parts) * 0.45
    )
    fast_market = (
        _date_rank(out, "excess_return_13w") * 0.70
        + _date_rank(out, "mae_13w") * 0.30
    )
    fast_excess = pd.to_numeric(out["excess_return_13w"], errors="coerce")
    fast_mae = pd.to_numeric(out["mae_13w"], errors="coerce")
    fast_industry = (
        fast_excess.groupby(industry_group).rank(method="average", pct=True) * 0.70
        + fast_mae.groupby(industry_group).rank(method="average", pct=True) * 0.30
    ).where(industry_size >= 5, fast_market)
    out["label_entry_utility_13w"] = fast_market * 0.55 + fast_industry * 0.45
    out["label_downside_rank_26w"] = _date_rank(out, "mae_26w")
    downside_floor = pd.to_numeric(out["mae_26w"], errors="coerce").groupby(out["date"]).transform(
        lambda values: values.quantile(0.25)
    )
    valid = out[[*components]].notna().all(axis=1)
    success = (
        out["label_entry_utility"].ge(0.70)
        & pd.to_numeric(out["excess_return_52w"], errors="coerce").gt(0)
        & pd.to_numeric(out["mae_26w"], errors="coerce").ge(downside_floor)
    )
    out["label_entry_success"] = np.where(valid, success.astype(float), np.nan)
    return out


def add_external_factor_transforms(frame: pd.DataFrame) -> pd.DataFrame:
    """Add scale-safe transforms after the PIT source merge."""

    out = frame.copy()
    total_mv_cny = pd.to_numeric(out.get("total_mv"), errors="coerce") * 10_000.0
    margin = pd.to_numeric(out.get("margin_balance"), errors="coerce")
    short = pd.to_numeric(out.get("short_balance"), errors="coerce")
    out["margin_balance_to_mv"] = margin / total_mv_cny.replace(0, np.nan)
    out["short_balance_to_mv"] = short / total_mv_cny.replace(0, np.nan)
    out["log_margin_balance"] = np.log1p(margin.clip(lower=0))
    out["log_short_balance"] = np.log1p(short.clip(lower=0))
    out["has_holder_trade_180d"] = (
        pd.to_numeric(out.get("holder_buy_event_count_180d"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out.get("holder_after_ratio_change_180d"), errors="coerce").fillna(0).ne(0)
    ).astype(float)
    out["has_top_list_20d"] = pd.to_numeric(
        out.get("top_list_count_20d"), errors="coerce"
    ).fillna(0).gt(0).astype(float)
    out["pledge_ratio_high"] = pd.to_numeric(out.get("pledge_ratio"), errors="coerce").ge(30).astype(float)
    if "large_net_5d_ratio" in out.columns and "return_120d_cross_section_pct" in out.columns:
        flow_rank = pd.to_numeric(out["large_net_5d_ratio"], errors="coerce").groupby(out["date"]).rank(pct=True)
        momentum_rank = pd.to_numeric(out["return_120d_cross_section_pct"], errors="coerce")
        out["large_flow_price_divergence_5d"] = flow_rank - momentum_rank
    return out.replace([np.inf, -np.inf], np.nan)


def select_industry_capped(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_n: int,
    max_per_industry: int,
) -> pd.DataFrame:
    """Select the highest scores with an explicit per-industry position cap."""

    selected: list[pd.DataFrame] = []
    for _, group in frame.dropna(subset=[score_column]).groupby("date", sort=True):
        counts: dict[str, int] = {}
        rows: list[int] = []
        for index, row in group.sort_values(score_column, ascending=False).iterrows():
            industry = str(row.get("industry") or "未知")
            if counts.get(industry, 0) >= max_per_industry:
                continue
            rows.append(index)
            counts[industry] = counts.get(industry, 0) + 1
            if len(rows) >= top_n:
                break
        if rows:
            selected.append(group.loc[rows])
    return pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0].copy()


def month_end_week_mask(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"])
    last_dates = dates.groupby(dates.dt.to_period("M")).transform("max")
    return dates.eq(last_dates)


def cooldown_cases(
    frame: pd.DataFrame,
    *,
    severity_column: str,
    cooldown_days: int = 91,
    maximum_rows: int = 120,
) -> pd.DataFrame:
    """Keep distinct episodes instead of adjacent weekly copies of one case."""

    chosen: list[int] = []
    accepted: dict[str, list[pd.Timestamp]] = {}
    ordered = frame.sort_values([severity_column, "date"], ascending=[False, False])
    for index, row in ordered.iterrows():
        code = str(row["ts_code"])
        date = pd.Timestamp(row["date"])
        if any(abs((date - previous).days) < cooldown_days for previous in accepted.get(code, [])):
            continue
        chosen.append(index)
        accepted.setdefault(code, []).append(date)
        if len(chosen) >= maximum_rows:
            break
    return frame.loc[chosen].sort_values(severity_column, ascending=False).reset_index(drop=True)


def classify_case_causes(frame: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """Assign an auditable first-pass cause; unresolved cases stay unresolved."""

    out = frame.copy()
    growth = pd.to_numeric(out.get("or_yoy"), errors="coerce")
    eps_growth = pd.to_numeric(out.get("basic_eps_yoy"), errors="coerce")
    revision = pd.to_numeric(out.get("analyst_eps_revision_180d"), errors="coerce")
    excess = pd.to_numeric(out.get("excess_return_52w"), errors="coerce")
    mae = pd.to_numeric(out.get("mae_26w"), errors="coerce")
    momentum = pd.to_numeric(out.get("return_120d"), errors="coerce")
    cause = pd.Series("unresolved", index=out.index, dtype="object")
    evidence = pd.Series("缺少可验证的事前或事后事件证据", index=out.index, dtype="object")
    if kind == "false_positive":
        prior_error = growth.lt(0) | eps_growth.lt(0) | revision.lt(-0.10)
        noise = excess.between(-0.05, 0.0, inclusive="both") & mae.gt(-0.20)
        limitation = ~prior_error & ~noise & (
            out["industry"].astype(str).str.contains("银行|食品|白酒|家用电器|医药|医疗", regex=True)
            | momentum.gt(0.20)
        )
        cause.loc[noise] = "noise"
        evidence.loc[noise] = "52周相对收益仅小幅为负且未发生20%级回撤"
        cause.loc[limitation] = "model_limitation"
        evidence.loc[limitation] = "风格/行业拥挤或高位延续未被现有标签充分惩罚"
        cause.loc[prior_error] = "prior_error"
        evidence.loc[prior_error] = "信号日已可见营收/EPS转负或一致预期下修"
    elif kind == "false_negative":
        rotation = momentum.gt(0) | out["industry"].astype(str).str.contains(
            "元器件|通信|半导体|软件|机械|电气|化工", regex=True
        )
        noise = excess.between(0.0, 0.05, inclusive="both")
        cause.loc[noise] = "noise"
        evidence.loc[noise] = "进入未来高分位但绝对超额幅度很小"
        cause.loc[rotation & ~noise] = "model_limitation"
        evidence.loc[rotation & ~noise] = "行业轮动/成长机会被防御型风险过滤压低"
    else:
        raise ValueError("kind must be false_positive or false_negative")
    out["case_cause"] = cause
    out["case_evidence"] = evidence
    out["new_information_status"] = "not_verified_without_post_signal_event_join"
    return out


def summarize_selection(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    return_column: str = "return_52w",
) -> dict[str, float | int]:
    def date_mean(frame: pd.DataFrame, column: str) -> float:
        usable = frame.dropna(subset=[column])
        return float(usable.groupby("date")[column].mean().mean()) if not usable.empty else np.nan

    selected_return = date_mean(selected, return_column)
    baseline_return = date_mean(baseline, return_column)
    return {
        "signals": int(selected["date"].nunique()),
        "observations": int(len(selected)),
        "mean_return_52w": selected_return,
        "baseline_return_52w": baseline_return,
        "return_delta": selected_return - baseline_return,
        "mean_excess_52w": date_mean(selected, "excess_return_52w"),
        "mean_mae_26w": date_mean(selected, "mae_26w"),
        "hit_rate_excess_52w": float(pd.to_numeric(selected["excess_return_52w"], errors="coerce").gt(0).mean()),
    }


def maximum_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float((values / values.cummax() - 1.0).min())


def equity_metrics(equity: pd.DataFrame) -> dict[str, float | int]:
    frame = equity.dropna(subset=["date", "equity"]).sort_values("date")
    if len(frame) < 2:
        return {}
    daily_return = pd.to_numeric(frame["equity"], errors="coerce").pct_change().dropna()
    years = max((frame["date"].iloc[-1] - frame["date"].iloc[0]).days / 365.25, 1 / 252)
    annual_return = float((frame["equity"].iloc[-1] / frame["equity"].iloc[0]) ** (1 / years) - 1)
    annual_volatility = float(daily_return.std() * np.sqrt(252))
    return {
        "start": frame["date"].iloc[0].date().isoformat(),
        "end": frame["date"].iloc[-1].date().isoformat(),
        "days": int(len(frame)),
        "total_return": float(frame["equity"].iloc[-1] / frame["equity"].iloc[0] - 1),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_zero_rf": annual_return / annual_volatility if annual_volatility > 0 else np.nan,
        "max_drawdown": maximum_drawdown(frame["equity"]),
    }
