"""Evidence-aware batch evaluation for the 112-company good-company universe.

The module applies the same GQS, scenario, technical and archival contract to
every company.  It intentionally keeps company quality, valuation and price
confirmation as separate outputs.  Missing evidence lowers coverage or makes a
valuation unavailable; it is never converted to a neutral zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.features.long_quality_factors import build_annual_quality_events


MODULE_WEIGHTS = {"A": 10.0, "B": 20.0, "C": 10.0, "D": 20.0, "E": 15.0, "F": 10.0, "G": 15.0}
FINANCIAL_PATTERN = "银行|多元金融|证券|保险"

# The 75-company niche file already contains a business-level scarcity
# hypothesis.  These are the 37 broad-screen-only companies, stated as
# hypotheses to test rather than verified moat claims.
BROAD_ONLY_SCARCITY = {
    "厦门空港": "厦门航空口岸区位、跑道与候机楼资源及非航商业运营权",
    "中信海直": "海上油气与低空场景的直升机运营资质、安全记录和机队组织能力",
    "申通地铁": "城市轨交运营经验、线路权益及运维管理能力",
    "新媒股份": "广东IPTV牌照、运营商入口与内容集成能力",
    "海看股份": "山东IPTV播控牌照、运营商入口与本地内容运营",
    "映翰通": "工业物联网边缘网关、设备云与长期现场适配",
    "军信股份": "固废处理特许经营项目、运营经验与区域客户关系",
    "顺控发展": "区域供水管网、特许经营权与稳定现金流",
    "陕西能源": "低成本煤电一体化资源与区域电力保障能力",
    "北大荒": "稀缺耕地资源、规模化农业组织与土地经营权",
    "安迪苏": "蛋氨酸、维生素等动物营养产品的工艺、规模和全球客户",
    "益丰药房": "区域药房密度、供应链、会员与并购整合能力",
    "新坐标": "精密冷成形零部件工艺、模具开发与汽车客户认证",
    "天目湖": "区域景区资源、目的地运营和交通区位",
    "中纺标": "纺织品检测认证资质、标准参与和客户公信力",
    "博士眼镜": "专业验光服务、核心商圈门店网络与供应链",
    "润贝航科": "航空化学品、航材分销资质与航空客户服务网络",
    "新余国科": "人工影响天气和军工火工品资质、技术与客户认证",
    "亚翔集成": "半导体洁净室工程经验、项目管理与客户认证",
    "广咨国际": "广东区域工程咨询资质、专家网络与政府项目经验",
    "上海建科": "工程检测、咨询和标准能力及区域实验室网络",
    "中国国贸": "北京CBD核心物业区位、会展酒店办公综合运营",
    "招商蛇口": "核心城市综合开发、园区和持有型资产运营平台",
    "浙江东日": "温州农批市场运营权、交易网络与区域商户黏性",
    "云铝股份": "云南水电铝资源、低碳成本与一体化产能",
    "金诚信": "深部地下矿山建设、采矿运营技术与全球项目经验",
    "紫金矿业": "全球矿产资源获取、低品位矿开发和项目建设运营",
    "世华科技": "功能性材料配方、精密涂布与消费电子客户认证",
    "康普顿": "润滑油品牌、配方和汽车后市场渠道",
    "陕西煤业": "优质煤炭资源、低成本矿区与铁路销售体系",
    "电投能源": "露天煤矿、坑口电厂和新能源协同的低成本一体化",
    "江苏金租": "厂商租赁渠道、小微资产定价和批量化风控能力",
    "常熟银行": "县域小微客群、线下网点和下沉风控模型",
    "宁波银行": "区域客群、轻资本业务与精细化风险管理",
    "贵州茅台": "酱香白酒品牌、产区微生态、基酒时间和渠道定价权",
    "红旗连锁": "成都高密度社区门店、供应链和便民服务入口",
    "山西汾酒": "清香白酒品牌、产区工艺和全国化渠道",
}


@dataclass(frozen=True)
class EvaluationPaths:
    raw_dir: Path
    broad_shortlist: Path
    niche_watchlist: Path
    daily_basic_snapshot: Path
    governance_dir: Path | None = None
    mcp_overrides_path: Path | None = None


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _round(value: Any, digits: int = 4) -> float | None:
    number = _num(value)
    return round(number, digits) if number is not None else None


def _mean(values: Iterable[Any]) -> float | None:
    usable = [number for number in (_num(value) for value in values) if number is not None]
    return float(np.mean(usable)) if usable else None


def _rating_by_thresholds(value: Any, thresholds: list[tuple[float, float]]) -> float | None:
    number = _num(value)
    if number is None:
        return None
    for threshold, rating in thresholds:
        if number >= threshold:
            return rating
    return 0.5


def _inverse_rating(value: Any, thresholds: list[tuple[float, float]]) -> float | None:
    number = _num(value)
    if number is None:
        return None
    for ceiling, rating in thresholds:
        if number <= ceiling:
            return rating
    return 0.5


def _weighted_rating(parts: list[tuple[float | None, float]]) -> tuple[float | None, float]:
    usable = [(value, weight) for value, weight in parts if value is not None]
    total_weight = sum(weight for _, weight in parts)
    covered_weight = sum(weight for _, weight in usable)
    if not usable or total_weight <= 0:
        return None, 0.0
    value = sum(float(rating) * weight for rating, weight in usable) / covered_weight
    return round(value * 2) / 2, covered_weight / total_weight


def broad_industry(industry: object) -> str:
    value = str(industry or "未知")
    rules = [
        ("金融", FINANCIAL_PATTERN),
        ("房地产与资产运营", "地产|房产|园区开发|商品城"),
        ("食品饮料与必选消费", "白酒|食品|乳制品|软饮料|超市连锁|日用化工"),
        ("可选消费与服务", "家用电器|家居用品|汽车|旅游|酒店餐饮|服饰|百货|文教休闲"),
        ("医药医疗", "医疗保健|化学制药|生物制药|中成药|医药商业"),
        ("信息技术与电子", "IT设备|元器件|半导体|互联网|软件服务|通信设备|电信运营"),
        ("工业制造", "专用机械|工程机械|机床制造|机械基件|电器仪表|电气设备|运输设备|航空|船舶"),
        ("材料与化工", "化工原料|化纤|塑料|橡胶|普钢|特种钢|钢加工|玻璃|水泥|陶瓷|矿物制品"),
        ("有色与贵金属", "小金属|铅锌|铜|铝|黄金"),
        ("能源", "煤炭开采|焦炭加工|石油加工|石油开采|石油贸易"),
        ("公用事业与环保", "供气供热|新型电力|水力发电|火力发电|水务|环境保护"),
        ("交通运输与物流", "仓储物流|公共交通|公路|机场|空运|港口|水运|铁路|路桥"),
        ("农业", "农业综合|农药化肥|林业|渔业|种植业|饲料"),
        ("传媒出版", "出版业|广告包装|影视音像"),
        ("建筑装饰", "建筑工程|装修装饰"),
        ("商贸与综合", "其他商业|商贸代理|批发业|综合类"),
    ]
    for label, pattern in rules:
        if pd.Series([value]).str.contains(pattern, regex=True, na=False).iloc[0]:
            return label
    return "其他"


def valuation_family(industry: str, broad: str) -> str:
    if any(token in industry for token in ("银行", "多元金融")):
        return "bank_pb_roe"
    if broad in {"有色与贵金属", "能源"} or industry in {"铝", "铜", "煤炭开采", "石油加工"}:
        return "resource_midcycle"
    if broad in {"公用事业与环保", "交通运输与物流"} or industry in {"园区开发"}:
        return "utility_infrastructure"
    if broad == "房地产与资产运营":
        return "real_estate_asset"
    if broad == "信息技术与电子":
        return "technology_pe"
    if broad == "医药医疗":
        return "medical_pe"
    if broad in {"食品饮料与必选消费", "可选消费与服务", "农业"}:
        return "consumer_pe"
    if broad in {"工业制造", "材料与化工", "建筑装饰"}:
        return "industrial_pe"
    return "service_pe"


def build_universe(paths: EvaluationPaths) -> pd.DataFrame:
    broad = pd.read_csv(paths.broad_shortlist).copy()
    niche = pd.read_csv(paths.niche_watchlist).copy()
    broad["ts_code"] = broad["ts_code"].astype(str)
    niche["ts_code"] = niche["ts_code"].astype(str)
    broad_cols = [
        column for column in (
            "ts_code", "name", "industry", "broad_industry", "stage1_proxy_score",
            "stage1_proxy_coverage", "within_industry_percentile", "roe_mean_5y",
            "cashflow_quality_3y", "free_cashflow_margin_3y", "fina_debt_to_assets",
        ) if column in broad.columns
    ]
    niche_cols = [
        column for column in (
            "ts_code", "name", "industry", "capability_category", "scarcity_hypothesis",
            "evidence_status", "scarcity_source_url", "stage1_proxy_score",
            "within_industry_percentile", "roe_mean_5y", "cashflow_quality_3y",
            "fina_debt_to_assets",
        ) if column in niche.columns
    ]
    base = pd.concat([broad[broad_cols], niche[niche_cols]], ignore_index=True, sort=False)
    base = base.sort_values("ts_code").groupby("ts_code", as_index=False).first()
    base["in_broad_48"] = base["ts_code"].isin(set(broad["ts_code"]))
    base["in_niche_75"] = base["ts_code"].isin(set(niche["ts_code"]))
    base["name"] = base["name"].astype(str).str.strip()
    base["industry"] = base["industry"].fillna("未知").astype(str).str.strip()
    base["broad_industry"] = base["broad_industry"].where(base["broad_industry"].notna(), base["industry"].map(broad_industry))
    base["scarcity_hypothesis"] = base["scarcity_hypothesis"].fillna(base["name"].map(BROAD_ONLY_SCARCITY))
    base["scarcity_hypothesis"] = base["scarcity_hypothesis"].fillna("业务稀缺性待通过年报与行业材料逐项核验")
    base["valuation_family"] = [valuation_family(i, b) for i, b in zip(base["industry"], base["broad_industry"])]
    if len(base) != 112 or base["ts_code"].nunique() != 112:
        raise ValueError(f"expected 112 unique companies, got rows={len(base)} unique={base['ts_code'].nunique()}")
    return base.reset_index(drop=True)


def _latest_annual(raw_dir: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    events = build_annual_quality_events(
        pd.read_parquet(raw_dir / "fina_indicator.parquet"),
        pd.read_parquet(raw_dir / "income.parquet"),
        pd.read_parquet(raw_dir / "cashflow.parquet"),
        pd.read_parquet(raw_dir / "balancesheet.parquet"),
    )
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    events = events[events["annual_quality_available_at"].le(cutoff_naive)].copy()
    return events.sort_values(["ts_code", "annual_quality_available_at", "period_end"]).groupby("ts_code", as_index=False).tail(1)


def _latest_period(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["_ann"] = pd.to_datetime(frame["ann_date"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8], format="%Y%m%d", errors="coerce")
    frame["_end"] = pd.to_datetime(frame["end_date"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8], format="%Y%m%d", errors="coerce")
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    frame = frame[frame["_ann"].le(cutoff_naive)].sort_values(["ts_code", "_end", "_ann"])
    return frame.groupby("ts_code", as_index=False).tail(1)


def _forecast_table(raw_dir: Path, universe: set[str], cutoff: pd.Timestamp, shares: pd.Series) -> pd.DataFrame:
    forecasts = pd.read_parquet(raw_dir / "analyst_forecasts.parquet").copy()
    forecasts["report_date"] = pd.to_datetime(forecasts["report_date"], errors="coerce")
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    forecasts = forecasts[
        forecasts["ts_code"].astype(str).isin(universe)
        & forecasts["report_date"].le(cutoff_naive)
        & forecasts["forecast_year"].isin([2026, 2027, 2028])
        & forecasts["source"].isin(["datayes_consensus", "akshare_em_snapshot", "akshare_em_research"])
    ].copy()
    forecasts["priority"] = forecasts["source"].map({"datayes_consensus": 3, "akshare_em_snapshot": 2, "akshare_em_research": 1}).fillna(0)
    latest_by_source = forecasts.sort_values("report_date").groupby(["ts_code", "forecast_year", "source"], as_index=False).tail(1)
    selected = latest_by_source.sort_values(["priority", "report_date"]).groupby(["ts_code", "forecast_year"], as_index=False).tail(1).copy()
    share_map = shares.to_dict()
    selected["eps_original"] = pd.to_numeric(selected["eps"], errors="coerce")
    selected["net_profit_cny"] = pd.to_numeric(selected["net_profit"], errors="coerce") * 10_000.0
    selected["shares"] = selected["ts_code"].map(share_map)
    selected["eps_share_adjusted"] = selected["net_profit_cny"] / selected["shares"].where(selected["shares"].gt(0))
    mismatch = (selected["eps_share_adjusted"] / selected["eps_original"] - 1.0).abs().gt(0.12)
    selected["eps"] = selected["eps_original"].where(~mismatch | selected["eps_share_adjusted"].isna(), selected["eps_share_adjusted"])
    selected["share_adjusted"] = mismatch & selected["eps_share_adjusted"].notna()

    alt = latest_by_source.pivot_table(index=["ts_code", "forecast_year"], columns="source", values="eps", aggfunc="last")
    if {"datayes_consensus", "akshare_em_snapshot"}.issubset(alt.columns):
        dispersion = (alt["datayes_consensus"] - alt["akshare_em_snapshot"]).abs() / alt[["datayes_consensus", "akshare_em_snapshot"]].mean(axis=1).abs().replace(0, np.nan)
        selected = selected.merge(dispersion.rename("source_dispersion").reset_index(), on=["ts_code", "forecast_year"], how="left")
    else:
        selected["source_dispersion"] = np.nan

    rows: list[dict[str, Any]] = []
    for code, group in selected.groupby("ts_code"):
        item: dict[str, Any] = {"ts_code": code}
        for row in group.itertuples(index=False):
            year = int(row.forecast_year)
            item[f"eps_{year}"] = _num(row.eps)
            item[f"net_profit_{year}"] = _num(row.net_profit_cny)
            item[f"sample_count_{year}"] = _num(row.report_count) or _num(row.analyst_count)
            item[f"forecast_source_{year}"] = str(row.source)
            item[f"forecast_as_of_{year}"] = row.report_date.strftime("%Y-%m-%d")
            item[f"dispersion_{year}"] = _num(row.source_dispersion)
            item[f"share_adjusted_{year}"] = bool(row.share_adjusted)
        rows.append(item)
    return pd.DataFrame(rows)


def _technical_table(raw_dir: Path, universe: set[str], price_date: str) -> pd.DataFrame:
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=raw_dir, mirror_parquet=True))
    end = pd.Timestamp(price_date).strftime("%Y%m%d")
    start = (pd.Timestamp(price_date) - pd.Timedelta(days=470)).strftime("%Y%m%d")
    daily = store.read_market_range("daily", start_date=start, end_date=end, symbols=sorted(universe))
    daily["trade_date"] = pd.to_datetime(daily["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    rows: list[dict[str, Any]] = []
    for code, group in daily.groupby("ts_code"):
        group = group.sort_values("trade_date").tail(300)
        close = pd.to_numeric(group["close"], errors="coerce")
        volume = pd.to_numeric(group.get("volume", group.get("vol")), errors="coerce")
        if group.empty or close.dropna().empty:
            continue
        current = float(close.iloc[-1])
        ma20 = close.tail(20).mean() if len(close) >= 20 else np.nan
        ma60 = close.tail(60).mean() if len(close) >= 60 else np.nan
        high250 = close.tail(250).max() if len(close) >= 20 else np.nan
        volume20 = volume.tail(21).iloc[:-1].mean() if len(volume) >= 21 else np.nan
        vol_ratio = volume.iloc[-1] / volume20 if _num(volume20) not in (None, 0) else np.nan
        ret20 = current / close.iloc[-21] - 1 if len(close) >= 21 and close.iloc[-21] else np.nan
        ret60 = current / close.iloc[-61] - 1 if len(close) >= 61 and close.iloc[-61] else np.nan
        ret120 = current / close.iloc[-121] - 1 if len(close) >= 121 and close.iloc[-121] else np.nan
        vol20 = close.pct_change().tail(20).std() * np.sqrt(252) if len(close) >= 21 else np.nan
        ma20_gap = current / ma20 - 1 if _num(ma20) not in (None, 0) else np.nan
        ma60_gap = current / ma60 - 1 if _num(ma60) not in (None, 0) else np.nan
        if _num(ma20_gap) is None or _num(ma60_gap) is None:
            state = "数据不足"
        elif ma20_gap >= 0.03 and ma60_gap >= 0 and _num(ret60) is not None and ret60 > 0:
            state = "强势确认"
        elif ma20_gap >= 0 and ma60_gap < 0:
            state = "修复中"
        elif ma20_gap < -0.03 and ma60_gap < -0.05:
            state = "弱势未确认"
        else:
            state = "区间震荡"
        rows.append({
            "ts_code": code,
            "price_date": group["trade_date"].iloc[-1].strftime("%Y-%m-%d"),
            "current_price": current,
            "return_20d": _num(ret20), "return_60d": _num(ret60), "return_120d": _num(ret120),
            "volatility_20d": _num(vol20), "drawdown_250d": _num(current / high250 - 1) if _num(high250) else None,
            "ma20_gap": _num(ma20_gap), "ma60_gap": _num(ma60_gap), "volume_ratio_20d": _num(vol_ratio),
            "technical_state": state,
        })
    return pd.DataFrame(rows)


def _governance_table(
    governance_dir: Path | None,
    cutoff: pd.Timestamp,
    price_date: str,
) -> pd.DataFrame:
    if governance_dir is None or not governance_dir.exists():
        return pd.DataFrame(columns=["ts_code"])
    cutoff_text = cutoff.strftime("%Y%m%d")
    parts: list[pd.DataFrame] = []

    audit_path = governance_dir / "fina_audit.parquet"
    if audit_path.exists():
        audit = pd.read_parquet(audit_path)
        audit = audit[audit["ann_date"].fillna("").astype(str).le(cutoff_text)]
        audit = audit.sort_values(["ts_code", "end_date", "ann_date"]).groupby("ts_code", as_index=False).tail(1)
        parts.append(audit[["ts_code", "ann_date", "end_date", "audit_result", "audit_agency"]].rename(columns={"ann_date": "audit_ann_date", "end_date": "audit_period"}))

    pledge_path = governance_dir / "pledge_stat.parquet"
    if pledge_path.exists():
        pledge = pd.read_parquet(pledge_path)
        pledge = pledge[pledge["end_date"].fillna("").astype(str).le(price_date)]
        pledge = pledge.sort_values(["ts_code", "end_date"]).groupby("ts_code", as_index=False).tail(1)
        parts.append(pledge[["ts_code", "end_date", "pledge_count", "pledge_ratio"]].rename(columns={"end_date": "pledge_date"}))

    dividend_path = governance_dir / "dividend.parquet"
    if dividend_path.exists():
        dividend = pd.read_parquet(dividend_path)
        available = dividend["imp_ann_date"].fillna(dividend["ann_date"]).fillna("").astype(str)
        dividend = dividend[available.le(cutoff_text)].copy()
        dividend["_implemented"] = dividend["div_proc"].astype(str).eq("实施").astype(int)
        dividend = dividend.sort_values(["ts_code", "end_date", "_implemented", "imp_ann_date", "ann_date"]).groupby("ts_code", as_index=False).tail(1)
        keep = [column for column in ("ts_code", "end_date", "div_proc", "cash_div_tax", "stk_div", "imp_ann_date") if column in dividend.columns]
        parts.append(dividend[keep].rename(columns={"end_date": "dividend_period", "imp_ann_date": "dividend_announcement"}))

    repurchase_path = governance_dir / "repurchase.parquet"
    if repurchase_path.exists():
        repurchase = pd.read_parquet(repurchase_path)
        repurchase = repurchase[repurchase["ann_date"].fillna("").astype(str).le(cutoff_text)].copy()
        if not repurchase.empty:
            repurchase["amount"] = pd.to_numeric(repurchase.get("amount"), errors="coerce")
            grouped = repurchase.groupby("ts_code", as_index=False).agg(
                repurchase_count=("ann_date", "count"),
                repurchase_amount=("amount", "max"),
                repurchase_latest=("ann_date", "max"),
            )
            parts.append(grouped)
    if not parts:
        return pd.DataFrame(columns=["ts_code"])
    output = parts[0]
    for part in parts[1:]:
        output = output.merge(part, on="ts_code", how="outer")
    return output


def _module_scores(row: pd.Series) -> dict[str, Any]:
    financial = bool(row.get("is_financial"))
    roe_rating = _rating_by_thresholds(row.get("roe_mean_5y"), [(25, 5), (20, 4.5), (15, 4), (12, 3.5), (9, 3), (6, 2), (0, 1)])
    if financial:
        roe_rating = _rating_by_thresholds(row.get("roe_mean_5y"), [(16, 5), (13, 4.5), (11, 4), (9, 3.5), (7, 3), (5, 2), (0, 1)])
    profit_persistence = _rating_by_thresholds(row.get("profit_positive_share_5y"), [(1.0, 5), (0.8, 4), (0.6, 3), (0.4, 2), (0.0, 1)])
    gross_margin = _rating_by_thresholds(row.get("fina_grossprofit_margin"), [(60, 5), (40, 4), (25, 3.5), (15, 3), (8, 2), (0, 1)])
    asset_turn = _rating_by_thresholds(row.get("fina_assets_turn"), [(1.2, 5), (0.8, 4), (0.5, 3.5), (0.3, 3), (0.15, 2), (0, 1)])
    a_rating, a_cov = _weighted_rating([(profit_persistence, 0.40), (gross_margin, 0.30), (asset_turn, 0.30)])

    stage1 = _num(row.get("stage1_proxy_score"))
    within = _num(row.get("within_industry_percentile"))
    b_raw = 2.75 + (0.35 if bool(row.get("in_niche_75")) else 0.0)
    if stage1 is not None:
        b_raw += _clip((stage1 - 65) / 35, 0, 1) * 0.55
    if within is not None:
        b_raw += _clip((within - 0.70) / 0.30, 0, 1) * 0.45
    b_rating = round(_clip(b_raw, 1.0, 4.5) * 2) / 2
    b_cov = 0.70 if str(row.get("scarcity_source_url") or "").strip() else 0.60

    rev_growth = _rating_by_thresholds(row.get("revenue_cagr_3y"), [(0.20, 5), (0.12, 4.5), (0.07, 4), (0.03, 3.5), (0, 3), (-0.05, 2), (-1, 1)])
    profit_growth = _rating_by_thresholds(row.get("net_income_cagr_3y"), [(0.25, 5), (0.15, 4.5), (0.08, 4), (0.03, 3.5), (0, 3), (-0.08, 2), (-1, 1)])
    eps26 = _num(row.get("eps_2026")); eps28 = _num(row.get("eps_2028"))
    forecast_cagr = (eps28 / eps26) ** 0.5 - 1 if eps26 and eps28 and eps26 > 0 and eps28 > 0 else None
    forecast_growth = _rating_by_thresholds(forecast_cagr, [(0.20, 5), (0.12, 4.5), (0.08, 4), (0.04, 3.5), (0, 3), (-0.05, 2), (-1, 1)])
    c_rating, c_cov = _weighted_rating([(rev_growth, 0.30), (profit_growth, 0.35), (forecast_growth, 0.35)])

    roa_rating = _rating_by_thresholds(row.get("fina_roa"), [(15, 5), (10, 4.5), (7, 4), (4, 3.5), (2, 3), (0, 2)])
    margin_rating = _rating_by_thresholds(row.get("fina_netprofit_margin"), [(30, 5), (20, 4.5), (12, 4), (8, 3.5), (4, 3), (0, 2)])
    roe_stability = _inverse_rating(row.get("roe_std_5y"), [(2, 5), (4, 4.5), (7, 4), (10, 3), (15, 2)])
    d_parts = [(roe_rating, 0.45), (profit_persistence, 0.20), (roe_stability, 0.15)]
    if not financial:
        d_parts.extend([(roa_rating, 0.10), (margin_rating, 0.10)])
    d_rating, d_cov = _weighted_rating(d_parts)

    cash_conversion = _rating_by_thresholds(row.get("cashflow_quality_3y"), [(1.3, 5), (1.0, 4.5), (0.8, 4), (0.6, 3), (0.3, 2), (-10, 1)])
    fcf_rating = _rating_by_thresholds(row.get("free_cashflow_margin_3y"), [(0.20, 5), (0.12, 4.5), (0.07, 4), (0.03, 3.5), (0, 3), (-0.05, 2), (-10, 1)])
    accrual_rating = _inverse_rating(row.get("accruals_to_assets_3y"), [(-0.05, 5), (0, 4.5), (0.03, 4), (0.06, 3), (0.10, 2)])
    cfo_persistence = _rating_by_thresholds(row.get("cfo_positive_share_5y"), [(1.0, 5), (0.8, 4), (0.6, 3), (0.4, 2), (0, 1)])
    goodwill = _inverse_rating(row.get("annual_goodwill_to_assets"), [(0.01, 5), (0.05, 4), (0.15, 3), (0.30, 2)])
    if financial:
        e_rating, e_cov = 3.0, 0.55
    else:
        e_rating, e_cov = _weighted_rating([(cash_conversion, 0.30), (fcf_rating, 0.25), (accrual_rating, 0.20), (cfo_persistence, 0.15), (goodwill, 0.10)])

    debt_rating = _inverse_rating(row.get("fina_debt_to_assets"), [(25, 5), (40, 4.5), (55, 4), (65, 3), (75, 2)])
    current_rating = _rating_by_thresholds(row.get("fina_current_ratio"), [(2.0, 5), (1.5, 4.5), (1.2, 4), (1.0, 3), (0.7, 2), (0, 1)])
    if financial:
        f_rating, f_cov = _weighted_rating([(roe_rating, 0.45), (profit_persistence, 0.55)])
        f_cov *= 0.65
    else:
        f_rating, f_cov = _weighted_rating([(debt_rating, 0.45), (current_rating, 0.20), (profit_persistence, 0.35)])

    dividend_yield = _num(row.get("dv_ttm"))
    listing_years = _num(row.get("listed_years"))
    g_rating = 3.0
    audit_result = str(row.get("audit_result") or "").strip()
    if "标准无保留" in audit_result:
        g_rating += 0.5
    elif "无保留意见" in audit_result:
        g_rating -= 0.5
    elif audit_result:
        g_rating -= 2.0
    pledge_ratio = _num(row.get("pledge_ratio"))
    if pledge_ratio is not None:
        g_rating += 0.25 if pledge_ratio <= 5 else -0.5 if pledge_ratio >= 20 else 0.0
        if pledge_ratio >= 40:
            g_rating -= 0.75
    if dividend_yield is not None and dividend_yield >= 2:
        g_rating += 0.25
    if str(row.get("div_proc") or "") == "实施":
        g_rating += 0.25
    if (_num(row.get("repurchase_count")) or 0) > 0:
        g_rating += 0.25
    if listing_years is not None and listing_years >= 10:
        g_rating += 0.25
    if _num(row.get("annual_goodwill_to_assets")) is not None and row["annual_goodwill_to_assets"] > 0.30:
        g_rating -= 0.5
    g_rating = round(_clip(g_rating, 1.0, 4.5) * 2) / 2
    governance_fields = [bool(audit_result), dividend_yield is not None, pledge_ratio is not None, _num(row.get("repurchase_count")) is not None]
    g_cov = 0.55 + 0.075 * sum(governance_fields)

    ratings = {"A": a_rating, "B": b_rating, "C": c_rating, "D": d_rating, "E": e_rating, "F": f_rating, "G": g_rating}
    coverages = {"A": a_cov * 0.85, "B": b_cov, "C": c_cov, "D": d_cov, "E": e_cov, "F": f_cov, "G": g_cov}
    points = {key: (round(ratings[key] / 5 * MODULE_WEIGHTS[key], 2) if ratings[key] is not None else None) for key in MODULE_WEIGHTS}
    covered_weight = sum(MODULE_WEIGHTS[key] * coverages[key] for key in MODULE_WEIGHTS)
    realized = sum(points[key] * coverages[key] for key in MODULE_WEIGHTS if points[key] is not None) / (covered_weight / 100) if covered_weight else None
    realized = _clip(realized, 0, 100) if realized is not None else None
    adjustment = 0.0
    if forecast_cagr is not None:
        adjustment += 2.0 if forecast_cagr >= 0.20 else 1.0 if forecast_cagr >= 0.10 else -2.0 if forecast_cagr < -0.10 else -1.0 if forecast_cagr < 0 else 0.0
    latest_profit_yoy = _num(row.get("basic_eps_yoy"))
    if latest_profit_yoy is not None:
        adjustment += 1.0 if latest_profit_yoy >= 20 else -1.0 if latest_profit_yoy <= -20 else 0.0
    adjustment = _clip(round(adjustment * 2) / 2, -5, 5)
    forward = _clip((realized or 0) + adjustment, 0, 100) if realized is not None else None
    coverage = covered_weight / 100
    hard_gate = any(
        marker in audit_result
        for marker in ("否定意见", "无法表示意见", "无法表示", "拒绝表示")
    ) or ("保留意见" in audit_result and "无保留意见" not in audit_result)
    if hard_gate:
        classification = "硬门槛排除"
    elif coverage < 0.60 or realized is None:
        classification = "未评级"
    elif forward >= 85 and realized >= 82 and coverage >= 0.90 and all((points.get(k) or 0) >= v for k, v in {"B": 16, "D": 16, "E": 11, "G": 12}.items()):
        classification = "卓越复利候选"
    elif forward >= 75 and realized >= 70 and coverage >= 0.80 and all((points.get(k) or 0) >= v for k, v in {"B": 14, "D": 14, "E": 9, "G": 10}.items()):
        classification = "优质公司"
    elif forward >= 65:
        classification = "潜力公司"
    elif forward >= 55:
        classification = "普通公司"
    else:
        classification = "较弱公司"
    confidence = "高" if coverage >= 0.90 else "中" if coverage >= 0.75 else "低"
    return {
        "ratings": ratings, "module_scores": points, "module_coverages": {k: round(v, 4) for k, v in coverages.items()},
        "gqs_r": _round(realized, 2), "forward_adjustment": adjustment, "gqs_f": _round(forward, 2),
        "coverage_ratio": round(coverage, 4), "confidence": confidence, "classification": classification,
        "forecast_eps_cagr_2026_2028": _round(forecast_cagr, 4),
        "hard_gate": "nonstandard_audit_opinion" if hard_gate else "none_identified",
    }


def _scenario_valuation(
    row: pd.Series,
    gqs: dict[str, Any],
    family_median_forward_pe: float | None,
    family_median_pb: float | None = None,
) -> dict[str, Any]:
    price = _num(row.get("current_price")); eps = _num(row.get("eps_2027")); shares = _num(row.get("total_shares"))
    sample_count = _num(row.get("sample_count_2027")); pb = _num(row.get("pb")); roe = _num(row.get("roe_mean_5y"))
    missing: list[str] = []
    if price is None or price <= 0: missing.append("缺少截止时点价格")
    if shares is None or shares <= 0: missing.append("缺少最新股本")
    if sample_count is None or sample_count < 2: missing.append("一致预期样本不足2个")
    family = str(row.get("valuation_family"))
    if family == "bank_pb_roe":
        if pb is None or pb <= 0: missing.append("缺少P/B")
        if roe is None or roe <= 0: missing.append("缺少可持续ROE")
    elif eps is None or eps <= 0:
        missing.append("缺少正的FY2027每股盈利预测")
    if missing:
        return {"status": "unavailable", "method_primary": family, "method_crosscheck": None, "forecast_basis": "FY2027", "bear": None, "base": None, "bull": None, "missing_reasons": missing}

    dividend_return = (_num(row.get("dv_ttm")) or 0) / 100
    if family == "bank_pb_roe":
        current_bvps = price / pb
        payout = _clip(dividend_return * price / max(current_bvps * roe / 100, 0.01), 0.15, 0.65)
        bvps_target = current_bvps * (1 + roe / 100 * (1 - payout))
        parameters = {
            # The last parameter is an explicit A-share bank franchise and
            # asset-quality discount to the textbook Gordon-growth P/B.  A
            # pure formula otherwise overstates fair P/B when the market
            # persistently prices credit-cycle and governance uncertainty.
            "bear": (roe * 0.80, 0.02, 0.12, 0.90),
            "base": (roe, 0.03, 0.105, 0.78),
            "bull": (roe * 1.10, 0.04, 0.095, 0.82),
        }
        scenarios: dict[str, Any] = {}
        peer_anchor = family_median_pb if family_median_pb and family_median_pb > 0 else pb
        for name, (scenario_roe, growth, cost, franchise_discount) in parameters.items():
            theoretical_pb = (scenario_roe / 100 - growth) / max(cost - growth, 0.01)
            discounted_pb = theoretical_pb * franchise_discount
            if name == "base":
                discounted_pb = discounted_pb * 0.8 + peer_anchor * 0.2
            target_pb = _clip(discounted_pb, 0.6, 1.8)
            target = round(bvps_target * target_pb, 2)
            price_upside = round(target / price - 1, 4)
            scenarios[name] = {
                "conditions": f"可持续ROE {scenario_roe:.1f}%，长期增长 {growth:.1%}，股权成本 {cost:.1%}",
                "earnings_or_cashflow": round(bvps_target, 3), "multiple_or_rate": round(target_pb, 2),
                "target_price": target, "price_upside": price_upside,
                "dividend_return": round(dividend_return, 4), "total_return": round(price_upside + dividend_return, 4),
            }
        return {"status": "available", "method_primary": "P/B–ROE", "method_crosscheck": "当前P/B与股息率", "forecast_basis": "FY2027 BVPS", **scenarios, "missing_reasons": []}

    bounds = {
        "resource_midcycle": (6.0, 15.0), "utility_infrastructure": (8.0, 22.0),
        "real_estate_asset": (6.0, 20.0), "consumer_pe": (12.0, 35.0),
        "industrial_pe": (12.0, 35.0), "technology_pe": (18.0, 50.0),
        "medical_pe": (15.0, 40.0), "service_pe": (12.0, 35.0),
    }
    low, high = bounds.get(family, (10.0, 35.0))
    current_forward_pe = price / eps
    if family_median_forward_pe and math.isfinite(family_median_forward_pe) and current_forward_pe > 0:
        # Geometric blending preserves peer comparability without assuming a
        # lowly valued company automatically rerates all the way to the peer
        # median.  This is especially important when consensus embeds a sharp
        # recovery that the latest quarter has not yet confirmed.
        anchor = math.sqrt(family_median_forward_pe * current_forward_pe)
    else:
        anchor = current_forward_pe
    quality_factor = _clip(0.85 + ((gqs.get("gqs_r") or 65) - 65) * 0.01, 0.75, 1.25)
    base_multiple = _clip(anchor * quality_factor, low, high)
    latest_profit_yoy = _num(row.get("basic_eps_yoy"))
    if latest_profit_yoy is not None and latest_profit_yoy <= -30:
        base_multiple = _clip(base_multiple * 0.88, low, high)
    elif latest_profit_yoy is not None and latest_profit_yoy >= 30:
        base_multiple = _clip(base_multiple * 1.04, low, high)
    if family == "resource_midcycle":
        earnings_factors = {"bear": 0.65, "base": 0.90, "bull": 1.20}
    elif family == "utility_infrastructure":
        earnings_factors = {"bear": 0.88, "base": 1.00, "bull": 1.10}
    else:
        earnings_factors = {"bear": 0.80, "base": 1.00, "bull": 1.20}
    multiple_factors = {"bear": 0.72, "base": 1.0, "bull": 1.28}
    descriptions = {
        "bear": "盈利低于一致预期且估值收缩",
        "base": "FY2027一致预期大致兑现，估值回到质量调整后的同类中枢",
        "bull": "盈利超预期且竞争优势获得更高定价",
    }
    scenarios = {}
    for name in ("bear", "base", "bull"):
        scenario_eps = eps * earnings_factors[name]
        multiple = _clip(base_multiple * multiple_factors[name], low, high)
        target = round(scenario_eps * multiple, 2)
        price_upside = round(target / price - 1, 4)
        scenarios[name] = {
            "conditions": descriptions[name], "earnings_or_cashflow": round(scenario_eps, 3),
            "multiple_or_rate": round(multiple, 2), "target_price": target,
            "price_upside": price_upside, "dividend_return": round(dividend_return, 4),
            "total_return": round(price_upside + dividend_return, 4),
        }
    crosscheck = "正常化FCF收益率" if family not in {"technology_pe", "medical_pe"} else "增长与现金流敏感性"
    method = "中周期盈利×P/E" if family == "resource_midcycle" else "FY2027稀释EPS×目标P/E"
    return {"status": "available", "method_primary": method, "method_crosscheck": crosscheck, "forecast_basis": "FY2027", **scenarios, "missing_reasons": []}


def _research_narrative(row: pd.Series, gqs: dict[str, Any], valuation: dict[str, Any]) -> dict[str, Any]:
    name = str(row["name"]); scarcity = str(row["scarcity_hypothesis"])
    roe = _num(row.get("roe_mean_5y")); cash = _num(row.get("cashflow_quality_3y")); growth = _num(gqs.get("forecast_eps_cagr_2026_2028"))
    pillar1 = f"稀缺能力假设：{scarcity}；需继续用份额、客户认证、替代难度和利润池归属验证。"
    pillar2 = f"历史质量：5年ROE均值{roe:.1f}%" if roe is not None else "历史质量：ROE长期序列不完整"
    if cash is not None:
        pillar2 += f"，3年经营现金/利润{cash:.2f}倍。"
    else:
        pillar2 += "，现金流质量口径暂不完整。"
    pillar3 = f"前瞻：2026–2028年一致预期EPS复合增速{growth:.1%}。" if growth is not None else "前瞻：一致预期增速缺少完整可比口径。"
    weakest = min((score if score is not None else 999, key) for key, score in gqs["module_scores"].items())[1]
    counter = f"最强反方：{weakest}维度是当前最低分；稀缺性可能无法转化为持续的超额资本回报，且一致预期可能下修。"
    risks = ["一致预期下修或目标年盈利未兑现", "竞争加剧使毛利率、份额或现金转化下降", "治理、资本开支、并购或股本稀释超出当前证据"]
    if str(row.get("valuation_family")) == "resource_midcycle": risks.insert(0, "商品价格与成本周期逆转，峰值利润均值回归")
    if valuation["status"] == "unavailable": risks.append("估值证据门槛未通过，不能用目标价判断安全边际")
    base_upside = valuation.get("base", {}).get("price_upside") if valuation.get("base") else None
    stance = "积极观察" if (gqs.get("gqs_f") or 0) >= 75 and base_upside is not None and base_upside >= 0.15 else "中性观察" if (gqs.get("gqs_f") or 0) >= 65 else "谨慎观察"
    summary = f"{name}的GQS-R为{gqs.get('gqs_r'):.1f}、GQS-F为{gqs.get('gqs_f'):.1f}，分类为{gqs['classification']}；"
    summary += f"中性情景空间{base_upside:+.1%}。" if base_upside is not None else "目标价因证据门槛不足暂不可用。"
    return {
        "stance": stance, "summary": summary,
        "thesis_pillars": [pillar1, pillar2, pillar3], "strongest_counterargument": counter,
        "falsifiers": ["连续两个报告期ROE和现金利润比同时恶化", "核心份额或关键客户认证出现可验证倒退", "目标年盈利预期较当前下修20%以上"],
        "catalysts": ["定期报告验证盈利与现金流", "新产品/新产能/海外业务兑现并提高每股价值", "分红、回购或资本配置改善"],
        "risks": risks,
        "monitoring": ["季度收入与归母净利润增速", "ROE、毛利率和经营现金/利润", "一致预期样本、分歧和修订方向", "核心竞争地位与稀缺性反证", "股本、质押、处罚、并购与资本开支"],
    }


def evaluate_companies(paths: EvaluationPaths, *, analysis_cutoff: str, target_date: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cutoff = pd.Timestamp(analysis_cutoff)
    universe = build_universe(paths)
    codes = set(universe["ts_code"])
    basic = pd.read_parquet(paths.raw_dir / "stock_basic.parquet")
    basic["list_date_dt"] = pd.to_datetime(basic["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    basic["listed_years"] = (cutoff_naive - basic["list_date_dt"]).dt.days / 365.25
    annual = _latest_annual(paths.raw_dir, cutoff)
    latest_fina = _latest_period(paths.raw_dir / "fina_indicator.parquet", cutoff)
    latest_income = _latest_period(paths.raw_dir / "income.parquet", cutoff)
    daily_basic = pd.read_parquet(paths.daily_basic_snapshot).copy()
    daily_basic["total_shares"] = pd.to_numeric(daily_basic["total_share"], errors="coerce") * 10_000.0
    daily_basic["market_cap"] = pd.to_numeric(daily_basic["total_mv"], errors="coerce") * 10_000.0
    technical = _technical_table(paths.raw_dir, codes, "20260807")
    governance = _governance_table(paths.governance_dir, cutoff, "20260807")

    frame = universe.merge(basic[["ts_code", "symbol", "market", "listed_years"]], on="ts_code", how="left")
    frame = frame.merge(annual, on="ts_code", how="left", suffixes=("", "_annual"))
    fina_keep = [column for column in ("ts_code", "_ann", "_end", "eps", "roe", "roe_waa", "roa", "netprofit_margin", "grossprofit_margin", "debt_to_assets", "current_ratio", "quick_ratio", "basic_eps_yoy", "or_yoy") if column in latest_fina.columns]
    frame = frame.merge(latest_fina[fina_keep].rename(columns={"_ann": "latest_finance_ann", "_end": "latest_finance_period", "eps": "latest_eps", "roe": "latest_roe", "roe_waa": "latest_roe_waa", "roa": "latest_roa", "netprofit_margin": "latest_netprofit_margin", "grossprofit_margin": "latest_grossprofit_margin", "debt_to_assets": "latest_debt_to_assets", "current_ratio": "latest_current_ratio", "quick_ratio": "latest_quick_ratio"}), on="ts_code", how="left")
    income_keep = [column for column in ("ts_code", "revenue", "n_income_attr_p") if column in latest_income.columns]
    frame = frame.merge(latest_income[income_keep].rename(columns={"revenue": "latest_revenue", "n_income_attr_p": "latest_net_profit"}), on="ts_code", how="left")
    db_keep = [column for column in ("ts_code", "trade_date", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_shares", "market_cap", "turnover_rate", "volume_ratio") if column in daily_basic.columns]
    frame = frame.merge(daily_basic[db_keep], on="ts_code", how="left")
    frame = frame.merge(technical, on="ts_code", how="left")
    frame = frame.merge(governance, on="ts_code", how="left")
    forecasts = _forecast_table(paths.raw_dir, codes, cutoff, frame.set_index("ts_code")["total_shares"])
    frame = frame.merge(forecasts, on="ts_code", how="left")
    mcp_payload: dict[str, Any] = {}
    if paths.mcp_overrides_path is not None and paths.mcp_overrides_path.exists():
        mcp_payload = json.loads(paths.mcp_overrides_path.read_text(encoding="utf-8"))
        for code, override in mcp_payload.get("forecast_overrides", {}).items():
            mask = frame["ts_code"].eq(code)
            for field, value in override.items():
                if field != "reason":
                    frame.loc[mask, field] = value
    frame["is_financial"] = frame["industry"].astype(str).str.contains(FINANCIAL_PATTERN, regex=True, na=False)

    # Prefer latest-period ratios for display, keep annual multi-year metrics for GQS.
    for target, candidates in {
        "fina_roa": ("latest_roa", "fina_roa"), "fina_netprofit_margin": ("latest_netprofit_margin", "fina_netprofit_margin"),
        "fina_grossprofit_margin": ("latest_grossprofit_margin", "fina_grossprofit_margin"),
        "fina_debt_to_assets": ("latest_debt_to_assets", "fina_debt_to_assets"), "fina_current_ratio": ("latest_current_ratio", "fina_current_ratio"),
    }.items():
        left = pd.to_numeric(frame.get(candidates[0]), errors="coerce")
        right = pd.to_numeric(frame.get(candidates[1]), errors="coerce")
        frame[target] = left.fillna(right)
    frame["basic_eps_yoy"] = pd.to_numeric(frame.get("basic_eps_yoy"), errors="coerce")

    frame["forward_pe_2027"] = pd.to_numeric(frame["current_price"], errors="coerce") / pd.to_numeric(frame["eps_2027"], errors="coerce").where(pd.to_numeric(frame["eps_2027"], errors="coerce").gt(0))
    medians = frame.groupby("valuation_family")["forward_pe_2027"].median().to_dict()
    pb_medians = frame.groupby("valuation_family")["pb"].median().to_dict()
    evaluations: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        gqs = _module_scores(row)
        valuation = _scenario_valuation(
            row,
            gqs,
            _num(medians.get(row["valuation_family"])),
            _num(pb_medians.get(row["valuation_family"])),
        )
        research = _research_narrative(row, gqs, valuation)
        company_mcp_evidence = mcp_payload.get("company_evidence", {}).get(row["ts_code"])
        if company_mcp_evidence:
            research["risks"].insert(0, company_mcp_evidence["risk"])
            research["monitoring"].insert(0, company_mcp_evidence["monitoring"])
        score_evidence = [
            f"截至{pd.Timestamp(row['period_end']).date() if pd.notna(row.get('period_end')) else '未知'}的年度财务质量序列",
            f"{row['price_date']}未复权收盘与Tushare每日估值快照",
            f"FY2026–FY2028一致预期，FY2027样本{int(row['sample_count_2027']) if _num(row.get('sample_count_2027')) else '未知'}",
            f"业务稀缺性假设：{row['scarcity_hypothesis']}",
        ]
        score_limitations = ["A/B/G仍需要逐家公司一级信源和反证补强", "ROIC、客户集中度、处罚与质押字段在批量本地数据中并不完整"]
        sources = [
            {"label": "项目日行情", "kind": "reproducible_project_data", "path": "data/raw/daily_partitioned", "available_at": row.get("price_date")},
            {"label": "Tushare daily_basic", "kind": "structured_market_data", "path": str(paths.daily_basic_snapshot), "available_at": row.get("price_date")},
            {"label": "项目财务报表与指标", "kind": "statutory_filing_dataset", "path": "data/raw/{income,cashflow,balancesheet,fina_indicator}.parquet", "available_at": row.get("latest_finance_ann").strftime("%Y-%m-%d") if pd.notna(row.get("latest_finance_ann")) else None},
            {"label": "分析师一致预期", "kind": "consensus_forecast", "path": "data/raw/analyst_forecasts.parquet", "available_at": row.get("forecast_as_of_2027")},
        ]
        if paths.governance_dir is not None:
            sources.append({
                "label": "Tushare审计、质押、分红与回购快照",
                "kind": "structured_governance_data",
                "path": str(paths.governance_dir),
                "available_at": analysis_cutoff,
                "audit_result": row.get("audit_result"),
                "pledge_ratio": _round(row.get("pledge_ratio"), 4),
                "dividend_process": row.get("div_proc"),
                "repurchase_count": _round(row.get("repurchase_count"), 0),
            })
        mcp_queries = [
            query for query in mcp_payload.get("queries", [])
            if row["ts_code"] in query.get("companies", [])
        ]
        mcp_conflicts = [
            conflict for conflict in mcp_payload.get("data_conflicts", [])
            if conflict.get("ts_code") == row["ts_code"]
        ]
        if mcp_queries and paths.mcp_overrides_path is not None:
            sources.append({
                "label": "妙想一致预期复核",
                "kind": "mcp_financial_data",
                "path": str(paths.mcp_overrides_path),
                "available_at": analysis_cutoff,
            })
        if company_mcp_evidence:
            sources.append({
                "label": company_mcp_evidence["label"],
                "kind": "statutory_filing",
                "url": company_mcp_evidence["url"],
                "published_at": company_mcp_evidence["published_at"],
                "available_at": company_mcp_evidence["published_at"],
                "fact": company_mcp_evidence["fact"],
            })
        if str(row.get("scarcity_source_url") or "").startswith("http"):
            sources.append({"label": str(row.get("evidence_status") or "稀缺性初核"), "kind": "primary_or_company_source", "url": row["scarcity_source_url"], "available_at": None})
        item = {
            "identity": {
                "ts_code": row["ts_code"], "symbol": str(row.get("symbol") or row["ts_code"].split(".")[0]), "name": row["name"],
                "exchange": row["ts_code"].split(".")[-1], "market": row.get("market"), "industry": row["industry"], "broad_industry": row["broad_industry"],
                "in_broad_48": bool(row["in_broad_48"]), "in_niche_75": bool(row["in_niche_75"]),
                "scarcity_hypothesis": row["scarcity_hypothesis"], "valuation_family": row["valuation_family"],
            },
            "cutoff": {
                "analysis_cutoff": analysis_cutoff, "target_date": target_date, "price_date": row.get("price_date"),
                "finance_available_at": row.get("latest_finance_ann").strftime("%Y-%m-%d") if pd.notna(row.get("latest_finance_ann")) else None,
                "forecast_available_at": row.get("forecast_as_of_2027"),
                "latest_annual_period": pd.Timestamp(row["period_end"]).strftime("%Y-%m-%d") if pd.notna(row.get("period_end")) else None,
                "latest_interim_period": row.get("latest_finance_period").strftime("%Y-%m-%d") if pd.notna(row.get("latest_finance_period")) else None,
            },
            "market": {key: _round(row.get(key), 4) for key in ("current_price", "market_cap", "total_shares", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "return_20d", "return_60d", "return_120d", "volatility_20d", "drawdown_250d", "ma20_gap", "ma60_gap", "volume_ratio_20d")},
            "financials": {
                "revenue_latest": _round(row.get("latest_revenue"), 2), "net_profit_latest": _round(row.get("latest_net_profit"), 2),
                "revenue_yoy": _round(row.get("or_yoy"), 4), "net_profit_yoy": _round(row.get("basic_eps_yoy"), 4),
                "revenue_cagr_3y": _round(row.get("revenue_cagr_3y"), 4), "net_profit_cagr_3y": _round(row.get("net_income_cagr_3y"), 4),
                "gross_margin": _round(row.get("fina_grossprofit_margin"), 4), "net_margin": _round(row.get("fina_netprofit_margin"), 4),
                "roe": _round(row.get("roe_mean_5y"), 4), "roe_latest": _round(row.get("latest_roe_waa") or row.get("latest_roe"), 4), "roa": _round(row.get("fina_roa"), 4), "roic": None,
                "ocf_to_net_profit": _round(row.get("cashflow_quality_3y"), 4), "fcf_margin": _round(row.get("free_cashflow_margin_3y"), 4),
                "debt_to_assets": _round(row.get("fina_debt_to_assets"), 4), "current_ratio": _round(row.get("fina_current_ratio"), 4),
                "goodwill_to_assets": _round(row.get("annual_goodwill_to_assets"), 4), "profit_positive_share_5y": _round(row.get("profit_positive_share_5y"), 4),
                "net_debt_to_ebitda": None, "dividend_payout": None, "dilution_3y": None,
            },
            "forecast": {
                "fy1": 2026, "fy2": 2027, "fy3": 2028,
                "eps_fy1": _round(row.get("eps_2026"), 4), "eps_fy2": _round(row.get("eps_2027"), 4), "eps_fy3": _round(row.get("eps_2028"), 4),
                "net_profit_fy1": _round(row.get("net_profit_2026"), 2), "net_profit_fy2": _round(row.get("net_profit_2027"), 2), "net_profit_fy3": _round(row.get("net_profit_2028"), 2),
                "sample_count_fy2": _round(row.get("sample_count_2027"), 0), "dispersion_fy2": _round(row.get("dispersion_2027"), 4),
                "source": row.get("forecast_source_2027"), "as_of": row.get("forecast_as_of_2027"), "share_adjusted": bool(row.get("share_adjusted_2027", False)),
            },
            "gqs": {
                "a_customer_business": gqs["module_scores"]["A"], "b_scarcity_moat": gqs["module_scores"]["B"],
                "c_growth_reinvestment": gqs["module_scores"]["C"], "d_returns_profitability": gqs["module_scores"]["D"],
                "e_cash_accounting": gqs["module_scores"]["E"], "f_resilience_risk": gqs["module_scores"]["F"],
                "g_governance_allocation": gqs["module_scores"]["G"], "ratings_0_to_5": gqs["ratings"],
                "module_coverages": gqs["module_coverages"], "gqs_r": gqs["gqs_r"], "forward_adjustment": gqs["forward_adjustment"], "gqs_f": gqs["gqs_f"],
                "classification": gqs["classification"], "hard_gate": gqs["hard_gate"], "coverage_ratio": gqs["coverage_ratio"], "confidence": gqs["confidence"],
                "score_evidence": score_evidence, "score_limitations": score_limitations,
            },
            "valuation": valuation,
            "research": research,
            "evidence": {"sources": sources, "primary_source_count": sum(1 for source in sources if source["kind"] in {"primary_or_company_source", "statutory_filing"}), "source_tier_mix": {"structured": 5 + int(bool(mcp_queries)), "primary": sum(1 for source in sources if source["kind"] in {"primary_or_company_source", "statutory_filing"})}, "data_conflicts": mcp_conflicts, "mcp_execution": {"attempted": len(mcp_queries) + int(bool(company_mcp_evidence)), "succeeded": sum(query.get("status") in {"success", "success_empty"} for query in mcp_queries) + int(bool(company_mcp_evidence)), "results_adopted": int(row["ts_code"] in mcp_payload.get("forecast_overrides", {})) + int(bool(company_mcp_evidence))}},
        }
        item["market"]["technical_state"] = row.get("technical_state")
        evaluations.append(item)
    return frame, evaluations


def flatten_evaluation(item: dict[str, Any]) -> dict[str, Any]:
    identity = item["identity"]; market = item["market"]; gqs = item["gqs"]; valuation = item["valuation"]
    return {
        "ts_code": identity["ts_code"], "name": identity["name"], "industry": identity["industry"], "broad_industry": identity["broad_industry"],
        "in_broad_48": identity["in_broad_48"], "in_niche_75": identity["in_niche_75"], "valuation_family": identity["valuation_family"],
        "current_price": market["current_price"], "market_cap": market["market_cap"], "pe_ttm": market["pe_ttm"], "pb": market["pb"], "dividend_yield": market["dv_ttm"],
        "gqs_r": gqs["gqs_r"], "gqs_f": gqs["gqs_f"], "coverage_ratio": gqs["coverage_ratio"], "confidence": gqs["confidence"], "classification": gqs["classification"],
        "score_a": gqs["a_customer_business"], "score_b": gqs["b_scarcity_moat"], "score_c": gqs["c_growth_reinvestment"], "score_d": gqs["d_returns_profitability"], "score_e": gqs["e_cash_accounting"], "score_f": gqs["f_resilience_risk"], "score_g": gqs["g_governance_allocation"],
        "valuation_status": valuation["status"], "valuation_method": valuation["method_primary"],
        "bear_target": valuation.get("bear", {}).get("target_price") if valuation.get("bear") else None,
        "bear_upside": valuation.get("bear", {}).get("price_upside") if valuation.get("bear") else None,
        "base_target": valuation.get("base", {}).get("target_price") if valuation.get("base") else None,
        "base_upside": valuation.get("base", {}).get("price_upside") if valuation.get("base") else None,
        "bull_target": valuation.get("bull", {}).get("target_price") if valuation.get("bull") else None,
        "bull_upside": valuation.get("bull", {}).get("price_upside") if valuation.get("bull") else None,
        "technical_state": market.get("technical_state"), "return_60d": market.get("return_60d"),
        "stance": item["research"]["stance"], "summary": item["research"]["summary"], "scarcity_hypothesis": identity["scarcity_hypothesis"],
    }


def render_company_report(item: dict[str, Any]) -> str:
    i=item["identity"]; c=item["cutoff"]; m=item["market"]; f=item["financials"]; q=item["gqs"]; v=item["valuation"]; r=item["research"]
    def pct(value: Any) -> str:
        return "不可用" if _num(value) is None else f"{float(value):+.1%}"
    lines = [
        f"# {i['name']}（{i['ts_code']}）统一口径完整评估", "",
        f"分析截止：{c['analysis_cutoff']}  ", f"目标日期：{c['target_date']}  ", f"价格：{m['current_price']}元（{c['price_date']}，未复权收盘）", "",
        "## 结论先行", "", f"研究倾向：{r['stance']}｜置信度：{q['confidence']}｜证据覆盖率：{q['coverage_ratio']:.1%}", "", r["summary"], "",
        f"最强反方：{r['strongest_counterargument']}", "", "## 好公司评分（GQS）", "",
        "| 维度 | 得分 | 满分 |", "|---|---:|---:|",
        f"| A 客户价值与商业模式 | {q['a_customer_business']} | 10 |", f"| B 稀缺性、护城河与竞争地位 | {q['b_scarcity_moat']} | 20 |",
        f"| C 成长质量与再投资跑道 | {q['c_growth_reinvestment']} | 10 |", f"| D 资本回报与盈利能力 | {q['d_returns_profitability']} | 20 |",
        f"| E 现金流与会计质量 | {q['e_cash_accounting']} | 15 |", f"| F 抗风险性 | {q['f_resilience_risk']} | 10 |", f"| G 治理与资本配置 | {q['g_governance_allocation']} | 15 |",
        "", f"GQS-R {q['gqs_r']}；前瞻调整 {q['forward_adjustment']:+.1f}；GQS-F {q['gqs_f']}；分类：{q['classification']}。", "",
        "## 业务、财务与前瞻", "", *[f"- {pillar}" for pillar in r["thesis_pillars"]], "",
        f"最新期收入：{f['revenue_latest']}元；归母净利润：{f['net_profit_latest']}元；5年ROE均值：{f['roe']}%；3年经营现金/利润：{f['ocf_to_net_profit']}倍。", "",
        "## 12个月三情景估值", "",
    ]
    if v["status"] == "available":
        lines.extend(["| 情景 | 条件 | 核心盈利/每股净资产 | 倍数/PB | 目标价 | 价格空间 | 含股息总回报 |", "|---|---|---:|---:|---:|---:|---:|"])
        for key, label in (("bear","悲观"),("base","中性"),("bull","乐观")):
            s=v[key]; lines.append(f"| {label} | {s['conditions']} | {s['earnings_or_cashflow']} | {s['multiple_or_rate']} | {s['target_price']}元 | {pct(s['price_upside'])} | {pct(s['total_return'])} |")
        lines.extend(["", f"主方法：{v['method_primary']}；交叉检验：{v['method_crosscheck']}。", ""])
    else:
        lines.extend([f"估值状态：不可用。原因：{'；'.join(v['missing_reasons'])}。", ""])
    lines.extend(["## 催化剂、风险与监测", "", "催化剂：", "", *[f"- {x}" for x in r["catalysts"]], "", "主要风险：", "", *[f"- {x}" for x in r["risks"]], "", "证伪与监测：", "", *[f"- {x}" for x in r["falsifiers"]], *[f"- {x}" for x in r["monitoring"]], "", "## 证据与局限", ""])
    for source in item["evidence"]["sources"]:
        target = source.get("url") or source.get("path"); lines.append(f"- {source['label']}：{target}（可得时间：{source.get('available_at') or '待补'}）")
    lines.extend(["", *[f"- 局限：{x}" for x in q["score_limitations"]], "", "本报告基于统一批量口径与明确假设，仅供研究与教育用途，不构成个性化投资建议、收益承诺或交易指令。"])
    return "\n".join(lines) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, tuple): return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    if value is pd.NA: return None
    return value
