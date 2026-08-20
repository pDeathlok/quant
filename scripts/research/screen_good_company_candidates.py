#!/usr/bin/env python3
"""Build a reproducible, industry-neutral GQS stage-one candidate list.

This is deliberately a financial-quality proxy, not a full GQS rating.  It
uses only annual information available by the requested cutoff, separates the
financial-sector model, and attaches (but does not score) current consensus
forecasts for later human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.long_quality_factors import build_annual_quality_events


FINANCIAL_PATTERN = "银行|证券|保险|多元金融"
NONFINANCIAL_WEIGHTS = {
    "profitability_proxy": 0.30,
    "cash_accrual_proxy": 0.25,
    "stability_proxy": 0.20,
    "leverage_liquidity_proxy": 0.15,
    "per_share_growth_proxy": 0.10,
}
FINANCIAL_WEIGHTS = {
    "profitability_proxy": 0.45,
    "stability_proxy": 0.35,
    "per_share_growth_proxy": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="ISO timestamp or YYYY-MM-DD")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-percentile", type=float, default=0.80)
    parser.add_argument("--minimum-coverage", type=float, default=0.70)
    parser.add_argument("--top-per-broad-industry", type=int, default=3)
    parser.add_argument("--top-per-fine-industry", type=int, default=3)
    return parser.parse_args()


def broad_industry(industry: object) -> str:
    value = str(industry or "未知")
    rules = [
        ("金融", FINANCIAL_PATTERN),
        ("房地产与资产运营", "地产|房产|园区开发|商品城"),
        ("食品饮料与必选消费", "白酒|啤酒|红黄酒|食品|乳制品|软饮料|超市连锁|日用化工"),
        ("可选消费与服务", "家用电器|家居用品|汽车|摩托车|旅游|酒店餐饮|服饰|百货|文教休闲|电器连锁"),
        ("医药医疗", "医疗保健|化学制药|生物制药|中成药|医药商业"),
        ("信息技术与电子", "IT设备|元器件|半导体|互联网|软件服务|通信设备|电信运营"),
        ("工业制造", "专用机械|农用机械|化工机械|工程机械|机床制造|机械基件|轻工机械|纺织机械|电器仪表|电气设备|运输设备|航空|船舶"),
        ("材料与化工", "化工原料|化纤|塑料|染料涂料|橡胶|普钢|特种钢|钢加工|玻璃|水泥|陶瓷|其他建材|矿物制品|造纸|纺织"),
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


def _industry_percentile(frame: pd.DataFrame, column: str, *, high: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    industry = frame["industry"].fillna("未知").astype(str)
    broad = frame["broad_industry"].fillna("其他").astype(str)
    local_size = values.groupby(industry).transform("count")
    group = industry.where(local_size >= 5, broad)

    def winsorized(group_values: pd.Series) -> pd.Series:
        clean = group_values.dropna()
        if len(clean) < 5:
            return group_values
        lower, upper = clean.quantile([0.025, 0.975])
        return group_values.clip(lower, upper)

    clipped = values.groupby(group, group_keys=False).apply(winsorized)
    rank = clipped.groupby(group).rank(method="average", pct=True, ascending=high)
    market = clipped.rank(method="average", pct=True, ascending=high)
    return rank.where(local_size >= 5, market).mul(100.0)


def _weighted_score(
    components: dict[str, tuple[pd.Series, float]],
) -> tuple[pd.Series, pd.Series]:
    numerator = pd.Series(0.0, index=next(iter(components.values()))[0].index)
    denominator = pd.Series(0.0, index=numerator.index)
    total_weight = sum(weight for _, weight in components.values())
    for series, weight in components.values():
        numeric = pd.to_numeric(series, errors="coerce")
        numerator += numeric.fillna(0.0) * weight
        denominator += numeric.notna().astype(float) * weight
    return numerator / denominator.replace(0, np.nan), denominator / total_weight


def _load_latest_annual(raw_dir: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    events = build_annual_quality_events(
        pd.read_parquet(raw_dir / "fina_indicator.parquet"),
        pd.read_parquet(raw_dir / "income.parquet"),
        pd.read_parquet(raw_dir / "cashflow.parquet"),
        pd.read_parquet(raw_dir / "balancesheet.parquet"),
    )
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    events = events[events["annual_quality_available_at"].le(cutoff_naive)].copy()
    return (
        events.sort_values(["ts_code", "annual_quality_available_at", "period_end"])
        .groupby("ts_code", as_index=False)
        .tail(1)
    )


def _attach_forecasts(frame: pd.DataFrame, raw_dir: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    path = raw_dir / "analyst_forecasts.parquet"
    if not path.exists():
        return frame
    forecasts = pd.read_parquet(path)
    forecasts["report_date"] = pd.to_datetime(forecasts["report_date"], errors="coerce")
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    forecasts = forecasts[
        forecasts["report_date"].le(cutoff_naive)
        & forecasts["forecast_year"].isin([2026, 2027])
        & forecasts["source"].isin(["datayes_consensus", "akshare_em_snapshot"])
    ].copy()
    forecasts["source_priority"] = forecasts["source"].map(
        {"datayes_consensus": 2, "akshare_em_snapshot": 1}
    ).fillna(0)
    forecasts = (
        forecasts.sort_values(
            ["ts_code", "forecast_year", "report_date", "source_priority"]
        )
        .groupby(["ts_code", "forecast_year"], as_index=False)
        .tail(1)
    )
    keep = ["eps", "net_profit", "report_count", "analyst_count", "report_date", "source"]
    wide_parts: list[pd.DataFrame] = []
    for year in (2026, 2027):
        part = forecasts[forecasts["forecast_year"].eq(year)][["ts_code", *keep]].copy()
        part = part.rename(columns={column: f"forecast_{year}_{column}" for column in keep})
        wide_parts.append(part)
    out = frame.copy()
    for part in wide_parts:
        out = out.merge(part, on="ts_code", how="left")
    eps_26 = pd.to_numeric(out.get("forecast_2026_eps"), errors="coerce")
    eps_27 = pd.to_numeric(out.get("forecast_2027_eps"), errors="coerce")
    out["forecast_eps_growth_2027"] = eps_27 / eps_26.where(eps_26.gt(0)) - 1.0
    return out


def build_screen(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cutoff = pd.Timestamp(args.cutoff)
    latest = _load_latest_annual(args.raw_dir, cutoff)
    basic = pd.read_parquet(args.raw_dir / "stock_basic.parquet")
    frame = basic.merge(latest, on="ts_code", how="inner")
    frame["broad_industry"] = frame["industry"].map(broad_industry)
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
    frame["listed_years"] = (cutoff_naive - frame["list_date"]).dt.days / 365.25
    frame["is_financial"] = frame["industry"].astype(str).str.contains(
        FINANCIAL_PATTERN, regex=True, na=False
    )
    frame = frame[
        frame["listed_years"].ge(3)
        & pd.to_numeric(frame["annual_history_years"], errors="coerce").ge(3)
        & ~frame["name"].astype(str).str.upper().str.contains("ST|退", regex=True, na=False)
    ].copy()

    profitability, profitability_coverage = _weighted_score(
        {
            "roe": (_industry_percentile(frame, "roe_mean_5y", high=True), 0.65),
            "roa": (_industry_percentile(frame, "fina_roa", high=True), 0.35),
        }
    )
    cash, cash_coverage = _weighted_score(
        {
            "cash_conversion": (_industry_percentile(frame, "cashflow_quality_3y", high=True), 0.35),
            "fcf_margin": (_industry_percentile(frame, "free_cashflow_margin_3y", high=True), 0.25),
            "low_accrual": (_industry_percentile(frame, "accruals_to_assets_3y", high=False), 0.20),
            "cfo_persistence": (_industry_percentile(frame, "cfo_positive_share_5y", high=True), 0.20),
        }
    )
    stability, stability_coverage = _weighted_score(
        {
            "profit_years": (_industry_percentile(frame, "profit_positive_share_5y", high=True), 0.40),
            "low_roe_volatility": (_industry_percentile(frame, "roe_std_5y", high=False), 0.35),
            "revenue_years": (_industry_percentile(frame, "revenue_growth_positive_share_5y", high=True), 0.25),
        }
    )
    leverage, leverage_coverage = _weighted_score(
        {
            "low_debt": (_industry_percentile(frame, "fina_debt_to_assets", high=False), 0.50),
            "current_ratio": (_industry_percentile(frame, "fina_current_ratio", high=True), 0.25),
            "cash_assets": (_industry_percentile(frame, "annual_cash_to_assets", high=True), 0.25),
        }
    )
    per_share_growth, growth_coverage = _weighted_score(
        {
            "eps_growth": (_industry_percentile(frame, "fina_basic_eps_yoy", high=True), 0.50),
            "profit_cagr": (_industry_percentile(frame, "net_income_cagr_3y", high=True), 0.30),
            "revenue_cagr": (_industry_percentile(frame, "revenue_cagr_3y", high=True), 0.20),
        }
    )
    frame["profitability_proxy"] = profitability
    frame["cash_accrual_proxy"] = cash
    frame["stability_proxy"] = stability
    frame["leverage_liquidity_proxy"] = leverage
    frame["per_share_growth_proxy"] = per_share_growth

    nonfinancial_score, nonfinancial_coverage = _weighted_score(
        {
            "profitability": (profitability, NONFINANCIAL_WEIGHTS["profitability_proxy"]),
            "cash": (cash, NONFINANCIAL_WEIGHTS["cash_accrual_proxy"]),
            "stability": (stability, NONFINANCIAL_WEIGHTS["stability_proxy"]),
            "leverage": (leverage, NONFINANCIAL_WEIGHTS["leverage_liquidity_proxy"]),
            "growth": (per_share_growth, NONFINANCIAL_WEIGHTS["per_share_growth_proxy"]),
        }
    )
    financial_score, financial_coverage = _weighted_score(
        {
            "profitability": (profitability, FINANCIAL_WEIGHTS["profitability_proxy"]),
            "stability": (stability, FINANCIAL_WEIGHTS["stability_proxy"]),
            "growth": (per_share_growth, FINANCIAL_WEIGHTS["per_share_growth_proxy"]),
        }
    )
    frame["stage1_proxy_score"] = nonfinancial_score.where(~frame["is_financial"], financial_score)
    frame["stage1_proxy_coverage"] = nonfinancial_coverage.where(
        ~frame["is_financial"], financial_coverage
    )
    frame["stage1_model"] = np.where(frame["is_financial"], "financial_proxy_v0_2", "nonfinancial_proxy_v0_2")
    frame["within_industry_percentile"] = frame.groupby("industry")["stage1_proxy_score"].rank(
        method="average", pct=True
    )
    frame["candidate"] = (
        frame["stage1_proxy_coverage"].ge(args.minimum_coverage)
        & frame["within_industry_percentile"].ge(args.candidate_percentile)
        & pd.to_numeric(frame["profit_positive_share_5y"], errors="coerce").ge(0.80)
        & pd.to_numeric(frame["roe_mean_5y"], errors="coerce").gt(0)
    )
    frame["candidate_fail_reasons"] = ""
    fail_rules = [
        (~frame["stage1_proxy_coverage"].ge(args.minimum_coverage), "coverage"),
        (~frame["within_industry_percentile"].ge(args.candidate_percentile), "industry_percentile"),
        (~pd.to_numeric(frame["profit_positive_share_5y"], errors="coerce").ge(0.80), "profit_history"),
        (~pd.to_numeric(frame["roe_mean_5y"], errors="coerce").gt(0), "nonpositive_roe"),
    ]
    for failed, label in fail_rules:
        frame.loc[failed.fillna(True), "candidate_fail_reasons"] += label + ";"
    frame.loc[frame["candidate"], "candidate_fail_reasons"] = ""
    frame = _attach_forecasts(frame, args.raw_dir, cutoff)

    candidates = frame[frame["candidate"]].sort_values(
        ["broad_industry", "stage1_proxy_score"], ascending=[True, False]
    )
    shortlist = (
        candidates.groupby("broad_industry", group_keys=False)
        .head(args.top_per_broad_industry)
        .reset_index(drop=True)
    )
    metadata = {
        "analysis_cutoff": cutoff.isoformat(),
        "latest_annual_period": str(frame["period_end"].max().date()),
        "universe_after_history_and_listing_filters": int(len(frame)),
        "candidate_count": int(len(candidates)),
        "shortlist_count": int(len(shortlist)),
        "fine_industry_count": int(candidates["industry"].nunique()),
        "top_per_fine_industry": int(args.top_per_fine_industry),
        "candidate_percentile_threshold": args.candidate_percentile,
        "minimum_proxy_coverage": args.minimum_coverage,
        "nonfinancial_weights": NONFINANCIAL_WEIGHTS,
        "financial_weights": FINANCIAL_WEIGHTS,
        "warning": "Stage-one financial-quality proxy only; not a full GQS score or investment recommendation.",
    }
    return candidates, shortlist, frame, metadata


def _display_columns() -> list[str]:
    return [
        "broad_industry", "industry", "ts_code", "name", "stage1_model",
        "stage1_proxy_score", "stage1_proxy_coverage", "within_industry_percentile",
        "period_end", "roe_mean_5y", "roe_std_5y", "cashflow_quality_3y",
        "free_cashflow_margin_3y", "profit_positive_share_5y", "fina_debt_to_assets",
        "net_income_cagr_3y", "forecast_2026_eps", "forecast_2027_eps",
        "forecast_eps_growth_2027", "forecast_2026_report_count",
        "forecast_2026_report_date", "forecast_2026_source",
    ]


def write_outputs(
    candidates: pd.DataFrame,
    shortlist: pd.DataFrame,
    screened_universe: pd.DataFrame,
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [column for column in _display_columns() if column in candidates.columns]
    fine_shortlist = (
        candidates.sort_values(
            ["broad_industry", "industry", "stage1_proxy_score"],
            ascending=[True, True, False],
        )
        .groupby("industry", group_keys=False)
        .head(int(metadata["top_per_fine_industry"]))
        .copy()
    )
    fine_shortlist["fine_industry_rank"] = (
        fine_shortlist.groupby("industry")["stage1_proxy_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    metadata["fine_shortlist_count"] = int(len(fine_shortlist))
    candidates[columns].to_csv(output_dir / "all_candidates.csv", index=False)
    universe_columns = [
        "candidate", "candidate_fail_reasons", *columns,
    ]
    screened_universe[
        [column for column in universe_columns if column in screened_universe.columns]
    ].to_csv(output_dir / "screened_universe.csv", index=False)
    shortlist[columns].to_csv(output_dir / "industry_shortlist.csv", index=False)
    fine_columns = ["fine_industry_rank", *columns]
    fine_shortlist[fine_columns].to_csv(
        output_dir / "fine_industry_shortlist.csv", index=False
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# A股好公司阶段一候选",
        "",
        f"分析截止：{metadata['analysis_cutoff']}",
        "",
        "> 这是行业中性的财务质量初筛代理，不是完整GQS，也不包含估值、价格或买卖建议。",
        "",
        f"初筛后样本：{metadata['universe_after_history_and_listing_filters']}｜候选：{metadata['candidate_count']}｜分行业短名单：{metadata['shortlist_count']}",
        "",
        "| 大类行业 | 细分行业 | 代码 | 公司 | 初筛分 | 覆盖率 | 5年ROE均值 | 3年现金利润比 | 5年盈利正值率 | 资产负债率 | FY2027E EPS增速 | 预测样本 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in shortlist.itertuples(index=False):
        def value(name: str, scale: float = 1.0, suffix: str = "") -> str:
            item = getattr(row, name, np.nan)
            if pd.isna(item):
                return "—"
            return f"{float(item) * scale:.1f}{suffix}"

        lines.append(
            "| {broad} | {industry} | {code} | {name} | {score} | {coverage} | {roe} | {cash} | {profit} | {debt} | {growth} | {sample} |".format(
                broad=row.broad_industry,
                industry=row.industry,
                code=row.ts_code,
                name=row.name,
                score=value("stage1_proxy_score"),
                coverage=value("stage1_proxy_coverage", 100, "%"),
                roe=value("roe_mean_5y", 1, "%"),
                cash=value("cashflow_quality_3y"),
                profit=value("profit_positive_share_5y", 100, "%"),
                debt=value("fina_debt_to_assets", 1, "%"),
                growth=value("forecast_eps_growth_2027", 100, "%"),
                sample=value("forecast_2026_report_count"),
            )
        )
    lines.extend(
        [
            "",
            "## 完整结果入口",
            "",
            f"- `fine_industry_catalog.md`：{metadata['fine_industry_count']}个细分行业、{metadata['fine_shortlist_count']}家逐行业候选。",
            f"- `all_candidates.csv`：{metadata['candidate_count']}家通过阶段一门槛的完整候选池。",
            f"- `screened_universe.csv`：{metadata['universe_after_history_and_listing_filters']}家公司及未入池原因，便于追溯遗漏。",
            "- `niche_capability_watchlist.md`：业务级稀缺能力观察池，独立于证券行业分类。",
            "",
            "## 使用边界",
            "",
            "- 非金融代理按盈利30%、现金/应计25%、稳定性20%、杠杆15%、每股增长10%合成；金融业使用单独的盈利、稳定与增长代理。",
            "- 行业内2.5%/97.5%缩尾并排名，取细分行业前20%；预测只附在结果中，未进入阶段一实际质量分。",
            "- 下一步仍需逐家核验红线、客户价值、稀缺性、增量ROIC、治理与预测离散度后，才能形成完整GQS。",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    catalog_lines = [
        "# A股好公司细分行业完整候选目录",
        "",
        f"分析截止：{metadata['analysis_cutoff']}",
        "",
        "> 覆盖本次可复算初筛中的全部细分行业；每个细分行业最多保留财务质量代理分前三。它是研究入口，不是完整GQS结论或投资建议。",
        "",
        (
            f"细分行业：{metadata['fine_industry_count']}｜目录公司："
            f"{metadata['fine_shortlist_count']}｜完整候选池：{metadata['candidate_count']}"
        ),
        "",
    ]
    for broad, broad_frame in fine_shortlist.groupby("broad_industry", sort=True):
        catalog_lines.extend([f"## {broad}", ""])
        for industry, industry_frame in broad_frame.groupby("industry", sort=True):
            catalog_lines.extend(
                [
                    f"### {industry}",
                    "",
                    "| 排名 | 代码 | 公司 | 初筛分 | 行业内百分位 | 5年ROE均值 | 3年现金利润比 | 资产负债率 | FY2027E EPS增速 | 预测样本 |",
                    "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in industry_frame.itertuples(index=False):
                def item(name: str, scale: float = 1.0, suffix: str = "") -> str:
                    value = getattr(row, name, np.nan)
                    if pd.isna(value):
                        return "—"
                    return f"{float(value) * scale:.1f}{suffix}"

                catalog_lines.append(
                    "| {rank} | {code} | {name} | {score} | {percentile} | {roe} | {cash} | {debt} | {growth} | {sample} |".format(
                        rank=row.fine_industry_rank,
                        code=row.ts_code,
                        name=row.name,
                        score=item("stage1_proxy_score"),
                        percentile=item("within_industry_percentile", 100, "%"),
                        roe=item("roe_mean_5y", 1, "%"),
                        cash=item("cashflow_quality_3y"),
                        debt=item("fina_debt_to_assets", 1, "%"),
                        growth=item("forecast_eps_growth_2027", 100, "%"),
                        sample=item("forecast_2026_report_count"),
                    )
                )
            catalog_lines.append("")
    catalog_lines.extend(
        [
            "## 阅读方法",
            "",
            "- 目录解决“大类前三遮蔽细分冠军”的问题；大类均衡短名单仍单独保留，不能替代本目录。",
            "- 初筛分只衡量已实现的财务质量代理；业务稀缺性、竞争壁垒、客户价值和治理仍需逐家深评。",
            "- 现金利润比、负债率等跨行业不可机械横比；本目录的核心排序是公司在自身细分行业中的相对位置。",
        ]
    )
    (output_dir / "fine_industry_catalog.md").write_text(
        "\n".join(catalog_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    candidates, shortlist, screened_universe, metadata = build_screen(args)
    write_outputs(candidates, shortlist, screened_universe, metadata, args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
