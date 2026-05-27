"""
批量下载因子建模所需的辅助数据 (AKShare 源)

数据分类及对应因子:
  1. 财务数据 (多期) → 价值/质量/成长/杠杆/盈利因子
  2. 指数日线      → 市场基准收益率 (RSTR)
  3. 宏观数据      → 择时因子
  4. 国债利率      → 无风险利率
  5. 北向资金      → 聪明钱因子
  6. 龙虎榜        → 异动/游资因子

保存目录: data/factors_raw/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import time
import logging
from datetime import datetime

import akshare as ak
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_DIR = Path("./data/factors_raw")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RETRY = 3
RETRY_DELAY = 3


def safe_download(name: str, func, **kwargs) -> pd.DataFrame | None:
    for attempt in range(1, RETRY + 1):
        try:
            time.sleep(0.5)
            df = func(**kwargs)
            if df is not None and not df.empty:
                logger.info(f"  OK {name}: {df.shape}")
                return df
            else:
                logger.warning(f"  EMPTY {name}")
                return None
        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{RETRY} {name} failed: {e}")
            if attempt < RETRY:
                time.sleep(RETRY_DELAY * attempt)
    logger.error(f"  FAIL {name}: all retries exhausted")
    return None


def save(df: pd.DataFrame, name: str):
    path = SAVE_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / 1024 / 1024
    logger.info(f"  Saved {path.name} ({df.shape[0]} rows, {df.shape[1]} cols, {size_mb:.1f} MB)")


def download_financial():
    """1. 多期业绩报表 + 资产负债表 (价值/质量/成长/杠杆/盈利因子)

    字段映射:
    - 每股收益     → PE, PEG
    - 每股净资产   → PB
    - 净资产收益率 → ROE
    - 销售毛利率   → GrossProfitMargin
    - 营业总收入   → PS, RevenueGrowth
    - 净利润       → NetMargin, NetProfitYoY
    - 总资产       → ROA
    - 资产负债率   → DebtToEquity
    """
    logger.info("=" * 60)
    logger.info("1. 财务数据 - 业绩报表 + 资产负债表 (多期)")

    periods = ["20251231", "20250930", "20250630", "20250331",
               "20241231", "20240930", "20240630", "20240331",
               "20231231", "20230930"]

    # 业绩报表
    all_yjbb = []
    for period in periods:
        df = safe_download(f"yjbb_{period}", ak.stock_yjbb_em, date=period)
        if df is not None:
            df["report_date"] = period
            all_yjbb.append(df)

    if all_yjbb:
        combined = pd.concat(all_yjbb, ignore_index=True)
        save(combined, "financial_yjbb_multi")

    # 资产负债表
    all_zcfz = []
    for period in periods:
        df = safe_download(f"zcfz_{period}", ak.stock_zcfz_em, date=period)
        if df is not None:
            df["report_date"] = period
            all_zcfz.append(df)

    if all_zcfz:
        combined = pd.concat(all_zcfz, ignore_index=True)
        save(combined, "balance_sheet_zcfz_multi")


def download_index():
    """2. 指数日线行情 (市场基准收益率 / RSTR)"""
    logger.info("=" * 60)
    logger.info("2. 指数日线行情 (市场基准)")

    indices = {
        "000001": "上证指数",
        "000300": "沪深300",
        "000905": "中证500",
        "000852": "中证1000",
        "399001": "深证成指",
        "399006": "创业板指",
    }

    all_idx = []
    for code, name in indices.items():
        df = safe_download(
            f"index_{code}",
            ak.index_zh_a_hist,
            symbol=code,
            period="daily",
            start_date="20100101",
            end_date="20260527",
        )
        if df is not None:
            df["index_code"] = code
            df["index_name"] = name
            all_idx.append(df)

    if all_idx:
        combined = pd.concat(all_idx, ignore_index=True)
        save(combined, "index_daily_multi")


def download_macro():
    """3. 宏观经济数据 (择时因子)"""
    logger.info("=" * 60)
    logger.info("3. 宏观经济数据")

    macro_funcs = {
        "macro_cpi": ak.macro_china_cpi,
        "macro_ppi": ak.macro_china_ppi,
        "macro_gdp": ak.macro_china_gdp,
        "macro_pmi": ak.macro_china_pmi_yearly,
        "macro_money_supply": ak.macro_china_money_supply,
        "macro_shrzgm": ak.macro_china_shrzgm,
        "macro_lpr": ak.macro_china_lpr,
        "macro_cnbs": ak.macro_cnbs,
        "macro_fx_reserves": ak.macro_china_fx_reserves_yearly,
    }

    for name, func in macro_funcs.items():
        df = safe_download(name, func)
        if df is not None:
            save(df, name)


def download_bond_yield():
    """4. 国债收益率 (无风险利率)"""
    logger.info("=" * 60)
    logger.info("4. 利率/国债收益率")

    df = safe_download("bond_yield", ak.bond_china_yield)
    if df is not None:
        save(df, "bond_yield_curve")

    df2 = safe_download("bond_zh_us_rate", ak.bond_zh_us_rate)
    if df2 is not None:
        save(df2, "bond_zh_us_rate")


def download_north_flow():
    """5. 北向资金 (聪明钱因子)"""
    logger.info("=" * 60)
    logger.info("5. 北向资金")

    df = safe_download("hsgt_fund_flow", ak.stock_hsgt_fund_flow_summary_em)
    if df is not None:
        save(df, "hsgt_fund_flow_summary")

    df2 = safe_download("hsgt_hold_stock", ak.stock_hsgt_hold_stock_em)
    if df2 is not None:
        save(df2, "hsgt_hold_stock")


def download_lhb():
    """6. 龙虎榜 (异动/游资因子)"""
    logger.info("=" * 60)
    logger.info("6. 龙虎榜")

    years = ["2023", "2024", "2025", "2026"]
    all_lhb = []
    for year in years:
        df = safe_download(
            f"lhb_{year}",
            ak.stock_lhb_detail_em,
            start_date=f"{year}0101",
            end_date=f"{year}1231",
        )
        if df is not None:
            all_lhb.append(df)

    if all_lhb:
        combined = pd.concat(all_lhb, ignore_index=True)
        save(combined, "lhb_detail_multi_year")


def download_board():
    """7. 板块分类 (行业因子)"""
    logger.info("=" * 60)
    logger.info("7. 板块分类")

    df = safe_download("board_industry", ak.stock_board_industry_name_em)
    if df is not None:
        save(df, "board_industry_name")

    df2 = safe_download("board_concept", ak.stock_board_concept_name_em)
    if df2 is not None:
        save(df2, "board_concept_name")


def download_spot():
    """8. 全市场实时快照 (截面因子: PE/PB/市值等)"""
    logger.info("=" * 60)
    logger.info("8. 全市场实时快照")

    df = safe_download("spot_em", ak.stock_zh_a_spot_em)
    if df is not None:
        save(df, "market_snapshot_spot")


def download_st():
    """9. ST 股票 (风控过滤)"""
    logger.info("=" * 60)
    logger.info("9. ST 列表")

    df = safe_download("st_list", ak.stock_zh_a_st_em)
    if df is not None:
        save(df, "st_stock_list")


def main():
    logger.info("开始下载因子建模辅助数据")
    logger.info(f"保存目录: {SAVE_DIR}")
    logger.info("")

    tasks = [
        ("1. 财务数据", download_financial),
        ("2. 指数行情", download_index),
        ("3. 宏观数据", download_macro),
        ("4. 国债利率", download_bond_yield),
        ("5. 北向资金", download_north_flow),
        ("6. 龙虎榜", download_lhb),
        ("7. 板块分类", download_board),
        ("8. 实时快照", download_spot),
        ("9. ST 列表", download_st),
    ]

    for name, func in tasks:
        try:
            func()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("用户中断")
            break
        except Exception as e:
            logger.error(f"{name} 异常: {e}")
            time.sleep(2)

    # 汇总
    logger.info("=" * 60)
    logger.info("下载完成!")
    files = list(SAVE_DIR.glob("*.parquet"))
    total_rows = 0
    for f in sorted(files):
        df = pd.read_parquet(f)
        total_rows += len(df)
        size_mb = f.stat().st_size / 1024 / 1024
        logger.info(f"  {f.name}: {df.shape} | {size_mb:.1f} MB")
    logger.info(f"合计: {len(files)} 个文件, {total_rows:,} 条记录")
    logger.info(f"总大小: {sum(f.stat().st_size for f in files) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
