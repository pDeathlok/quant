"""
批量下载沪深主板+创业板股票日线行情数据 (字段完整版)

主要数据源: akshare.stock_zh_a_daily (9个字段)
  - date, open, high, low, close, volume, amount, outstanding_share, turnover

备用数据源: akshare.stock_zh_a_hist (12个字段, 含振幅/涨跌幅/换手率等)
  - 当 stock_zh_a_daily 不可用时自动切换

用法:
    # 下载全部沪深主板+创业板股票 (默认从2010年至今)
    python scripts/data/download_all_daily.py

    # 仅下载创业板
    python scripts/data/download_all_daily.py --board gem

    # 仅下载主板
    python scripts/data/download_all_daily.py --board main

    # 指定日期范围
    python scripts/data/download_all_daily.py --start 20200101 --end 20260527

    # 调整并发数
    python scripts/data/download_all_daily.py --workers 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak
import pandas as pd
from tqdm import tqdm

from quant.data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 全局标记: 自动检测到的可用 API
_api_prefers_hist = None


def _probe_api() -> str:
    """探测哪个数据源可用, 返回 'daily' 或 'hist'"""
    global _api_prefers_hist
    if _api_prefers_hist is not None:
        return "hist" if _api_prefers_hist else "daily"
    try:
        df = ak.stock_zh_a_daily(symbol="sz000001", start_date="20250101", end_date="20250105", adjust="qfq")
        if df is not None and len(df) > 0:
            _api_prefers_hist = False
            logger.info("API 探测: stock_zh_a_daily 可用")
            return "daily"
    except Exception:
        pass
    try:
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20250101", end_date="20250105", adjust="qfq")
        if df is not None and len(df) > 0:
            _api_prefers_hist = True
            logger.info("API 探测: stock_zh_a_hist 可用")
            return "hist"
    except Exception:
        pass
    logger.warning("API 探测: 所有数据源均不可用!")
    return "daily"  # 默认


def _to_exchange_symbol(code: str) -> str:
    """将6位纯代码转换为带 sh/sz 前缀的格式"""
    if code.startswith("6"):
        return f"sh{code}"
    else:
        return f"sz{code}"


def get_stock_symbols(board: str = "all") -> list[tuple[str, str]]:
    """获取股票代码列表 (代码不含 sh/sz 前缀, 适配 stock_zh_a_hist)

    Args:
        board: "all"=全部主板+创业板, "gem"=仅创业板

    Returns:
        [(code, name), ...] 如 [("600000", "浦发银行"), ("300750", "宁德时代")]
    """
    results: list[tuple[str, str]] = []

    # 深市 (主板 + 创业板)
    try:
        logger.info("获取深市股票列表...")
        df_sz = ak.stock_info_sz_name_code()
        for _, row in df_sz.iterrows():
            code = str(row["A股代码"]).zfill(6)
            name = str(row["A股简称"]).strip()
            sector = str(row.get("板块", ""))

            # 创业板: 300xxx, 301xxx
            is_gem = code.startswith("30")
            # 深市主板: 000xxx ~ 003xxx
            is_sz_main = code.startswith("00")

            if board == "gem" and not is_gem:
                continue
            if board == "all" and not (is_gem or is_sz_main):
                continue

            results.append((code, name))
        logger.info(f"深市获取到 {len(results)} 只股票")
    except Exception as e:
        logger.error(f"获取深市列表失败: {e}")

    # 沪市 (主板)
    if board != "gem":
        try:
            logger.info("获取沪市股票列表...")
            df_sh = ak.stock_info_sh_name_code()
            sh_count = 0
            for _, row in df_sh.iterrows():
                code = str(row["证券代码"]).zfill(6)
                name = str(row["证券简称"]).strip()
                if code.startswith("6"):
                    results.append((code, name))
                    sh_count += 1
            logger.info(f"沪市获取到 {sh_count} 只股票")
        except Exception as e:
            logger.error(f"获取沪市列表失败: {e}")

    logger.info(f"合计获取到 {len(results)} 只股票")
    return results


def fetch_single_stock_daily(
    code: str,
    name: str,
    start_date: str,
    end_date: str,
    adjust: str,
    save_dir: Path,
    api: str = "daily",
    max_retries: int = 5,
) -> tuple[str, str, int, str | None]:
    """下载单只股票日线数据

    Args:
        api: "daily"=stock_zh_a_daily (9字段), "hist"=stock_zh_a_hist (12字段)

    Returns:
        (code, name, row_count, error_or_None)
    """
    storage = DataStorage(data_dir=str(save_dir))

    for attempt in range(1, max_retries + 1):
        try:
            if api == "daily":
                sym = _to_exchange_symbol(code)
                df = ak.stock_zh_a_daily(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                    "amount": "turnover",
                    "outstanding_share": "outstanding_share",
                    "turnover": "turnover_rate",
                })
            else:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
                df = df.rename(columns={
                    "日期": "date",
                    "股票代码": "code",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "turnover",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "price_change",
                    "换手率": "turnover_rate",
                })

            if df is None or df.empty:
                return code, name, 0, "无数据"

            # 添加元信息
            df["symbol"] = code
            df["name"] = name
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            # 数值类型
            for col in df.columns:
                if col not in ("date", "symbol", "name", "code"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("date").reset_index(drop=True)
            storage.save_parquet(df, filename=code)
            return code, name, len(df), None

        except Exception as e:
            err_msg = str(e)
            if attempt < max_retries:
                time.sleep(attempt * 2)
                continue
            return code, name, 0, err_msg


def download_all_daily(
    board: str = "all",
    start_date: str = "20100101",
    end_date: str | None = None,
    adjust: str = "qfq",
    save_dir: str = "./data/stocks_daily",
    workers: int = 5,
    sleep_between: float = 0.3,
):
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 探测可用 API
    api = _probe_api()
    if api == "daily":
        logger.info("使用 stock_zh_a_daily 接口 (9字段: OHLCV + 成交额 + 流通股本 + 换手率)")
    else:
        logger.info("使用 stock_zh_a_hist 接口 (12字段: OHLCV + 振幅 + 涨跌幅 + 涨跌额 + 换手率)")

    # 获取股票列表
    symbols = get_stock_symbols(board)
    if not symbols:
        logger.error("未获取到任何股票列表, 退出")
        return

    logger.info(
        f"开始下载: board={board}, start={start_date}, end={end_date}, "
        f"adjust={adjust}, workers={workers}"
    )
    logger.info(f"保存目录: {save_path}")

    success = 0
    fail = 0
    total_rows = 0
    failed_list: list[tuple[str, str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_single_stock_daily,
                code, name, start_date, end_date, adjust, save_path, api,
            ): (code, name)
            for code, name in symbols
        }

        with tqdm(total=len(futures), desc="下载进度") as pbar:
            for future in as_completed(futures):
                code, name, rows, error = future.result()
                time.sleep(sleep_between)

                if error is None and rows > 0:
                    success += 1
                    total_rows += rows
                    tqdm.write(f"  [{success}] {code} {name}: {rows} 条")
                else:
                    fail += 1
                    failed_list.append((code, name, error or "无数据"))
                    tqdm.write(f"  [FAIL] {code} {name}: {error or '无数据'}")

                pbar.update(1)

    # 汇总统计
    logger.info("=" * 60)
    logger.info(f"下载完成!")
    logger.info(f"成功: {success} 只")
    logger.info(f"失败: {fail} 只")
    logger.info(f"总记录数: {total_rows:,}")
    logger.info(f"保存位置: {save_path}")

    if failed_list:
        logger.warning(f"失败列表 ({len(failed_list)} 只):")
        for code, name, reason in failed_list[:30]:
            logger.warning(f"  {code} {name}: {reason}")
        if len(failed_list) > 30:
            logger.warning(f"  ... 还有 {len(failed_list) - 30} 个")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="批量下载沪深主板+创业板股票日线行情数据"
    )
    parser.add_argument(
        "--board", "-b",
        default="all",
        choices=["all", "gem", "main"],
        help="板块: all=沪深主板+创业板, gem=仅创业板, main=仅主板",
    )
    parser.add_argument("--start", "-s", default="20100101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", "-e", default=None, help="结束日期 YYYYMMDD, 默认今天")
    parser.add_argument(
        "--adjust", "-a",
        default="qfq",
        choices=["qfq", "hfq", "none"],
        help="复权方式: qfq=前复权, hfq=后复权, none=不复权",
    )
    parser.add_argument("--save-dir", "-d", default="./data/stocks_daily", help="保存目录")
    parser.add_argument("--workers", "-w", type=int, default=5, help="并发线程数 (建议3-8)")
    parser.add_argument("--sleep", type=float, default=0.3, help="每只股票请求间隔(秒), 防封")

    args = parser.parse_args()

    adjust = args.adjust if args.adjust != "none" else None

    download_all_daily(
        board=args.board,
        start_date=args.start,
        end_date=args.end,
        adjust=adjust,
        save_dir=args.save_dir,
        workers=args.workers,
        sleep_between=args.sleep,
    )


if __name__ == "__main__":
    main()
