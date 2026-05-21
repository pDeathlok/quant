"""
批量下载股票数据 (多线程并发)

Usage: python scripts/data/batch_download.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import akshare as ak
from datetime import datetime
from quant.data.fetcher import DataFetcher
from quant.data.storage import DataStorage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_top_stocks(n: int = 200) -> list:
    """获取市值排名前 n 的股票列表"""
    print(f"正在获取市值排名前 {n} 的股票列表...")
    
    try:
        # 获取所有 A 股股票信息
        stock_list = ak.stock_zh_a_spot()
        
        # 按市值排序，取前 n 只
        if "总市值" in stock_list.columns:
            stock_list = stock_list.sort_values("总市值", ascending=False).head(n)
        elif "流通市值" in stock_list.columns:
            stock_list = stock_list.sort_values("流通市值", ascending=False).head(n)
        else:
            # 如果没有市值列，按代码排序取前 n 只
            stock_list = stock_list.head(n)
        
        symbols = stock_list["代码"].tolist()
        names = stock_list["名称"].tolist()
        
        print(f"已选择 {len(symbols)} 只股票")
        return list(zip(symbols, names))
    
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        # 返回默认的股票列表
        default_stocks = [
            "sh600000", "sh600519", "sh601318", "sh601398", "sh601988",
            "sz000001", "sz000858", "sz002594", "sz300750", "sz300059"
        ]
        return [(s, s) for s in default_stocks]


def download_single_stock(symbol_name, start_date, end_date, adjust, cache_dir):
    """下载单只股票数据"""
    symbol, name = symbol_name
    try:
        fetcher = DataFetcher(cache_dir=cache_dir)
        df = fetcher.get_stock_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        return symbol, name, len(df), None
    except Exception as e:
        return symbol, name, 0, str(e)


def batch_download_stocks(
    n_stocks: int = 200,
    start_date: str = "20100101",
    end_date: str = None,
    adjust: str = "qfq",
    cache_dir: str = "./data/cache",
    max_workers: int = 5
):
    """
    批量下载股票数据
    
    Args:
        n_stocks: 股票数量
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权方式
        cache_dir: 缓存目录
        max_workers: 并发线程数
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    # 获取股票列表
    stocks = get_top_stocks(n_stocks)
    
    print(f"\n开始批量下载 {len(stocks)} 只股票数据")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"复权方式: {adjust}")
    print(f"并发线程: {max_workers}")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    failed_stocks = []
    total_records = 0

    # 使用多线程并发下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_stock, stock, start_date, end_date, adjust, cache_dir): stock
            for stock in stocks
        }

        with tqdm(total=len(futures), desc="下载进度") as pbar:
            for future in as_completed(futures):
                symbol, name, records, error = future.result()
                pbar.update(1)
                
                if error is None and records > 0:
                    success_count += 1
                    total_records += records
                    tqdm.write(f"✓ {symbol} ({name}): {records} 条")
                else:
                    fail_count += 1
                    failed_stocks.append((symbol, name, error or "无数据"))
                    tqdm.write(f"✗ {symbol} ({name}): {error or '无数据'}")

    print("=" * 60)
    print(f"\n批量下载完成!")
    print(f"成功: {success_count} 只股票")
    print(f"失败: {fail_count} 只股票")
    print(f"总数据条数: {total_records:,}")

    if failed_stocks:
        print("\n失败列表 (前10个):")
        for symbol, name, reason in failed_stocks[:10]:
            print(f"  {symbol} ({name}): {reason}")

        if len(failed_stocks) > 10:
            print(f"  ... 还有 {len(failed_stocks) - 10} 个失败")

    # 将成功下载的数据复制到 stocks 目录
    storage = DataStorage(data_dir="./data/stocks")
    fetcher = DataFetcher(cache_dir=cache_dir)
    for symbol, name in stocks:
        try:
            df = pd.read_parquet(f"{cache_dir}/{symbol}_{start_date}_{end_date}_{adjust}.parquet")
            storage.save_parquet(df, filename=f"{symbol}")
        except:
            pass

    return success_count, fail_count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="批量下载沪深股票日线数据")
    parser.add_argument("--count", "-n", type=int, default=200, help="股票数量")
    parser.add_argument("--start", "-b", default="20100101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", "-e", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--adjust", "-adj", default="qfq", choices=["qfq", "hfq", "none"], help="复权方式")
    parser.add_argument("--workers", "-w", type=int, default=5, help="并发线程数")
    parser.add_argument("--cache-dir", "-d", default="./data/cache", help="缓存目录")

    args = parser.parse_args()

    adjust = args.adjust if args.adjust != "none" else None

    batch_download_stocks(
        n_stocks=args.count,
        start_date=args.start,
        end_date=args.end,
        adjust=adjust,
        cache_dir=args.cache_dir,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
