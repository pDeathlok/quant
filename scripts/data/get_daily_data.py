"""
获取单只或全部股票日线数据

Usage: python scripts/data/get_daily_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import akshare as ak
from datetime import datetime
from quant.data.fetcher import DataFetcher
from quant.data.storage import DataStorage


def get_all_stock_symbols():
    """获取所有 A 股股票代码列表"""
    print("正在获取所有 A 股股票列表...")
    stock_list = ak.stock_zh_a_spot()
    symbols = stock_list["代码"].tolist()
    names = stock_list["名称"].tolist()
    print(f"共获取到 {len(symbols)} 只股票")
    return list(zip(symbols, names))


def fetch_all_daily_data(
    symbols: list = None,
    start_date: str = "19901219",
    end_date: str = None,
    save_dir: str = "./data/stocks",
    adjust: str = "qfq"
):
    """
    获取历史日线数据
    
    Args:
        symbols: 股票代码列表，如果为 None 则获取所有 A 股
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD，默认为今天
        save_dir: 保存目录
        adjust: 复权方式，qfq(前复权)/hfq(后复权)/None(不复权)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if symbols is None:
        symbols = get_all_stock_symbols()
    elif isinstance(symbols, list) and not isinstance(symbols[0], tuple):
        symbols = [(s, s) for s in symbols]

    fetcher = DataFetcher(cache_dir="./data/cache")
    storage = DataStorage(data_dir=save_dir)

    success_count = 0
    fail_count = 0
    failed_stocks = []

    print(f"\n开始获取历史日线数据 (日期范围: {start_date} ~ {end_date})")
    print("=" * 60)

    for i, (symbol, name) in enumerate(symbols, 1):
        try:
            print(f"[{i}/{len(symbols)}] 正在获取: {symbol} ({name})")
            
            df = fetcher.get_stock_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            if len(df) > 0:
                storage.save_parquet(df, filename=f"{symbol}")
                success_count += 1
                print(f"    ✓ 成功获取 {len(df)} 条数据")
            else:
                print(f"    ✗ 无数据")
                fail_count += 1
                failed_stocks.append((symbol, name, "无数据"))

        except Exception as e:
            fail_count += 1
            failed_stocks.append((symbol, name, str(e)))
            print(f"    ✗ 失败: {str(e)}")

    print("=" * 60)
    print(f"\n数据获取完成!")
    print(f"成功: {success_count} 只股票")
    print(f"失败: {fail_count} 只股票")

    if failed_stocks:
        print("\n失败列表:")
        for symbol, name, reason in failed_stocks:
            print(f"  {symbol} ({name}): {reason}")

    return success_count, fail_count


def fetch_single_stock(
    symbol: str,
    start_date: str = "19901219",
    end_date: str = None,
    adjust: str = "qfq",
    show_stats: bool = True
) -> pd.DataFrame:
    """
    获取单只股票的历史日线数据
    
    Args:
        symbol: 股票代码，如 "sh600000" 或 "sz000001"
        start_date: 开始日期
        end_date: 结束日期，默认为今天
        adjust: 复权方式
        show_stats: 是否显示统计信息
    
    Returns:
        包含日线数据的 DataFrame
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    print(f"获取股票 {symbol} 的历史数据 ({start_date} ~ {end_date})...")

    fetcher = DataFetcher(cache_dir="./data/cache")
    df = fetcher.get_stock_daily(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust
    )

    if show_stats and not df.empty:
        print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"数据条数: {len(df)}")
        print(f"平均日成交量: {df['volume'].mean():,.0f}")
        print(f"平均日成交额: {df['turnover'].mean():,.0f}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="使用 AKShare 获取历史日线行情数据")
    parser.add_argument("--symbol", "-s", help="单只股票代码，如 sh600000")
    parser.add_argument("--all", "-a", action="store_true", help="获取所有 A 股数据")
    parser.add_argument("--start", "-b", default="19901219", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", "-e", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--adjust", "-adj", default="qfq", choices=["qfq", "hfq", "none"], help="复权方式")
    parser.add_argument("--save-dir", "-d", default="./data/stocks", help="保存目录")
    parser.add_argument("--show-stats", action="store_true", help="显示统计信息")

    args = parser.parse_args()

    adjust = args.adjust if args.adjust != "none" else None

    if args.symbol:
        df = fetch_single_stock(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            adjust=adjust,
            show_stats=True
        )
        print("\n数据预览:")
        print(df.head())

    elif args.all:
        fetch_all_daily_data(
            symbols=None,
            start_date=args.start,
            end_date=args.end,
            save_dir=args.save_dir,
            adjust=adjust
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
