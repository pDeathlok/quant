#!/usr/bin/env python3
"""测试 B1 策略回测"""

import sys
sys.path.insert(0, "src")

from quant.data.fetcher import DataFetcher
from quant.strategies.custom.b1 import B1Strategy
from quant.backtest.engine import BacktestEngine

SYMBOL = "sz300750"
START_DATE = "20200101"
END_DATE = "20260521"

print(f"=== B1 策略回测测试 ===")
print(f"股票: {SYMBOL}")
print(f"区间: {START_DATE} ~ {END_DATE}")

# 1. 获取数据
fetcher = DataFetcher()
df = fetcher.get_stock_daily(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)
print(f"数据: {len(df)} 条日K记录")

# 2. 创建策略
strategy = B1Strategy()
print(f"策略: {strategy.name}")
print(f"  持有天数: {strategy.hold_days}")
print(f"  止损: 前低向下 {strategy.stop_loss_pct*100:.1f}%")
print(f"  止盈回撤: {strategy.take_profit_drawdown*100:.1f}% (需 J>80)")
print(f"  时间止损回撤: {strategy.time_stop_drawdown*100:.1f}% (需 >= {strategy.hold_days}天)")

# 3. 运行回测
engine = BacktestEngine(data=df, strategy=strategy)
result = engine.run(
    symbols=[SYMBOL],
    show_progress=True,
    report_filename="backtest_b1_test.html"
)

# 4. 打印结果
print("\n=== 回测结果 ===")
print(result)

# 详细指标
if result:
    print("\n=== 业绩指标 ===")
    print(result)

# 交易记录
trades = result.trades_df
if not trades.empty:
    print(f"\n=== 交易记录 ({len(trades)} 笔) ===")
    cols = ['entry_time', 'exit_time', 'entry_price', 'exit_price', 'quantity', 'side', 'pnl', 'return_pct', 'commission', 'duration']
    available = [c for c in cols if c in trades.columns]
    print(trades[available].to_string(index=False))
else:
    print("\n无交易记录")

print("\n报告已保存: backtest_b1_test.html")
