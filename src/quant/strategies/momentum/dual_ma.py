"""
双均线策略 (Dual Moving Average Strategy)

策略逻辑：
1. 计算 5 日均线 (MA5) 和 20 日均线 (MA20)
2. 金叉买入：当 MA5 上穿 MA20，且当前无持仓时，买入
3. 死叉卖出：当 MA5 下穿 MA20，且当前有持仓时，卖出
"""

import akquant as aq
from akquant import Bar, Strategy
import pandas as pd


class DualMAStrategy(Strategy):
    """
    双均线策略实现
    
    参数:
        short_window: 短期均线窗口（默认 5）
        long_window: 长期均线窗口（默认 20）
        position_ratio: 仓位比例（默认 0.95）
    """

    def __init__(self, short_window: int = 5, long_window: int = 20, position_ratio: float = 0.95):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.position_ratio = position_ratio
        self.warmup_period = long_window
        
    def on_start(self) -> None:
        """策略初始化时调用"""
        print(f"双均线策略初始化 - 短期窗口: {self.short_window}, 长期窗口: {self.long_window}")

    def on_bar(self, bar: Bar) -> None:
        """核心交易逻辑"""
        symbol = bar.symbol

        # 1. 获取历史数据
        closes = self.get_history(count=self.long_window, symbol=symbol, field="close")
        
        if len(closes) < self.long_window:
            return  # 数据不足，等待
        
        # 2. 计算均线
        ma5_curr = closes[-self.short_window:].mean()
        ma20_curr = closes[-self.long_window:].mean()
        
        # 3. 获取持仓
        position = self.get_position(symbol)

        # 4. 交易信号
        if ma5_curr > ma20_curr and position == 0:
            print(
                f"[{bar.timestamp_str}] 金叉买入 (MA5={ma5_curr:.2f}, "
                f"MA20={ma20_curr:.2f})"
            )
            self.order_target_percent(self.position_ratio, symbol)

        elif ma5_curr < ma20_curr and position > 0:
            print(
                f"[{bar.timestamp_str}] 死叉卖出 (MA5={ma5_curr:.2f}, "
                f"MA20={ma20_curr:.2f})"
            )
            self.order_target_percent(0.0, symbol)


def run_backtest():
    """
    运行回测示例
    """
    import sys
    from pathlib import Path
    
    # 添加项目路径
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # 1. 准备数据（使用本地缓存）
    print("正在加载数据...")
    df = pd.read_parquet("./data/stocks/sh600000.parquet")
    
    # 筛选日期范围
    from datetime import date
    df = df[(df["date"] >= date(2020, 1, 1)) & (df["date"] <= date(2023, 12, 31))].copy()
    
    # 确保 symbol 列正确
    df["symbol"] = "sh600000"
    
    print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"数据条数: {len(df)}")

    # 2. 运行回测
    print("\n开始回测...")
    result = aq.run_backtest(
        strategy=DualMAStrategy,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,
        lot_size=100,
    )

    # 3. 打印结果
    print("\n" + "=" * 50)
    print("双均线策略回测结果")
    print("=" * 50)
    print(result)
    
    # 4. 生成报告
    result.report(filename="backtest_report_dual_ma.html")
    print("\n回测报告已生成: backtest_report_dual_ma.html")
    
    return result


if __name__ == "__main__":
    run_backtest()
