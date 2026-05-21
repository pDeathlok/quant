"""
AKQuant 快速回测示例

Usage: python scripts/backtest/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import akshare as ak
import akquant as aq
from quant.strategies.momentum import DualMAStrategy
from quant.backtest.engine import BacktestEngine


def main():
    print("=" * 50)
    print("AKQuant 快速回测示例")
    print("=" * 50)

    print("\n[1/4] 准备数据...")
    df = ak.stock_zh_a_daily(
        symbol="sh600000",
        start_date="20250101",
        end_date="20260501",
        adjust="qfq"
    )
    print(f"获取数据: {len(df)} 行")

    print("\n[2/4] 初始化策略...")
    strategy = DualMAStrategy(
        fast_ma_period=5,
        slow_ma_period=20,
        holding_bars=5
    )
    print(f"策略: {strategy.name}")

    print("\n[3/4] 运行回测...")
    engine = BacktestEngine(
        initial_cash=1000000.0,
        commission_rate=0.0003
    )
    engine.set_strategy(strategy)
    engine.run(
        symbol="600000",
        start_date="20250101",
        end_date="20260501"
    )

    print("\n[4/4] 回测结果...")
    metrics = engine.get_metrics()
    print(f"总收益率: {metrics.get('total_return_pct', 0):.2%}")
    print(f"年化收益率: {metrics.get('annualized_return', 0):.2%}")
    print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"最大回撤: {metrics.get('max_drawdown_pct', 0):.2%}")
    print(f"交易次数: {metrics.get('trade_count', 0)}")

    report_file = "backtest_report.html"
    print(f"\n生成报告: {report_file}")
    engine.generate_report(output_file=report_file)

    print("\n完成!")


if __name__ == "__main__":
    main()
