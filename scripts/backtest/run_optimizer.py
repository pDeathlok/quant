"""
参数优化示例

Usage: python scripts/backtest/run_optimizer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from quant.data.fetcher import DataFetcher
from quant.strategies.momentum import DualMAStrategy
from quant.backtest.engine import BacktestEngine
from quant.backtest.optimizer import GridSearchOptimizer


def main():
    print("=" * 50)
    print("参数优化示例")
    print("=" * 50)

    print("\n[1/4] 准备数据...")
    fetcher = DataFetcher(cache_dir="./data/cache")
    df = fetcher.get_stock_daily(
        symbol="sh600000",
        start_date="20240101",
        end_date="20260501"
    )
    print(f"获取数据: {len(df)} 行")

    print("\n[2/4] 设置参数网格...")
    param_grid = {
        "fast_ma_period": [3, 5, 10],
        "slow_ma_period": [15, 20, 30],
        "holding_bars": [3, 5, 7]
    }
    print(f"参数组合数: {3 * 3 * 3} = 27")

    print("\n[3/4] 运行网格搜索...")
    optimizer = GridSearchOptimizer(
        strategy_class=DualMAStrategy,
        param_grid=param_grid,
        data=df,
        metric="sharpe_ratio",
        workers=4
    )

    result = optimizer.run()

    print("\n[4/4] 优化结果...")
    print(f"最佳参数: {result['best_params']}")
    print(f"最佳夏普比率: {result['best_score']:.4f}")

    print("\nTop 5 参数组合:")
    for i, r in enumerate(result["all_results"][:5], 1):
        print(f"  {i}. sharpe={r.get('sharpe_ratio', 0):.4f}, params={r}")

    print("\n完成!")


if __name__ == "__main__":
    main()
