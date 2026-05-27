#!/usr/bin/env python3
"""
Quant 量化系统统一入口

Usage:
    python main.py backtest -s b1 -sym 600000 -sd 20200101 -ed 20231231
    python main.py optimize -s dual_ma -sym 600000
    python main.py analyze -s b1
"""

import argparse
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quant import (
    DualMAStrategy,
    MomentumStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    B1Strategy,
    TemplateStrategy,
    RightSideBottomFishingStrategy,
    BacktestEngine
)


def main():
    parser = argparse.ArgumentParser(
        description="Quant 量化系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["backtest", "optimize", "analyze"],
        help="执行命令"
    )
    parser.add_argument(
        "--strategy", "-s",
        required=True,
        help="策略名称: dual_ma, momentum, breakout, mean_reversion, b1, template, right_side_bottom"
    )
    parser.add_argument(
        "--symbol", "-sym",
        default="600000",
        help="股票代码"
    )
    parser.add_argument(
        "--start-date", "-sd",
        default="20200101",
        help="开始日期 (YYYYMMDD)"
    )
    parser.add_argument(
        "--end-date", "-ed",
        default="20231231",
        help="结束日期 (YYYYMMDD)"
    )
    parser.add_argument(
        "--output", "-o",
        default="backtest_report.html",
        help="输出报告文件名"
    )
    
    args = parser.parse_args()
    
    # 策略映射
    strategy_map = {
        "dual_ma": DualMAStrategy,
        "momentum": MomentumStrategy,
        "breakout": BreakoutStrategy,
        "mean_reversion": MeanReversionStrategy,
        "b1": B1Strategy,
        "template": TemplateStrategy,
        "right_side_bottom": RightSideBottomFishingStrategy
    }
    
    # 验证策略
    if args.strategy not in strategy_map:
        print(f"错误: 未知策略 '{args.strategy}'")
        print(f"可用策略: {list(strategy_map.keys())}")
        sys.exit(1)
    
    # 创建策略实例
    strategy_class = strategy_map[args.strategy]
    strategy = strategy_class()
    print(f"策略: {strategy.name}")
    
    # 执行命令
    if args.command == "backtest":
        print(f"股票: {args.symbol}")
        print(f"时间范围: {args.start_date} ~ {args.end_date}")
        print("=" * 50)
        
        # 运行回测
        engine = BacktestEngine()
        engine.set_strategy(strategy)
        engine.run(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # 生成报告
        engine.generate_report(output_file=args.output)
        print(f"\n回测报告已生成: {args.output}")
    
    elif args.command == "optimize":
        print("策略优化功能开发中...")
    
    elif args.command == "analyze":
        print("策略分析功能开发中...")


if __name__ == "__main__":
    main()
