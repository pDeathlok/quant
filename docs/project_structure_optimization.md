# 项目结构优化方案

## 当前结构分析

### 存在的问题
1. **策略目录分散**：`my_strategies` 和 `examples` 分离，不易管理
2. **脚本目录杂乱**：`scripts/` 下文件较多，缺乏分类
3. **数据与代码混合**：数据文件和代码文件在同一层级
4. **缺乏清晰的入口文件**：没有统一的启动入口

### 当前结构
```
quant/
├── strategies/
│   ├── my_strategies/        # 用户策略
│   └── examples/            # 示例策略
├── scripts/                 # 零散脚本
├── data/                    # 数据（混合代码和数据文件）
├── backtest/                # 回测引擎
├── analysis/                # 分析模块
├── ml/                      # 机器学习模块
├── trading/                 # 交易模块
├── risk/                    # 风控模块
└── monitor/                 # 监控模块
```

---

## 优化后结构

### 优化目标
1. **清晰的模块划分**：按功能职责划分目录
2. **策略统一管理**：所有策略集中管理
3. **脚本分类**：按用途分类脚本
4. **数据与代码分离**：数据文件单独存放
5. **统一入口**：提供统一的命令行入口

### 优化后结构
```
quant/
├── src/                     # 源代码（核心逻辑）
│   ├── quant/               # 主包
│   │   ├── strategies/      # 策略模块
│   │   │   ├── __init__.py
│   │   │   ├── base.py      # 策略基类
│   │   │   ├── momentum/    # 动量策略
│   │   │   ├── mean_reversion/  # 均值回归策略
│   │   │   ├── technical/   # 技术指标策略
│   │   │   └── custom/      # 用户自定义策略
│   │   ├── backtest/        # 回测引擎
│   │   ├── data/            # 数据模块
│   │   ├── analysis/        # 分析模块
│   │   ├── ml/              # 机器学习模块
│   │   ├── trading/         # 交易模块
│   │   ├── risk/            # 风控模块
│   │   ├── monitor/         # 监控模块
│   │   └── __init__.py
│   └── __init__.py
├── data/                    # 数据文件（独立目录）
│   ├── stocks/              # 股票行情数据
│   ├── factors/             # 因子数据
│   ├── financial/           # 财务数据
│   └── cache/               # 缓存数据
├── scripts/                 # 脚本
│   ├── backtest/            # 回测脚本
│   ├── data/                # 数据下载脚本
│   ├── analysis/            # 分析脚本
│   └── utils/               # 工具脚本
├── config/                  # 配置文件
├── docs/                    # 文档
├── tests/                   # 测试用例
├── logs/                    # 日志文件
├── README.md
└── main.py                  # 统一入口
```

---

## 策略模块重构方案

### 按策略类型分类
```
src/quant/strategies/
├── __init__.py              # 导出所有策略
├── base.py                  # 策略基类
├── momentum/                # 动量策略
│   ├── __init__.py
│   ├── dual_ma.py           # 双均线策略
│   ├── momentum.py          # 动量策略
│   └── breakout.py          # 突破策略
├── mean_reversion/          # 均值回归策略
│   ├── __init__.py
│   └── mean_reversion.py    # 均值回归
├── technical/               # 技术指标策略
│   ├── __init__.py
│   ├── kdj_strategy.py      # KDJ策略
│   ├── macd_strategy.py     # MACD策略
│   └── rsi_strategy.py      # RSI策略
└── custom/                  # 用户自定义策略
    ├── __init__.py
    └── b1.py                # B1策略
```

### 策略模块导出
```python
# src/quant/strategies/__init__.py
from .base import BaseStrategy

# 动量策略
from .momentum import DualMAStrategy, MomentumStrategy, BreakoutStrategy

# 均值回归策略
from .mean_reversion import MeanReversionStrategy

# 技术指标策略
from .technical import KDJStrategy, MACDStrategy, RSIStrategy

# 自定义策略
from .custom import B1Strategy

__all__ = [
    "BaseStrategy",
    "DualMAStrategy",
    "MomentumStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "KDJStrategy",
    "MACDStrategy",
    "RSIStrategy",
    "B1Strategy"
]
```

---

## 脚本目录重构方案

### 按用途分类
```
scripts/
├── backtest/                # 回测脚本
│   ├── run_backtest.py      # 运行回测
│   ├── run_optimizer.py     # 策略优化
│   └── compare_strategies.py # 策略对比
├── data/                    # 数据脚本
│   ├── download_stocks.py   # 下载股票数据
│   ├── download_financial.py # 下载财务数据
│   ├── compute_factors.py   # 计算因子
│   └── update_data.py       # 更新数据
├── analysis/                # 分析脚本
│   ├── analyze_backtest.py  # 回测分析
│   ├── factor_analysis.py   # 因子分析
│   └── performance_report.py # 绩效报告
└── utils/                   # 工具脚本
    ├── data_converter.py    # 数据格式转换
    └── logger_config.py     # 日志配置
```

---

## 实现步骤

### 步骤1：创建新目录结构
```bash
# 创建源代码目录
mkdir -p src/quant/strategies/{momentum,mean_reversion,technical,custom}
mkdir -p scripts/{backtest,data,analysis,utils}
mkdir -p data/{stocks,factors,financial,cache}
```

### 步骤2：迁移策略文件
```bash
# 迁移策略文件
mv strategies/base.py src/quant/strategies/
mv strategies/my_strategies/dual_ma.py src/quant/strategies/momentum/
mv strategies/my_strategies/b1.py src/quant/strategies/custom/
mv strategies/examples/momentum.py src/quant/strategies/momentum/
mv strategies/examples/breakout.py src/quant/strategies/momentum/
mv strategies/examples/mean_reversion.py src/quant/strategies/mean_reversion/
```

### 步骤3：迁移数据文件
```bash
# 迁移数据文件
mv data/stocks/*.parquet data/stocks/
mv data/factors/*.parquet data/factors/
mv data/*.parquet data/
```

### 步骤4：创建统一入口
```python
# main.py
"""
Quant 量化系统统一入口
"""

import argparse
from src.quant.strategies import *
from src.quant.backtest import BacktestEngine

def main():
    parser = argparse.ArgumentParser(description="Quant 量化系统")
    parser.add_argument("command", choices=["backtest", "optimize", "analyze"],
                        help="执行命令")
    parser.add_argument("--strategy", "-s", required=True,
                        help="策略名称")
    parser.add_argument("--symbol", "-sym", default="600000",
                        help="股票代码")
    parser.add_argument("--start-date", "-sd", default="20200101",
                        help="开始日期")
    parser.add_argument("--end-date", "-ed", default="20231231",
                        help="结束日期")
    
    args = parser.parse_args()
    
    # 根据策略名称创建策略实例
    strategy_map = {
        "dual_ma": DualMAStrategy,
        "momentum": MomentumStrategy,
        "b1": B1Strategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy
    }
    
    if args.strategy not in strategy_map:
        print(f"未知策略: {args.strategy}")
        return
    
    strategy = strategy_map[args.strategy]()
    
    if args.command == "backtest":
        engine = BacktestEngine()
        engine.set_strategy(strategy)
        engine.run(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )
        engine.generate_report()

if __name__ == "__main__":
    main()
```

### 步骤5：更新导入路径
```python
# 更新所有文件的导入路径
# 例如：from strategies.base import BaseStrategy 
# 改为：from src.quant.strategies.base import BaseStrategy
```

---

## 优化后的使用方式

### 命令行方式
```bash
# 运行回测
python main.py backtest -s b1 -sym 600000 -sd 20200101 -ed 20231231

# 运行策略优化
python main.py optimize -s dual_ma -sym 600000

# 分析回测结果
python main.py analyze -s b1
```

### Python 代码方式
```python
from src.quant.strategies import B1Strategy
from src.quant.backtest import BacktestEngine

# 创建策略
strategy = B1Strategy()

# 运行回测
engine = BacktestEngine()
engine.set_strategy(strategy)
engine.run(symbol="600000", start_date="20200101", end_date="20231231")

# 生成报告
report = engine.generate_report()
```

---

## 优势对比

| 优化项 | 优化前 | 优化后 |
|--------|--------|--------|
| 策略管理 | 分散在两个目录 | 按类型分类管理 |
| 脚本管理 | 零散存放 | 按用途分类 |
| 数据管理 | 混合代码 | 独立数据目录 |
| 入口方式 | 多个脚本 | 统一入口 |
| 导入路径 | 相对路径复杂 | 标准包导入 |
| 扩展性 | 较差 | 良好 |

---

## 实施建议

### 第一阶段：结构重构
1. 创建新目录结构
2. 迁移文件
3. 更新导入路径

### 第二阶段：代码优化
1. 统一代码风格
2. 添加类型注解
3. 完善文档

### 第三阶段：测试覆盖
1. 添加单元测试
2. 集成测试
3. 回归测试

### 第四阶段：部署优化
1. 添加依赖管理
2. 创建 Docker 镜像
3. CI/CD 配置