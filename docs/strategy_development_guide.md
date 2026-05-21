# 策略开发标准文档与代码框架

## 1. 策略开发规范

### 1.1 策略结构

一个标准的 AKQuant 策略应包含以下核心组件：

| 组件 | 说明 | 必要性 |
|------|------|--------|
| `__init__()` | 策略初始化，设置参数 | 必需 |
| `on_start()` | 策略启动时的初始化操作 | 可选 |
| `on_bar(bar)` | 逐 K 线处理逻辑 | 必需 |
| `on_order(order)` | 订单状态变化回调 | 可选 |
| `on_stop()` | 策略结束时的清理操作 | 可选 |

### 1.2 关键方法说明

#### 1.2.1 数据获取

```python
# 获取历史数据
closes = self.get_history(
    count=20,          # 获取数量
    symbol="600000",   # 标的代码
    field="close"      # 字段: open, high, low, close, volume
)

# 获取多字段历史数据
df = self.get_history_df(
    count=20,
    symbol="600000",
    fields=["open", "high", "low", "close", "volume"]
)
```

#### 1.2.2 持仓管理

```python
# 获取当前持仓
position = self.get_position(symbol="600000")

# 获取账户资金
cash = self.get_cash()
equity = self.get_equity()
```

#### 1.2.3 订单操作

```python
# 目标仓位下单 (推荐)
self.order_target_percent(0.95, symbol)  # 仓位占比 95%
self.order_target_percent(0.0, symbol)   # 清仓

# 固定数量下单
self.buy(symbol="600000", quantity=100)
self.sell(symbol="600000", quantity=100)

# 限价单
self.buy_limit(symbol="600000", price=10.0, quantity=100)
self.sell_limit(symbol="600000", price=10.0, quantity=100)

# 止损单
self.stop_loss(symbol="600000", stop_price=9.5)

# 关闭持仓
self.close_position(symbol="600000")
```

### 1.3 时间心智模型

**AKQuant 核心原则：**

1. **UTC 时间戳**：`bar.timestamp` 使用 UTC 纳秒时间戳
2. **本地时间显示**：`bar.timestamp_iso` 或 `self.now` 转换为本地时间
3. **信号生成时机**：当前 Bar 的信号基于历史数据，在下一根 Bar 开盘成交
4. **避免未来函数**：`on_bar` 中只能访问当前及之前的数据

---

## 2. 策略模板代码

### 2.1 基础策略模板

```python
"""
策略模板 - 通用框架

使用方法：
1. 复制此文件，重命名为你的策略名称
2. 在 `__init__` 中设置策略参数
3. 在 `on_bar` 中实现交易逻辑
4. 运行回测：python main.py backtest -s your_strategy
"""

import akquant as aq
from akquant import Bar, Strategy
import pandas as pd
import numpy as np
from typing import Optional


class TemplateStrategy(Strategy):
    """
    策略模板 - 请替换为策略名称
    
    策略描述：
    - 策略类型：【动量/均值回归/技术指标/多因子】
    - 核心逻辑：简要描述策略原理
    - 适用市场：A股/美股/期货等
    
    参数说明：
    - param1: 参数1说明
    - param2: 参数2说明
    """
    
    def __init__(self, 
                 param1: int = 5, 
                 param2: int = 20,
                 param3: float = 0.95):
        """
        策略初始化
        
        Args:
            param1: 短期参数（默认 5）
            param2: 长期参数（默认 20）
            param3: 仓位比例（默认 0.95，即 95%）
        """
        super().__init__()
        
        # === 策略参数 ===
        self.param1 = param1
        self.param2 = param2
        self.position_ratio = param3
        
        # === 预热期（确保有足够历史数据计算指标）===
        self.warmup_period = max(param1, param2)
        
        # === 状态变量 ===
        self.signal = 0  # 1: 买入信号, -1: 卖出信号, 0: 无信号
        self.entry_price = 0.0
        self.last_trade_date = None
        
        # === 日志开关 ===
        self.verbose = True
    
    def on_start(self) -> None:
        """
        策略启动时调用（仅一次）
        
        可用于：
        - 初始化额外数据
        - 加载模型
        - 设置初始状态
        """
        if self.verbose:
            print(f"[{self.now}] 策略启动 - 参数: param1={self.param1}, param2={self.param2}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（可扩展）
        
        Args:
            df: 包含 open, high, low, close, volume 的 DataFrame
        
        Returns:
            添加指标后的 DataFrame
        """
        # 示例：计算均线
        df['ma_short'] = df['close'].rolling(self.param1).mean()
        df['ma_long'] = df['close'].rolling(self.param2).mean()
        
        # 示例：计算 RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 示例：计算 MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> int:
        """
        生成交易信号
        
        Args:
            df: 包含指标的 DataFrame
        
        Returns:
            signal: 1 (买入), -1 (卖出), 0 (无信号)
        """
        if len(df) < self.warmup_period:
            return 0
        
        # === 策略核心逻辑（请在此处实现）===
        # 示例：双均线策略
        ma_short = df['ma_short'].iloc[-1]
        ma_long = df['ma_long'].iloc[-1]
        prev_ma_short = df['ma_short'].iloc[-2]
        prev_ma_long = df['ma_long'].iloc[-2]
        
        # 金叉买入
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            return 1
        
        # 死叉卖出
        if prev_ma_short >= prev_ma_long and ma_short < ma_long:
            return -1
        
        return 0
    
    def on_bar(self, bar: Bar) -> None:
        """
        核心交易逻辑 - 每根 K 线调用一次
        
        Args:
            bar: 当前 K 线数据，包含:
                - symbol: 标的代码
                - timestamp: UTC 时间戳
                - timestamp_iso: ISO 时间字符串
                - open/high/low/close: 价格
                - volume: 成交量
        """
        symbol = bar.symbol
        
        # === 1. 获取历史数据 ===
        # 获取最近 N 根 K 线的完整数据
        history = self.get_history_df(
            count=self.warmup_period + 1,
            symbol=symbol,
            fields=["open", "high", "low", "close", "volume"]
        )
        
        if history is None or len(history) < self.warmup_period:
            return
        
        # === 2. 计算指标 ===
        df = self.calculate_indicators(history)
        
        # === 3. 生成信号 ===
        signal = self.generate_signal(df)
        
        # === 4. 获取当前持仓 ===
        position = self.get_position(symbol)
        
        # === 5. 执行交易 ===
        if signal == 1 and position == 0:
            # 买入信号且当前无持仓
            if self.verbose:
                print(f"[{bar.timestamp_iso}] 买入信号 - 标的: {symbol}, 收盘价: {bar.close:.2f}")
            self.order_target_percent(self.position_ratio, symbol)
            self.entry_price = bar.close
            self.last_trade_date = bar.timestamp_iso
        
        elif signal == -1 and position > 0:
            # 卖出信号且当前有持仓
            if self.verbose:
                print(f"[{bar.timestamp_iso}] 卖出信号 - 标的: {symbol}, 收盘价: {bar.close:.2f}")
            self.order_target_percent(0.0, symbol)
            self.entry_price = 0.0
    
    def on_order(self, order) -> None:
        """
        订单状态变化回调
        
        Args:
            order: 订单对象，包含:
                - symbol: 标的代码
                - side: 买卖方向
                - quantity: 数量
                - price: 成交价格
                - status: 订单状态
                - timestamp: 时间戳
        """
        if order.status == aq.OrderStatus.Filled:
            if self.verbose:
                print(f"[{order.timestamp}] 订单成交 - {order.side} {order.symbol} "
                      f"数量: {order.quantity} 价格: {order.price:.2f}")
    
    def on_stop(self) -> None:
        """
        策略结束时调用（仅一次）
        
        可用于：
        - 输出统计信息
        - 保存结果
        """
        if self.verbose:
            print(f"[{self.now}] 策略结束")


# ==============================================================================
# 回测运行代码（预设好，直接运行即可）
# ==============================================================================
def get_data(symbol: str = "600000", 
             start_date: str = "20200101", 
             end_date: str = "20231231") -> pd.DataFrame:
    """
    获取回测数据
    
    Args:
        symbol: 股票代码（不含 sh/sz 前缀）
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    
    Returns:
        包含历史行情的 DataFrame
    """
    import akshare as ak
    
    print(f"正在获取 {symbol} 的历史数据...")
    
    # 根据代码判断市场
    if symbol.startswith("6"):
        market_symbol = f"sh{symbol}"
    else:
        market_symbol = f"sz{symbol}"
    
    # 获取前复权数据
    df = ak.stock_zh_a_daily(
        symbol=market_symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    
    # 确保列名正确
    df["symbol"] = symbol
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    
    print(f"数据获取完成 - {len(df)} 条记录")
    return df


def run_backtest(strategy_class, 
                 data: pd.DataFrame,
                 initial_cash: float = 100_000.0,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.001,
                 lot_size: int = 100) -> aq.BacktestResult:
    """
    运行回测
    
    Args:
        strategy_class: 策略类（不是实例）
        data: 回测数据
        initial_cash: 初始资金（默认 10万）
        commission_rate: 佣金率（默认 0.03%）
        stamp_tax_rate: 印花税（默认 0.1%）
        lot_size: 最小交易单位（默认 100 股）
    
    Returns:
        回测结果对象
    """
    print("\n" + "=" * 50)
    print("开始回测")
    print("=" * 50)
    
    result = aq.run_backtest(
        strategy=strategy_class,
        data=data,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        lot_size=lot_size,
        fill_policy={
            "price_basis": "open",      # 成交基准价：open/close
            "bar_offset": 1,            # 下一根 K 线成交
            "temporal": "same_cycle"    # 时间周期
        }
    )
    
    return result


def print_results(result: aq.BacktestResult) -> None:
    """
    打印回测结果
    
    Args:
        result: 回测结果对象
    """
    print("\n" + "=" * 50)
    print("回测结果摘要")
    print("=" * 50)
    print(result)
    
    # 打印详细指标
    print("\n" + "-" * 50)
    print("详细业绩指标")
    print("-" * 50)
    
    # 提取关键指标
    metrics = result.metrics_df
    if not metrics.empty:
        print(metrics)
    
    # 打印交易记录
    trades = result.trades_df
    if not trades.empty:
        print("\n" + "-" * 50)
        print(f"交易记录（共 {len(trades)} 笔）")
        print("-" * 50)
        print(trades[["time", "symbol", "side", "quantity", "price", "pnl"]])


if __name__ == "__main__":
    # === 配置参数 ===
    SYMBOL = "600000"           # 股票代码
    START_DATE = "20200101"     # 开始日期
    END_DATE = "20231231"       # 结束日期
    INITIAL_CASH = 100_000.0    # 初始资金
    STRATEGY = TemplateStrategy  # 使用的策略类
    
    # === 运行流程 ===
    # 1. 获取数据
    df = get_data(SYMBOL, START_DATE, END_DATE)
    
    # 2. 运行回测
    result = run_backtest(STRATEGY, df, initial_cash=INITIAL_CASH)
    
    # 3. 打印结果
    print_results(result)
    
    # 4. 保存结果
    result.to_csv(f"./backtest_result_{SYMBOL}.csv")
    print(f"\n回测结果已保存到: ./backtest_result_{SYMBOL}.csv")
```

### 2.2 策略注册（添加到策略模块）

```python
# 在 src/quant/strategies/custom/__init__.py 中添加
from .template import TemplateStrategy

__all__ = ["B1Strategy", "TemplateStrategy"]
```

```python
# 在 src/quant/strategies/__init__.py 中添加
from .custom import TemplateStrategy

__all__ = [
    # ... 其他策略 ...
    "TemplateStrategy"
]
```

---

## 3. 策略开发工作流

### 3.1 快速开始步骤

```bash
# 1. 复制模板文件
cp src/quant/strategies/custom/template.py src/quant/strategies/custom/my_strategy.py

# 2. 修改策略名称和逻辑
vi src/quant/strategies/custom/my_strategy.py

# 3. 更新 __init__.py 导出
vi src/quant/strategies/custom/__init__.py

# 4. 运行回测
python main.py backtest -s my_strategy -sym 600000
```

### 3.2 策略实现要点

| 步骤 | 内容 | 说明 |
|------|------|------|
| 1 | 参数定义 | 在 `__init__` 中定义可配置参数 |
| 2 | 指标计算 | 在 `calculate_indicators` 中实现 |
| 3 | 信号生成 | 在 `generate_signal` 中实现核心逻辑 |
| 4 | 交易执行 | 在 `on_bar` 中调用订单方法 |
| 5 | 日志记录 | 使用 `self.verbose` 控制输出 |

### 3.3 常见策略模式

#### 模式 1：动量策略

```python
def generate_signal(self, df: pd.DataFrame) -> int:
    # 价格突破 N 日新高买入
    if df['close'].iloc[-1] == df['close'].rolling(20).max().iloc[-1]:
        return 1
    # 跌破 MA20 卖出
    if df['close'].iloc[-1] < df['ma_long'].iloc[-1]:
        return -1
    return 0
```

#### 模式 2：均值回归策略

```python
def generate_signal(self, df: pd.DataFrame) -> int:
    # 计算 Z-score
    mean = df['close'].rolling(20).mean().iloc[-1]
    std = df['close'].rolling(20).std().iloc[-1]
    z_score = (df['close'].iloc[-1] - mean) / std
    
    # Z-score < -2 买入（超卖）
    if z_score < -2:
        return 1
    # Z-score > 2 卖出（超买）
    if z_score > 2:
        return -1
    return 0
```

#### 模式 3：多因子策略

```python
def generate_signal(self, df: pd.DataFrame) -> int:
    # 综合多个因子评分
    score = 0
    
    # 价值因子：低 PE
    if df['pe_ratio'].iloc[-1] < 15:
        score += 1
    
    # 动量因子：近 20 日收益为正
    if df['close'].iloc[-1] > df['close'].iloc[-21]:
        score += 1
    
    # 质量因子：ROE > 10%
    if df['roe'].iloc[-1] > 10:
        score += 1
    
    # 综合评分
    if score >= 3:
        return 1
    elif score <= 1:
        return -1
    return 0
```

---

## 4. 回测配置说明

### 4.1 成交策略配置

```python
fill_policy = {
    "price_basis": "open",      # 成交基准价
                                # - "open": 下一根 K 线开盘价
                                # - "close": 当前 K 线收盘价（慎用，可能引入未来函数）
                                # - "vwap": 成交量加权平均价
    
    "bar_offset": 1,            # 成交延迟
                                # - 0: 当前 K 线内成交
                                # - 1: 下一根 K 线成交（推荐）
    
    "temporal": "same_cycle"    # 时间周期类型
}
```

### 4.2 手续费配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `commission_rate` | 0.0003 | 佣金率（0.03%） |
| `stamp_tax_rate` | 0.001 | 印花税率（0.1%），仅卖出时收取 |
| `slippage` | 0.0 | 滑点（每手额外成本） |

### 4.3 资金配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_cash` | 100000.0 | 初始资金 |
| `lot_size` | 100 | 最小交易单位（A股为100股） |
| `short_enabled` | False | 是否允许卖空 |

---

## 5. 策略评估指标

### 5.1 收益指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| 累计收益率 | 总收益 | (最终权益 - 初始资金) / 初始资金 |
| 年化收益率 | 年度化收益 | (1 + 累计收益率)^(365/天数) - 1 |
| 夏普比率 | 风险调整后收益 | 超额收益 / 收益标准差 |
| 最大回撤 | 最大亏损幅度 | max(1 - 权益/历史最高权益) |

### 5.2 风险指标

| 指标 | 说明 |
|------|------|
| 波动率 | 收益标准差 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 最大连续亏损 | 连续亏损交易次数 |

---

## 6. 注意事项

### 6.1 避免未来函数

```python
# 错误：使用了未来数据
df['signal'] = np.where(df['ma5'] > df['ma20'], 1, 0)  # 当天信号当天用

# 正确：信号后移一天
df['signal'] = np.where(df['ma5'] > df['ma20'], 1, 0).shift(1)
```

### 6.2 数据对齐

确保多标的回测时数据时间对齐：

```python
# 获取多个标的数据时，AKQuant 会自动对齐时间
df1 = get_data("600000")
df2 = get_data("600001")
combined_df = pd.concat([df1, df2])  # 按时间戳对齐
```

### 6.3 日志控制

```python
# 策略运行时会产生大量日志，建议在 __init__ 中设置 verbose
self.verbose = False  # 生产环境关闭详细日志
```

---

## 附录：策略模板文件路径

```
src/
└── quant/
    └── strategies/
        └── custom/
            ├── __init__.py       # 导出策略
            ├── template.py       # 策略模板
            └── your_strategy.py  # 你的策略实现
```

---

**版本**: v1.0  
**日期**: 2024年  
**框架**: AKQuant  
**文档参考**: 
- https://akquant.akfamily.xyz/textbook/04_backtest_engine/
- https://akquant.akfamily.xyz/textbook/01_foundations/