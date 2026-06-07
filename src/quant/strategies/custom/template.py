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
    - short_window: 短期均线窗口（默认 5）
    - long_window: 长期均线窗口（默认 20）
    - position_ratio: 仓位比例（默认 0.95）
    """
    
    def __init__(self, 
                 short_window: int = 5, 
                 long_window: int = 20,
                 position_ratio: float = 0.95):
        """
        策略初始化
        
        Args:
            short_window: 短期参数（默认 5）
            long_window: 长期参数（默认 20）
            position_ratio: 仓位比例（默认 0.95，即 95%）
        """
        super().__init__()
        
        # === 策略参数 ===
        self.short_window = short_window
        self.long_window = long_window
        self.position_ratio = position_ratio
        
        # === 预热期（确保有足够历史数据计算指标）===
        self.warmup_period = max(short_window, long_window)
        
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
            print(f"[{self.now}] 策略启动 - 参数: short_window={self.short_window}, long_window={self.long_window}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标（可扩展）
        
        Args:
            df: 包含 open, high, low, close, volume 的 DataFrame
        
        Returns:
            添加指标后的 DataFrame
        """
        # 示例：计算均线
        df['ma_short'] = df['close'].rolling(self.short_window).mean()
        df['ma_long'] = df['close'].rolling(self.long_window).mean()
        
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
    from quant.data import TushareDataFetcher
    
    print(f"正在获取 {symbol} 的历史数据...")
    fetcher = TushareDataFetcher()
    df = fetcher.get_stock_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq")
    
    # 确保列名正确
    df["symbol"] = symbol
    
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
