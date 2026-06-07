"""
T+1交易制度下的Label生成器

Label设计原则：
1. 符合A股T+1交易制度（T+1买入，最早T+2卖出）
2. 真实交易场景模拟
3. 多层次标签（分类 + 回归）

Label列表：
1. label_max_profit: T+2到T+6期间最大收益（回归）
2. label_max_drawdown: T+2到T+6期间最大回撤（回归）
3. label_hit_profit: 是否达到收益目标（二分类）
4. label_hit_stop: 是否跌破止损位（二分类）
5. label_exit_type: 退出类型（分类）
6. label_future_return: T+6收盘价收益（回归）
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class T1TradingLabelMaker:
    """
    T+1交易制度下的Label生成器
    
    交易规则：
    - 买入点：T+1日开盘价
    - 最早卖出：T+2日（T+1买入后次日才能卖出）
    - 持有期：T+2到T+6（共5个交易日）
    """

    def __init__(self, profit_target: float = 0.05, stop_loss: float = 0.02, max_days: int = 5):
        """
        Args:
            profit_target: 收益目标（默认5%）
            stop_loss: 止损比例（默认2%）
            max_days: 最大持有天数（默认5天）
        """
        self.profit_target = profit_target  # 收益目标
        self.stop_loss = stop_loss          # 止损比例
        self.max_days = max_days            # 最大持有天数

    def make(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成T+1交易场景下的Label

        Args:
            df: 包含 open, high, low, close 的DataFrame，需要按symbol分组且索引连续

        Returns:
            包含各类Label的DataFrame
        """
        if 'symbol' not in df.columns:
            raise ValueError("DataFrame must contain 'symbol' column")

        result_dfs = []
        for symbol in df['symbol'].unique():
            stock_df = df[df['symbol'] == symbol].copy()
            stock_df = stock_df.sort_values('date').reset_index(drop=True)

            buy_price = stock_df['open'].shift(-1)

            high_cols = []
            low_cols = []

            for i in range(2, self.max_days + 2):
                high_col = f'high_t{i}'
                low_col = f'low_t{i}'
                high_cols.append(high_col)
                low_cols.append(low_col)
                stock_df[high_col] = stock_df['high'].shift(-i)
                stock_df[low_col] = stock_df['low'].shift(-i)

            max_high = stock_df[high_cols].max(axis=1)
            max_profit = (max_high / buy_price - 1) * 100

            min_low = stock_df[low_cols].min(axis=1)
            max_drawdown = (min_low / buy_price - 1) * 100

            hit_profit = (max_profit >= self.profit_target * 100).astype(int)
            hit_stop = (max_drawdown <= -self.stop_loss * 100).astype(int)

            conditions = [
                hit_profit == 1,
                hit_stop == 1,
            ]
            choices = ['止盈', '止损']
            exit_type = np.select(conditions, choices, default='时间退出')

            future_price = stock_df['close'].shift(-self.max_days - 1)
            future_return = (future_price / buy_price - 1) * 100

            holding_days = pd.Series(self.max_days, index=stock_df.index)

            labels = pd.DataFrame({
                'symbol': stock_df['symbol'].values,
                'date': stock_df['date'].values,
                'label_max_profit': max_profit,
                'label_max_drawdown': max_drawdown,
                'label_hit_profit': hit_profit,
                'label_hit_stop': hit_stop,
                'label_exit_type': exit_type,
                'label_future_return': future_return,
                'label_holding_days': holding_days
            })

            result_dfs.append(labels)

        all_labels = pd.concat(result_dfs, ignore_index=True)
        return all_labels

    def make_with_multiple_thresholds(self, df: pd.DataFrame, thresholds: list) -> pd.DataFrame:
        """
        生成多个阈值组合的Label

        Args:
            df: 包含 open, high, low, close 的DataFrame
            thresholds: 阈值组合列表，每个元素为(profit_target, stop_loss, suffix)

        Returns:
            包含多组Label的DataFrame
        """
        all_labels = pd.DataFrame(index=df.index)

        for profit_target, stop_loss, suffix in thresholds:
            maker = T1TradingLabelMaker(profit_target=profit_target, stop_loss=stop_loss)
            labels = maker.make(df)

            labeled_cols = {
                'symbol': labels['symbol'].values,
                'date': labels['date'].values,
                f'label_max_profit{suffix}': labels['label_max_profit'].values,
                f'label_max_drawdown{suffix}': labels['label_max_drawdown'].values,
                f'label_hit_profit{suffix}': labels['label_hit_profit'].values,
                f'label_hit_stop{suffix}': labels['label_hit_stop'].values,
                f'label_exit_type{suffix}': labels['label_exit_type'].values,
                f'label_future_return{suffix}': labels['label_future_return'].values,
            }

            for col_name, col_data in labeled_cols.items():
                all_labels[col_name] = col_data

        return all_labels

