"""
B1策略Label生成器

根据B1策略的卖点设计Label，融入短期大涨强化
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class B1QualityLabelMaker:
    """
    B1策略强化版LabelMaker

    Label设计原则：
    1. 与B1卖点对齐（持有5天，时间止损/止盈/止损）
    2. 融入短期大涨（让模型学习有爆发力的买点）
    3. 多层次标签（分类 + 回归）

    Label层级：
    - Level 3: 大涨优质买点（5日涨>=5% OR 期间有单日涨>=7%）
    - Level 2: 中等赚钱买点（5日涨>=2%）
    - Level 1: 微利买点（5日涨>0%）
    - Level 0: 亏损买点（5日涨<=0%）
    """

    def __init__(self, forward_days: int = 5):
        self.forward_days = forward_days

    def make(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成多层Label

        Args:
            df: 包含 close, high, low, volume 的DataFrame

        Returns:
            包含各类Label的DataFrame
        """
        future_price = df['close'].shift(-self.forward_days)
        future_return = (future_price / df['close'] - 1) * 100

        max_intraday = self._calc_max_intraday(df, self.forward_days)
        max_return = self._calc_max_return(df, self.forward_days)

        labels = pd.DataFrame({
            'future_return': future_return,
            'max_intraday': max_intraday,
            'max_return': max_return,
        })

        labels['quality'] = self._calc_quality_label(
            labels['future_return'],
            labels['max_intraday']
        )

        labels['is_good'] = (
            (labels['future_return'] >= 2) |
            ((labels['future_return'] >= 0) & (labels['max_intraday'] >= 5))
        ).astype(int)

        labels['has_surge_5'] = (labels['max_intraday'] >= 5).astype(int)
        labels['has_surge_7'] = (labels['max_intraday'] >= 7).astype(int)
        labels['has_surge_9'] = (labels['max_intraday'] >= 9.5).astype(int)

        labels['tp_potential_5'] = (labels['max_return'] >= 5).astype(int)
        labels['tp_potential_7'] = (labels['max_return'] >= 7).astype(int)

        labels['quality_score'] = self._calc_quality_score(
            labels['future_return'],
            labels['max_intraday'],
            labels['max_return']
        )

        labels['return_bin'] = pd.cut(
            labels['future_return'],
            bins=[-np.inf, -2, 0, 2, 5, np.inf],
            labels=['止损区', '亏损区', '微利区', '中赚区', '大涨区']
        )

        labels['quality_str'] = labels['quality'].map({
            0: '亏损',
            1: '微利',
            2: '中赚',
            3: '大涨'
        })

        return labels

    def _calc_max_intraday(self, df: pd.DataFrame, days: int) -> pd.Series:
        """计算持有期间最大单日涨幅"""
        max_ret = pd.Series(0.0, index=df.index)

        for i in range(1, days + 1):
            daily = (df['close'].shift(-i) / df['close'].shift(-i + 1) - 1) * 100
            max_ret = pd.concat([max_ret, daily], axis=1).max(axis=1)

        return max_ret

    def _calc_max_return(self, df: pd.DataFrame, days: int) -> pd.Series:
        """计算持有期间最高收益"""
        max_price = df['close'].copy()

        for i in range(1, days + 1):
            high = df['high'].shift(-i)
            max_price = pd.concat([max_price, high], axis=1).max(axis=1)

        return (max_price / df['close'] - 1) * 100

    def _calc_quality_label(self, future_return: pd.Series, max_intraday: pd.Series) -> pd.Series:
        """计算质量分类"""
        conditions = [
            (future_return >= 5) | (max_intraday >= 7),
            (future_return >= 2) | (max_intraday >= 5),
            future_return > 0,
        ]
        choices = [3, 2, 1]
        return np.select(conditions, choices, default=0)

    def _calc_quality_score(self, future_return: pd.Series, max_intraday: pd.Series, max_return: pd.Series) -> pd.Series:
        """计算质量分数 (0-100)"""
        return (
            future_return.clip(lower=0, upper=10) * 3 +
            max_intraday.clip(lower=0, upper=5) * 6 +
            max_return.clip(lower=0, upper=5) * 4
        )

    def get_classification_labels(self, labels: pd.DataFrame) -> Dict[str, pd.Series]:
        """获取分类Label"""
        return {
            'quality': labels['quality'],
            'is_good': labels['is_good'],
            'has_surge_5': labels['has_surge_5'],
            'has_surge_7': labels['has_surge_7'],
        }

    def get_regression_labels(self, labels: pd.DataFrame) -> Dict[str, pd.Series]:
        """获取回归Label"""
        return {
            'future_return': labels['future_return'],
            'quality_score': labels['quality_score'],
            'max_intraday': labels['max_intraday'],
        }

    def get_entry_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        获取满足B1入场条件的Mask

        用于只对满足入场条件的位置打Label
        """
        if len(df) < 60:
            return pd.Series(False, index=df.index)

        close = df['close']
        volume = df['volume']

        ma3 = close.rolling(3).mean()
        ma6 = close.rolling(6).mean()
        ma12 = close.rolling(12).mean()
        ma24 = close.rolling(24).mean()
        ma60 = close.rolling(60).mean()
        bbi = (ma3 + ma6 + ma12 + ma24) / 4

        pct_change = close.pct_change() * 100
        amplitude = (df['high'] - df['low']) / df['low'] * 100

        low = df['low']
        high = df['high']
        window = 9
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        rsv = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d

        mask = (
            (pct_change >= -2) & (pct_change <= 2) &
            (amplitude < 7) &
            (bbi > ma60) &
            (j < -5) &
            (volume > volume.shift(1))
        )

        return mask


class B1ExitAwareLabelMaker(B1QualityLabelMaker):
    """
    B1策略出场感知LabelMaker

    在B1QualityLabelMaker基础上，增加出场方式感知
    """

    def __init__(self, forward_days: int = 5):
        super().__init__(forward_days)

    def make(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成包含出场方式感知的Label"""
        labels = super().make(df)

        future_low = pd.concat([
            df['low'].shift(-i) for i in range(1, self.forward_days + 1)
        ], axis=1).min(axis=1)
        min_return = (future_low / df['close'] - 1) * 100

        labels['min_return'] = min_return

        def classify_exit(row):
            fr = row['future_return']
            mr = row['max_return']
            mir = row['max_intraday']
            min_r = row['min_return']

            if pd.isna(fr):
                return 'unknown'

            if mr >= 5:
                return '止盈'
            elif mir >= 9.5:
                return '涨停'
            elif min_r <= -2:
                return '止损'
            elif fr < 0:
                return '时间止损'
            elif fr < 2:
                return '微利'
            else:
                return '赚钱'

        labels['exit_type'] = labels.apply(classify_exit, axis=1)

        labels['exit_is_profitable'] = labels['exit_type'].isin(['止盈', '涨停', '赚钱']).astype(int)

        return labels


def create_b1_labels(df: pd.DataFrame, forward_days: int = 5, exit_aware: bool = False) -> pd.DataFrame:
    """
    便捷函数：创建B1策略Label

    Args:
        df: K线数据
        forward_days: 持有天数（默认5天）
        exit_aware: 是否包含出场感知Label

    Returns:
        包含Label的DataFrame
    """
    if exit_aware:
        maker = B1ExitAwareLabelMaker(forward_days)
    else:
        maker = B1QualityLabelMaker(forward_days)

    return maker.make(df)