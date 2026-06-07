"""
B1策略LabelMaker测试脚本

用于验证Label生成是否正确
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np

from quant.data.source_merge import normalize_ts_code, normalize_tushare_daily
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.ml import B1QualityLabelMaker, B1ExitAwareLabelMaker, create_b1_labels


def load_test_data(symbol: str = "600000", days: int = 1000) -> pd.DataFrame:
    """加载 Tushare 测试数据，优先使用本地例行任务产物。"""
    print(f"加载 {symbol} 测试数据...")

    ts_code = normalize_ts_code(symbol)
    root = Path(__file__).resolve().parents[2]
    local_path = root / "data" / "raw" / "daily" / f"{ts_code}.parquet"
    if local_path.exists():
        df = pd.read_parquet(local_path)
    else:
        fetcher = TushareDataFetcher(cache_dir=root / "data" / "cache" / "source_merge" / "tushare")
        df = fetcher.get_stock_daily(ts_code, "20150101", "20231231", adjust=None)
    df = normalize_tushare_daily(df, ts_code).tail(days).reset_index(drop=True)

    print(f"数据加载完成 - {len(df)} 条记录")
    return df


def test_basic_label_maker():
    """测试基础LabelMaker"""
    print("\n" + "=" * 60)
    print("测试1: B1QualityLabelMaker")
    print("=" * 60)

    df = load_test_data()

    maker = B1QualityLabelMaker(forward_days=5)
    labels = maker.make(df)

    print("\nLabel列:")
    print(labels.columns.tolist())

    print("\nLabel分布 (quality):")
    print(labels['quality'].value_counts().sort_index())

    print("\nLabel分布 (is_good):")
    print(labels['is_good'].value_counts())

    print("\nLabel分布 (return_bin):")
    print(labels['return_bin'].value_counts())

    print("\n统计信息:")
    print(labels[['future_return', 'max_intraday', 'max_return', 'quality_score']].describe())

    return labels


def test_exit_aware_label_maker():
    """测试出场感知LabelMaker"""
    print("\n" + "=" * 60)
    print("测试2: B1ExitAwareLabelMaker")
    print("=" * 60)

    df = load_test_data()

    maker = B1ExitAwareLabelMaker(forward_days=5)
    labels = maker.make(df)

    print("\n出场类型分布:")
    print(labels['exit_type'].value_counts())

    print("\n出场盈亏分布:")
    print(labels.groupby('exit_type')['future_return'].agg(['mean', 'count']))

    return labels


def test_entry_mask():
    """测试入场条件Mask"""
    print("\n" + "=" * 60)
    print("测试3: B1入场条件Mask")
    print("=" * 60)

    df = load_test_data()
    df["pct_change"] = df["close"].pct_change() * 100

    maker = B1QualityLabelMaker()
    mask = maker.get_entry_mask(df)

    print(f"\n总记录数: {len(df)}")
    print(f"满足B1入场条件: {mask.sum()}")
    print(f"占比: {mask.mean():.2%}")

    entry_df = df[mask].copy()
    print(f"\n入场样本示例:")
    print(entry_df[['date', 'close', 'volume', 'pct_change']].head(10))


def test_label_with_entry_filter():
    """测试只对入场点打Label"""
    print("\n" + "=" * 60)
    print("测试4: 只对B1入场点打Label")
    print("=" * 60)

    df = load_test_data()

    maker = B1QualityLabelMaker()
    labels = maker.make(df)
    entry_mask = maker.get_entry_mask(df)

    entry_labels = labels[entry_mask]

    print(f"\n入场点数量: {len(entry_labels)}")

    if len(entry_labels) > 0:
        print("\n入场点Label分布 (quality):")
        print(entry_labels['quality'].value_counts().sort_index())

        print("\n入场点Label分布 (is_good):")
        print(entry_labels['is_good'].value_counts())

        print("\n入场点Label分布 (return_bin):")
        print(entry_labels['return_bin'].value_counts())

        print("\n入场点收益统计:")
        print(entry_labels['future_return'].describe())

        print("\n大涨比例:")
        print(f"has_surge_5: {entry_labels['has_surge_5'].mean():.2%}")
        print(f"has_surge_7: {entry_labels['has_surge_7'].mean():.2%}")
        print(f"has_surge_9: {entry_labels['has_surge_9'].mean():.2%}")


def test_convenience_function():
    """测试便捷函数"""
    print("\n" + "=" * 60)
    print("测试5: create_b1_labels 便捷函数")
    print("=" * 60)

    df = load_test_data()
    labels = create_b1_labels(df, forward_days=5)

    print(f"\n生成的Label列: {labels.columns.tolist()}")
    print(f"\n样本数: {len(labels)}")

    valid_labels = labels.dropna(subset=['quality'])
    print(f"有效Label样本: {len(valid_labels)}")


def visualize_label_distribution():
    """可视化Label分布"""
    print("\n" + "=" * 60)
    print("测试6: Label分布可视化")
    print("=" * 60)

    df = load_test_data()

    maker = B1QualityLabelMaker()
    labels = maker.make(df)
    entry_mask = maker.get_entry_mask(df)

    entry_labels = labels[entry_mask]

    print("\n【B1入场点收益分布】")
    print("-" * 40)

    bins = [-np.inf, -2, 0, 2, 5, np.inf]
    names = ['<-2%', '-2~0%', '0~2%', '2~5%', '>5%']

    entry_labels['return_range'] = pd.cut(
        entry_labels['future_return'],
        bins=bins,
        labels=names
    )

    dist = entry_labels['return_range'].value_counts().sort_index()
    total = len(entry_labels)

    for name, count in dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"{name:>8}: {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n【短期大涨统计】")
    print("-" * 40)
    surge_cols = ['has_surge_5', 'has_surge_7', 'has_surge_9']
    for col in surge_cols:
        pct = entry_labels[col].mean() * 100
        bar = "█" * int(pct / 2)
        print(f"{col:>12}: {pct:>5.1f}% {bar}")


def main():
    print("=" * 60)
    print("B1策略LabelMaker测试")
    print("=" * 60)

    test_basic_label_maker()
    test_exit_aware_label_maker()
    test_entry_mask()
    test_label_with_entry_filter()
    test_convenience_function()
    visualize_label_distribution()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
