#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 B1 策略构建训练数据 - 并行版本

并行处理：
1. 使用 ProcessPoolExecutor 并发处理每只股票
2. 并发数可配置（默认40，用户建议值）
3. 每只股票独立完成所有计算后再合并
4. 最后划分 train/test/oot
5. 确保按股票单独处理，避免跨股票数据泄露
6. 确保数据按日期排序
"""

import os
import sys
sys.path.insert(0, '/Users/didi/Project/quant/src')

import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import argparse

from quant.data.factors import (
    MA, EMA, MACD, RSI, ATR, BollingerBands, KDJ, WilliamsR, BIAS,
    Momentum, OBV, CCI,
    Volatility, DownsideVolatility,
    Factor,
    LogMarketCap, Amplitude,
)
from quant.data.factors.technical import (
    PSY, VR, MassIndex, ParabolicSAR, VortexIndicator,
    ChaikinMoneyFlow, EaseOfMovement, KeltnerChannel,
)
from quant.data.factors.alpha101 import (
    Alpha001Factor, Alpha002Factor, Alpha003Factor, Alpha004Factor, Alpha005Factor,
    Alpha006Factor, Alpha007Factor, Alpha008Factor, Alpha009Factor, Alpha010Factor
)
from quant.data.factors.alpha191 import (
    Alpha191_01Factor, Alpha191_02Factor, Alpha191_03Factor, 
    Alpha191_04Factor, Alpha191_05Factor, Alpha191_06Factor,
    Alpha191_07Factor, Alpha191_08Factor, Alpha191_09Factor,
    Alpha191_10Factor, Alpha191_11Factor, Alpha191_12Factor,
    Alpha191_13Factor, Alpha191_14Factor, Alpha191_15Factor
)
from quant.data.factors.momentum import (
    ReturnFactor, MomentumSkip5Factor, RiskAdjustedMomentumFactor, ReversalFactor
)
from quant.ml.label_maker import create_b1_labels


def calculate_b1_signal_single_stock(df: pd.DataFrame) -> pd.Series:
    """计算单只股票的 B1 策略入场信号"""
    # 确保数据按日期排序
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    elif 'trade_date' in df.columns:
        df = df.sort_values('trade_date').reset_index(drop=True)
    
    pct_change = df["close"].pct_change() * 100
    cond1 = (pct_change >= -3) & (pct_change <= 3)
    
    amplitude = (df["high"] - df["low"]) / df["low"] * 100
    cond2 = amplitude < 10
    
    kdj = KDJ().compute(df)
    cond4 = kdj["J"] < 10
    
    cond5 = df["volume"] > df["volume"].shift(1) * 0.8
    
    return (cond1 & cond2 & cond4 & cond5).astype(int)


def calculate_factors_single_stock(df: pd.DataFrame) -> pd.DataFrame:
    """计算单只股票的所有因子
    
    注意：
    1. 每只股票独立计算，避免跨股票数据泄露
    2. 使用rolling计算时，确保只使用当前股票的历史数据
    3. 数据必须按日期排序
    """
    # 确保数据按日期排序
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    elif 'trade_date' in df.columns:
        df = df.sort_values('trade_date').reset_index(drop=True)
    
    factors = pd.DataFrame(index=df.index)
    
    # 基础技术指标
    factors["ma_5"] = MA(5).compute(df)
    factors["ma_10"] = MA(10).compute(df)
    factors["ma_20"] = MA(20).compute(df)
    factors["ma_60"] = MA(60).compute(df)
    factors["ma_120"] = MA(120).compute(df)
    
    factors["ema_5"] = EMA(5).compute(df)
    factors["ema_10"] = EMA(10).compute(df)
    factors["ema_20"] = EMA(20).compute(df)
    
    factors["macd"] = MACD().compute(df)
    
    factors["rsi_6"] = RSI(6).compute(df)
    factors["rsi_12"] = RSI(12).compute(df)
    factors["rsi_24"] = RSI(24).compute(df)
    
    kdj = KDJ().compute(df)
    factors["kdj_k"] = kdj["K"]
    factors["kdj_d"] = kdj["D"]
    factors["kdj_j"] = kdj["J"]
    
    bb = BollingerBands().compute(df)
    if isinstance(bb, pd.DataFrame):
        factors["bb_upper"] = bb.iloc[:, 0]
        factors["bb_middle"] = bb.iloc[:, 1]
        factors["bb_lower"] = bb.iloc[:, 2]
    else:
        factors["bb_middle"] = bb
    
    factors["atr_14"] = ATR(14).compute(df)
    factors["williams_r_14"] = WilliamsR(14).compute(df)
    factors["cci"] = CCI().compute(df)
    factors["bias_6"] = BIAS(6).compute(df)
    factors["bias_12"] = BIAS(12).compute(df)
    factors["bias_24"] = BIAS(24).compute(df)
    
    factors["obv"] = OBV().compute(df)
    
    # 更多技术指标
    factors["psy_12"] = PSY(12).compute(df)
    factors["psy_24"] = PSY(24).compute(df)
    factors["vr_6"] = VR(6).compute(df)
    factors["vr_12"] = VR(12).compute(df)
    factors["vr_24"] = VR(24).compute(df)
    factors["mass_index"] = MassIndex().compute(df)
    factors["parabolic_sar"] = ParabolicSAR().compute(df)
    
    vortex = VortexIndicator().compute(df)
    if isinstance(vortex, pd.DataFrame):
        factors["vortex_plus"] = vortex.iloc[:, 0]
        factors["vortex_minus"] = vortex.iloc[:, 1]
    
    factors["cmf"] = ChaikinMoneyFlow().compute(df)
    factors["eom"] = EaseOfMovement().compute(df)
    
    kc = KeltnerChannel().compute(df)
    if isinstance(kc, pd.DataFrame):
        factors["keltner_upper"] = kc.iloc[:, 0]
        factors["keltner_lower"] = kc.iloc[:, 1]
        factors["keltner_width"] = (kc.iloc[:, 0] - kc.iloc[:, 1]) / kc.iloc[:, 2]
    
    factors["amplitude_1"] = Amplitude(1).compute(df)
    factors["amplitude_20"] = Amplitude(20).compute(df)
    
    # Alpha因子
    factors["alpha001"] = Alpha001Factor().compute(df)
    factors["alpha002"] = Alpha002Factor().compute(df)
    factors["alpha003"] = Alpha003Factor().compute(df)
    factors["alpha004"] = Alpha004Factor().compute(df)
    factors["alpha005"] = Alpha005Factor().compute(df)
    factors["alpha006"] = Alpha006Factor().compute(df)
    factors["alpha007"] = Alpha007Factor().compute(df)
    factors["alpha008"] = Alpha008Factor().compute(df)
    factors["alpha009"] = Alpha009Factor().compute(df)
    factors["alpha010"] = Alpha010Factor().compute(df)
    
    factors["alpha191_01"] = Alpha191_01Factor().compute(df)
    factors["alpha191_02"] = Alpha191_02Factor().compute(df)
    factors["alpha191_03"] = Alpha191_03Factor().compute(df)
    factors["alpha191_04"] = Alpha191_04Factor().compute(df)
    factors["alpha191_05"] = Alpha191_05Factor().compute(df)
    factors["alpha191_06"] = Alpha191_06Factor().compute(df)
    factors["alpha191_07"] = Alpha191_07Factor().compute(df)
    factors["alpha191_08"] = Alpha191_08Factor().compute(df)
    factors["alpha191_09"] = Alpha191_09Factor().compute(df)
    factors["alpha191_10"] = Alpha191_10Factor().compute(df)
    factors["alpha191_11"] = Alpha191_11Factor().compute(df)
    factors["alpha191_12"] = Alpha191_12Factor().compute(df)
    factors["alpha191_13"] = Alpha191_13Factor().compute(df)
    factors["alpha191_14"] = Alpha191_14Factor().compute(df)
    factors["alpha191_15"] = Alpha191_15Factor().compute(df)
    
    # 动量因子
    factors["return_1d"] = ReturnFactor(1).compute(df)
    factors["return_5d"] = ReturnFactor(5).compute(df)
    factors["return_10d"] = ReturnFactor(10).compute(df)
    factors["return_20d"] = ReturnFactor(20).compute(df)
    factors["return_60d"] = ReturnFactor(60).compute(df)
    factors["return_120d"] = ReturnFactor(120).compute(df)
    
    factors["momentum_5d"] = MomentumSkip5Factor(5).compute(df)
    factors["momentum_20d"] = MomentumSkip5Factor(20).compute(df)
    factors["momentum_60d"] = MomentumSkip5Factor(60).compute(df)
    
    factors["risk_adjusted_momentum"] = RiskAdjustedMomentumFactor().compute(df)
    factors["reversal_5d"] = ReversalFactor(5).compute(df)
    factors["reversal_20d"] = ReversalFactor(20).compute(df)
    
    # 波动率因子
    factors["volatility_20d"] = Volatility(20).compute(df)
    factors["volatility_60d"] = Volatility(60).compute(df)
    factors["downside_volatility_20d"] = DownsideVolatility(20).compute(df)
    factors["downside_volatility_60d"] = DownsideVolatility(60).compute(df)
    
    # 市值相关因子
    factors["price_level"] = df["close"]
    factors["price_log"] = np.log(df["close"] + 1)
    factors["price_volume_ratio"] = df["close"] / (df["volume"] + 1)
    factors["volume_relative_60d"] = df["volume"] / df["volume"].rolling(60).mean().replace(0, np.nan)
    
    return factors


def calculate_labels_single_stock(df: pd.DataFrame) -> pd.DataFrame:
    """计算单只股票的所有标签
    
    注意：
    1. 每只股票独立计算Label，避免跨股票数据泄露
    2. 使用新版Label定义（用户要求的6个Label）
    """
    # 确保数据按日期排序
    if 'date' in df.columns:
        df_sorted = df.sort_values('date').reset_index(drop=True)
    elif 'trade_date' in df.columns:
        df_sorted = df.sort_values('trade_date').reset_index(drop=True)
    else:
        df_sorted = df.copy()
    
    # 使用新版Label定义（use_new_labels=True）
    b1_labels = create_b1_labels(df_sorted, forward_days=5, exit_aware=True, use_new_labels=True)
    
    # 添加一些额外的简单Label
    extra_labels = pd.DataFrame(index=df_sorted.index)
    
    # 不同周期的收益
    extra_labels["label_1d_return"] = df_sorted["close"].pct_change(1).shift(-1)
    extra_labels["label_3d_return"] = df_sorted["close"].pct_change(3).shift(-3)
    extra_labels["label_10d_return"] = df_sorted["close"].pct_change(10).shift(-10)
    extra_labels["label_20d_return"] = df_sorted["close"].pct_change(20).shift(-20)
    
    # 不同周期上涨标签
    extra_labels["label_1d_up"] = (df_sorted["close"].shift(-1) > df_sorted["close"]).astype(int)
    extra_labels["label_3d_up"] = (df_sorted["close"].shift(-3) > df_sorted["close"]).astype(int)
    extra_labels["label_10d_up"] = (df_sorted["close"].shift(-10) > df_sorted["close"]).astype(int)
    extra_labels["label_20d_up"] = (df_sorted["close"].shift(-20) > df_sorted["close"]).astype(int)
    
    # B1策略信号
    extra_labels["label_b1_signal"] = calculate_b1_signal_single_stock(df_sorted)
    
    # 合并所有Label
    labels = pd.concat([b1_labels, extra_labels], axis=1)
    
    # 重命名部分列以保持一致性
    labels = labels.rename(columns={
        'future_return': 'label_5d_return',
        'max_intraday': 'label_max_intraday',
        'max_return': 'label_max_return',
        'quality': 'label_quality',
        'is_good': 'label_is_good',
        'has_surge_5': 'label_has_surge_5',
        'has_surge_7': 'label_has_surge_7',
        'has_surge_9': 'label_has_surge_9',
        'tp_potential_5': 'label_tp_potential_5',
        'tp_potential_7': 'label_tp_potential_7',
        'quality_score': 'label_quality_score',
        'min_return': 'label_min_return',
        'exit_type': 'label_exit_type',
        'exit_is_profitable': 'label_exit_is_profitable'
    })
    
    return labels


def get_market_type(ts_code: str) -> str:
    """根据股票代码判断市场类型"""
    if pd.isna(ts_code):
        return '未知'
    
    ts_code = str(ts_code).strip().lower()
    
    # 格式1: sh600000 或 sz000001
    if ts_code.startswith('sh'):
        code = ts_code[2:]
        if code.startswith('60'):
            return '沪市主板'
        elif code.startswith('688'):
            return '科创板'
    elif ts_code.startswith('sz'):
        code = ts_code[2:]
        if code.startswith('000') or code.startswith('001'):
            return '深市主板'
        elif code.startswith('002'):
            return '中小板'
        elif code.startswith('300') or code.startswith('301'):
            return '创业板'
    
    # 格式2: 600000.SH 或 000001.SZ
    if '.sh' in ts_code:
        code = ts_code.split('.')[0]
        if code.startswith('60'):
            return '沪市主板'
        elif code.startswith('688'):
            return '科创板'
    elif '.sz' in ts_code:
        code = ts_code.split('.')[0]
        if code.startswith('000') or code.startswith('001'):
            return '深市主板'
        elif code.startswith('002'):
            return '中小板'
        elif code.startswith('300') or code.startswith('301'):
            return '创业板'
    elif '.bj' in ts_code:
        return '北交所'
    
    # 格式3: 纯数字代码
    code = ''.join([c for c in ts_code if c.isdigit()])
    if len(code) >= 6:
        code = code[:6]
        if code.startswith('60'):
            return '沪市主板'
        elif code.startswith('688'):
            return '科创板'
        elif code.startswith('000') or code.startswith('001'):
            return '深市主板'
        elif code.startswith('002'):
            return '中小板'
        elif code.startswith('300') or code.startswith('301'):
            return '创业板'
        elif code.startswith('8'):
            return '北交所'
    
    return '未知'


def process_single_stock(file_path: str) -> pd.DataFrame:
    """处理单只股票的完整流程
    
    每只股票独立处理，避免跨股票数据泄露：
    1. 加载数据
    2. 按日期排序
    3. 计算因子（仅使用当前股票数据）
    4. 计算标签（仅使用当前股票数据）
    5. 合并结果
    
    Args:
        file_path: 单只股票数据文件路径
    
    Returns:
        处理后的DataFrame，包含因子和标签
    """
    try:
        # 1. 加载数据
        df = pd.read_parquet(file_path)
        
        # 提取股票代码
        filename = os.path.basename(file_path)
        symbol = filename.replace('.parquet', '')
        
        # 2. 数据预处理 - 确保按日期排序
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        elif 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        else:
            print(f"警告: 股票 {symbol} 缺少日期列")
            return None
        
        if 'vol' in df.columns and 'volume' not in df.columns:
            df = df.rename(columns={'vol': 'volume'})
        elif 'vol' in df.columns and 'volume' in df.columns:
            df = df.drop(columns=['vol'])
        df['symbol'] = symbol
        
        # 3. 添加市场分类
        df['market'] = get_market_type(symbol)
        
        # 4. 计算因子（每只股票独立计算）
        factors = calculate_factors_single_stock(df)
        
        # 5. 计算标签（每只股票独立计算）
        labels = calculate_labels_single_stock(df)
        
        # 6. 合并数据
        result = pd.concat([df, factors, labels], axis=1)
        
        # 7. 筛选满足B1条件的样本
        if 'label_b1_signal' in result.columns:
            result = result[result['label_b1_signal'] == 1]
        
        return result
        
    except Exception as e:
        print(f"处理股票失败 {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def split_train_test_oot(df: pd.DataFrame, train_end_date: str = '2023-12-31', 
                         test_end_date: str = '2024-12-31') -> dict:
    """划分训练集、测试集和OOT集
    
    按时间顺序划分，确保没有数据泄露：
    - 训练集: 截止到train_end_date
    - 测试集: train_end_date之后到test_end_date
    - OOT集: test_end_date之后
    
    Args:
        df: 完整的数据集
        train_end_date: 训练集结束日期
        test_end_date: 测试集结束日期
    
    Returns:
        包含train, test, oot的字典
    """
    df['date'] = pd.to_datetime(df['date'])
    
    train_mask = df['date'] <= train_end_date
    test_mask = (df['date'] > train_end_date) & (df['date'] <= test_end_date)
    oot_mask = df['date'] > test_end_date
    
    return {
        'train': df[train_mask].copy(),
        'test': df[test_mask].copy(),
        'oot': df[oot_mask].copy()
    }


def main():
    parser = argparse.ArgumentParser(description='基于 B1 策略构建训练数据 - 并行版本')
    parser.add_argument('--data_dir', required=True, help='股票数据目录路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--max_workers', type=int, default=40, help='并发数（默认40，用户建议值）')
    parser.add_argument('--n_stocks', type=int, default=None, help='处理股票数量（不指定则处理全部）')
    parser.add_argument('--train_end', type=str, default='2023-12-31', help='训练集结束日期')
    parser.add_argument('--test_end', type=str, default='2024-12-31', help='测试集结束日期')
    
    args = parser.parse_args()
    
    # 获取所有股票文件
    pattern = os.path.join(args.data_dir, '*.parquet')
    files = sorted(glob.glob(pattern))
    
    if args.n_stocks:
        files = files[:args.n_stocks]
    
    print(f"找到 {len(files)} 个股票数据文件")
    print(f"并发数: {args.max_workers}")
    print(f"训练集结束日期: {args.train_end}")
    print(f"测试集结束日期: {args.test_end}")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 并行处理所有股票
    results = []
    completed = 0
    total = len(files)
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_stock, file): file for file in files}
        
        # 处理完成的任务
        for future in as_completed(futures):
            file = futures[future]
            try:
                result = future.result()
                if result is not None and len(result) > 0:
                    results.append(result)
            except Exception as e:
                print(f"处理股票异常 {file}: {e}")
                import traceback
                traceback.print_exc()
            
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"已处理: {completed}/{total}")
    
    # 合并所有结果
    print(f"\n合并 {len(results)} 只股票的数据...")
    combined = pd.concat(results, ignore_index=True)
    
    # 确保合并后数据按日期排序
    if 'date' in combined.columns:
        combined = combined.sort_values('date').reset_index(drop=True)
    
    print(f"合并后总记录数: {len(combined)}")
    
    # 划分训练集、测试集、OOT集
    print(f"\n划分数据集...")
    splits = split_train_test_oot(combined, args.train_end, args.test_end)
    
    # 保存数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for name, data in splits.items():
        output_path = os.path.join(args.output, f'{name}_data_{timestamp}.parquet')
        data.to_parquet(output_path)
        print(f"{name}集已保存到: {output_path}")
        print(f"  样本数量: {len(data)}")
    
    # 保存完整数据
    full_path = os.path.join(args.output, f'full_data_{timestamp}.parquet')
    combined.to_parquet(full_path)
    print(f"完整数据已保存到: {full_path}")
    
    # 输出统计信息
    print("\n=== 数据统计 ===")
    print(f"总样本数: {len(combined)}")
    print(f"训练集: {len(splits['train'])} ({len(splits['train'])/len(combined)*100:.1f}%)")
    print(f"测试集: {len(splits['test'])} ({len(splits['test'])/len(combined)*100:.1f}%)")
    print(f"OOT集: {len(splits['oot'])} ({len(splits['oot'])/len(combined)*100:.1f}%)")
    
    print(f"\n市场分布:")
    market_counts = combined['market'].value_counts()
    for market, count in market_counts.items():
        print(f"  {market}: {count}")
    
    # 输出新增Label的统计信息
    new_labels = [
        'label_t1_open_max_high_5pct',
        'label_t1_open_max_high_8pct',
        'label_t1_open_max_high_10pct',
        'label_t1_open_min_low_2pct_below_t0_low',
        'label_t1_open_min_low_3pct_below_t0_low',
        'label_t1_open_min_close_2pct_below_t0_low',
        'label_t1_open_min_close_3pct_below_t0_low'
    ]
    
    print(f"\n新增Label分布:")
    for label in new_labels:
        if label in combined.columns:
            counts = combined[label].value_counts()
            print(f"  {label}:")
            for val, cnt in counts.items():
                print(f"    {val}: {cnt} ({cnt/len(combined)*100:.1f}%)")


if __name__ == "__main__":
    main()
