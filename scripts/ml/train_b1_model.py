"""
B1策略ML训练脚本

功能：
1. 加载股票数据
2. 生成Label
3. 计算特征
4. 构建训练数据集
5. 训练模型
6. 保存模型和结果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime

from quant.ml import (
    B1QualityLabelMaker,
    B1ExitAwareLabelMaker,
    create_b1_labels,
    MLDataSet
)
from quant.data.source_merge import normalize_ts_code, normalize_tushare_daily
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.data.factors import KDJ, MACD, BOLL, RSI, MA, Volume


SYMBOL = "600000"
START_DATE = "20150101"
END_DATE = "20231231"
FORWARD_DAYS = 5


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载 Tushare 日线数据，优先使用本地例行任务产物。"""
    print(f"正在加载 {symbol} 的数据...")

    ts_code = normalize_ts_code(symbol)
    local_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily" / f"{ts_code}.parquet"
    if local_path.exists():
        df = pd.read_parquet(local_path)
    else:
        fetcher = TushareDataFetcher(cache_dir=Path(__file__).resolve().parents[2] / "data" / "cache" / "source_merge" / "tushare")
        df = fetcher.get_stock_daily(ts_code, start_date, end_date, adjust=None)

    df = normalize_tushare_daily(df, ts_code)
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)].copy()
    df["symbol"] = ts_code

    print(f"数据加载完成 - {len(df)} 条记录")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算特征"""
    print("正在计算特征...")

    df = df.copy()

    df["pct_change"] = df["close"].pct_change() * 100
    df["amplitude"] = (df["high"] - df["low"]) / df["low"] * 100

    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    ma60 = df["close"].rolling(60).mean()

    df["bbi"] = (ma3 + ma6 + ma12 + ma24) / 4
    df["ma3"] = ma3
    df["ma6"] = ma6
    df["ma12"] = ma12
    df["ma24"] = ma24
    df["ma60"] = ma60

    df["ma3_ma60_diff"] = ma3 - ma60
    df["bbi_ma60_diff"] = df["bbi"] - ma60
    df["close_ma60_diff"] = df["close"] - ma60
    df["ma3_ma60_ratio"] = ma3 / ma60
    df["bbi_ma60_ratio"] = df["bbi"] / ma60

    kdj = KDJ()
    kdj_result = kdj.compute(df[["high", "low", "close"]])
    df["kdj_k"] = kdj_result["K"]
    df["kdj_d"] = kdj_result["D"]
    df["kdj_j"] = kdj_result["J"]

    macd = MACD()
    macd_result = macd.compute(df[["close"]])
    df["macd"] = macd_result["macd"]
    df["signal"] = macd_result["signal"]
    df["histogram"] = macd_result["histogram"]

    boll = BOLL()
    boll_result = boll.compute(df[["close"]])
    df["boll_upper"] = boll_result["upper"]
    df["boll_middle"] = boll_result["middle"]
    df["boll_lower"] = boll_result["lower"]
    df["boll_width"] = (boll_result["upper"] - boll_result["lower"]) / boll_result["middle"]

    rsi = RSI()
    rsi_result = rsi.compute(df[["close"]])
    df["rsi"] = rsi_result["rsi"]

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean()

    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]

    df["volatility_5"] = df["close"].pct_change().rolling(5).std() * 100
    df["volatility_10"] = df["close"].pct_change().rolling(10).std() * 100
    df["volatility_20"] = df["close"].pct_change().rolling(20).std() * 100

    for i in [1, 2, 3, 5]:
        df[f"return_{i}d"] = df["close"].pct_change(i) * 100

    print(f"特征计算完成 - {len([c for c in df.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']])} 个特征")
    return df


def build_dataset(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """构建带Label的数据集"""
    print("正在生成Label...")

    df = df.copy()
    labels = create_b1_labels(df, forward_days=forward_days)

    df = pd.concat([df, labels], axis=1)

    df["entry_mask"] = B1QualityLabelMaker().get_entry_mask(df)

    feature_cols = [
        "pct_change", "amplitude", "volume_ratio",
        "bbi", "ma3", "ma6", "ma12", "ma24", "ma60",
        "ma3_ma60_diff", "bbi_ma60_diff", "close_ma60_diff",
        "ma3_ma60_ratio", "bbi_ma60_ratio",
        "kdj_k", "kdj_d", "kdj_j",
        "macd", "signal", "histogram",
        "boll_upper", "boll_middle", "boll_lower", "boll_width",
        "rsi",
        "high_low_ratio", "close_open_ratio",
        "volatility_5", "volatility_10", "volatility_20",
        "return_1d", "return_2d", "return_3d", "return_5d"
    ]

    available_features = [c for c in feature_cols if c in df.columns]

    df = df.dropna(subset=available_features)

    print(f"数据集构建完成 - {len(df)} 条记录")
    print(f"有效入场点: {df['entry_mask'].sum()} 个")

    return df, available_features


def train_model(X_train, y_train, model_type: str = "gbdt"):
    """训练模型"""
    print(f"正在训练 {model_type} 模型...")

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            class_weight="balanced",
            random_state=42
        )
    elif model_type == "gbdt":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

    model.fit(X_train, y_train)
    print("模型训练完成")

    return model


def evaluate_model(model, X_test, y_test, label_name: str = "quality"):
    """评估模型"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print(f"模型评估 - {label_name}")
    print("=" * 50)

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nAUC-ROC: {auc:.4f}")
    except:
        pass


def main():
    print("=" * 60)
    print("B1策略ML训练流程")
    print("=" * 60)

    df = load_data(SYMBOL, START_DATE, END_DATE)

    df = compute_features(df)

    df, feature_cols = build_dataset(df, forward_days=FORWARD_DAYS)

    entry_df = df[df["entry_mask"] == True].copy()
    print(f"\n满足B1入场条件的样本数: {len(entry_df)}")

    label_type = "is_good"
    print(f"使用Label: {label_type}")

    X = entry_df[feature_cols].values
    y = entry_df[label_type].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"正样本比例: 训练集 {y_train.mean():.2%}, 测试集 {y_test.mean():.2%}")

    model = train_model(X_train, y_train, model_type="gbdt")

    evaluate_model(model, X_test, y_test, label_type)

    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"b1_model_{label_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
    joblib.dump(model, model_path)
    print(f"\n模型已保存: {model_path}")

    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_path = output_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d')}.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"特征重要性已保存: {importance_path}")

    print("\n" + "=" * 60)
    print("Top 10 重要特征:")
    print("=" * 60)
    print(feature_importance.head(10).to_string(index=False))

    return model, feature_cols, feature_importance


if __name__ == "__main__":
    main()
