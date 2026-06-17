from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.strategies.convertible_bond.backtest import (  # noqa: E402
    ConvertibleBondBacktestConfig,
    write_backtest_outputs,
)
from quant.strategies.convertible_bond.grid import (  # noqa: E402
    ConservativeGridConfig,
    add_low_position_features,
    backtest_conservative_grid,
)


DATA_DIR = PROJECT_ROOT / "data/convertible_bond/tushare"
DAILY_PATH = DATA_DIR / "cb_daily_20180101_20260616.parquet"
BASIC_PATH = DATA_DIR / "cb_basic_all.parquet"
CALL_PATH = DATA_DIR / "cb_call_20180101_20260616.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "reports/convertible_bond/grid_iteration"


def variant_configs() -> list[ConservativeGridConfig]:
    return [
        ConservativeGridConfig(name="grid_balanced"),
        ConservativeGridConfig(
            name="grid_strict_low",
            top_n=6,
            max_total_weight=0.70,
            max_entry_price=114.0,
            max_premium_rate=20.0,
            max_double_low=130.0,
            max_price_position_252=0.25,
            min_drawdown_from_252_high=0.12,
            min_amount=3_000.0,
        ),
        ConservativeGridConfig(
            name="grid_deep_value",
            top_n=8,
            max_total_weight=0.90,
            max_entry_price=112.0,
            max_premium_rate=18.0,
            max_double_low=126.0,
            max_price_position_252=0.20,
            min_drawdown_from_252_high=0.15,
            grid_full_price=104.0,
            grid_large_price=108.0,
            grid_half_price=112.0,
            grid_small_price=115.0,
        ),
        ConservativeGridConfig(
            name="grid_recent_defensive",
            top_n=5,
            max_total_weight=0.55,
            max_position_weight=0.12,
            max_entry_price=112.0,
            max_premium_rate=16.0,
            max_double_low=124.0,
            max_price_position_252=0.18,
            min_drawdown_from_252_high=0.18,
            min_amount=5_000.0,
            min_credit_rating="AA",
        ),
        ConservativeGridConfig(
            name="grid_strict_stabilized",
            top_n=6,
            max_total_weight=0.60,
            max_position_weight=0.12,
            max_entry_price=114.0,
            min_premium_rate=0.0,
            max_premium_rate=20.0,
            max_double_low=132.0,
            max_price_position_252=0.25,
            min_drawdown_from_252_high=0.10,
            min_amount=3_000.0,
            min_momentum_20d=-0.08,
        ),
        ConservativeGridConfig(
            name="grid_balanced_stabilized",
            top_n=8,
            max_total_weight=0.75,
            max_position_weight=0.12,
            max_entry_price=116.0,
            min_premium_rate=0.0,
            max_premium_rate=24.0,
            max_double_low=138.0,
            max_price_position_252=0.32,
            min_drawdown_from_252_high=0.08,
            min_amount=3_000.0,
            min_momentum_20d=-0.10,
        ),
        ConservativeGridConfig(
            name="grid_loose_low",
            top_n=10,
            max_total_weight=0.95,
            max_entry_price=120.0,
            max_premium_rate=28.0,
            max_double_low=145.0,
            max_price_position_252=0.40,
            min_drawdown_from_252_high=0.05,
            min_amount=2_000.0,
        ),
    ]


def run_variant(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame,
    grid_config: ConservativeGridConfig,
    start_date: str,
    end_date: str,
    rebalance: str = "daily",
) -> dict:
    backtest_config = ConvertibleBondBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance=rebalance,
        commission_rate=0.0002,
        slippage_rate=0.0002,
        initial_cash=1_000_000.0,
    )
    result = backtest_conservative_grid(
        daily=daily,
        basic=basic,
        call=call,
        backtest_config=backtest_config,
        grid_config=grid_config,
    )
    out_dir = OUTPUT_ROOT / f"{grid_config.name}_{start_date}_{end_date}_{rebalance}"
    paths = write_backtest_outputs(result, out_dir)
    cases = build_case_analysis(result.equity, result.targets)
    cases_path = out_dir / "cases.csv"
    cases.to_csv(cases_path, index=False)
    summary = dict(result.summary)
    summary["variant"] = grid_config.name
    summary["rebalance"] = rebalance
    summary["case_path"] = str(cases_path)
    summary["summary_path"] = str(paths["summary"])
    return summary


def build_case_analysis(equity: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    frame = equity.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    recent = frame[frame["trade_date"] >= "20240101"].copy()
    if recent.empty:
        recent = frame.tail(500).copy()
    worst_days = recent.sort_values("daily_return").head(20)
    rows = []
    for row in worst_days.to_dict(orient="records"):
        trade_date = str(row["trade_date"])
        day_targets = targets[targets["trade_date"].astype(str) == trade_date] if not targets.empty else pd.DataFrame()
        rows.append(
            {
                "trade_date": trade_date,
                "daily_return": row["daily_return"],
                "equity": row["equity"],
                "exposure": row.get("exposure", np.nan),
                "positions": row["positions"],
                "top_codes": ",".join(day_targets.head(5).get("ts_code", pd.Series(dtype=str)).astype(str).tolist()),
                "avg_close": day_targets["close"].mean() if "close" in day_targets else np.nan,
                "avg_premium_rate": day_targets["premium_rate"].mean() if "premium_rate" in day_targets else np.nan,
                "avg_price_position_252": day_targets["price_position_252"].mean() if "price_position_252" in day_targets else np.nan,
                "avg_target_weight": day_targets["target_weight"].mean() if "target_weight" in day_targets else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_model_diagnostics(daily: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = add_low_position_features(daily)
    features["future_20d_return"] = (
        features.groupby("ts_code")["close"].shift(-20) / features["close"] - 1.0
    )
    sample = features[
        (features["trade_date"] >= "20150101")
        & (features["trade_date"] <= "20260616")
        & features["future_20d_return"].notna()
        & features["price_position_252"].notna()
    ].copy()
    sample["premium_rate"] = pd.to_numeric(
        sample.get("premium_rate", sample.get("bond_over_rate", np.nan)),
        errors="coerce",
    )
    sample["double_low"] = sample["close"] + sample["premium_rate"].fillna(0.0)
    sample["log_amount"] = np.log1p(pd.to_numeric(sample.get("amount", 0.0), errors="coerce").fillna(0.0))
    columns = [
        "close",
        "premium_rate",
        "double_low",
        "price_position_252",
        "drawdown_from_252_high",
        "momentum_20d",
        "log_amount",
    ]
    sample[columns] = sample[columns].replace([np.inf, -np.inf], np.nan)
    sample = sample.dropna(subset=columns + ["future_20d_return"])
    correlations = sample[columns + ["future_20d_return"]].corr(numeric_only=True)[
        "future_20d_return"
    ].drop("future_20d_return")
    payload = {
        "rows": int(len(sample)),
        "date_min": str(sample["trade_date"].min()) if not sample.empty else None,
        "date_max": str(sample["trade_date"].max()) if not sample.empty else None,
        "future_20d_return_correlations": correlations.to_dict(),
    }
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        train = sample[sample["trade_date"] < "20240101"]
        test = sample[sample["trade_date"] >= "20240101"]
        if len(train) > 100 and len(test) > 100:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train[columns])
            x_test = scaler.transform(test[columns])
            y_train = (train["future_20d_return"] > 0).astype(int)
            y_test = (test["future_20d_return"] > 0).astype(int)
            model = LogisticRegression(max_iter=500)
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]
            payload["logistic_auc_recent"] = float(roc_auc_score(y_test, probability))
            payload["logistic_coefficients"] = dict(zip(columns, model.coef_[0].tolist()))
    except Exception as exc:
        payload["model_error"] = str(exc)
    path = output_dir / "simple_model_diagnostics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    daily = pd.read_parquet(DAILY_PATH)
    basic = pd.read_parquet(BASIC_PATH)
    call = pd.read_parquet(CALL_PATH) if CALL_PATH.exists() else pd.DataFrame()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for config in variant_configs():
        for rebalance in ["daily", "weekly"]:
            summaries.append(
                run_variant(
                    daily=daily,
                    basic=basic,
                    call=call,
                    grid_config=config,
                    start_date="20150101",
                    end_date="20260616",
                    rebalance=rebalance,
                )
            )
            summaries.append(
                run_variant(
                    daily=daily,
                    basic=basic,
                    call=call,
                    grid_config=config,
                    start_date="20240101",
                    end_date="20260616",
                    rebalance=rebalance,
                )
            )
    summary_frame = pd.DataFrame(summaries)
    summary_path = OUTPUT_ROOT / "iteration_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    model_path = add_model_diagnostics(daily, OUTPUT_ROOT)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "model_path": str(model_path),
                "top": summary_frame.sort_values(
                    ["start_date", "sharpe", "max_drawdown"],
                    ascending=[True, False, False],
                )
                .head(12)[
                    [
                        "variant",
                        "rebalance",
                        "start_date",
                        "end_date",
                        "total_return",
                        "annual_return",
                        "max_drawdown",
                        "sharpe",
                        "average_exposure",
                        "invested_days",
                        "trade_count",
                    ]
                ]
                .to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
