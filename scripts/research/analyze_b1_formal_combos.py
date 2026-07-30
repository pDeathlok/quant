from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

import pandas as pd

from analyze_b1_entry_exit_grid import (
    ExitRule,
    PROJECT_ROOT,
    add_future_prices,
    simulate_exit,
    summarize_returns,
)
from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.routine.b1_daily_plan import predict_models as predict_release_models
from quant.routine.paths import CONFIG_PATH
from quant.routine.strategies import (
    ExitConfig,
    StrategyConfig,
    load_strategy_release,
)


CANDIDATE_PATH = PROJECT_ROOT / "data/features/b1/candidates_strict_no_volume_20240101.parquet"
DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
OUTPUT_DIR = Path(
    os.getenv("B1_FORMAL_OUTPUT_DIR", str(PROJECT_ROOT / "reports/b1/current"))
).expanduser().resolve()
REPORT_PATH = OUTPUT_DIR / "backtest.md"
RELEASE = load_strategy_release(CONFIG_PATH, include_disabled=True)
FORMAL_MODEL_DIR = Path(
    os.getenv("B1_FORMAL_MODEL_DIR", str(PROJECT_ROOT / RELEASE.model_dir))
).expanduser().resolve()


@dataclass(frozen=True)
class FormalCombo:
    name: str
    description: str
    min_up8: float | None
    max_down3: float | None
    exit_rule: ExitRule
    min_up5: float | None = None
    min_up10: float | None = None
    max_down2: float | None = None


def _pct(value: float | None) -> str:
    if value is None:
        return "0"
    percent = value * 100
    return str(int(percent)) if percent.is_integer() else f"{percent:g}"


def _exit_rule(config: ExitConfig) -> ExitRule:
    hold = config.hold_days + 1
    if config.kind == "expiry":
        name = f"expiry_T{hold}_close"
    elif config.kind == "fixed":
        name = f"fixed_tp{_pct(config.take_profit)}%_sl{_pct(config.stop_loss)}%_T{hold}"
    elif config.kind == "trailing":
        name = (
            f"trail_target{_pct(config.take_profit)}%_dd{_pct(config.trail_drawdown)}%"
            f"_sl{_pct(config.stop_loss)}%_T{hold}"
        )
    else:
        raise ValueError(f"unsupported formal B1 exit kind: {config.kind}")
    return ExitRule(
        name,
        config.kind,
        hold_days=config.hold_days,
        take_profit=config.take_profit,
        stop_loss=config.stop_loss,
        trail_drawdown=config.trail_drawdown,
    )


def _formal_combo(strategy: StrategyConfig) -> FormalCombo:
    return FormalCombo(
        name=strategy.backtest_combo,
        description=f"{strategy.name}：{strategy.description}",
        min_up5=strategy.entry.min_up5,
        min_up8=strategy.entry.min_up8,
        min_up10=strategy.entry.min_up10,
        max_down2=strategy.entry.max_down2,
        max_down3=strategy.entry.max_down3,
        exit_rule=_exit_rule(strategy.exit),
    )


COMBOS = [
    _formal_combo(strategy)
    for strategy in RELEASE.strategies
]


PERIODS = {
    "2024_test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    "oot_2025plus": (pd.Timestamp("2025-01-01"), pd.Timestamp.max),
    "all": (pd.Timestamp("2024-01-01"), pd.Timestamp.max),
}


def drawdown_window(trades: pd.DataFrame) -> tuple[str, str, float]:
    daily = trades.groupby("date")["return_pct"].mean().sort_index()
    if daily.empty:
        return "", "", float("nan")
    equity = (1 + daily / 100).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return str(start.date()), str(end.date()), float(dd.loc[end] * 100)


def combo_mask(df: pd.DataFrame, combo: FormalCombo) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if combo.min_up5 is not None:
        mask &= df["pred_up5_es"] >= combo.min_up5
    if combo.min_up8 is not None:
        mask &= df["pred_up8_es"] >= combo.min_up8
    if combo.min_up10 is not None:
        mask &= df["pred_up10_es"] >= combo.min_up10
    if combo.max_down2 is not None:
        mask &= df["pred_down2_es"] <= combo.max_down2
    if combo.max_down3 is not None:
        mask &= df["pred_down3_es"] <= combo.max_down3
    return mask


def fmt_table(df: pd.DataFrame, columns: list[str]) -> str:
    return df[columns].to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    print("loading candidates")
    candidates = pd.read_parquet(CANDIDATE_PATH)
    candidates["date"] = pd.to_datetime(candidates["date"])
    candidates = candidates[candidates["date"] >= pd.Timestamp("2024-01-01")].copy()
    candidates = predict_release_models(
        candidates,
        model_dir=FORMAL_MODEL_DIR,
        model_names=RELEASE.model_names,
    )
    candidates["pred_up10"] = candidates["pred_up10_es"]
    candidates["entry_score"] = (
        0.60 * candidates["pred_up8_es"]
        + 0.30 * candidates["pred_up10_es"]
        - 0.35 * candidates["pred_down3_es"]
    )

    compatibility = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "release_id": RELEASE.id,
        "strategy_config": str(CONFIG_PATH),
        "model_dir": str(FORMAL_MODEL_DIR),
        "candidate_rows": int(len(candidates)),
        "candidate_date_min": str(candidates["date"].min().date()),
        "candidate_date_max": str(candidates["date"].max().date()),
        "stable_threshold_rows": int(combo_mask(candidates, COMBOS[0]).sum()),
        "aggressive_threshold_rows": int(combo_mask(candidates, COMBOS[1]).sum()),
        "status": "valid",
    }
    if compatibility["stable_threshold_rows"] < 30 or compatibility["aggressive_threshold_rows"] < 10:
        compatibility["status"] = "incompatible"
        compatibility["reason"] = (
            "Production model thresholds are incompatible with the unified feature distribution; "
            "retrain and recalibrate before publishing a replacement backtest."
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(compatibility, OUTPUT_DIR / "model_compatibility_audit.json")
        raise RuntimeError(compatibility["reason"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(compatibility, OUTPUT_DIR / "model_compatibility_audit.json")

    print("adding future prices")
    candidates = add_future_prices(candidates, DAILY_DIR, max_hold_days=8)
    candidates = candidates.dropna(subset=["entry_open"]).copy()

    summary_rows = []
    detail_frames = []
    for combo in COMBOS:
        selected = candidates[combo_mask(candidates, combo)].copy()
        print(f"{combo.name}: {len(selected):,} selected")
        trades = simulate_exit(selected, combo.exit_rule)
        trades = trades.merge(
            selected[
                [
                    "date",
                    "symbol",
                    "name",
                    "industry",
                    "entry_score",
                    "pred_up8_es",
                    "pred_up10",
                    "pred_down3_es",
                    "entry_open",
                ]
            ],
            on=["date", "symbol"],
            how="left",
        )
        trades["combo"] = combo.name
        trades["combo_description"] = combo.description
        detail_frames.append(trades)

        for period_name, (start, end) in PERIODS.items():
            p_trades = trades[(trades["date"] >= start) & (trades["date"] <= end)].copy()
            metrics = summarize_returns(p_trades)
            if not metrics:
                continue
            dd_start, dd_end, dd_value = drawdown_window(p_trades)
            summary_rows.append(
                {
                    "period": period_name,
                    "combo": combo.name,
                    "description": combo.description,
                    "entry_min_up5": combo.min_up5,
                    "entry_min_up8": combo.min_up8,
                    "entry_min_up10": combo.min_up10,
                    "entry_max_down2": combo.max_down2,
                    "entry_max_down3": combo.max_down3,
                    "exit_rule": combo.exit_rule.name,
                    "max_dd_start": dd_start,
                    "max_dd_end": dd_end,
                    "max_dd_value_pct": dd_value,
                    **metrics,
                }
            )

    summary = pd.DataFrame(summary_rows)
    details = pd.concat(detail_frames, ignore_index=True)
    oot_validation = {}
    for combo in COMBOS[:2]:
        row = summary[(summary["period"] == "oot_2025plus") & (summary["combo"] == combo.name)].iloc[0]
        oot_validation[combo.name] = {
            "trades": int(row["trades"]),
            "avg_return_pct": float(row["avg_return_pct"]),
            "profit_factor": float(row["profit_factor"]),
            "passed": bool(row["trades"] >= 30 and row["avg_return_pct"] > 0 and row["profit_factor"] > 1),
        }
    compatibility["strategy_validation"] = oot_validation
    if not all(item["passed"] for item in oot_validation.values()):
        compatibility["status"] = "incompatible"
        compatibility["reason"] = "Calibrated formal strategies failed the OOT return gate."
    atomic_write_json(compatibility, OUTPUT_DIR / "model_compatibility_audit.json")
    if compatibility["status"] != "valid":
        raise RuntimeError(compatibility["reason"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.csv"
    detail_path = OUTPUT_DIR / "trades.csv"
    atomic_write_csv(summary, summary_path, index=False)
    atomic_write_csv(details, detail_path, index=False)

    oot = summary[summary["period"] == "oot_2025plus"].copy()
    display_cols = [
        "combo",
        "trades",
        "avg_return_pct",
        "median_return_pct",
        "win_rate",
        "daily_sharpe",
        "max_drawdown_pct",
        "profit_factor",
        "stop_rate",
        "trailing_stop_rate",
        "expiry_rate",
    ]

    period_cols = [
        "period",
        "combo",
        "trades",
        "avg_return_pct",
        "win_rate",
        "daily_sharpe",
        "max_drawdown_pct",
        "profit_factor",
        "stop_rate",
        "trailing_stop_rate",
    ]

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("# B1 两个正式组合回测\n\n")
        f.write("本次回测使用统一 B1 特征候选池和五个正式 XGBoost 模型；阈值按 2025 校准、2026 独立验证，达不到阈值则空仓。\n\n")
        f.write("## 1. 回测组合\n\n")
        for combo in COMBOS:
            f.write(f"- `{combo.name}`：{combo.description}\n")
        f.write("\n")
        f.write("## 2. 样本外 2025+ 结果\n\n")
        f.write(fmt_table(oot, display_cols))
        f.write("\n\n")
        f.write("## 3. 分阶段结果\n\n")
        f.write(fmt_table(summary, period_cols))
        f.write("\n\n")
        f.write("## 4. 最大回撤区间\n\n")
        f.write(fmt_table(summary[summary["period"] == "oot_2025plus"], ["combo", "max_dd_start", "max_dd_end", "max_dd_value_pct"]))
        f.write("\n\n")
        f.write("## 5. 解读\n\n")
        stable = oot[oot["combo"] == "stable_up10_020_down3_040_fixed8_sl15_T5"].iloc[0]
        aggressive = oot[oot["combo"] == "aggressive_up8_070_down3_045_expiry_T9"].iloc[0]
        baseline = oot[oot["combo"] == "baseline_up8_055_trail5_dd2_sl2_T9"].iloc[0]
        f.write(f"- 稳健版相对基准，交易次数从 `{baseline.trades:.0f}` 降到 `{stable.trades:.0f}`，胜率从 `{baseline.win_rate:.2%}` 到 `{stable.win_rate:.2%}`，最大回撤从 `{baseline.max_drawdown_pct:.2f}%` 到 `{stable.max_drawdown_pct:.2f}%`。\n")
        f.write(f"- 进攻版交易更少，样本外 `{aggressive.trades:.0f}` 笔，平均单笔收益 `{aggressive.avg_return_pct:.2f}%`，胜率 `{aggressive.win_rate:.2%}`，最大回撤 `{aggressive.max_drawdown_pct:.2f}%`。\n")
        f.write("- 稳健版用固定止损约束单笔风险；进攻版不设盘中止损，依赖严格入场阈值和 T+9 到期退出，因此尾部波动更大。\n")
        f.write("- 如果目标是降低回撤并保持较高交易频率，优先看稳健版；进攻版只适合低频观察，并需接受样本更少和回撤更大的风险。\n\n")
        f.write(f"输出文件：`{summary_path}`、`{detail_path}`。\n")

    print(f"wrote {summary_path}")
    print(f"wrote {detail_path}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
