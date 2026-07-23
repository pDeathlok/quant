from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd

from analyze_b1_entry_exit_grid import (
    ExitRule,
    PROJECT_ROOT,
    add_future_prices,
    predict_models,
    simulate_exit,
    summarize_returns,
)


CANDIDATE_PATH = PROJECT_ROOT / "data/features/b1/candidates_strict_no_volume_20240101.parquet"
DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
OUTPUT_DIR = Path(
    os.getenv("B1_FORMAL_OUTPUT_DIR", str(PROJECT_ROOT / "reports/b1/current"))
).expanduser().resolve()
REPORT_PATH = OUTPUT_DIR / "backtest.md"


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


COMBOS = [
    FormalCombo(
        name="stable_up10_020_down3_040_fixed8_sl15_T5",
        description="稳健版：pred_up10_es>=0.20 + pred_down3_es<=0.40 + 8%止盈 + 1.5%止损 + T+5到期",
        min_up8=None,
        min_up10=0.20,
        max_down3=0.40,
        exit_rule=ExitRule(
            "fixed_tp8%_sl1.5%_T5",
            "fixed",
            hold_days=4,
            take_profit=0.08,
            stop_loss=0.015,
        ),
    ),
    FormalCombo(
        name="aggressive_up8_070_down3_045_expiry_T9",
        description="进攻版：pred_up8_es>=0.70 + pred_down3_es<=0.45 + T+9收盘到期退出",
        min_up8=0.70,
        max_down3=0.45,
        exit_rule=ExitRule(
            "expiry_T9_close",
            "expiry",
            hold_days=8,
        ),
    ),
    FormalCombo(
        name="baseline_up8_055_trail5_dd2_sl2_T9",
        description="对照基准：pred_up8_es>=0.55 + 5%目标后2%回撤止盈 + 2%止损 + T+9到期",
        min_up8=0.55,
        max_down3=None,
        exit_rule=ExitRule(
            "trail_target5%_dd2%_sl2%_T9",
            "trailing",
            hold_days=8,
            take_profit=0.05,
            stop_loss=0.02,
            trail_drawdown=0.02,
        ),
    ),
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
    candidates = predict_models(candidates)

    compatibility = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
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
        (OUTPUT_DIR / "model_compatibility_audit.json").write_text(
            json.dumps(compatibility, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(compatibility["reason"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "model_compatibility_audit.json").write_text(
        json.dumps(compatibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
    (OUTPUT_DIR / "model_compatibility_audit.json").write_text(
        json.dumps(compatibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if compatibility["status"] != "valid":
        raise RuntimeError(compatibility["reason"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.csv"
    detail_path = OUTPUT_DIR / "trades.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(detail_path, index=False)

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
