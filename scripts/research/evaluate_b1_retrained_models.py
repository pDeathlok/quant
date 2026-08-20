#!/usr/bin/env python
"""Evaluate a retrained B1 model without using OOT data for rule selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from quant.research.b1_backtest import ExitRule, add_future_prices, simulate_exit, summarize_returns
from quant.routine.strategies import StrategyConfig, load_strategy_configs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("up5_es", "up8_es", "up10_es", "down2_es", "down3_es")


class EntryRule:
    def __init__(
        self,
        name: str,
        *,
        min_up5: float | None = None,
        min_up8: float | None = None,
        min_up10: float | None = None,
        max_down2: float | None = None,
        max_down3: float | None = None,
    ) -> None:
        self.name = name
        self.min_up5 = min_up5
        self.min_up8 = min_up8
        self.min_up10 = min_up10
        self.max_down2 = max_down2
        self.max_down3 = max_down3

    def parameters(self) -> dict[str, float | str | None]:
        return {
            "entry_rule": self.name,
            "min_up5": self.min_up5,
            "min_up8": self.min_up8,
            "min_up10": self.min_up10,
            "max_down2": self.max_down2,
            "max_down3": self.max_down3,
        }


def apply_entry_rule(data: pd.DataFrame, rule: EntryRule) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for column, threshold, operation in (
        ("pred_up5_es", rule.min_up5, "min"),
        ("pred_up8_es", rule.min_up8, "min"),
        ("pred_up10_es", rule.min_up10, "min"),
        ("pred_down2_es", rule.max_down2, "max"),
        ("pred_down3_es", rule.max_down3, "max"),
    ):
        if threshold is None:
            continue
        mask &= data[column].ge(threshold) if operation == "min" else data[column].le(threshold)
    return mask


def predict_models(data: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    out = data.copy()
    schemas = set(out.get("factor_schema_version", pd.Series(dtype=str)).dropna().astype(str))
    for model_name in MODEL_NAMES:
        model_path = model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        model = joblib.load(model_path)
        model_schema = getattr(model, "factor_schema_version_", None)
        if model_schema and schemas != {model_schema}:
            raise RuntimeError(
                f"{model_name} factor schema mismatch: model={model_schema} data={sorted(schemas) or ['missing']}"
            )
        feature_columns = list(model.feature_names_in_)
        missing = [column for column in feature_columns if column not in out]
        if missing:
            raise ValueError(f"{model_path} missing feature columns: {missing[:20]}")
        out[f"pred_{model_name}"] = model.predict_proba(out[feature_columns])[:, 1]
    return out


def entry_rule_from_strategy(strategy: StrategyConfig) -> EntryRule:
    return EntryRule(
        f"current__{strategy.id}",
        min_up5=strategy.entry.min_up5,
        min_up8=strategy.entry.min_up8,
        min_up10=strategy.entry.min_up10,
        max_down2=strategy.entry.max_down2,
        max_down3=strategy.entry.max_down3,
    )


def exit_rule_from_strategy(strategy: StrategyConfig) -> ExitRule:
    return ExitRule(
        name=f"current__{strategy.id}__{strategy.exit.rule_name}",
        kind=strategy.exit.kind,
        hold_days=strategy.exit.hold_days,
        take_profit=strategy.exit.take_profit,
        stop_loss=strategy.exit.stop_loss,
        trail_drawdown=strategy.exit.trail_drawdown,
    )


def _quantile(data: pd.DataFrame, column: str, q: float) -> float:
    value = data[column].quantile(q)
    if pd.isna(value):
        raise RuntimeError(f"Cannot calculate {column} q={q} on the test split")
    return float(value)


def _optional_threshold(value: float | None) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def build_test_calibrated_entry_rules(data: pd.DataFrame) -> list[EntryRule]:
    """Build probability grids using only the pre-OOT test distribution."""

    test = data[data["split"] == "test"]
    if test.empty:
        raise RuntimeError("Cannot calibrate B1 thresholds without a test split")
    up_quantiles = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    down_quantiles = (0.10, 0.20, 0.30, 0.40, 0.50)
    rules: list[EntryRule] = []

    for up_name, up_arg in (("up5", "min_up5"), ("up8", "min_up8"), ("up10", "min_up10")):
        for up_q in up_quantiles:
            for down_name, down_arg in (("down2", "max_down2"), ("down3", "max_down3")):
                for down_q in down_quantiles:
                    values = {
                        up_arg: _quantile(test, f"pred_{up_name}_es", up_q),
                        down_arg: _quantile(test, f"pred_{down_name}_es", down_q),
                    }
                    rules.append(
                        EntryRule(
                            f"testq__{up_name}_{up_q:.2f}__{down_name}_{down_q:.2f}",
                            **values,
                        )
                    )

    for up_q in (0.50, 0.70, 0.85, 0.90):
        for down_q in down_quantiles:
            rules.append(
                EntryRule(
                    f"testq__up5_up8_{up_q:.2f}__down2_{down_q:.2f}",
                    min_up5=_quantile(test, "pred_up5_es", up_q),
                    min_up8=_quantile(test, "pred_up8_es", up_q),
                    max_down2=_quantile(test, "pred_down2_es", down_q),
                )
            )
            rules.append(
                EntryRule(
                    f"testq__up8_up10_{up_q:.2f}__down3_{down_q:.2f}",
                    min_up8=_quantile(test, "pred_up8_es", up_q),
                    min_up10=_quantile(test, "pred_up10_es", up_q),
                    max_down3=_quantile(test, "pred_down3_es", down_q),
                )
            )
    return rules


def attach_exit_returns(data: pd.DataFrame, exit_rules: Iterable[ExitRule]) -> dict[str, pd.DataFrame]:
    if data.duplicated(["date", "symbol"]).any():
        raise RuntimeError("B1 evaluation candidates contain duplicate symbol/date rows")
    base_columns = ["date", "symbol", "split"]
    frames: dict[str, pd.DataFrame] = {}
    for rule in exit_rules:
        trades = simulate_exit(data, rule)
        frames[rule.name] = data[base_columns].merge(
            trades,
            on=["date", "symbol"],
            how="left",
            validate="one_to_one",
        )
    return frames


def summarize_with_confidence(trades: pd.DataFrame) -> dict[str, float | int]:
    valid = trades.dropna(subset=["return_pct"])
    metrics = summarize_returns(valid)
    if not metrics:
        return {}
    daily = valid.groupby("date")["return_pct"].mean().sort_index()
    standard_error = daily.std(ddof=1) / np.sqrt(len(daily)) if len(daily) > 1 else np.nan
    metrics["daily_return_lcb95_pct"] = (
        float(daily.mean() - 1.645 * standard_error) if pd.notna(standard_error) else np.nan
    )
    return metrics


def empty_trade_metrics() -> dict[str, float | int]:
    return {
        "trades": 0,
        "days": 0,
        "avg_return_pct": np.nan,
        "median_return_pct": np.nan,
        "win_rate": np.nan,
        "daily_sharpe": np.nan,
        "daily_return_lcb95_pct": np.nan,
        "max_drawdown_pct": np.nan,
        "profit_factor": np.nan,
    }


def evaluate_grid(
    candidates: pd.DataFrame,
    entry_rules: Iterable[EntryRule],
    exit_rules: Iterable[ExitRule],
    exit_returns: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict] = []
    for entry_rule in entry_rules:
        mask = apply_entry_rule(candidates, entry_rule)
        for exit_rule in exit_rules:
            trades = exit_returns[exit_rule.name].loc[mask]
            for split in ("test", "oot"):
                metrics = summarize_with_confidence(trades[trades["split"] == split])
                if metrics:
                    rows.append(
                        {
                            **entry_rule.parameters(),
                            "exit_rule": exit_rule.name,
                            "exit_kind": exit_rule.kind,
                            "hold_days": exit_rule.hold_days,
                            "take_profit": exit_rule.take_profit,
                            "stop_loss": exit_rule.stop_loss,
                            "trail_drawdown": exit_rule.trail_drawdown,
                            "split": split,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def select_rules_on_test(grid: pd.DataFrame, min_trades: int, min_days: int) -> pd.DataFrame:
    test = grid[(grid["split"] == "test") & (grid["trades"] >= min_trades) & (grid["days"] >= min_days)].copy()
    if test.empty:
        raise RuntimeError(
            f"No B1 entry/exit combination meets test coverage: min_trades={min_trades}, min_days={min_days}"
        )
    test["selection_rank"] = test.groupby("exit_rule")["daily_return_lcb95_pct"].rank(
        method="first", ascending=False
    )
    selected_test = test[test["selection_rank"] == 1].copy()
    keys = ["entry_rule", "exit_rule"]
    selected_oot = grid[grid["split"] == "oot"].merge(selected_test[keys], on=keys, how="inner")
    selected_test["selected_on"] = "test"
    selected_oot["selected_on"] = "test"
    return pd.concat([selected_test, selected_oot], ignore_index=True).sort_values(keys + ["split"])


def evaluate_named_periods(
    candidates: pd.DataFrame,
    rules: Iterable[EntryRule],
    exits: dict[str, ExitRule],
    exit_returns: dict[str, pd.DataFrame],
    rule_to_exit: dict[str, str],
    rule_set: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    periods = {
        "test": candidates["split"].eq("test"),
        "oot": candidates["split"].eq("oot"),
        "oot_2025": candidates["split"].eq("oot") & candidates["date"].dt.year.eq(2025),
        "oot_2026": candidates["split"].eq("oot") & candidates["date"].dt.year.eq(2026),
    }
    for rule in rules:
        entry_mask = apply_entry_rule(candidates, rule)
        exit_name = rule_to_exit[rule.name]
        exit_rule = exits[exit_name]
        for period, period_mask in periods.items():
            metrics = summarize_with_confidence(exit_returns[exit_name].loc[entry_mask & period_mask])
            rows.append(
                {
                    "rule_set": rule_set,
                    **rule.parameters(),
                    "exit_rule": exit_name,
                    "exit_kind": exit_rule.kind,
                    "period": period,
                    **(metrics or empty_trade_metrics()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_stable_activity_sensitivity(
    activity_candidates: pd.DataFrame,
    trade_candidates: pd.DataFrame,
    stable_strategy: StrategyConfig,
    stable_exit_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Measure the cost of relaxing B1's current down3 risk ceiling."""

    if stable_strategy.entry.min_up10 is None or stable_strategy.entry.max_down3 is None:
        raise ValueError("stable activity sensitivity requires min_up10 and max_down3")
    thresholds = sorted(
        {
            float(stable_strategy.entry.max_down3),
            0.45,
            0.50,
            0.55,
            0.60,
        }
    )
    period_specs = (
        ("test", "test", None),
        ("oot", "oot", None),
        ("oot_2025", "oot", 2025),
        ("oot_2026", "oot", 2026),
    )
    latest_date = activity_candidates["date"].max()
    rows = []
    for threshold in thresholds:
        rule = EntryRule(
            f"stable_down3_le_{threshold:.2f}",
            min_up10=stable_strategy.entry.min_up10,
            max_down3=threshold,
        )
        activity_entry = apply_entry_rule(activity_candidates, rule)
        trade_entry = apply_entry_rule(trade_candidates, rule)
        latest_signals = int((activity_entry & activity_candidates["date"].eq(latest_date)).sum())
        for period, split, year in period_specs:
            activity_period = activity_candidates["split"].eq(split)
            trade_period = trade_candidates["split"].eq(split)
            if year is not None:
                activity_period &= activity_candidates["date"].dt.year.eq(year)
                trade_period &= trade_candidates["date"].dt.year.eq(year)
            period_dates = pd.Index(sorted(activity_candidates.loc[activity_period, "date"].unique()))
            counts = (
                activity_candidates.loc[activity_period & activity_entry]
                .groupby("date")
                .size()
                .reindex(period_dates, fill_value=0)
            )
            metrics = summarize_with_confidence(
                stable_exit_returns.loc[trade_period & trade_entry]
            ) or empty_trade_metrics()
            rows.append(
                {
                    **rule.parameters(),
                    "period": period,
                    "raw_gate_days": len(period_dates),
                    "signal_days": int((counts > 0).sum()),
                    "empty_day_rate": float((counts == 0).mean()) if len(counts) else np.nan,
                    "avg_signals_per_day": float(counts.mean()) if len(counts) else np.nan,
                    "latest_date": str(latest_date.date()),
                    "latest_signal_count": latest_signals,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def prediction_quantiles(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("test", "oot"):
        part = data[data["split"] == split]
        for model_name in MODEL_NAMES:
            values = part[f"pred_{model_name}"]
            rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "rows": len(values),
                    **{f"q{int(q * 100):02d}": float(values.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
                }
            )
    return pd.DataFrame(rows)


def read_model_quality(training_report: Path | None) -> pd.DataFrame:
    if training_report is None:
        return pd.DataFrame()
    payload = json.loads(training_report.read_text(encoding="utf-8"))
    rows = []
    for model_name in MODEL_NAMES:
        report = payload[model_name]
        for split in ("test", "oot"):
            metrics = report["splits"][split]
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "auc": metrics["auc"],
                    "pr_auc": metrics["pr_auc"],
                    "brier_score": metrics.get("brier_score"),
                    "log_loss": metrics.get("log_loss"),
                    "positive_rate": metrics["positive_rate"],
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(
    path: Path,
    *,
    variant: str,
    dataset: Path,
    model_dir: Path,
    rows: int,
    model_quality: pd.DataFrame,
    current_periods: pd.DataFrame,
    selected: pd.DataFrame,
    selected_periods: pd.DataFrame,
    stable_activity: pd.DataFrame,
) -> None:
    columns = [
        "entry_rule", "exit_rule", "split", "trades", "avg_return_pct", "win_rate",
        "profit_factor", "daily_sharpe", "daily_return_lcb95_pct", "max_drawdown_pct",
    ]
    period_columns = [
        "entry_rule", "period", "trades", "avg_return_pct", "win_rate", "profit_factor",
        "daily_sharpe", "daily_return_lcb95_pct", "max_drawdown_pct",
    ]
    lines = [
        f"# B1 重训练评估：{variant}",
        "",
        f"- 数据集：`{dataset}`（{rows:,} 个 B1 原始门槛候选）",
        f"- 模型目录：`{model_dir}`",
        "- 阈值校准仅使用 2025 年前的 test 股票；2025 年起 OOT 未参与选择。",
        "- 交易口径：信号日筛选，T+1 开盘买入，沿用当前三个 B1 出场规则；收益未计交易成本。",
        "",
    ]
    if not model_quality.empty:
        lines.extend(["## 分类质量", "", model_quality.to_markdown(index=False, floatfmt=".4f"), ""])
    lines.extend(
        [
            "## 原生产阈值直接迁移到新模型",
            "",
            current_periods[period_columns].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## 仅按 test 期选择的规则及其 OOT 表现",
            "",
            selected[columns].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## 入选规则分年表现",
            "",
            selected_periods[period_columns].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## 稳健版 down3 风险上限的活跃度—绩效敏感性",
            "",
            stable_activity[
                [
                    "max_down3", "period", "latest_signal_count", "empty_day_rate",
                    "avg_signals_per_day", "trades", "avg_return_pct", "profit_factor",
                    "daily_sharpe", "max_drawdown_pct",
                ]
            ].to_markdown(index=False, floatfmt=".4f"),
            "",
            "> 判定是否升级生产模型时，应优先看 OOT 与分年稳定性，不应因为 test 最优而直接发布。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, default=None)
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--strategy-config", type=Path, default=PROJECT_ROOT / "configs/strategies/b1_selected.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-test-trades", type=int, default=100)
    parser.add_argument("--min-test-days", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(args.dataset)
    candidates["date"] = pd.to_datetime(candidates["date"], errors="raise")
    candidates = predict_models(candidates, args.model_dir)
    activity_candidates = candidates.copy()
    strategies = load_strategy_configs(args.strategy_config, include_disabled=True)
    current_rules = [entry_rule_from_strategy(strategy) for strategy in strategies]
    exit_rules = [exit_rule_from_strategy(strategy) for strategy in strategies]
    exits = {rule.name: rule for rule in exit_rules}

    candidates = add_future_prices(candidates, args.daily_dir, max(rule.hold_days for rule in exit_rules))
    candidates = candidates.reset_index(drop=True)
    exit_returns = attach_exit_returns(candidates, exit_rules)

    adaptive_rules = build_test_calibrated_entry_rules(candidates)
    grid = evaluate_grid(candidates, adaptive_rules, exit_rules, exit_returns)
    selected = select_rules_on_test(grid, args.min_test_trades, args.min_test_days)

    current_rule_to_exit = {
        rule.name: exit_rule_from_strategy(strategy).name
        for rule, strategy in zip(current_rules, strategies)
    }
    current_periods = evaluate_named_periods(
        candidates, current_rules, exits, exit_returns, current_rule_to_exit, "current_thresholds"
    )
    stable_strategy = next(strategy for strategy in strategies if strategy.id == "b1_stable")
    stable_exit_name = exit_rule_from_strategy(stable_strategy).name
    stable_activity = evaluate_stable_activity_sensitivity(
        activity_candidates,
        candidates,
        stable_strategy,
        exit_returns[stable_exit_name],
    )

    selected_test_rows = selected[selected["split"] == "test"]
    selected_rules = []
    selected_rule_to_exit = {}
    for row in selected_test_rows.itertuples(index=False):
        selected_name = f"selected__{row.exit_rule}__{row.entry_rule}"
        selected_rules.append(
            EntryRule(
                selected_name,
                min_up5=_optional_threshold(row.min_up5),
                min_up8=_optional_threshold(row.min_up8),
                min_up10=_optional_threshold(row.min_up10),
                max_down2=_optional_threshold(row.max_down2),
                max_down3=_optional_threshold(row.max_down3),
            )
        )
        selected_rule_to_exit[selected_name] = row.exit_rule
    selected_periods = evaluate_named_periods(
        candidates, selected_rules, exits, exit_returns, selected_rule_to_exit, "test_selected"
    )

    quantiles = prediction_quantiles(candidates)
    model_quality = read_model_quality(args.training_report)
    grid.to_csv(args.output_dir / "entry_exit_grid.csv", index=False)
    selected.to_csv(args.output_dir / "selected_by_test.csv", index=False)
    current_periods.to_csv(args.output_dir / "current_threshold_periods.csv", index=False)
    selected_periods.to_csv(args.output_dir / "selected_periods.csv", index=False)
    stable_activity.to_csv(args.output_dir / "stable_activity_sensitivity.csv", index=False)
    quantiles.to_csv(args.output_dir / "prediction_quantiles.csv", index=False)
    if not model_quality.empty:
        model_quality.to_csv(args.output_dir / "model_quality.csv", index=False)
    metadata = {
        "variant": args.variant,
        "dataset": str(args.dataset),
        "model_dir": str(args.model_dir),
        "rows_with_future_prices": len(candidates),
        "date_min": str(candidates["date"].min().date()),
        "date_max": str(candidates["date"].max().date()),
        "selection_split": "test",
        "oot_start": "2025-01-01",
        "min_test_trades": args.min_test_trades,
        "min_test_days": args.min_test_days,
        "exit_rules": [asdict(rule) for rule in exit_rules],
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(
        args.output_dir / "summary.md",
        variant=args.variant,
        dataset=args.dataset,
        model_dir=args.model_dir,
        rows=len(candidates),
        model_quality=model_quality,
        current_periods=current_periods,
        selected=selected,
        selected_periods=selected_periods,
        stable_activity=stable_activity,
    )
    print(f"B1 evaluation written: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
