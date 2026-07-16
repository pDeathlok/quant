#!/usr/bin/env python
"""Point-in-time walk-forward validation for optimized similar-pattern signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.research.similar_patterns import (  # noqa: E402
    SimilarPatternConfig,
    build_probability_variant_cases,
    build_pattern_vector,
    classify_forecast_signal,
    latest_snapshot,
    load_daily_file,
    load_stock_basic,
    load_stock_vector_cache,
    make_cached_candidate_row,
    optimize_similar_cases,
    resample_close_series,
    select_best_positions_from_contiguous_matches,
    summarize_forecast,
    vector_cache_key,
)
from quant.research.similar_patterns_validation import (  # noqa: E402
    HORIZON_COLUMNS,
    HORIZON_DAYS,
    apply_global_expanding_calibration,
    build_industry_regime,
    filter_cases_mature_at_signal,
    load_market_regime,
    summarize_walk_forward_records,
    validation_config_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate optimized similar-pattern decisions")
    parser.add_argument("--targets", nargs="+", default=["002594.SZ", "002788.SZ"])
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--anchor-step", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/similar_patterns/validation_2025")
    parser.add_argument("--max-symbols", type=int, default=None)
    return parser.parse_args()


def _asof_value(frame: pd.DataFrame, date: pd.Timestamp, column: str, default: str) -> str:
    eligible = frame.loc[frame["date"] <= date, column] if not frame.empty and column in frame.columns else pd.Series(dtype=object)
    return str(eligible.iloc[-1]) if not eligible.empty else default


def _query_contexts(
    targets: list[str],
    start_date: str,
    end_date: str | None,
    anchor_step: int,
    config: SimilarPatternConfig,
    basic: pd.DataFrame,
    market_regime: pd.DataFrame,
    industry_regimes: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, pd.DataFrame]]:
    queries: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    target_daily: dict[str, pd.DataFrame] = {}
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    for symbol in targets:
        daily = load_daily_file(PROJECT_ROOT / "data/raw/daily" / f"{symbol}.parquet")
        target_daily[symbol] = daily
        weekly, monthly = resample_close_series(daily)
        mask = daily["date"].ge(pd.Timestamp(start_date))
        if end_date:
            mask &= daily["date"].le(pd.Timestamp(end_date))
        indices = daily.index[mask].tolist()[:: max(1, anchor_step)]
        latest_eligible = daily.index[mask].tolist()
        if latest_eligible and latest_eligible[-1] not in indices:
            indices.append(latest_eligible[-1])
        industry = str(basic_map.get(symbol, {}).get("industry", ""))
        industry_regime = industry_regimes.get(industry, pd.DataFrame())
        for end_idx in indices:
            vector = build_pattern_vector(daily, int(end_idx), config, weekly, monthly)
            if vector is None:
                continue
            signal_date = pd.Timestamp(daily.iloc[end_idx]["date"])
            queries.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "end_idx": int(end_idx),
                    "industry": industry,
                    "market_regime": _asof_value(market_regime, signal_date, "market_regime", "neutral"),
                    "industry_regime": _asof_value(industry_regime, signal_date, "industry_regime", "neutral"),
                    "snapshot": latest_snapshot(daily, int(end_idx)),
                }
            )
            vectors.append(vector.astype(np.float32))
    return queries, np.vstack(vectors), target_daily


def _collect_matches(
    queries: list[dict[str, object]],
    query_vectors: np.ndarray,
    config: SimilarPatternConfig,
    max_symbols: int | None,
) -> list[list[dict[str, object]]]:
    cache_root = PROJECT_ROOT / "data/research/similar_patterns/vector_cache" / vector_cache_key(config)
    files = sorted(cache_root.glob("*.npz"))
    if max_symbols is not None:
        files = files[:max_symbols]
    matches: list[list[dict[str, object]]] = [[] for _ in queries]
    target_symbols = {str(query["symbol"]).upper() for query in queries}
    for file_no, path in enumerate(files, 1):
        cached = load_stock_vector_cache(path)
        if str(cached["symbol"]).upper() in target_symbols:
            continue
        matrix = np.asarray(cached["vectors"], dtype=np.float32)
        if matrix.size == 0:
            continue
        indices = [int(value) for value in cached["indices"]]
        dates = pd.to_datetime(np.asarray(cached["dates"]).astype(str))
        matrix_norm = np.sum(matrix * matrix, axis=1, dtype=np.float32)[:, None]
        query_norm = np.sum(query_vectors * query_vectors, axis=1, dtype=np.float32)[None, :]
        distance_sq = np.maximum(matrix_norm + query_norm - 2.0 * (matrix @ query_vectors.T), 0.0)
        distances = np.sqrt(distance_sq).astype(np.float32)
        similarities = 1.0 / (1.0 + distances)
        for query_no, query in enumerate(queries):
            cutoff = pd.Timestamp(query["signal_date"]) - pd.offsets.BDay(1)
            eligible_count = int(np.searchsorted(dates.values, np.datetime64(cutoff), side="right"))
            if eligible_count <= 0:
                continue
            selected = select_best_positions_from_contiguous_matches(
                indices[:eligible_count],
                similarities[:eligible_count, query_no],
                float(config.similarity_threshold),
                config.candidate_step_days,
            )
            for position in selected:
                matches[query_no].append(
                    make_cached_candidate_row(
                        cached,
                        position,
                        float(distances[position, query_no]),
                        float(similarities[position, query_no]),
                    )
                )
        if file_no % 500 == 0 or file_no == len(files):
            print(f"scanned vector caches {file_no}/{len(files)}", flush=True)
    return matches


def _forecast_probability(cases: pd.DataFrame, horizon: str) -> float | None:
    if cases.empty:
        return None
    forecast = summarize_forecast(cases).set_index("horizon")
    value = forecast.loc[horizon, "up_probability"]
    return float(value) if pd.notna(value) else None


def _single_condition_probabilities(
    cases: pd.DataFrame,
    query: dict[str, object],
    horizon: str,
    config: SimilarPatternConfig,
) -> dict[str, float | None]:
    """Evaluate one optimization condition at a time on the same mature cases."""
    variants = build_probability_variant_cases(
        cases,
        config,
        target_date=pd.Timestamp(query["signal_date"]),
        target_industry=str(query["industry"]),
        target_market_regime=str(query["market_regime"]),
        target_industry_regime=str(query["industry_regime"]),
    )
    return {
        name: _forecast_probability(variant_cases, horizon)
        for name, variant_cases in variants.items()
    }


def _build_records(
    queries: list[dict[str, object]],
    matches: list[list[dict[str, object]]],
    target_daily: dict[str, pd.DataFrame],
    market_regime: pd.DataFrame,
    industry_regimes: dict[str, pd.DataFrame],
    config: SimilarPatternConfig,
) -> pd.DataFrame:
    market_map = market_regime.set_index("date")["market_regime"] if not market_regime.empty else pd.Series(dtype=object)
    records: list[dict[str, object]] = []
    for query_no, query in enumerate(queries):
        cases = pd.DataFrame(matches[query_no])
        if cases.empty:
            continue
        cases["date"] = pd.to_datetime(cases["date"])
        cases["market_regime"] = cases["date"].map(market_map).fillna("neutral")
        industry = str(query["industry"])
        industry_frame = industry_regimes.get(industry, pd.DataFrame())
        industry_map = (
            industry_frame.set_index("date")["industry_regime"]
            if not industry_frame.empty
            else pd.Series(dtype=object)
        )
        cases["industry_regime"] = np.where(
            cases["industry"].astype(str).eq(industry),
            cases["date"].map(industry_map).fillna("neutral"),
            "cross_industry",
        )
        daily = target_daily[str(query["symbol"])]
        end_idx = int(query["end_idx"])
        for horizon, days in HORIZON_DAYS.items():
            mature_cases = filter_cases_mature_at_signal(cases, pd.Timestamp(query["signal_date"]), horizon)
            optimized, sample_summary = optimize_similar_cases(
                mature_cases,
                config,
                target_date=pd.Timestamp(query["signal_date"]),
                target_industry=industry,
                target_market_regime=str(query["market_regime"]),
                target_industry_regime=str(query["industry_regime"]),
            )
            raw_forecast = summarize_forecast(mature_cases).set_index("horizon")
            optimized_forecast = summarize_forecast(optimized).set_index("horizon")
            if horizon not in optimized_forecast.index or optimized.empty:
                continue
            actual_return = np.nan
            outcome_date = pd.NaT
            if end_idx + days < len(daily):
                actual_return = float(daily.iloc[end_idx + days]["close"] / daily.iloc[end_idx]["close"] - 1.0)
                outcome_date = pd.Timestamp(daily.iloc[end_idx + days]["date"])
            ablation = _single_condition_probabilities(mature_cases, query, horizon, config)
            records.append(
                {
                    "symbol": query["symbol"],
                    "signal_date": query["signal_date"],
                    "outcome_date": outcome_date,
                    "horizon": horizon,
                    "horizon_days": days,
                    "close": query["snapshot"]["close"],
                    "market_regime": query["market_regime"],
                    "industry_regime": query["industry_regime"],
                    "raw_baseline_up_probability": raw_forecast.loc[horizon, "up_probability"],
                    "raw_up_probability": optimized_forecast.loc[horizon, "up_probability"],
                    "event_dedupe_up_probability": ablation["event_dedupe"],
                    "nonlinear_up_probability": ablation["nonlinear"],
                    "regime_industry_up_probability": ablation["regime_industry"],
                    "recency_up_probability": ablation["recency"],
                    "median_return": optimized_forecast.loc[horizon, "median"],
                    "actual_return": actual_return,
                    "raw_cases": sample_summary["raw_cases"],
                    "effective_cases": sample_summary["deduplicated_cases"],
                    "effective_sample_size": sample_summary["effective_sample_size"],
                    "sample_status": sample_summary["sample_status"],
                    "snapshot": query["snapshot"],
                    "outcome_column": HORIZON_COLUMNS[horizon],
                }
            )
    return pd.DataFrame(records)


def _apply_decisions(records: pd.DataFrame, calibrations_min_samples: int, config: SimilarPatternConfig) -> tuple[pd.DataFrame, dict]:
    calibrated, calibrations = apply_global_expanding_calibration(records, min_samples=calibrations_min_samples)
    decisions: list[dict[str, object]] = []
    for row in calibrated.itertuples(index=False):
        decision = classify_forecast_signal(
            float(row.calibrated_up_probability),
            dict(row.snapshot),
            str(row.market_regime),
            config,
        )
        decisions.append(decision)
    calibrated["signal"] = [decision["signal"] for decision in decisions]
    calibrated["raw_signal"] = [decision["raw_signal"] for decision in decisions]
    calibrated["risk_gate"] = [decision["risk_gate"] for decision in decisions]
    calibrated["risk_reasons"] = ["；".join(decision["reasons"]) for decision in decisions]
    return calibrated.drop(columns=["snapshot"]), calibrations


def _augment_summary(summary: pd.DataFrame, records: pd.DataFrame, target_daily: dict[str, pd.DataFrame], start_date: str) -> pd.DataFrame:
    out = summary.copy()
    raw_accuracy: dict[tuple[str, str], float] = {}
    forced_accuracy: dict[tuple[str, str], float] = {}
    for (symbol, horizon), group in records.groupby(["symbol", "horizon"]):
        mature = group[group["actual_return"].notna()]
        actual_up = mature["actual_return"].gt(0)
        raw_accuracy[(symbol, horizon)] = float(mature["raw_baseline_up_probability"].ge(50).eq(actual_up).mean() * 100)
        forced_accuracy[(symbol, horizon)] = float(mature["calibrated_up_probability"].ge(50).eq(actual_up).mean() * 100)
    out["raw_baseline_accuracy"] = [round(raw_accuracy.get((row.symbol, row.horizon), np.nan), 2) for row in out.itertuples()]
    out["optimized_forced_accuracy"] = [round(forced_accuracy.get((row.symbol, row.horizon), np.nan), 2) for row in out.itertuples()]
    buy_hold: dict[str, float] = {}
    for symbol, daily in target_daily.items():
        period = daily[daily["date"] >= pd.Timestamp(start_date)]
        buy_hold[symbol] = float(period.iloc[-1]["close"] / period.iloc[0]["close"] - 1.0) * 100 if len(period) > 1 else 0.0
    out["buy_hold_return"] = [round(buy_hold.get(str(symbol), 0.0), 2) for symbol in out["symbol"]]
    return out


def _select_probability_models(records: pd.DataFrame, config: SimilarPatternConfig) -> dict[str, dict[str, object]]:
    """Choose one transferable policy per horizon from pooled watchlist samples."""
    selections: dict[str, dict[str, object]] = {}
    sources = {
        "raw_baseline": "raw_baseline_up_probability",
        "event_dedupe": "event_dedupe_up_probability",
        "nonlinear": "nonlinear_up_probability",
        "regime_industry": "regime_industry_up_probability",
        "recency": "recency_up_probability",
        "full_weighting": "raw_up_probability",
        "calibrated": "calibrated_up_probability",
    }
    thresholds = [(50.0, 50.0), (49.0, 51.0), (47.0, 53.0), (45.0, 55.0), (40.0, 60.0)]
    for horizon, group in records.groupby("horizon"):
        mature = group[group["actual_return"].notna()].copy()
        candidates: list[dict[str, object]] = []
        for source, column in sources.items():
            for bearish_max, bullish_min in thresholds:
                valid = mature[column].notna()
                actionable = valid & (mature[column].le(bearish_max) | mature[column].ge(bullish_min))
                sample = mature[actionable]
                predicted_up = sample[column].ge(bullish_min)
                accuracy = float(predicted_up.eq(sample["actual_return"].gt(0)).mean() * 100) if len(sample) else 0.0
                by_symbol: dict[str, dict[str, object]] = {}
                for symbol, symbol_group in mature.groupby("symbol"):
                    symbol_sample = symbol_group.loc[actionable.reindex(symbol_group.index, fill_value=False)]
                    symbol_accuracy = (
                        float(
                            symbol_sample[column]
                            .ge(bullish_min)
                            .eq(symbol_sample["actual_return"].gt(0))
                            .mean()
                            * 100
                        )
                        if len(symbol_sample)
                        else 0.0
                    )
                    by_symbol[str(symbol)] = {
                        "accuracy": round(symbol_accuracy, 2),
                        "coverage": round(len(symbol_sample) / len(symbol_group) * 100, 2),
                        "signals": int(len(symbol_sample)),
                    }
                candidates.append(
                    {
                        "source": source,
                        "bearish_max": bearish_max,
                        "bullish_min": bullish_min,
                        "enable_risk_gate": False,
                        "accuracy": round(accuracy, 2),
                        "coverage": round(len(sample) / len(mature) * 100, 2) if len(mature) else 0.0,
                        "signals": int(len(sample)),
                        "worst_symbol_accuracy": min(
                            (float(item["accuracy"]) for item in by_symbol.values()), default=0.0
                        ),
                        "worst_symbol_coverage": min(
                            (float(item["coverage"]) for item in by_symbol.values()), default=0.0
                        ),
                        "by_symbol": by_symbol,
                    }
                )
        eligible = [
            item
            for item in candidates
            if item["signals"] >= 40
            and float(item["coverage"]) >= 50.0
            and float(item["worst_symbol_accuracy"]) >= 50.0
            and float(item["worst_symbol_coverage"]) >= 50.0
        ] or candidates
        selected = max(
            eligible,
            key=lambda item: (
                float(item["accuracy"]),
                float(item["worst_symbol_accuracy"]),
                float(item["coverage"]),
            ),
        )
        selections[str(horizon)] = {"selected": selected, "candidates": candidates}
    return selections


def _build_ablation_summary(records: pd.DataFrame, config: SimilarPatternConfig) -> pd.DataFrame:
    probability_variants = {
        "baseline": "raw_baseline_up_probability",
        "event_dedupe_only": "event_dedupe_up_probability",
        "nonlinear_weight_only": "nonlinear_up_probability",
        "market_industry_only": "regime_industry_up_probability",
        "recency_only": "recency_up_probability",
        "full_weighting": "raw_up_probability",
        "full_plus_calibration": "calibrated_up_probability",
    }
    rows: list[dict[str, object]] = []
    for (symbol, horizon), group in records.groupby(["symbol", "horizon"]):
        mature = group[group["actual_return"].notna()].copy()
        actual_up = mature["actual_return"].gt(0)
        baseline_accuracy = float(mature["raw_baseline_up_probability"].ge(50).eq(actual_up).mean() * 100)
        for variant, column in probability_variants.items():
            valid = mature[column].notna()
            sample = mature[valid]
            predicted_up = sample[column].ge(50)
            accuracy = float(predicted_up.eq(actual_up[valid]).mean() * 100) if len(sample) else np.nan
            probability = sample[column].astype(float) / 100.0
            brier = float(np.mean(np.square(probability - actual_up[valid].astype(float)))) if len(sample) else np.nan
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "variant": variant,
                    "signals": int(len(sample)),
                    "coverage": round(len(sample) / len(mature) * 100, 2) if len(mature) else 0.0,
                    "direction_accuracy": round(accuracy, 2),
                    "accuracy_delta_vs_baseline": round(accuracy - baseline_accuracy, 2),
                    "brier_score": round(brier, 4),
                }
            )
        for variant, column in [
            ("observe_zone_on_baseline", "raw_baseline_up_probability"),
            ("observe_zone_on_full", "raw_up_probability"),
            ("calibration_plus_observe", "calibrated_up_probability"),
        ]:
            actionable = mature[column].le(config.signal_bearish_max) | mature[column].ge(config.signal_bullish_min)
            sample = mature[actionable]
            accuracy = float(
                sample[column].ge(config.signal_bullish_min).eq(sample["actual_return"].gt(0)).mean() * 100
            ) if len(sample) else np.nan
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "variant": variant,
                    "signals": int(len(sample)),
                    "coverage": round(len(sample) / len(mature) * 100, 2) if len(mature) else 0.0,
                    "direction_accuracy": round(accuracy, 2),
                    "accuracy_delta_vs_baseline": round(accuracy - baseline_accuracy, 2),
                    "brier_score": None,
                }
            )
        final = mature[mature["signal"].isin(["bullish", "bearish"])]
        final_accuracy = float(
            final["signal"].eq("bullish").eq(final["actual_return"].gt(0)).mean() * 100
        ) if len(final) else np.nan
        rows.append(
            {
                "symbol": symbol,
                "horizon": horizon,
                "variant": "full_plus_risk_gate",
                "signals": int(len(final)),
                "coverage": round(len(final) / len(mature) * 100, 2) if len(mature) else 0.0,
                "direction_accuracy": round(final_accuracy, 2),
                "accuracy_delta_vs_baseline": round(final_accuracy - baseline_accuracy, 2),
                "brier_score": None,
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(
    output_dir: Path,
    records: pd.DataFrame,
    summary: pd.DataFrame,
    calibrations: dict,
    args: argparse.Namespace,
    config: SimilarPatternConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_selection = _select_probability_models(records, config)
    summary = summary.copy()
    summary["selected_model"] = [
        model_selection.get(str(row.horizon), {}).get("selected", {}).get("source")
        for row in summary.itertuples()
    ]
    summary["selected_model_accuracy"] = [
        (
            model_selection.get(str(row.horizon), {})
            .get("selected", {})
            .get("by_symbol", {})
            .get(str(row.symbol), {})
            .get("accuracy")
        )
        for row in summary.itertuples()
    ]
    summary["selected_model_coverage"] = [
        (
            model_selection.get(str(row.horizon), {})
            .get("selected", {})
            .get("by_symbol", {})
            .get(str(row.symbol), {})
            .get("coverage")
        )
        for row in summary.itertuples()
    ]
    summary["global_model_accuracy"] = [
        model_selection.get(str(row.horizon), {}).get("selected", {}).get("accuracy")
        for row in summary.itertuples()
    ]
    summary["global_model_coverage"] = [
        model_selection.get(str(row.horizon), {}).get("selected", {}).get("coverage")
        for row in summary.itertuples()
    ]
    records.to_csv(output_dir / "walk_forward_records.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    ablation = _build_ablation_summary(records, config)
    ablation.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    calibration_payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "anchor_step": args.anchor_step,
        "targets": args.targets,
        "config": validation_config_payload(config),
        "calibrations": calibrations,
        "model_selection": model_selection,
        "summary": summary.replace({np.nan: None}).to_dict("records"),
        "ablation": ablation.replace({np.nan: None}).to_dict("records"),
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 相似走势优化版 2025 年后走步验证",
        "",
        f"- 标的：{', '.join(args.targets)}",
        f"- 区间：{args.start_date} 至 {args.end_date or records['signal_date'].max().strftime('%Y-%m-%d')}",
        f"- 锚点：每 {args.anchor_step} 个交易日；候选结果必须在信号日已经成熟。",
        "- 模型策略：按预测周期在全自选池合并样本上统一选择，并要求每只验证股票准确率、覆盖率均不低于 50%。",
        f"- 候选观望区：50/50、49/51、47/53、45/55、40/60；单次方向成本：{config.transaction_cost:.2%}。",
        "",
        summary.to_markdown(index=False),
        "",
        "## 单条件消融",
        "",
        ablation[ablation["horizon"] == "next_1d"].to_markdown(index=False),
        "",
        "## 口径",
        "",
        "- 同日同行业只保留相似度最高案例，同日最多保留三个市场事件。",
        "- 相似度边际采用平方权重，并叠加同行业、同市场状态和时间衰减权重。",
        "- 1/20/60 日使用各自已经成熟的历史样本池；概率校准使用全池当时已经兑现的旧预测，便于新股票直接复用。",
        "- bullish/bearish 计入方向收益，observe 不计入覆盖率；收益字段是每个可执行锚点的平均方向收益，不是重叠持有期复合净值。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = SimilarPatternConfig(
        candidate_step_days=5,
        candidate_start_date="2018-01-01",
        similarity_threshold=0.055,
        take_profit_3d=0.03,
        stop_loss_3d=0.03,
    )
    basic = load_stock_basic(PROJECT_ROOT / "data/raw/stock_basic.parquet")
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    market_regime = load_market_regime(PROJECT_ROOT / "data/raw/index_000300.SH.parquet")
    industries = {str(basic_map.get(symbol, {}).get("industry", "")) for symbol in args.targets}
    industry_regimes = {
        industry: build_industry_regime(PROJECT_ROOT / "data/raw/daily", basic, industry)
        for industry in industries
        if industry
    }
    queries, query_vectors, target_daily = _query_contexts(
        args.targets,
        args.start_date,
        args.end_date,
        args.anchor_step,
        config,
        basic,
        market_regime,
        industry_regimes,
    )
    print(f"prepared {len(queries)} point-in-time queries", flush=True)
    matches = _collect_matches(queries, query_vectors, config, args.max_symbols)
    records = _build_records(queries, matches, target_daily, market_regime, industry_regimes, config)
    records, calibrations = _apply_decisions(records, 20, config)
    summary = summarize_walk_forward_records(records, transaction_cost=config.transaction_cost)
    summary = _augment_summary(summary, records, target_daily, args.start_date)
    _write_outputs(args.output_dir, records, summary, calibrations, args, config)
    print(summary.to_string(index=False), flush=True)
    print(args.output_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
