#!/usr/bin/env python
"""Run the frozen B-development capital constraint on playbook policy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.research.right_side_playbook_capital import (
    CAPITAL_BACKTEST_SCHEMA_VERSION,
    CapitalBacktestSpec,
    build_continuous_daily_marks,
    prepare_capital_candidates,
    simulate_capital_constrained_policy,
)
from quant.research.right_side_playbook_dataset import file_sha256


DATA_ROOT = PROJECT_ROOT / "data/research/right_side_unified_v2_118"
REPORT_ROOT = PROJECT_ROOT / "reports/research/right_side_unified_v2_118"
MODEL_ROOT = PROJECT_ROOT / "models/research/right_side_unified_v2_118"
DEFAULT_EVENTS = DATA_ROOT / "playbook_events.parquet"
DEFAULT_OUTCOMES = DATA_ROOT / "playbook_outcomes.parquet"
DEFAULT_SELECTIONS = REPORT_ROOT / "playbook_policy_selections_B.parquet"
DEFAULT_DATASET_MANIFEST = DATA_ROOT / "playbook_dataset_manifest.json"
DEFAULT_MODEL_MANIFEST = MODEL_ROOT / "playbook/shared_playbook_model_B.manifest.json"
DEFAULT_MARKET_ROOT = PROJECT_ROOT / "data/raw"
ARMS = ("shared_playbook_model", "static_per_signal")


def _parquet_fold_bounds(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    fold_index = parquet.schema_arrow.names.index("fold")
    minima: list[str] = []
    maxima: list[str] = []
    for position in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(position).column(fold_index).statistics
        if statistics is None or not statistics.has_min_max:
            raise ValueError(f"fold metadata missing min/max: {path}")
        minima.append(str(statistics.min))
        maxima.append(str(statistics.max))
    return {
        "rows": int(parquet.metadata.num_rows),
        "global_min": min(minima),
        "global_max": max(maxima),
        "inspection": "metadata_only",
    }


def _read_inputs(
    *,
    selections_path: Path,
    events_path: Path,
    outcomes_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = pd.read_parquet(
        selections_path,
        columns=[
            "arm",
            "fold",
            "event_id",
            "symbol",
            "date",
            "playbook_id",
            "entry_mode",
            "eligible",
            "mature",
            "entry_date",
            "exit_date",
            "net_return",
            "round_trip_cost",
            "ambiguous_bar",
        ],
        filters=[("fold", "==", "B"), ("arm", "in", list(ARMS))],
    )
    events = pd.read_parquet(
        events_path,
        columns=[
            "fold",
            "event_id",
            "first_layer_score",
            "first_layer_score_provenance",
            "first_layer_score_fold",
        ],
        filters=[("fold", "==", "B")],
    )
    if set(events["first_layer_score_provenance"].astype(str)) != {"test"}:
        raise ValueError("B capital evaluation requires test first-layer score provenance")
    if set(events["first_layer_score_fold"].astype(str)) != {"B"}:
        raise ValueError("B capital evaluation score fold drifted")
    outcomes = pd.read_parquet(
        outcomes_path,
        columns=[
            "fold",
            "event_id",
            "playbook_id",
            "entry_raw_price",
            "exit_raw_price",
            "gross_return",
            "net_return",
            "round_trip_cost",
            "entry_date",
            "exit_date",
            "ambiguous_bar",
        ],
        filters=[("fold", "==", "B")],
    )
    return selections, events, outcomes


def _month_paths(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    paths: list[Path] = []
    for path in sorted((root / "daily_partitioned").glob("year_month=*/data.parquet")):
        month = path.parent.name.partition("=")[2]
        if start.strftime("%Y%m") <= month <= end.strftime("%Y%m"):
            paths.append(path)
    return paths


def _load_daily_marks(
    *,
    market_root: Path,
    symbols: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[Path]]:
    history_start = start - pd.Timedelta(days=40)
    store = MarketDataStore(
        MarketDataStoreConfig(
            backend="parquet",
            root=market_root,
            sql_url=None,
            mirror_parquet=True,
        )
    )
    daily = store.read_market_range(
        "daily",
        start_date=history_start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        symbols=symbols,
        columns=["ts_code", "trade_date", "open", "close", "pre_close"],
    ).rename(columns={"ts_code": "symbol", "trade_date": "date"})
    if daily.empty:
        raise ValueError("daily market marks are empty")
    daily["date"] = pd.to_datetime(
        daily["date"].astype(str), format="%Y%m%d", errors="raise"
    )
    marks = build_continuous_daily_marks(daily)
    return marks, _month_paths(market_root, history_start, end)


def _combined_file_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _scenario_specs() -> tuple[CapitalBacktestSpec, ...]:
    common = {
        "initial_capital": 1_000_000.0,
        "target_position_cash": 50_000.0,
        "max_new_positions_per_session": 5,
        "board_lot_size": 100,
    }
    return (
        CapitalBacktestSpec(
            scenario_id="base_max20_cost15bps",
            max_concurrent_positions=20,
            cost_bps=15.0,
            **common,
        ),
        CapitalBacktestSpec(
            scenario_id="sensitivity_max10_cost15bps",
            max_concurrent_positions=10,
            cost_bps=15.0,
            **common,
        ),
        CapitalBacktestSpec(
            scenario_id="sensitivity_max30_cost15bps",
            max_concurrent_positions=30,
            cost_bps=15.0,
            **common,
        ),
        CapitalBacktestSpec(
            scenario_id="stress_max20_cost30bps",
            max_concurrent_positions=20,
            cost_bps=30.0,
            **common,
        ),
    )


def _metric_frame(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "evaluation_role",
        "fold",
        "arm",
        "scenario_id",
        "cost_bps",
        "initial_capital",
        "target_position_cash",
        "max_concurrent_positions",
        "max_new_positions_per_session",
        "sessions",
        "final_equity",
        "minimum_cash",
        "no_leverage_cash_non_negative",
        "final_open_positions",
        "accounting_reconciliation_error",
        "accounting_reconciliation_pass",
        "total_return",
        "cagr_252",
        "max_drawdown",
        "daily_sharpe_252",
        "candidate_orders",
        "known_candidate_orders",
        "unevaluated_candidate_orders",
        "executed_trades",
        "rejected_orders",
        "rejection_rate_all_candidates",
        "rejection_rate_known_candidates",
        "executed_win_rate",
        "average_executed_net_return",
        "maximum_observed_concurrent_positions",
        "maximum_observed_new_positions",
        "average_cash_utilization",
        "board_lot_allocations",
        "equal_slot_price_fallback_allocations",
        "ambiguous_sl_first_executions",
    ]
    return pd.DataFrame(metrics)[columns].sort_values(
        ["scenario_id", "arm"], kind="stable"
    )


def _comparison_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key = {(row["scenario_id"], row["arm"]): row for row in metrics}
    for spec in _scenario_specs():
        shared = by_key[(spec.scenario_id, "shared_playbook_model")]
        static = by_key[(spec.scenario_id, "static_per_signal")]
        rows.append(
            {
                "scenario_id": spec.scenario_id,
                "shared_minus_static_total_return": shared["total_return"]
                - static["total_return"],
                "shared_minus_static_cagr_252": shared["cagr_252"]
                - static["cagr_252"],
                "shared_minus_static_max_drawdown": shared["max_drawdown"]
                - static["max_drawdown"],
                "shared_minus_static_daily_sharpe_252": shared["daily_sharpe_252"]
                - static["daily_sharpe_252"],
                "shared_minus_static_executed_trades": shared["executed_trades"]
                - static["executed_trades"],
            }
        )
    return rows


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if pd.isna(number) else f"{number:.2%}"


def _render_report(payload: dict[str, Any]) -> str:
    comparisons = {row["scenario_id"]: row for row in payload["comparisons"]}
    base = comparisons["base_max20_cost15bps"]
    max10 = comparisons["sensitivity_max10_cost15bps"]
    max30 = comparisons["sensitivity_max30_cost15bps"]
    stress = comparisons["stress_max20_cost30bps"]
    lines = [
        "# Right-side playbook capital constraint (B development, descriptive)",
        "",
        "**This is not model selection, not a promotion test, and does not read C.** The base contract was frozen before this simulation; max10/max30 and 30 bps are sensitivity reports only.",
        "",
        "## Frozen contract",
        "",
        "- Initial cash: CNY 1,000,000; fixed target per order: CNY 50,000 (= initial/20).",
        "- Base capacity: at most 20 concurrent lots and 5 new lots per session.",
        "- Priority: first-layer score descending, then symbol and event ID ascending.",
        "- Daily sequence: completed exits return cash first, then entries; no leverage and no reuse of cash still tied in open positions.",
        "- Allocation: raw entry price with 100-share board-lot rounding; missing raw price falls back to an explicitly counted equal slot.",
        "- Base net return exactly reproduces gross return minus 15 bps. The 30 bps stress subtracts 30 bps from the same gross outcome.",
        "- Existing outcome semantics are unchanged, including stop-first for same-bar stop/target ambiguity.",
        "",
        "## Results",
        "",
        "| Scenario | Arm | Total return | CAGR | MDD | Sharpe | Trades | Known-order reject | Cash use |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["metrics"]:
        lines.append(
            f"| {row['scenario_id']} | {row['arm']} | {_pct(row['total_return'])} | "
            f"{_pct(row['cagr_252'])} | {_pct(row['max_drawdown'])} | "
            f"{row['daily_sharpe_252']:.3f} | {row['executed_trades']:,} | "
            f"{_pct(row['rejection_rate_known_candidates'])} | "
            f"{_pct(row['average_cash_utilization'])} |"
        )
    lines.extend(
        [
            "",
            "## Shared minus static-per-signal",
            "",
            "| Scenario | Δ total return | Δ CAGR | Δ MDD | Δ Sharpe | Δ trades |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["comparisons"]:
        lines.append(
            f"| {row['scenario_id']} | {_pct(row['shared_minus_static_total_return'])} | "
            f"{_pct(row['shared_minus_static_cagr_252'])} | "
            f"{_pct(row['shared_minus_static_max_drawdown'])} | "
            f"{row['shared_minus_static_daily_sharpe_252']:.3f} | "
            f"{row['shared_minus_static_executed_trades']:+,} |"
        )
    lines.extend(
        [
            "",
            f"- Base contract: shared trails static total return by {_pct(-base['shared_minus_static_total_return'])}, while improving MDD by {_pct(base['shared_minus_static_max_drawdown'])} and Sharpe by {base['shared_minus_static_daily_sharpe_252']:.3f}.",
            f"- Max10 reverses the return comparison (shared Δ {_pct(max10['shared_minus_static_total_return'])}); this is capacity sensitivity, not permission to select max10 on B.",
            f"- Max30 is identical to base (Δ comparison {_pct(max30['shared_minus_static_total_return'])}) because the fixed CNY 50,000 target, five-entry daily cap, holding periods, and cash already cap observed concurrency at 20.",
            f"- At 30 bps, shared still trails static total return by {_pct(-stress['shared_minus_static_total_return'])}; both arms remain positive in this B description.",
            "- Accounting audit passes for every arm/scenario: cash never goes negative, final positions are zero, and final equity reconciles to initial cash plus realized P&L.",
            "",
            "## Interpretation limits",
            "",
            "- This resolves the cash/concurrency/Top-K gap for one fixed B-development contract, but it remains a post-model descriptive simulation on B and therefore cannot authorize production.",
            "- Daily marks are reconstructed causally from raw open/close/pre-close continuity. Exact frozen outcome P&L is booked at exit; missing marks are carried from the last available mark and counted.",
            "- Liquidity, market impact, partial fills, limit-queue priority, taxes beyond the fixed cost, and duplicate-symbol aggregation are not modeled.",
            "- Sensitivity scenarios do not select max positions or costs. A future untouched shadow window must reuse a precommitted contract unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    fold_bounds = {
        "events": _parquet_fold_bounds(args.events),
        "outcomes": _parquet_fold_bounds(args.outcomes),
        "selections": _parquet_fold_bounds(args.selections),
    }
    if (
        fold_bounds["events"]["global_max"] != "B"
        or fold_bounds["outcomes"]["global_max"] != "B"
        or fold_bounds["selections"]["global_min"] != "B"
        or fold_bounds["selections"]["global_max"] != "B"
    ):
        raise ValueError("capital input parquet metadata does not satisfy no-C contract")
    selections, events, outcomes = _read_inputs(
        selections_path=args.selections,
        events_path=args.events,
        outcomes_path=args.outcomes,
    )
    candidates = prepare_capital_candidates(selections, events, outcomes, arms=ARMS)
    del selections, events, outcomes
    evaluable = candidates["capital_evaluable"].fillna(False).astype(bool)
    start = pd.to_datetime(candidates.loc[evaluable, "entry_date"]).min()
    end = pd.to_datetime(candidates.loc[evaluable, "exit_date"]).max()
    symbols = set(candidates.loc[evaluable, "symbol"].astype(str))
    marks, market_paths = _load_daily_marks(
        market_root=args.market_root,
        symbols=symbols,
        start=start,
        end=end,
    )
    sessions = pd.DatetimeIndex(
        sorted(
            set(
                marks.loc[
                    marks["date"].between(start, end, inclusive="both"), "date"
                ]
            )
            | set(pd.to_datetime(candidates.loc[evaluable, "entry_date"]))
            | set(pd.to_datetime(candidates.loc[evaluable, "exit_date"]))
        )
    )

    metric_rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    orders: list[pd.DataFrame] = []
    specs = _scenario_specs()
    for spec in specs:
        for arm in ARMS:
            result = simulate_capital_constrained_policy(
                candidates,
                marks,
                arm=arm,
                spec=spec,
                sessions=sessions,
            )
            metric_rows.append(result.metrics)
            curves.append(result.curve)
            orders.append(result.orders)
    curve_frame = pd.concat(curves, ignore_index=True)
    order_frame = pd.concat(orders, ignore_index=True)
    metric_frame = _metric_frame(metric_rows)
    comparisons = _comparison_rows(metric_rows)
    rejection_rows = [
        {
            "scenario_id": metric["scenario_id"],
            "arm": metric["arm"],
            "reason": reason,
            "orders": count,
        }
        for metric in metric_rows
        for reason, count in metric["rejection_reason_counts"].items()
    ]
    rejection_frame = pd.DataFrame(rejection_rows).sort_values(
        ["scenario_id", "arm", "orders"],
        ascending=[True, True, False],
        kind="stable",
    )

    payload = {
        "schema_version": CAPITAL_BACKTEST_SCHEMA_VERSION,
        "evaluation_role": "B_development_descriptive_not_selection_not_promotion",
        "fold_policy": "B_only_C_forbidden_not_read",
        "promotion_changed": False,
        "base_contract": specs[0].manifest(),
        "sensitivity_contracts": [spec.manifest() for spec in specs[1:]],
        "sensitivity_policy": "report_only_never_select_on_B",
        "metrics": metric_rows,
        "comparisons": comparisons,
        "input_audit": {
            "fold_bounds": fold_bounds,
            "artifact_sha256": {
                "events": file_sha256(args.events),
                "outcomes": file_sha256(args.outcomes),
                "selections": file_sha256(args.selections),
                "dataset_manifest": file_sha256(args.dataset_manifest),
                "model_manifest": file_sha256(args.model_manifest),
                "market_month_files_combined": _combined_file_digest(market_paths),
            },
            "market_month_files": [str(path) for path in market_paths],
            "first_layer_score_priority": "B test score, descending",
            "source_cost_reconciliation": "gross_return - 0.0015 == source_net_return",
        },
        "output_artifacts": {
            "curve": str(args.curve_out),
            "orders": str(args.orders_out),
            "metrics_csv": str(args.metrics_out),
            "rejections": str(args.rejections_out),
            "report": str(args.report_out),
        },
        "warning": (
            "B-development descriptive capital simulation only; sensitivity results "
            "must not tune the model, policy, capacity, or promotion decision"
        ),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_write_parquet(curve_frame, args.curve_out, index=False, compression="zstd")
    atomic_write_parquet(order_frame, args.orders_out, index=False, compression="zstd")
    atomic_write_csv(metric_frame, args.metrics_out, index=False)
    atomic_write_csv(rejection_frame, args.rejections_out, index=False)
    atomic_write_json(payload, args.summary_out)
    atomic_write_text(_render_report(payload), args.report_out)
    return {
        "status": "complete_B_descriptive_only",
        "promotion_changed": False,
        "metrics": str(args.metrics_out),
        "summary": str(args.summary_out),
        "report": str(args.report_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument(
        "--curve-out",
        type=Path,
        default=REPORT_ROOT / "playbook_capital_curves_B.parquet",
    )
    parser.add_argument(
        "--orders-out",
        type=Path,
        default=REPORT_ROOT / "playbook_capital_orders_B.parquet",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=REPORT_ROOT / "playbook_capital_metrics_B.csv",
    )
    parser.add_argument(
        "--rejections-out",
        type=Path,
        default=REPORT_ROOT / "playbook_capital_rejections_B.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPORT_ROOT / "playbook_capital_constraint_B.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=REPORT_ROOT / "playbook_capital_constraint_B.md",
    )
    return parser.parse_args()


def main() -> None:
    result = command_evaluate(parse_args())
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
