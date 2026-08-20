#!/usr/bin/env python
"""Build normalized A->B right-side playbook events and outcomes.

The command reads only the frozen first-layer A/B prediction slice.  Fold C is
not an accepted option and is never loaded.  Wide T-close factors are stored
once in ``playbook_events.parquet``; the nine action outcomes remain narrow in
``playbook_outcomes.parquet``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_json
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.research.right_side_playbook_dataset import (
    EVENT_FACTOR_COLUMNS,
    FirstLayerPredictionContract,
    OutcomeBuildAudit,
    PLAYBOOK_DEVELOPMENT_FOLDS,
    StreamingParquetWriter,
    audit_narrow_playbook_tables,
    audit_reusable_playbook_events,
    dataset_manifest_payload,
    load_first_layer_predictions,
    narrow_playbook_outcomes,
    stable_event_ids,
)
from quant.research.right_side_playbook_model import (
    FIRST_LAYER_FOLD_COLUMN,
    FIRST_LAYER_PROVENANCE_COLUMN,
    FIRST_LAYER_SCORE_COLUMN,
)
from quant.research.right_side_playbook_policy import build_playbook_outcomes


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/research/right_side_unified_v2_118"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/right_side_unified_v2_118"
DEFAULT_FACTOR_DATASET = DEFAULT_DATA_ROOT / "unified_right_side_dataset.parquet"
DEFAULT_PREDICTIONS = DEFAULT_REPORT_ROOT / "test_predictions.parquet"
DEFAULT_EVENTS = DEFAULT_DATA_ROOT / "playbook_events.parquet"
DEFAULT_OUTCOMES = DEFAULT_DATA_ROOT / "playbook_outcomes.parquet"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "playbook_dataset_manifest.json"
DEFAULT_AUDIT = DEFAULT_REPORT_ROOT / "playbook_dataset_audit.json"
DEFAULT_DAILY_ROOT = PROJECT_ROOT / "data/raw/daily_partitioned"
DEFAULT_TRADABILITY_ROOT = PROJECT_ROOT / "data/raw/tradability"


def _ensure_output_scope(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to replace existing versioned playbook outputs without --force: "
            f"{[str(path) for path in existing]}"
        )


def _date_filter_expression(dataset: Any, start: pd.Timestamp, end: pd.Timestamp) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    date_type = dataset.schema.field("date").type
    if pa.types.is_timestamp(date_type):
        lower = pa.scalar(start.to_pydatetime(), type=date_type)
        upper = pa.scalar(end.to_pydatetime(), type=date_type)
        return (ds.field("date") >= lower) & (ds.field("date") <= upper)
    if pa.types.is_date(date_type):
        lower = pa.scalar(start.date(), type=date_type)
        upper = pa.scalar(end.date(), type=date_type)
        return (ds.field("date") >= lower) & (ds.field("date") <= upper)
    if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
        lower = pc.cast(pa.scalar(start.strftime("%Y-%m-%d")), date_type)
        upper = pc.cast(pa.scalar(end.strftime("%Y-%m-%d")), date_type)
        return (ds.field("date") >= lower) & (ds.field("date") <= upper)
    raise TypeError(f"unsupported factor date type: {date_type}")


def _write_event_table(
    factor_dataset: Path,
    predictions: pd.DataFrame,
    output: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(factor_dataset), format="parquet")
    factor_columns = tuple(EVENT_FACTOR_COLUMNS)
    required = {"symbol", "date", *factor_columns}
    available = set(dataset.schema.names)
    missing = required - available
    if missing:
        raise ValueError(
            "118-factor event source is incomplete; missing columns: "
            f"{sorted(missing)}"
        )
    if len([column for column in factor_columns if column.startswith("rs_")]) != 118:
        raise ValueError("event build is not using the frozen 118-rule-factor contract")

    optional_schema = ["factor_schema_version"] if "factor_schema_version" in available else []
    scan_columns = ["symbol", "date", *optional_schema, *factor_columns]
    start = pd.Timestamp(predictions["date"].min()).normalize()
    end = pd.Timestamp(predictions["date"].max()).normalize()
    scanner = dataset.scanner(
        columns=scan_columns,
        filter=_date_filter_expression(dataset, start, end),
        batch_size=batch_size,
        use_threads=True,
    )
    lookup = predictions.set_index(["symbol", "date"])[
        [
            "fold",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        ]
    ]
    seen_event_ids: set[str] = set()
    fold_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    factor_schema_versions: set[str] = set()
    writer = StreamingParquetWriter(output)
    try:
        for record_batch in scanner.to_batches():
            factors = record_batch.to_pandas()
            factors["symbol"] = factors["symbol"].astype("string").str.strip()
            factors["date"] = pd.to_datetime(factors["date"], errors="coerce").dt.normalize()
            if factors["date"].isna().any():
                raise ValueError("factor dataset contains invalid event dates")
            keys = pd.MultiIndex.from_frame(factors[["symbol", "date"]])
            metadata = lookup.reindex(keys).reset_index(drop=True)
            matched = metadata["fold"].notna().to_numpy()
            if not matched.any():
                continue
            factors = factors.loc[matched].reset_index(drop=True)
            metadata = metadata.loc[matched].reset_index(drop=True)
            events = pd.concat([metadata, factors], axis=1)
            events["fold"] = events["fold"].astype("string")
            events[FIRST_LAYER_PROVENANCE_COLUMN] = events[
                FIRST_LAYER_PROVENANCE_COLUMN
            ].astype("string")
            events[FIRST_LAYER_FOLD_COLUMN] = events[FIRST_LAYER_FOLD_COLUMN].astype(
                "string"
            )
            events[FIRST_LAYER_SCORE_COLUMN] = pd.to_numeric(
                events[FIRST_LAYER_SCORE_COLUMN], errors="coerce"
            ).astype("float32")
            events["event_id"] = stable_event_ids(events["symbol"], events["date"])
            duplicate_batch = events["event_id"].duplicated()
            if duplicate_batch.any():
                raise ValueError("factor source contains duplicate event rows in one batch")
            current_ids = set(events["event_id"].astype(str))
            overlap = current_ids & seen_event_ids
            if overlap:
                raise ValueError(
                    "factor source contains duplicate event rows across batches: "
                    f"{sorted(overlap)[:3]}"
                )
            seen_event_ids.update(current_ids)
            if "factor_schema_version" in events:
                factor_schema_versions.update(
                    events["factor_schema_version"].dropna().astype(str).unique()
                )
            fold_counts.update(events["fold"].astype(str).value_counts().to_dict())
            provenance_counts.update(
                events[FIRST_LAYER_PROVENANCE_COLUMN]
                .astype(str)
                .value_counts()
                .to_dict()
            )
            ordered = [
                "fold",
                "event_id",
                "symbol",
                "date",
                FIRST_LAYER_SCORE_COLUMN,
                FIRST_LAYER_PROVENANCE_COLUMN,
                FIRST_LAYER_FOLD_COLUMN,
                *optional_schema,
                *factor_columns,
            ]
            writer.write(events[ordered])
        if writer.rows != len(predictions):
            missing_count = len(predictions) - writer.rows
            missing_samples = (
                predictions.loc[
                    ~stable_event_ids(predictions["symbol"], predictions["date"])
                    .astype(str)
                    .isin(seen_event_ids),
                    ["symbol", "date", "fold"],
                ]
                .head(3)
                .to_dict("records")
            )
            raise ValueError(
                f"first-layer/factor coverage is incomplete: missing={missing_count} "
                f"samples={missing_samples}"
            )
        if set(fold_counts) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
            raise ValueError("event output must contain A/B only")
        writer.close(commit=True)
    except Exception:
        writer.close(commit=False)
        raise
    return {
        "rows": int(writer.rows),
        "prediction_coverage": float(writer.rows / len(predictions)),
        "fold_rows": dict(sorted(fold_counts.items())),
        "first_layer_provenance_rows": dict(sorted(provenance_counts.items())),
        "date_min": start,
        "date_max": end,
        "project_factor_count": len(PROJECT_FACTOR_COLUMNS),
        "event_factor_count": len(factor_columns),
        "factor_schema_versions": sorted(factor_schema_versions),
    }


def _daily_partition_paths(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    months = pd.period_range(start=start, end=end, freq="M")
    paths = [root / f"year_month={month.strftime('%Y%m')}" / "data.parquet" for month in months]
    available = [path for path in paths if path.is_file()]
    if not available:
        raise FileNotFoundError(
            f"no daily partitions matched {start.date()} through {end.date()} under {root}"
        )
    return available


def _load_daily_market(
    root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: set[str],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    import pyarrow.parquet as pq

    required = {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close"}
    optional = {"up_limit", "is_suspended"}
    frames: list[pd.DataFrame] = []
    calendar_values: list[pd.Series] = []
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    for path in _daily_partition_paths(root, start, end):
        available = set(pq.ParquetFile(path).schema.names)
        missing = required - available
        if missing:
            raise ValueError(f"daily partition {path} missing columns: {sorted(missing)}")
        frame = pd.read_parquet(path, columns=sorted(required | (optional & available)))
        date_text = frame["trade_date"].astype(str)
        mask = date_text.between(start_text, end_text)
        frame = frame.loc[mask].copy()
        if frame.empty:
            continue
        calendar_values.append(frame["trade_date"].astype(str))
        frame = frame[frame["ts_code"].astype(str).isin(symbols)]
        if not frame.empty:
            frames.append(frame)
    if not frames or not calendar_values:
        raise ValueError("daily market has no rows for the playbook event window")
    market = pd.concat(frames, ignore_index=True, sort=False)
    market["ts_code"] = market["ts_code"].astype("string")
    market = market.sort_values(["ts_code", "trade_date"], kind="stable").drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            pd.concat(calendar_values, ignore_index=True).drop_duplicates(),
            format="%Y%m%d",
            errors="raise",
        )
    ).sort_values()
    return market.reset_index(drop=True), calendar


def _load_tradability(
    root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbols: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    for path in sorted(root.glob("*.parquet")):
        if start_text <= path.stem <= end_text:
            frame = pd.read_parquet(path)
            if "ts_code" in frame:
                frame = frame[frame["ts_code"].astype(str).isin(symbols)]
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    tradability = pd.concat(frames, ignore_index=True, sort=False)
    return tradability.sort_values(["ts_code", "trade_date"], kind="stable").drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )


def _process_symbol_outcomes(
    task: tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DatetimeIndex,
    ],
) -> tuple[pd.DataFrame, str | None]:
    symbol, signals, daily, tradability, calendar = task
    try:
        wide = build_playbook_outcomes(
            signals[["fold", "event_id", "symbol", "date"]],
            daily,
            calendar,
            tradability,
            event_id_column="event_id",
            signal_symbol_column="symbol",
            signal_date_column="date",
            daily_symbol_column="ts_code",
            daily_date_column="trade_date",
            tradability_symbol_column="ts_code",
            tradability_date_column="trade_date",
        )
        return narrow_playbook_outcomes(wide), None
    except Exception as exc:
        return pd.DataFrame(), f"{symbol}: {exc}"


def _write_outcome_table(
    events_path: Path,
    output: Path,
    *,
    daily_root: Path,
    tradability_root: Path,
    end_buffer_days: int,
    workers: int,
    max_pending_per_worker: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_pending_per_worker <= 0:
        raise ValueError("max_pending_per_worker must be positive")
    event_keys = pd.read_parquet(
        events_path,
        columns=["fold", "event_id", "symbol", "date"],
    )
    event_keys["fold"] = event_keys["fold"].astype("string")
    event_keys["event_id"] = event_keys["event_id"].astype("string")
    event_keys["symbol"] = event_keys["symbol"].astype("string")
    event_keys["date"] = pd.to_datetime(event_keys["date"], errors="coerce").dt.normalize()
    if event_keys["date"].isna().any() or event_keys.duplicated(["fold", "event_id"]).any():
        raise ValueError("event key table is invalid or duplicated")
    if set(event_keys["fold"].astype(str)) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
        raise ValueError("outcome build accepts A/B event keys only")
    start = pd.Timestamp(event_keys["date"].min())
    end = pd.Timestamp(event_keys["date"].max()) + pd.Timedelta(days=end_buffer_days)
    symbols = set(event_keys["symbol"].astype(str))
    market, calendar = _load_daily_market(
        daily_root,
        start=start,
        end=end,
        symbols=symbols,
    )
    tradability = _load_tradability(
        tradability_root,
        start=start,
        end=end,
        symbols=symbols,
    )
    by_market = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in market.groupby("ts_code", sort=False)
    }
    by_tradability = (
        {
            str(symbol): group.reset_index(drop=True)
            for symbol, group in tradability.groupby("ts_code", sort=False)
        }
        if not tradability.empty
        else {}
    )
    writer = StreamingParquetWriter(output)
    audit = OutcomeBuildAudit()
    grouped_events = event_keys.sort_values(["symbol", "date"], kind="stable").groupby(
        "symbol", sort=True
    )
    total_symbols = int(event_keys["symbol"].nunique())
    empty_daily = pd.DataFrame()

    def iter_tasks() -> Any:
        for symbol, signals in grouped_events:
            symbol_text = str(symbol)
            daily = by_market.get(symbol_text, empty_daily)
            if daily.empty:
                yield symbol_text, pd.DataFrame(), pd.DataFrame(), None, calendar
            else:
                yield (
                    symbol_text,
                    signals.reset_index(drop=True),
                    daily,
                    by_tradability.get(symbol_text),
                    calendar,
                )

    errors: list[str] = []
    completed = 0

    def consume(result: tuple[pd.DataFrame, str | None]) -> None:
        nonlocal completed
        narrow, error = result
        completed += 1
        if error is not None:
            errors.append(error)
        elif narrow.empty:
            errors.append("outcome worker returned an empty result")
        else:
            audit.update(narrow)
            writer.write(narrow)
        if completed % 250 == 0:
            print(
                f"playbook outcomes {completed:,}/{total_symbols:,} symbols; "
                f"rows={writer.rows:,}",
                flush=True,
            )

    try:
        tasks = iter(iter_tasks())
        if workers == 1:
            for task in tasks:
                consume(_process_symbol_outcomes(task))
        else:
            max_pending = max(workers, workers * max_pending_per_worker)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                pending = set()
                for _ in range(min(max_pending, total_symbols)):
                    try:
                        pending.add(executor.submit(_process_symbol_outcomes, next(tasks)))
                    except StopIteration:
                        break
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            consume(future.result())
                        except Exception as exc:
                            completed += 1
                            errors.append(f"outcome process failure: {exc}")
                        try:
                            pending.add(executor.submit(_process_symbol_outcomes, next(tasks)))
                        except StopIteration:
                            pass
        if errors:
            raise RuntimeError(
                f"playbook outcome build failed for {len(errors)} symbols; "
                f"first errors={errors[:5]}"
            )
        outcome_summary = audit.finalize(expected_events=len(event_keys))
        if writer.rows != outcome_summary["outcome_rows"]:
            raise ValueError("streaming writer row count differs from outcome audit")
        writer.close(commit=True)
    except Exception:
        writer.close(commit=False)
        raise
    tradability_source_rows = int(len(tradability))
    del market, tradability
    outcome_summary.update(
        {
            "date_min": start,
            "date_max": pd.Timestamp(event_keys["date"].max()),
            "daily_window_end": end,
            "market_calendar_sessions": int(len(calendar)),
            "tradability_source_rows": tradability_source_rows,
        }
    )
    return outcome_summary


def command_build(args: argparse.Namespace) -> dict[str, Any]:
    if args.reuse_events and not args.events_out.is_file():
        raise FileNotFoundError(
            f"--reuse-events requires an atomically published event file: {args.events_out}"
        )
    protected_outputs = [args.outcomes_out, args.manifest_out]
    if not args.reuse_events:
        protected_outputs.insert(0, args.events_out)
    _ensure_output_scope(
        protected_outputs,
        force=args.force,
    )
    contract = FirstLayerPredictionContract(
        score_column=args.score_column,
        entry_mode=args.entry_mode,
        horizon=args.horizon,
        label=args.label,
        selected_candidate=args.selected_candidate,
    )
    predictions = load_first_layer_predictions(args.predictions, contract=contract)
    if args.reuse_events:
        import pyarrow.parquet as pq

        available_event_columns = pq.ParquetFile(args.events_out).schema.names
        metadata_columns = [
            "fold",
            "event_id",
            "symbol",
            "date",
            FIRST_LAYER_SCORE_COLUMN,
            FIRST_LAYER_PROVENANCE_COLUMN,
            FIRST_LAYER_FOLD_COLUMN,
        ]
        if "factor_schema_version" in available_event_columns:
            metadata_columns.append("factor_schema_version")
        reusable_events = pd.read_parquet(
            args.events_out,
            columns=metadata_columns,
            filters=[("fold", "in", list(PLAYBOOK_DEVELOPMENT_FOLDS))],
        )
        event_summary = audit_reusable_playbook_events(
            reusable_events,
            predictions,
            events_path=args.events_out,
            predictions_path=args.predictions,
            factor_dataset_path=args.factor_dataset,
            available_event_columns=available_event_columns,
        )
        del reusable_events
    else:
        event_summary = _write_event_table(
            args.factor_dataset,
            predictions,
            args.events_out,
            batch_size=args.batch_size,
        )
    outcome_summary = _write_outcome_table(
        args.events_out,
        args.outcomes_out,
        daily_root=args.daily_root,
        tradability_root=args.tradability_root,
        end_buffer_days=args.end_buffer_days,
        workers=args.workers,
        max_pending_per_worker=args.max_pending_per_worker,
    )
    manifest = dataset_manifest_payload(
        events_path=args.events_out,
        outcomes_path=args.outcomes_out,
        predictions_path=args.predictions,
        factor_dataset_path=args.factor_dataset,
        prediction_contract=contract,
        event_summary=event_summary,
        outcome_summary=outcome_summary,
    )
    manifest["built_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["fold_policy"] = "A_train_to_B_test_only_C_forbidden"
    atomic_write_json(manifest, args.manifest_out)
    return {
        "events": str(args.events_out),
        "outcomes": str(args.outcomes_out),
        "manifest": str(args.manifest_out),
        "event_rows": event_summary["rows"],
        "outcome_rows": outcome_summary["outcome_rows"],
        "fold_rows": event_summary["fold_rows"],
    }


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    events = pd.read_parquet(
        args.events,
        filters=[("fold", "in", list(PLAYBOOK_DEVELOPMENT_FOLDS))],
    )
    outcomes = pd.read_parquet(
        args.outcomes,
        filters=[("fold", "in", list(PLAYBOOK_DEVELOPMENT_FOLDS))],
    )
    audit = audit_narrow_playbook_tables(events, outcomes)
    audit.update(
        {
            "events": str(args.events),
            "outcomes": str(args.outcomes),
            "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    atomic_write_json(audit, args.output)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--factor-dataset", type=Path, default=DEFAULT_FACTOR_DATASET)
    build.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    build.add_argument("--events-out", type=Path, default=DEFAULT_EVENTS)
    build.add_argument("--outcomes-out", type=Path, default=DEFAULT_OUTCOMES)
    build.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--daily-root", type=Path, default=DEFAULT_DAILY_ROOT)
    build.add_argument("--tradability-root", type=Path, default=DEFAULT_TRADABILITY_ROOT)
    build.add_argument("--score-column", default="pred_unified_long_task_deep")
    build.add_argument(
        "--selected-candidate",
        default=None,
        help=(
            "frozen first-layer candidate identifier; when supplied, the score "
            "column must be exactly pred_<selected-candidate>"
        ),
    )
    build.add_argument("--entry-mode", choices=["next_open", "next_close"], default="next_close")
    build.add_argument("--horizon", type=int, default=5)
    build.add_argument("--label", default="good_path5")
    build.add_argument("--batch-size", type=int, default=65_536)
    build.add_argument("--end-buffer-days", type=int, default=40)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--max-pending-per-worker", type=int, default=2)
    build.add_argument("--force", action="store_true")
    build.add_argument(
        "--reuse-events",
        action="store_true",
        help=(
            "resume outcomes from an existing atomically published event table "
            "after exact A/B key, row, score, provenance, schema, and hash checks"
        ),
    )

    audit = subparsers.add_parser("audit")
    audit.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    audit.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    audit.add_argument("--output", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = command_build(args)
    else:
        result = command_audit(args)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
