#!/usr/bin/env python
"""Run similar-pattern research for selected A-share stocks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.research.similar_patterns import (  # noqa: E402
    SimilarPatternConfig,
    analyze_target,
    analyze_targets_by_threshold,
    build_candidate_library,
    build_vector_caches_parallel,
    load_daily_file,
    load_stock_basic,
    write_result,
)

DEFAULT_TARGETS = ["002594.SZ", "002788.SZ"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze historical similar price-volume patterns")
    parser.add_argument("--daily-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--stock-basic", type=Path, default=PROJECT_ROOT / "data/raw/stock_basic.parquet")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/similar_patterns")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD. Default: latest local row per target.")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="When set, keep every historical fragment with similarity >= threshold instead of Top K.",
    )
    parser.add_argument(
        "--candidate-start-date",
        default=None,
        help="YYYY-MM-DD. In threshold mode default is 2018-01-01; in Top K mode default scans recent samples.",
    )
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional cap for faster dry runs.")
    parser.add_argument("--candidate-step-days", type=int, default=None)
    parser.add_argument("--max-candidates-per-symbol", type=int, default=120)
    parser.add_argument(
        "--vector-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data/research/similar_patterns/vector_cache",
        help="Directory for per-stock vector matrix caches used by threshold mode.",
    )
    parser.add_argument("--cache-workers", type=int, default=1, help="Parallel workers for vector cache building.")
    parser.add_argument("--force-vector-cache", action="store_true", help="Rebuild vector caches even if present.")
    parser.add_argument("--take-profit-3d", type=float, default=0.03, help="3-day upside trigger, e.g. 0.03 for +3%.")
    parser.add_argument("--stop-loss-3d", type=float, default=0.03, help="3-day downside trigger, e.g. 0.03 for -3%.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold_mode = args.similarity_threshold is not None
    config = SimilarPatternConfig(
        top_k=args.top_k,
        candidate_step_days=args.candidate_step_days or (1 if threshold_mode else 5),
        max_candidates_per_symbol=args.max_candidates_per_symbol,
        candidate_start_date=args.candidate_start_date or ("2018-01-01" if threshold_mode else None),
        similarity_threshold=args.similarity_threshold,
        take_profit_3d=args.take_profit_3d,
        stop_loss_3d=args.stop_loss_3d,
    )
    basic = load_stock_basic(args.stock_basic)
    target_symbols = {symbol.upper() for symbol in args.targets}
    if threshold_mode:
        print(
            "threshold scanning historical pattern fragments "
            f"from {config.candidate_start_date}, threshold={config.similarity_threshold}...",
            flush=True,
        )
        cache_report = build_vector_caches_parallel(
            args.daily_dir,
            basic,
            config,
            args.vector_cache_dir,
            target_symbols=target_symbols,
            max_symbols=args.max_symbols,
            workers=max(1, args.cache_workers),
            force=args.force_vector_cache,
        )
        cache_status = cache_report["status"].value_counts().to_dict() if not cache_report.empty else {}
        print(f"vector cache status: {cache_status}", flush=True)
        results = analyze_targets_by_threshold(
            args.daily_dir,
            basic,
            config,
            target_symbols=args.targets,
            target_date=args.target_date,
            max_symbols=args.max_symbols,
            vector_cache_dir=args.vector_cache_dir,
        )
        manifest = []
        for symbol, result in results.items():
            paths = write_result(result, args.output_dir)
            manifest.append(
                {
                    "symbol": symbol,
                    "name": result.target.name,
                    "target_date": result.target.target_date.strftime("%Y-%m-%d"),
                    "report": str(paths["report"]),
                    "cases": str(paths["cases"]),
                    "forecast": str(paths["forecast"]),
                }
            )
            print(
                f"wrote {symbol} threshold report: {paths['report']} "
                f"matches={len(result.similar_cases):,}",
                flush=True,
            )
        if manifest:
            manifest_path = args.output_dir / "manifest.csv"
            pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")
            print(f"manifest: {manifest_path}", flush=True)
        return

    print("building historical pattern library...", flush=True)
    library = build_candidate_library(
        args.daily_dir,
        basic,
        config,
        target_symbols=target_symbols,
        max_symbols=args.max_symbols,
    )
    print(f"candidate library rows: {len(library):,}", flush=True)
    if library.empty:
        raise SystemExit("No candidate rows built. Check daily data directory.")

    manifest: list[dict[str, str]] = []
    for symbol in args.targets:
        path = args.daily_dir / f"{symbol}.parquet"
        daily = load_daily_file(path)
        if daily.empty:
            print(f"missing target daily data: {symbol}", flush=True)
            continue
        result = analyze_target(symbol, daily, library, config, basic, target_date=args.target_date)
        paths = write_result(result, args.output_dir)
        manifest.append(
            {
                "symbol": symbol,
                "name": result.target.name,
                "target_date": result.target.target_date.strftime("%Y-%m-%d"),
                "report": str(paths["report"]),
                "cases": str(paths["cases"]),
                "forecast": str(paths["forecast"]),
            }
        )
        print(f"wrote {symbol} report: {paths['report']}", flush=True)

    if manifest:
        manifest_path = args.output_dir / "manifest.csv"
        pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
