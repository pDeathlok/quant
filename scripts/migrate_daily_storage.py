#!/usr/bin/env python
"""Migrate legacy per-symbol daily files into the canonical SQL/month-partition store."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import normalize_tushare_market_daily


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _frame_hash(frame: pd.DataFrame) -> int:
    columns = sorted(frame.columns)
    normalized = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    return int(pd.util.hash_pandas_object(normalized, index=False).sum())


def _load_batch(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if "ts_code" not in frame.columns or frame["ts_code"].isna().all():
            frame["ts_code"] = path.stem
        frames.append(frame)
    return normalize_tushare_market_daily(pd.concat(frames, ignore_index=True, sort=False))


def migrate(
    legacy_dir: Path,
    batch_size: int,
    write_sql: bool,
    remove_legacy: bool,
    remove_legacy_sql: bool,
) -> dict[str, object]:
    legacy_paths = sorted(legacy_dir.glob("*.parquet"))
    if not legacy_paths:
        raise RuntimeError(f"no legacy symbol parquet files under {legacy_dir}")
    root = legacy_dir.parent
    canonical_root = root / f"{legacy_dir.name}_partitioned"
    if canonical_root.exists():
        raise RuntimeError(f"canonical partition root already exists: {canonical_root}")
    staging_root = root / f".{legacy_dir.name}_partitioned_migration"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=root))
    started = perf_counter()
    source_rows = 0
    source_hash = 0
    part_index = 0
    sql_rows = 0
    for offset in range(0, len(legacy_paths), batch_size):
        paths = legacy_paths[offset : offset + batch_size]
        batch = _load_batch(paths)
        source_rows += len(batch)
        source_hash = (source_hash + _frame_hash(batch)) % (2**64)
        batch["year_month"] = batch["trade_date"].astype(str).str[:6]
        for month, month_frame in batch.groupby("year_month", sort=False):
            month_dir = staging_root / f"year_month={month}"
            month_dir.mkdir(parents=True, exist_ok=True)
            month_frame.drop(columns=["year_month"]).to_parquet(
                month_dir / f"part-{part_index:06d}.parquet",
                index=False,
            )
            part_index += 1
        if write_sql:
            sql_rows += store._write_sql_batch(
                batch.drop(columns=["year_month"]),
                legacy_dir.name,
                "trade_date",
                update_existing=False,
            )
        print(
            f"migration load: {min(offset + len(paths), len(legacy_paths))}/{len(legacy_paths)} "
            f"files rows={source_rows}",
            flush=True,
        )

    canonical_rows = 0
    canonical_hash = 0
    months = sorted(staging_root.glob("year_month=*"))
    for index, month_dir in enumerate(months, start=1):
        parts = sorted(month_dir.glob("part-*.parquet"))
        month = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True, sort=False)
        month = (
            month.drop_duplicates(["ts_code", "trade_date"], keep="last")
            .sort_values(["trade_date", "ts_code"])
            .reset_index(drop=True)
        )
        canonical_rows += len(month)
        canonical_hash = (canonical_hash + _frame_hash(month)) % (2**64)
        month.to_parquet(month_dir / "data.parquet", index=False)
        for path in parts:
            path.unlink()
        if index % 12 == 0 or index == len(months):
            print(f"migration compact: {index}/{len(months)} months rows={canonical_rows}", flush=True)

    consistent = source_rows == canonical_rows and source_hash == canonical_hash
    if not consistent:
        raise RuntimeError(
            f"canonical verification failed: source_rows={source_rows} canonical_rows={canonical_rows} "
            f"source_hash={source_hash} canonical_hash={canonical_hash}"
        )
    sql_verified = not write_sql
    sql_total_rows = 0
    sql_unique_rows = 0
    if write_sql:
        from sqlalchemy import text

        engine = store._engine()
        try:
            with engine.connect() as conn:
                counts = conn.execute(
                    text(
                        "SELECT COUNT(*) AS total_rows, "
                        "COUNT(DISTINCT ts_code, trade_date) AS unique_rows FROM market_daily"
                    )
                ).mappings().one()
            sql_total_rows = int(counts["total_rows"])
            sql_unique_rows = int(counts["unique_rows"])
            sql_verified = sql_total_rows == canonical_rows and sql_unique_rows == canonical_rows
        finally:
            engine.dispose()
        if not sql_verified:
            raise RuntimeError(
                f"SQL verification failed: canonical_rows={canonical_rows} "
                f"sql_total_rows={sql_total_rows} sql_unique_rows={sql_unique_rows}"
            )
    os.replace(staging_root, canonical_root)

    removed_files = 0
    removed_bytes = 0
    if remove_legacy:
        for path in legacy_paths:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1

    removed_sql_tables = 0
    if remove_legacy_sql:
        if not write_sql or not sql_verified:
            raise RuntimeError("legacy SQL tables can only be removed after verified SQL migration")
        from sqlalchemy import text

        engine = store._engine()
        try:
            with engine.connect() as conn:
                table_names = conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema=DATABASE()")
                ).scalars().all()
            legacy_tables = sorted(
                name for name in table_names if re.fullmatch(r"daily_[0-9]{6}_(?:sz|sh|bj)", str(name))
            )
            for offset in range(0, len(legacy_tables), 100):
                names = legacy_tables[offset : offset + 100]
                statement = "DROP TABLE IF EXISTS " + ", ".join(f"`{name}`" for name in names)
                with engine.begin() as conn:
                    conn.exec_driver_sql(statement)
                removed_sql_tables += len(names)
        finally:
            engine.dispose()

    result = {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "legacy_dir": str(legacy_dir),
        "canonical_root": str(canonical_root),
        "legacy_files": len(legacy_paths),
        "months": len(months),
        "source_rows": int(source_rows),
        "canonical_rows": int(canonical_rows),
        "source_hash": int(source_hash),
        "canonical_hash": int(canonical_hash),
        "consistent": consistent,
        "sql_rows": int(sql_rows),
        "sql_total_rows": sql_total_rows,
        "sql_unique_rows": sql_unique_rows,
        "sql_verified": sql_verified,
        "removed_legacy_files": removed_files,
        "removed_legacy_sql_tables": removed_sql_tables,
        "reclaimed_legacy_bytes": removed_bytes,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    audit_path = root / "source_audit" / f"{datetime.now():%Y%m%d_%H%M%S}_daily_storage_migration.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["audit_path"] = str(audit_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, default=PROJECT_ROOT / "data/raw/daily")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--skip-sql", action="store_true")
    parser.add_argument("--remove-legacy", action="store_true")
    parser.add_argument("--remove-legacy-sql", action="store_true")
    args = parser.parse_args()
    _load_env(PROJECT_ROOT / ".env")
    result = migrate(
        legacy_dir=args.legacy_dir,
        batch_size=max(1, args.batch_size),
        write_sql=not args.skip_sql,
        remove_legacy=args.remove_legacy,
        remove_legacy_sql=args.remove_legacy_sql,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
