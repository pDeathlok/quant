from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pandas as pd

from scripts.research import rebuild_strategy_signal_cache as signal_cache


def _write_market_partition(
    daily_dir: Path,
    month: str,
    rows: list[dict[str, object]],
) -> Path:
    path = (
        daily_dir.parent
        / f"{daily_dir.name}_partitioned"
        / f"year_month={month}"
        / "data.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _candidate(
    symbol: str,
    date: str,
    signal: str,
    value: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "date": [pd.Timestamp(date)],
            signal: [value],
        }
    )


def test_semantic_params_identity_includes_numeric_runtime_versions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(signal_cache.pd, "__version__", "pandas-test-a")
    first = signal_cache._semantic_params_fingerprint(
        rebuild_from=pd.Timestamp("2026-08-12"),
        start_date="2020-01-01",
        factor_mode="stateful",
    )
    monkeypatch.setattr(signal_cache.pd, "__version__", "pandas-test-b")
    second = signal_cache._semantic_params_fingerprint(
        rebuild_from=pd.Timestamp("2026-08-12"),
        start_date="2020-01-01",
        factor_mode="stateful",
    )
    assert first != second


def test_partition_identity_reuses_unchanged_hash_and_detects_late_correction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daily_dir = tmp_path / "raw" / "daily"
    partition = _write_market_partition(
        daily_dir,
        "202608",
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260811",
                "close": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "trade_date": "20260812",
                "close": 10.5,
            },
        ],
    )
    first = signal_cache._partitioned_source_identity(
        daily_dir,
        history_start=pd.Timestamp("2026-01-01"),
        processed_through=pd.Timestamp("2026-08-12"),
    )

    def unexpected_hash(_: Path) -> str:
        raise AssertionError("unchanged partition should reuse its digest")

    monkeypatch.setattr(signal_cache, "_sha256_file", unexpected_hash)
    repeated = signal_cache._partitioned_source_identity(
        daily_dir,
        history_start=pd.Timestamp("2026-01-01"),
        processed_through=pd.Timestamp("2026-08-12"),
        previous=first,
    )
    assert repeated["fingerprint"] == first["fingerprint"]

    monkeypatch.undo()
    corrected = pd.read_parquet(partition)
    corrected.loc[corrected["trade_date"].eq("20260811"), "close"] = 9.75
    corrected.to_parquet(partition, index=False)
    # Some filesystems have coarse timestamp resolution. Force a stat change
    # so the test exercises correction invalidation deterministically.
    stat = partition.stat()
    os.utime(partition, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    after_correction = signal_cache._partitioned_source_identity(
        daily_dir,
        history_start=pd.Timestamp("2026-01-01"),
        processed_through=pd.Timestamp("2026-08-12"),
        previous=first,
    )
    assert after_correction["fingerprint"] != first["fingerprint"]


def test_sql_frame_identity_is_order_independent_and_detects_late_correction() -> None:
    market = pd.DataFrame(
        {
            "ts_code": ["000002.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["20260812", "20260812", "20260811"],
            "name": ["乙", "甲", "甲"],
            "open": [8.0, 10.0, 9.8],
            "high": [8.2, 10.3, 10.0],
            "low": [7.9, 9.9, 9.7],
            "close": [8.1, 10.2, 9.9],
            "vol": [100.0, 200.0, 180.0],
        }
    )
    first = signal_cache._market_frame_source_identity(
        market,
        history_start=pd.Timestamp("2026-01-01"),
        processed_through=pd.Timestamp("2026-08-12"),
    )
    reordered = signal_cache._market_frame_source_identity(
        market.iloc[::-1].reset_index(drop=True),
        history_start=pd.Timestamp("2026-01-01"),
        processed_through=pd.Timestamp("2026-08-12"),
    )
    assert reordered["fingerprint"] == first["fingerprint"]

    corrected = market.copy()
    corrected.loc[corrected["trade_date"].eq("20260811"), "close"] = 9.7
    after_correction = signal_cache._market_frame_source_identity(
        corrected,
        history_start=pd.Timestamp("2026-01-01"),
        processed_through=pd.Timestamp("2026-08-12"),
    )
    assert after_correction["fingerprint"] != first["fingerprint"]


def test_exact_identity_fast_path_reuses_outputs_and_rejects_output_change(
    tmp_path: Path,
) -> None:
    outputs = {
        "family": tmp_path / "family.parquet",
        "extended": tmp_path / "extended.parquet",
        "b1_gate": tmp_path / "gate.parquet",
    }
    empty = pd.DataFrame(columns=["symbol", "date", "signal"])
    for path in outputs.values():
        empty.to_parquet(path, index=False)
    output_identities = signal_cache._output_identities(outputs)
    assert output_identities is not None
    source = {"fingerprint": "source-v1", "partitions": {}}
    manifest = {
        "incremental_start_date": "2026-08-12",
        "processed_through_date": "2026-08-12",
        "source_symbol_count": 5_557,
        "candidate_rows": 0,
        "candidate_symbols": 0,
        "factor_mode": "stateful",
        "family": {
            "rows": 0,
            "latest_date": None,
            "latest_rows": 0,
            "latest_hits": {},
        },
        "extended": {
            "rows": 0,
            "latest_date": None,
            "latest_rows": 0,
            "latest_hits": {},
        },
        "cache_identity": {
            "schema_version": signal_cache.CACHE_IDENTITY_SCHEMA_VERSION,
            "source_fingerprint": "source-v1",
            "contract_fingerprint": "contract-v1",
            "params_fingerprint": "params-v1",
            "outputs": output_identities,
        },
    }

    reused = signal_cache._fast_path_result(
        manifest=manifest,
        source_identity=source,
        contract_fingerprint="contract-v1",
        params_fingerprint="params-v1",
        output_paths=outputs,
    )
    assert reused is not None
    assert reused["execution_mode"] == "input_contract_cache_hit"
    assert reused["checkpoint_reused"] is True
    assert reused["b1_gate_rows"] == 0

    _candidate("000001.SZ", "2026-08-12", "signal").to_parquet(
        outputs["family"],
        index=False,
    )
    rejected = signal_cache._fast_path_result(
        manifest=manifest,
        source_identity=source,
        contract_fingerprint="contract-v1",
        params_fingerprint="params-v1",
        output_paths=outputs,
    )
    assert rejected is None


def test_batched_incremental_output_matches_unbatched_full_golden(
    monkeypatch,
) -> None:
    rebuild_from = pd.Timestamp("2026-08-12")
    tasks = [
        ("000001.SZ", pd.DataFrame()),
        ("000002.SZ", pd.DataFrame()),
        ("000003.SZ", pd.DataFrame()),
    ]

    def fake_process(
        symbol: str,
        _: pd.DataFrame,
        __: str,
        ___: str,
        ____: Path,
    ) -> dict[str, object]:
        if symbol == "000003.SZ":
            family = None
            extended = None
        else:
            family = pd.concat(
                [
                    _candidate(symbol, "2026-08-11", "family_signal"),
                    _candidate(symbol, "2026-08-12", "family_signal"),
                ],
                ignore_index=True,
            )
            extended = pd.concat(
                [
                    _candidate(symbol, "2026-08-11", "z_signal"),
                    _candidate(symbol, "2026-08-12", "z_signal"),
                ],
                ignore_index=True,
            )
        return {
            "symbol": symbol,
            "family": family,
            "extended": extended,
            "b1_gate": pd.DataFrame(),
            "errors": [],
            "factor_cache_mode": "golden",
        }

    monkeypatch.setattr(signal_cache, "_process_symbol", fake_process)
    full_results = signal_cache._process_symbol_batch(
        tasks,
        "2026-08-12",
        "stateful",
        Path("unused"),
    )
    batched_results = [
        item
        for batch in signal_cache._task_batches(tasks, 2)
        for item in signal_cache._process_symbol_batch(
            batch,
            "2026-08-12",
            "stateful",
            Path("unused"),
        )
    ]
    full_family = signal_cache._merge_incremental_cache(
        None,
        [item["family"] for item in full_results if item["family"] is not None],
        rebuild_from,
        empty_columns={"family_signal"},
    )
    batched_family = signal_cache._merge_incremental_cache(
        None,
        [
            item["family"]
            for item in batched_results
            if item["family"] is not None
        ],
        rebuild_from,
        empty_columns={"family_signal"},
    )
    pd.testing.assert_frame_equal(full_family, batched_family)
    assert set(full_family["date"]) == {rebuild_from}
    assert len(signal_cache._task_batches(tasks, 2)) == 2
    assert len(signal_cache._task_batches(tasks, 1)) == 3


def test_incremental_merge_handles_late_correction_repeat_and_empty_candidates() -> None:
    cached = pd.concat(
        [
            _candidate("000001.SZ", "2026-08-11", "signal"),
            _candidate("000001.SZ", "2026-08-12", "signal", False),
        ],
        ignore_index=True,
    )
    corrected = _candidate("000001.SZ", "2026-08-12", "signal", True)
    first = signal_cache._merge_incremental_cache(
        cached,
        [corrected],
        pd.Timestamp("2026-08-12"),
        empty_columns={"signal"},
    )
    repeated = signal_cache._merge_incremental_cache(
        first,
        [corrected],
        pd.Timestamp("2026-08-12"),
        empty_columns={"signal"},
    )
    pd.testing.assert_frame_equal(first, repeated)
    assert bool(first.loc[first["date"].eq("2026-08-12"), "signal"].iloc[0])

    legally_empty = signal_cache._merge_incremental_cache(
        None,
        [],
        pd.Timestamp("2026-08-12"),
        empty_columns={"signal"},
    )
    assert legally_empty.empty
    assert legally_empty.columns.tolist() == ["symbol", "date", "signal"]


def test_publish_lock_serializes_concurrent_publishers(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    entered: list[str] = []
    first_inside = threading.Event()

    def worker(name: str, hold: float) -> None:
        with signal_cache._publish_lock(manifest):
            entered.append(name)
            if name == "first":
                first_inside.set()
            time.sleep(hold)

    first = threading.Thread(target=worker, args=("first", 0.08))
    second = threading.Thread(target=worker, args=("second", 0.0))
    first.start()
    assert first_inside.wait(timeout=1)
    second.start()
    time.sleep(0.02)
    assert entered == ["first"]
    first.join(timeout=1)
    second.join(timeout=1)
    assert entered == ["first", "second"]


def test_publish_cache_set_rolls_back_every_output_on_replace_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_paths = {
        "family": tmp_path / "family.parquet",
        "extended": tmp_path / "extended.parquet",
        "b1_gate": tmp_path / "gate.parquet",
    }
    old = _candidate("OLD", "2026-08-11", "signal")
    new = _candidate("NEW", "2026-08-12", "signal")
    for path in output_paths.values():
        old.to_parquet(path, index=False)
    manifest_path = tmp_path / "manifest.json"
    signal_cache.atomic_write_json({"generation": "old"}, manifest_path)
    before = {
        key: signal_cache._sha256_file(path)
        for key, path in {**output_paths, "manifest": manifest_path}.items()
    }
    real_replace = signal_cache.os.replace
    replacement_count = 0

    def failing_replace(source: Path | str, target: Path | str) -> None:
        nonlocal replacement_count
        source_path = Path(source)
        if source_path.name.endswith(".stage"):
            replacement_count += 1
            if replacement_count == 2:
                raise OSError("injected publish failure")
        real_replace(source, target)

    monkeypatch.setattr(signal_cache.os, "replace", failing_replace)
    manifest = {
        "generation": "new",
        "cache_identity": {"outputs": {}},
    }
    try:
        signal_cache._publish_cache_set(
            {key: new for key in output_paths},
            output_paths,
            manifest,
            manifest_path,
        )
    except OSError as exc:
        assert "injected publish failure" in str(exc)
    else:
        raise AssertionError("expected injected publish failure")
    after = {
        key: signal_cache._sha256_file(path)
        for key, path in {**output_paths, "manifest": manifest_path}.items()
    }
    assert after == before


def test_source_change_check_raises_before_publish() -> None:
    before = {"fingerprint": "source-before"}
    after = {"fingerprint": "source-after"}
    try:
        signal_cache._assert_source_unchanged(
            before,
            after,
            phase="during test",
        )
    except RuntimeError as exc:
        assert "source changed during test" in str(exc)
    else:
        raise AssertionError("late source correction must abort publication")
