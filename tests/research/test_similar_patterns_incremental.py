from __future__ import annotations

import contextvars
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.research import similar_patterns as patterns


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.delenv("MARKET_DATA_ROOT", raising=False)


def history(symbol="A.SZ", rows=560):
    dates = pd.bdate_range("2010-01-04", periods=rows + 25).delete(slice(310, 335))[:rows]
    x = np.arange(rows)
    close = 20 + x * .025 + np.sin(x / 7) + .2 * np.cos(x / 3)
    raw_close = close.copy()
    raw_close[505:] /= 3  # An append can reanchor the entire adjusted history.
    return pd.DataFrame({
        "ts_code": symbol, "trade_date": dates.strftime("%Y%m%d"),
        "date": dates, "name": f"Name {symbol}",
        "open": raw_close * (.995 + .002 * np.sin(x)),
        "high": raw_close * 1.02, "low": raw_close * .98, "close": raw_close,
        "vol": np.where(x % 31 == 0, 0, 1000 + x * 2 + 100 * np.sin(x / 4)),
        "pct_chg": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
        "unused_wide_feature": "x" * 200,
    })


def write_legacy(tmp_path, frame):
    path = tmp_path / "daily" / f"{frame.ts_code.iloc[0]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def config(**kwargs):
    return replace(patterns.SimilarPatternConfig(candidate_step_days=1, candidate_start_date="2011-01-01"), **kwargs)


def assert_equivalent(actual, expected):
    assert actual["symbol"] == expected["symbol"]
    assert actual["name"] == expected["name"]
    assert actual["industry"] == expected["industry"]
    for field in ("indices", "dates", "close"):
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
    for field in ("vectors", *patterns._MATRIX_CACHE_FLOAT_FIELDS):
        # Reanchoring uses float64 division; the stored float32 ratios can differ
        # in their final bit, while all output fields and NaN maturities agree.
        np.testing.assert_allclose(actual[field], expected[field], rtol=2e-6, atol=2e-7,
                                   equal_nan=True, err_msg=field)


@pytest.mark.parametrize("append_rows", [1, 5, 21, 65])
def test_append_matches_full_vectors_and_maturing_labels(tmp_path, append_rows):
    frame = history(rows=580)
    path = write_legacy(tmp_path, frame.iloc[:500])
    cfg = config()
    first = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    old = patterns.load_stock_vector_cache(Path(first["cache_path"]))
    frame.iloc[:500 + append_rows].to_parquet(path, index=False)
    result = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    full = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "full", force=True)
    actual = patterns.load_stock_vector_cache(Path(result["cache_path"]))
    expected = patterns.load_stock_vector_cache(Path(full["cache_path"]))
    assert result["build_mode"] == "append"
    assert result["vectors_built"] == append_rows
    assert result["vectors_reused"] == len(old["indices"])
    np.testing.assert_array_equal(actual["vectors"][:len(old["indices"])], old["vectors"])
    assert np.isnan(old["fwd_60d"][-1])
    if append_rows >= 60:
        assert np.isfinite(actual["fwd_60d"][len(old["indices"]) - 1])
    assert_equivalent(actual, expected)
    # The full computation remains the vector oracle, independent of cache reuse.
    indices, matrix = patterns.build_stock_candidate_matrix(patterns.load_daily_file(path), cfg)
    np.testing.assert_array_equal(actual["indices"], indices)
    np.testing.assert_allclose(actual["vectors"], matrix, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("append_rows", [1, 5])
def test_rolling_candidate_phase_and_long_monthly_history_preserved(tmp_path, append_rows):
    frame = history(rows=1410)
    path = write_legacy(tmp_path, frame.iloc[:1400])
    cfg = config(candidate_start_date=None, max_candidates_per_symbol=4,
                 candidate_step_days=5, weekly_lookback=150, monthly_lookback=48)
    first = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    assert first["vectors"] > 0
    frame.iloc[:1400 + append_rows].to_parquet(path, index=False)
    appended = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    full = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "full", force=True)
    assert appended["vectors_reused"] == (0 if append_rows == 1 else 4)
    assert_equivalent(patterns.load_stock_vector_cache(Path(appended["cache_path"])),
                      patterns.load_stock_vector_cache(Path(full["cache_path"])))


def test_unchanged_rerun_builds_zero_and_metadata_is_part_of_identity(tmp_path, monkeypatch):
    path = write_legacy(tmp_path, history(rows=420))
    cfg = config()
    first = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    old_stat = Path(first["cache_path"]).stat()
    monkeypatch.setattr(patterns, "build_pattern_vector", lambda *a, **k: pytest.fail("unexpected vector build"))
    monkeypatch.setattr(patterns, "partitioned_daily_source_fingerprint", lambda *a: pytest.fail("global market scan"))
    second = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache", source_fingerprint="untrusted-global")
    assert second["status"] == "cache_hit"
    assert second["vectors_built"] == 0
    assert Path(first["cache_path"]).stat().st_mtime_ns == old_stat.st_mtime_ns
    renamed = patterns.build_stock_vector_cache(path, {"industry": "New industry"}, cfg, tmp_path / "cache")
    assert renamed["status"] == "built"
    assert renamed["vectors_built"] == 0
    assert patterns.load_stock_vector_cache(Path(renamed["cache_path"]))["industry"] == "New industry"
    assert patterns.vector_cache_key(cfg) != patterns.vector_cache_key(replace(cfg, max_candidates_per_symbol=1))


def test_one_symbol_correction_does_not_invalidate_partition_peers(tmp_path, monkeypatch):
    frame = pd.concat([history(symbol, 420) for symbol in ("A.SZ", "B.SZ", "C.SZ")], ignore_index=True)
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=tmp_path))
    store.write_market_batch(frame)
    cfg = config()
    args = (tmp_path / "daily", pd.DataFrame(), cfg, tmp_path / "cache")
    initial = patterns.build_vector_caches_parallel(*args).set_index("symbol")
    before = Path(initial.loc["B.SZ", "cache_path"]).read_bytes()
    correction = frame.copy()
    correction.loc[300, "high"] *= 1.1
    store.write_market_batch(correction)
    reads = []
    read_parquet = pd.read_parquet

    def projected_read(path, **kwargs):
        reads.append(kwargs)
        assert kwargs["columns"]
        assert "unused_wide_feature" not in kwargs["columns"]
        return read_parquet(path, **kwargs)

    monkeypatch.setattr(patterns.pd, "read_parquet", projected_read)
    # Listing is tested separately by the store; measure only projected vector reads.
    monkeypatch.setattr(patterns, "list_partitioned_symbol_paths", lambda _: [tmp_path / "daily" / f"{s}.parquet" for s in ("A.SZ", "B.SZ", "C.SZ")])
    result = patterns.build_vector_caches_parallel(*args).set_index("symbol")
    assert result.loc["A.SZ", "build_mode"] == "full"
    assert set(result.loc[["B.SZ", "C.SZ"], "status"]) == {"cache_hit"}
    assert Path(initial.loc["B.SZ", "cache_path"]).read_bytes() == before
    assert any(len(read["filters"][0][2]) > 1 for read in reads)
    oracle = patterns.build_stock_vector_cache(tmp_path / "daily/A.SZ.parquet", {}, cfg, tmp_path / "oracle", force=True)
    assert_equivalent(patterns.load_stock_vector_cache(Path(result.loc["A.SZ", "cache_path"])),
                      patterns.load_stock_vector_cache(Path(oracle["cache_path"])))


@pytest.mark.parametrize("change", ["delete", "same_stat", "aliases", "too_short"])
def test_corrections_deletions_aliases_and_shortening_match_full(tmp_path, change):
    frame = history(rows=420)
    path = write_legacy(tmp_path, frame)
    cfg = config()
    initial = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    stat = path.stat()
    if change == "delete":
        frame = frame.drop(index=280)
    elif change == "same_stat":
        frame.loc[300, "pct_chg"] += .1
    elif change == "aliases":
        frame = frame.rename(columns={"vol": "volume", "pct_chg": "pct_change"})
        frame.loc[300, "date"] += pd.Timedelta(days=1)
    else:
        frame = frame.iloc[:100]
    frame.to_parquet(path, index=False)
    if change == "same_stat":
        import os
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    if change == "too_short":
        assert result["status"] == "too_short"
        assert not Path(initial["cache_path"]).exists()
    else:
        oracle = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "full", force=True)
        assert result["build_mode"] == "full"
        assert_equivalent(patterns.load_stock_vector_cache(Path(result["cache_path"])),
                          patterns.load_stock_vector_cache(Path(oracle["cache_path"])))


@pytest.mark.parametrize("phase", ["listing", "read", "worker", "publication"])
def test_midrun_source_changes_preserve_all_live_caches(tmp_path, monkeypatch, phase):
    frame = history(rows=425)
    paths = [write_legacy(tmp_path, frame.iloc[:420].assign(ts_code=s)) for s in ("A.SZ", "B.SZ")]
    cfg = config()
    args = (tmp_path / "daily", pd.DataFrame(), cfg, tmp_path / "cache")
    records = patterns.build_vector_caches_parallel(*args)
    original = {Path(row.cache_path): Path(row.cache_path).read_bytes() for row in records.itertuples()}
    for path in paths:
        frame.assign(ts_code=path.stem).to_parquet(path, index=False)
    fired = False

    def mutate():
        nonlocal fired
        if not fired:
            fired = True
            frame.assign(close=frame.close * 1.1, ts_code="A.SZ").to_parquet(paths[0], index=False)

    if phase == "listing":
        listing = patterns.list_partitioned_symbol_paths
        def changing_listing(directory):
            value = listing(directory)
            mutate()
            return value
        monkeypatch.setattr(patterns, "list_partitioned_symbol_paths", changing_listing)
    elif phase == "read":
        read = patterns._VectorSource.read
        def changing_read(self, batch):
            value = read(self, batch)
            mutate()
            return value
        monkeypatch.setattr(patterns._VectorSource, "read", changing_read)
    elif phase == "worker":
        save = patterns.save_stock_vector_cache
        def changing_save(*args, **kwargs):
            save(*args, **kwargs)
            mutate()
        monkeypatch.setattr(patterns, "save_stock_vector_cache", changing_save)
    else:
        rename = Path.replace
        def changing_replace(self, destination):
            value = rename(self, destination)
            if self.suffix == ".npz" and ".vectors-building-" in str(self):
                mutate()
            return value
        monkeypatch.setattr(Path, "replace", changing_replace)
    with pytest.raises(patterns.VectorSourceChangedError):
        patterns.build_vector_caches_parallel(*args)
    assert fired
    for path, content in original.items():
        assert path.read_bytes() == content
    assert not list((tmp_path / "cache").rglob(".vectors-building-*"))


def test_thread_workers_keep_context_and_bounded_batches(tmp_path, monkeypatch):
    paths = [write_legacy(tmp_path, history(f"S{n}.SZ", 280)) for n in range(9)]
    context = contextvars.ContextVar("vector_test_context", default="missing")
    context.set("pinned-source")
    seen = []
    batch_sizes = []
    worker = patterns._build_stock_vector_cache_worker
    read = patterns._VectorSource.read

    def context_worker(task):
        seen.append(context.get())
        return worker(task)

    def batch_read(self, batch):
        batch_sizes.append(len(batch))
        return read(self, batch)

    monkeypatch.setattr(patterns, "_should_use_thread_pool_for_vector_cache", lambda: True)
    monkeypatch.setattr(patterns, "_build_stock_vector_cache_worker", context_worker)
    monkeypatch.setattr(patterns._VectorSource, "read", batch_read)
    result = patterns.build_vector_caches_parallel(tmp_path / "daily", pd.DataFrame(), config(), tmp_path / "cache", workers=2)
    assert len(result) == len(paths)
    assert seen == ["pinned-source"] * len(paths)
    assert max(batch_sizes) <= 4


def test_spawn_workers_build_from_parent_source_frames(tmp_path):
    for symbol in ("A.SZ", "B.SZ"):
        write_legacy(tmp_path, history(symbol, 300))
    result = patterns.build_vector_caches_parallel(
        tmp_path / "daily", pd.DataFrame(), config(), tmp_path / "cache", workers=2,
    )
    assert len(result) == 2
    assert set(result.status) == {"built"}
    assert result.vectors_built.sum() > 0


def test_partition_batch_preserves_store_ordering_without_supplied_returns(tmp_path):
    frame = history(rows=420).drop(columns=["pct_chg"]).iloc[::-1]
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=tmp_path))
    store.write_market_batch(frame)
    path = tmp_path / "daily/A.SZ.parquet"
    source = patterns._VectorSource(path.parent)
    projected = patterns.normalize_daily_frame(source.read([path])[path.stem], path.stem)
    canonical = patterns.load_daily_file(path)
    pd.testing.assert_frame_equal(projected, canonical)
    cfg = config()
    result = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    cached = patterns.load_stock_vector_cache(Path(result["cache_path"]))
    indices, matrix = patterns.build_stock_candidate_matrix(canonical, cfg)
    np.testing.assert_array_equal(cached["indices"], indices)
    np.testing.assert_array_equal(cached["vectors"], matrix)


def test_sql_revision_and_unjournaled_content_races_fail_closed(tmp_path, monkeypatch):
    state = {"frame": history(rows=420), "revision": 1, "reads": 0, "change": False}

    class SqlStore:
        def __init__(self, config):
            self.config = config
        def _engine(self):
            raise RuntimeError("no schema inspection in synthetic store")
        def dataset_revision(self, dataset):
            return state["revision"]
        def _read_sql_range(self, dataset, *, symbols=None, columns=None):
            assert symbols == ["A.SZ"]
            assert columns and "unused_wide_feature" not in columns
            state["reads"] += 1
            return state["frame"].copy()

    monkeypatch.setenv("MARKET_DATA_BACKEND", "sql")
    monkeypatch.setenv("MARKET_DATA_SQL_URL", "sqlite://synthetic-never-connected")
    monkeypatch.setattr(patterns, "MarketDataStore", SqlStore)
    monkeypatch.setattr(patterns, "_sql_vector_columns", lambda *args: list(patterns._VECTOR_SOURCE_COLUMNS))
    cfg = config()
    path = tmp_path / "daily/A.SZ.parquet"
    initial = patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    assert initial["status"] == "built"
    before = Path(initial["cache_path"]).read_bytes()
    save = patterns.save_stock_vector_cache

    def changing_save(*args, **kwargs):
        save(*args, **kwargs)
        if state["change"] == "revision":
            state["revision"] += 1
        else:
            state["frame"].loc[300, "high"] *= 1.1

    monkeypatch.setattr(patterns, "save_stock_vector_cache", changing_save)
    for change in ("revision", "content"):
        state["change"] = change
        with pytest.raises(patterns.VectorSourceChangedError):
            patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache", force=True)
        assert Path(initial["cache_path"]).read_bytes() == before


def test_append_and_full_threshold_outputs_agree(tmp_path):
    frame = history(rows=560)
    path = write_legacy(tmp_path, frame.iloc[:500])
    cfg = config(similarity_threshold=.04, min_effective_cases=1)
    patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    frame.to_parquet(path, index=False)
    patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "cache")
    patterns.build_stock_vector_cache(path, {}, cfg, tmp_path / "full", force=True)
    write_legacy(tmp_path, history("TARGET.SZ", rows=560))
    args = (tmp_path / "daily", pd.DataFrame(columns=["ts_code", "name", "industry"]), cfg, ["TARGET.SZ"])
    actual = patterns.analyze_targets_by_threshold(*args, vector_cache_dir=tmp_path / "cache")["TARGET.SZ"]
    expected = patterns.analyze_targets_by_threshold(*args, vector_cache_dir=tmp_path / "full")["TARGET.SZ"]
    assert not actual.similar_cases.empty
    assert actual.latest_snapshot == expected.latest_snapshot
    assert actual.status_probs == expected.status_probs
    pd.testing.assert_frame_equal(actual.similar_cases, expected.similar_cases, rtol=2e-6, atol=2e-7)
    pd.testing.assert_frame_equal(actual.forecast, expected.forecast, rtol=2e-6, atol=2e-7)


def test_small_synthetic_benchmark(tmp_path, capsys):
    frames = [history(f"S{n}.SZ", 560) for n in range(3)]
    paths = [write_legacy(tmp_path, frame.iloc[:550]) for frame in frames]
    cfg = config()
    args = (tmp_path / "daily", pd.DataFrame(), cfg, tmp_path / "cache")
    measured = {}
    for mode in ("full_seed", "unchanged", "append", "full_same_input"):
        if mode == "append":
            for path, frame in zip(paths, frames):
                frame.to_parquet(path, index=False)
        started = perf_counter()
        result = patterns.build_vector_caches_parallel(*args, force=mode == "full_same_input")
        measured[mode] = {"seconds": round(perf_counter() - started, 4),
                          "vectors_built": int(result.vectors_built.sum()),
                          "vectors_reused": int(result.vectors_reused.sum()),
                          "windows_evaluated": int(result.vector_windows_evaluated.sum()),
                          "rows_read": int(result.source_rows_read.sum())}
    assert measured["unchanged"]["vectors_built"] == 0
    assert measured["append"]["vectors_built"] == 30
    with capsys.disabled():
        print(f"\nSynthetic vector benchmark (3 symbols, 550 + 10 rows, 1 worker): {measured}")
