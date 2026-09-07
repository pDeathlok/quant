from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.research import similar_patterns as patterns
from quant.routine.vector_refresh_policy import vector_cache_refresh_decision
from test_similar_patterns_incremental import config, history, write_legacy


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.delenv("MARKET_DATA_ROOT", raising=False)


def seeded(tmp_path):
    frames = [history(symbol, 310) for symbol in ("A.SZ", "B.SZ")]
    paths = [write_legacy(tmp_path, frame.iloc[:300]) for frame in frames]
    cfg = config()
    args = (tmp_path / "daily", pd.DataFrame(), cfg, tmp_path / "cache")
    result = patterns.build_vector_caches_parallel(*args)
    config_dir = Path(result.iloc[0].cache_path).parent
    before = {path.name: path.read_bytes() for path in config_dir.glob("*.npz")}
    return cfg, args, paths, frames, config_dir, before


def test_process_death_quarantines_partial_library_and_repairs_complete_scope(tmp_path):
    cfg, args, paths, frames, config_dir, before = seeded(tmp_path)
    for path, frame in zip(paths, frames):
        frame.to_parquet(path, index=False)
    code = """
import os, sys
from pathlib import Path
import pandas as pd
from quant.research import similar_patterns as p
root = Path(sys.argv[1])
cfg = p.SimilarPatternConfig(candidate_step_days=1, candidate_start_date='2011-01-01')
destination = root / 'cache' / p.vector_cache_key(cfg)
replace = Path.replace
def crash_after_first(self, target):
    result = replace(self, target)
    if self.suffix == '.npz' and Path(target).parent == destination:
        assert (destination / p.VECTOR_CACHE_PENDING_FILENAME).is_file()
        os._exit(91)
    return result
Path.replace = crash_after_first
p.build_vector_caches_parallel(root / 'daily', pd.DataFrame(), cfg, root / 'cache')
"""
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=False,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 91, completed.stderr
    marker = config_dir / patterns.VECTOR_CACHE_PENDING_FILENAME
    assert marker.exists()
    assert json.loads(marker.read_text())["expected_symbols"] == ["A.SZ", "B.SZ"]
    assert sum(path.read_bytes() != before[path.name] for path in config_dir.glob("*.npz")) == 1
    with pytest.raises(patterns.VectorCachePendingError):
        patterns.load_stock_vector_cache(config_dir / "A_SZ.npz")
    decision = vector_cache_refresh_decision(config_dir, now=datetime(2026, 9, 7, 9),
        metadata={"refreshed_at": "2026-09-06T15:00:00", "cached_files": 2})
    assert decision["due"] and decision["repair_reason"] == "cache_publication_pending"
    # This caller asks for just one symbol. Repair must still validate/rebuild B.
    repaired = patterns.build_vector_caches_parallel(*args, max_symbols=1)
    assert set(repaired.symbol) == {"A.SZ", "B.SZ"}
    assert not marker.exists()
    commit = json.loads((config_dir / patterns._VECTOR_CACHE_COMMIT_FILENAME).read_text())
    assert commit["phase"] == "committed" and set(commit["symbol_sources"]) == {"A.SZ", "B.SZ"}
    for path in paths:
        cached = patterns.load_stock_vector_cache(patterns.vector_cache_path(tmp_path / "cache", path.stem, cfg))
        source = patterns._normalize_daily_source(pd.read_parquet(path), path.stem)
        assert cached["source_fingerprint"] == "symbol-semantic:" + patterns._source_rows_fingerprint(source)


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_exception_clears_marker_only_after_complete_rollback(tmp_path, monkeypatch, rollback_fails):
    _, args, paths, frames, config_dir, before = seeded(tmp_path)
    for path, frame in zip(paths, frames):
        frame.to_parquet(path, index=False)
    replace = Path.replace
    observed_marker = []

    def failing_replace(self, target):
        if self.suffix == ".npz" and Path(target).parent == config_dir:
            observed_marker.append((config_dir / patterns.VECTOR_CACHE_PENDING_FILENAME).exists())
            if self.name == "B_SZ.npz":
                raise OSError("injected publication failure")
        if rollback_fails and self.name == "A_SZ.npz.previous":
            raise OSError("injected rollback failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="publication failure"):
        patterns.build_vector_caches_parallel(*args)
    assert observed_marker == [True, True]
    marker = config_dir / patterns.VECTOR_CACHE_PENDING_FILENAME
    assert marker.exists() == rollback_fails
    if not rollback_fails:
        assert {path.name: path.read_bytes() for path in config_dir.glob("*.npz")} == before


def test_commit_failure_keeps_marker_and_failed_repair_cannot_clear_it(tmp_path, monkeypatch):
    _, args, paths, frames, config_dir, _ = seeded(tmp_path)
    for path, frame in zip(paths, frames):
        frame.to_parquet(path, index=False)
    registry_type = type(patterns._vector_artifact_registry(tmp_path / "cache"))
    commit = registry_type.commit
    with monkeypatch.context() as patch:
        patch.setattr(registry_type, "commit", lambda *a, **k: (_ for _ in ()).throw(OSError("commit failed")))
        with pytest.raises(OSError, match="commit failed"):
            patterns.build_vector_caches_parallel(*args)
    marker = config_dir / patterns.VECTOR_CACHE_PENDING_FILENAME
    assert marker.exists()
    with monkeypatch.context() as patch:
        patch.setattr(patterns, "save_stock_vector_cache", lambda *a, **k: (_ for _ in ()).throw(OSError("repair failed")))
        failed = patterns.build_vector_caches_parallel(*args, max_symbols=1)
        assert "error" in set(failed.status)
    assert marker.exists()
    assert registry_type.commit == commit
    patterns.build_vector_caches_parallel(*args, max_symbols=1)
    assert not marker.exists()


@pytest.mark.parametrize("reader", ["npz", "inventory", "compile", "scan", "analysis"])
def test_all_reader_paths_refuse_pending_without_legacy_fallback(tmp_path, reader):
    cfg, _, paths, _, config_dir, _ = seeded(tmp_path)
    entries = patterns._vector_cache_inventory(paths, tmp_path / "cache", cfg)
    compiled = patterns._ensure_compiled_vector_cache(entries, tmp_path / "cache", cfg)
    (config_dir / patterns.VECTOR_CACHE_PENDING_FILENAME).write_text("not valid json")
    with pytest.raises(patterns.VectorCachePendingError):
        if reader == "npz":
            patterns.load_stock_vector_cache(entries[0].path)
        elif reader == "inventory":
            patterns._vector_cache_inventory(paths, tmp_path / "cache", cfg)
        elif reader == "compile":
            patterns._ensure_compiled_vector_cache(entries, tmp_path / "cache", cfg)
        elif reader == "scan":
            patterns._scan_compiled_threshold_cache(compiled,
                {"T": np.zeros(int(compiled.manifest["vector_dim"]), dtype=np.float32)}, cfg, set())
        else:
            patterns.analyze_targets_by_threshold(tmp_path / "daily", pd.DataFrame(), cfg, [],
                                                  vector_cache_dir=tmp_path / "cache")


def test_compiled_generations_bound_growth_but_protect_leases_and_unknown(tmp_path):
    cfg, _, paths, _, config_dir, _ = seeded(tmp_path)
    registry = patterns._vector_artifact_registry(tmp_path / "cache")
    registry.set_references("similar_patterns:active_config", [config_dir])
    cache_root = config_dir / patterns._MATRIX_CACHE_DIRNAME

    def generation(number):
        path = patterns.vector_cache_path(tmp_path / "cache", "A.SZ", cfg)
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        payload["source_fingerprint"] = np.array(f"synthetic-revision:{number}")
        np.savez(path, **payload)
        entries = patterns._vector_cache_inventory(paths, tmp_path / "cache", cfg)
        return patterns._ensure_compiled_vector_cache(entries, tmp_path / "cache", cfg)

    first = generation(0)
    unknown = cache_root / "unknown-experiment"
    unknown.mkdir()
    with registry.lease([first.generation_dir], owner="test:reader", kind="read"):
        for number in range(1, 5):
            latest = generation(number)
        assert first.generation_dir.exists()
        generations = [path for path in cache_root.iterdir() if path.is_dir() and not path.name.startswith(".") and path != unknown]
        assert len(generations) == 3
    patterns._collect_compiled_vector_caches(tmp_path / "cache", cfg)
    assert not first.generation_dir.exists()
    assert latest.generation_dir.exists() and unknown.exists()
    generations = [path for path in cache_root.iterdir() if path.is_dir() and not path.name.startswith(".") and path != unknown]
    assert len(generations) == 2


def test_compiled_retry_retires_generation_from_interrupted_commit(tmp_path, monkeypatch):
    cfg, args, paths, frames, config_dir, _ = seeded(tmp_path)
    cache_dir = tmp_path / "cache"
    registry = patterns._vector_artifact_registry(cache_dir)

    def compile_cache():
        entries = patterns._vector_cache_inventory(paths, cache_dir, cfg)
        return patterns._ensure_compiled_vector_cache(entries, cache_dir, cfg)

    first = compile_cache()
    for path, frame in zip(paths, frames):
        frame.to_parquet(path, index=False)
    patterns.build_vector_caches_parallel(*args)
    with monkeypatch.context() as patch:
        retire = type(registry).retire

        def interrupt_retirement(self, path):
            if path == first.generation_dir:
                raise RuntimeError("interrupted after compiled commit")
            return retire(self, path)

        patch.setattr(type(registry), "retire", interrupt_retirement)
        with pytest.raises(RuntimeError, match="interrupted after compiled commit"):
            compile_cache()
    # The next build advances the pointer again, so only retiring :previous
    # would miss the first generation, still registered as committed.
    os.utime(config_dir / "A_SZ.npz", ns=(1, 1))
    with registry.lease([first.generation_dir], owner="reader"):
        latest = compile_cache()
        assert first.generation_dir.exists()
    patterns._collect_compiled_vector_caches(cache_dir, cfg)
    assert not first.generation_dir.exists()
    assert latest.generation_dir.exists()
    assert len([path for path in latest.generation_dir.parent.iterdir() if path.is_dir()]) == 2


def test_truncated_npz_is_rebuilt_by_anyday_repair(tmp_path):
    _, args, _, _, config_dir, _ = seeded(tmp_path)
    damaged = config_dir / "A_SZ.npz"
    damaged.write_bytes(b"PK\x03\x04truncated")
    decision = vector_cache_refresh_decision(
        config_dir, now=datetime(2026, 9, 7, 9),
        metadata={"refreshed_at": "2026-09-06T15:00:00", "cached_files": 2},
    )
    assert decision["due"] and decision["repair_reason"] == "cache_corrupt"
    repaired = patterns.build_vector_caches_parallel(*args)
    assert dict(zip(repaired.symbol, repaired.status)) == {"A.SZ": "built", "B.SZ": "cache_hit"}
    assert patterns.load_stock_vector_cache(damaged)["symbol"] == "A.SZ"
