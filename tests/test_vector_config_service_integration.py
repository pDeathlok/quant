from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant.infrastructure.artifact_registry import ArtifactRegistry
from quant.research.similar_patterns import SimilarPatternConfig
from quant.webapp import services


class AnalysisReached(RuntimeError):
    """Stop at the downstream boundary without running unrelated workspaces."""


@pytest.fixture
def isolated_service(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.delenv("MARKET_DATA_ROOT", raising=False)
    monkeypatch.setenv("SIMILAR_PATTERN_CACHE_WORKERS", "1")
    # The real vector builder discovers the same registry root as services.
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.setattr(services, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(services, "DAILY_DIR", tmp_path / "data/raw/daily")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR",
                        tmp_path / "data/research/similar_patterns/vector_cache")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_CONFIG", SimilarPatternConfig(
        candidate_step_days=1, candidate_start_date="2011-01-01",
    ))
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2011-03-11")
    monkeypatch.setattr(services, "_similar_pattern_vector_cache_refresh_decision",
                        lambda **kwargs: {"due": True})

    def stop_analysis(*args, **kwargs):
        raise AnalysisReached("analysis reached")

    monkeypatch.setattr(services, "analyze_targets_by_threshold", stop_analysis)
    return tmp_path


def test_validated_build_persists_metadata_and_really_activates_config(isolated_service, monkeypatch):
    root = isolated_service
    daily = services.DAILY_DIR
    daily.mkdir(parents=True)
    x = np.arange(310)
    close = 20 + x * .025 + np.sin(x / 7)
    pd.DataFrame({
        "ts_code": "000001.SZ", "name": "Example",
        "trade_date": pd.bdate_range("2010-01-04", periods=len(x)).strftime("%Y%m%d"),
        "open": close * .995, "high": close * 1.02, "low": close * .98,
        "close": close, "vol": 1000 + x * 2,
        "pct_chg": pd.Series(close).pct_change().fillna(0) * 100,
    }).to_parquet(daily / "000001.SZ.parquet", index=False)
    registry = ArtifactRegistry(root)
    previous = services.SIMILAR_PATTERN_VECTOR_CACHE_DIR / "previous"
    previous.mkdir(parents=True)
    registry.register(previous, producer="similar_patterns", input_versions={},
                      retention_class="rebuildable", state="committed")
    registry.commit("similar_patterns:active_config", previous)
    config_dir = services._similar_pattern_vector_cache_state_dir()
    activate = services._activate_similar_pattern_vector_cache_config

    def assert_metadata_then_activate():
        metadata = json.loads((config_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).read_text())
        assert metadata["errors"] == 0 and metadata["rebuilt"] == 1
        assert not (config_dir / "_publication_pending.json").exists()
        return activate()

    monkeypatch.setattr(services, "_activate_similar_pattern_vector_cache_config",
                        assert_metadata_then_activate)
    with pytest.raises(AnalysisReached):
        services._refresh_similar_pattern_analysis_once([])
    assert registry.referenced_paths("similar_patterns:active_config") == (config_dir,)
    assert registry.referenced_paths("similar_patterns:active_config:previous") == (previous,)
    assert registry.referenced_paths(f"similar_patterns:{config_dir.name}:vectors") == (config_dir,)


@pytest.mark.parametrize("failure", ["build", "metadata", "activation"])
def test_failed_build_metadata_or_activation_never_reaches_analysis(isolated_service, monkeypatch, failure):
    calls = []

    def build(*args, **kwargs):
        calls.append("build")
        return pd.DataFrame([{"symbol": "000001.SZ", "status": "error" if failure == "build" else "built",
                              "error": "injected"}])

    def metadata(**kwargs):
        calls.append("metadata")
        if failure == "metadata":
            raise OSError("metadata failed")
        return {}

    def activate():
        calls.append("activation")
        raise OSError("activation failed")

    monkeypatch.setattr(services, "build_vector_caches_parallel", build)
    monkeypatch.setattr(services, "_write_similar_pattern_vector_cache_metadata", metadata)
    monkeypatch.setattr(services, "_activate_similar_pattern_vector_cache_config", activate)
    with pytest.raises((RuntimeError, OSError), match="injected|metadata failed|activation failed"):
        services._refresh_similar_pattern_analysis_once([])
    assert calls == {"build": ["build"], "metadata": ["build", "metadata"],
                     "activation": ["build", "metadata", "activation"]}[failure]


def test_scheduled_skip_does_not_activate(isolated_service, monkeypatch):
    monkeypatch.setattr(services, "_similar_pattern_vector_cache_refresh_decision",
                        lambda **kwargs: {"due": False, "metadata": {}})

    def unexpected(*args, **kwargs):
        pytest.fail("scheduled skip must not build, write metadata, or activate")

    monkeypatch.setattr(services, "build_vector_caches_parallel", unexpected)
    monkeypatch.setattr(services, "_write_similar_pattern_vector_cache_metadata", unexpected)
    monkeypatch.setattr(services, "_activate_similar_pattern_vector_cache_config", unexpected)
    with pytest.raises(AnalysisReached):
        services._refresh_similar_pattern_analysis_once([])


@pytest.mark.parametrize("invalid", ["pending", "unregistered"])
def test_real_activation_does_not_silently_skip_invalid_registration(isolated_service, invalid):
    config_dir = services._similar_pattern_vector_cache_state_dir()
    config_dir.mkdir(parents=True)
    if invalid == "pending":
        (config_dir / "_publication_pending.json").write_text("{")
    with pytest.raises(ValueError, match="pending|committed"):
        services._activate_similar_pattern_vector_cache_config()
    assert ArtifactRegistry(isolated_service).referenced_paths("similar_patterns:active_config") == ()
