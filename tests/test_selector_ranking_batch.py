from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from quant.application import left_side_ranking
from quant.application import selector_ranking as ranking
from quant.application.left_side_ranking import (
    DEFAULT_LEFT_SIDE_RANKING_CONFIG,
)
from quant.features.left_side_factor_contract import (
    LEFT_SIDE_FACTOR_CONTRACT_SHA256,
    LEFT_SIDE_SCORE_SCHEMA_VERSION,
)
from quant.infrastructure.publication import PublicationView, current_publication, publication_context
from tests.test_selector_ranking import _promoted_config, _write_scores


DATE = "2026-08-12"


def _rows(*items):
    return [
        {
            "symbol": symbol,
            "selector_score": 13.0,
            "opportunity_score": 21.0,
            "holding_score": 34.0,
            "signals": [{"strategy_key": key, "buy_plan": "original"} for key in keys],
        }
        for symbol, keys in items
    ]


def _write_left(config, score=0.7):
    config.paths.score_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "symbol": symbol,
            "date": DATE,
            config.score_field: score,
            config.normalized_score_field: score * 100,
        }
        for symbol in ("both", "left")
    ]).to_parquet(config.paths.score_output, index=False)
    config.paths.score_manifest.write_text(json.dumps({
        "schema_version": LEFT_SIDE_SCORE_SCHEMA_VERSION,
        "target_date": DATE,
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "output_sha256": hashlib.sha256(config.paths.score_output.read_bytes()).hexdigest(),
        "artifact_sha256": "left-artifact",
        "policy_excluded_candidate_symbols": ["left-excluded"],
    }))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    right = _promoted_config(tmp_path)
    _write_scores(right, [
        {"symbol": "both", "date": DATE, "ranking_score": 0.9},
        {"symbol": "right", "date": DATE, "ranking_score": 0.6},
    ])
    manifest = json.loads(right.paths.score_manifest.read_text())
    manifest["policy_excluded_candidate_symbols"] = ["right-excluded"]
    manifest["policy_excluded_candidate_count"] = 1
    right.paths.score_manifest.write_text(json.dumps(manifest))
    left = replace(
        DEFAULT_LEFT_SIDE_RANKING_CONFIG,
        enabled=True,
        paths=replace(
            DEFAULT_LEFT_SIDE_RANKING_CONFIG.paths,
            score_output=tmp_path / "left/scores.parquet",
            score_manifest=tmp_path / "left/manifest.json",
        ),
    )
    _write_left(left)
    monkeypatch.setattr(ranking, "DEFAULT_SELECTOR_RANKING_CONFIG", right)
    monkeypatch.setattr(ranking, "DEFAULT_LEFT_SIDE_RANKING_CONFIG", left)
    return right, left


def _spy_loaders(monkeypatch):
    spies = []
    for name in (
        "load_right_side_ranking_scores",
        "load_left_side_ranking_scores",
    ):
        spy = Mock(wraps=getattr(ranking, name))
        monkeypatch.setattr(ranking, name, spy)
        spies.append(spy)
    return spies


def test_fourteen_pools_match_standalone_with_one_load_per_input(inputs, monkeypatch):
    rows = _rows(
        ("both", ["B2", "B1"]), ("right", ["B2"]), ("left", ["B1"]),
        ("right-excluded", ["B2", "B1"]), ("left-excluded", ["B1"]),
        ("unranked", ["UNKNOWN"]),
    )
    expected = ranking.apply_selector_ranking_source(
        deepcopy(rows), DATE, require_all_ranked_candidates=True
    )
    spies = _spy_loaders(monkeypatch)
    hash_spy = Mock(wraps=ranking._sha256)
    monkeypatch.setattr(ranking, "_sha256", hash_spy)
    left_hash_spy = Mock(wraps=left_side_ranking._sha256)
    monkeypatch.setattr(left_side_ranking, "_sha256", left_hash_spy)
    batch = ranking.SelectorRankingBatch(DATE)
    for pool in range(14):
        result = ranking.apply_selector_ranking_source(
            deepcopy(rows), DATE, ranking_batch=batch, require_all_ranked_candidates=True
        )
        assert result == expected
        if pool == 0:
            first_hash_calls = hash_spy.call_count
            first_left_hash_calls = left_hash_spy.call_count
        assert hash_spy.call_count == first_hash_calls
        assert left_hash_spy.call_count == first_left_hash_calls
    assert [spy.call_count for spy in spies] == [1, 1]
    assert [row["symbol"] for row in result] == ["both", "right", "left", "unranked"]
    assert result[0]["selector_score"] == 90.0


@pytest.mark.parametrize("side", ["right", "left", "neither", "empty"])
def test_loads_only_actually_needed_sides(inputs, monkeypatch, side):
    spies = _spy_loaders(monkeypatch)
    rows = {
        "right": _rows(("both", ["B2", "B1"])),
        "left": _rows(("left", ["B1"])),
        "neither": _rows(("unranked", ["UNKNOWN"])),
        "empty": [],
    }[side]
    batch = ranking.SelectorRankingBatch(DATE)
    for _ in range(3):
        ranking.apply_selector_ranking_source(deepcopy(rows), DATE, ranking_batch=batch)
    assert [spy.call_count for spy in spies] == {
        "right": [1, 0], "left": [0, 1],
        "neither": [0, 0], "empty": [0, 0],
    }[side]


@pytest.mark.parametrize("date", [None, "2026-08-13", "NaT"])
def test_batch_rejects_other_dates_even_for_empty_pool(inputs, monkeypatch, date):
    spies = _spy_loaders(monkeypatch)
    with pytest.raises(RuntimeError, match="signal_date mismatch"):
        ranking.apply_selector_ranking_source(
            [], date, ranking_batch=ranking.SelectorRankingBatch(DATE)
        )
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize("date", [None, "NaT"])
def test_batch_requires_constructor_date(date):
    with pytest.raises(ValueError, match="requires signal_date"):
        ranking.SelectorRankingBatch(date)


@pytest.mark.parametrize("change", ["right_policy", "right_path", "left_policy", "left_path", "no_left"])
def test_batch_rejects_different_configs(inputs, change):
    right, left = inputs
    batch = ranking.SelectorRankingBatch(DATE, config=right, left_config=left)
    if change == "right_policy":
        right = replace(right, selection_policy="changed")
    elif change == "right_path":
        right = replace(right, paths=replace(right.paths, score_manifest=Path("other.json")))
    elif change == "left_policy":
        left = replace(left, enabled=False)
    elif change == "left_path":
        left = replace(left, paths=replace(left.paths, score_output=Path("other.parquet")))
    else:
        left = None
    with pytest.raises(RuntimeError, match="config mismatch"):
        ranking.apply_selector_ranking_source(
            [], DATE, config=right, left_config=left, ranking_batch=batch
        )


def test_explicit_right_config_preserves_disabled_default_left(inputs, monkeypatch):
    right, _ = inputs
    spies = _spy_loaders(monkeypatch)
    result = ranking.apply_selector_ranking_source(
        _rows(("left", ["B1"])), DATE, config=right,
        ranking_batch=ranking.SelectorRankingBatch(DATE, config=right),
    )
    assert result[0]["ranking_source"] == "unified_ranker_not_applicable"
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize("side", ["right", "left"])
def test_stale_first_load_is_validated_and_failure_is_not_cached(inputs, monkeypatch, side):
    config = inputs[0 if side == "right" else 1]
    original = config.paths.score_manifest.read_text()
    stale = json.loads(original)
    stale["target_date"] = "2026-08-11"
    config.paths.score_manifest.write_text(json.dumps(stale))
    spies = _spy_loaders(monkeypatch)
    batch = ranking.SelectorRankingBatch(DATE)
    rows = _rows((side, ["B2" if side == "right" else "B1"]))
    with pytest.raises(RuntimeError, match="drifted|stale"):
        ranking.apply_selector_ranking_source(deepcopy(rows), DATE, ranking_batch=batch)
    config.paths.score_manifest.write_text(original)
    result = ranking.apply_selector_ranking_source(rows, DATE, ranking_batch=batch)
    assert result[0]["ranking_source"] == f"{side}_side_unified"
    assert spies[0 if side == "right" else 1].call_count == 2


@pytest.mark.parametrize("field", [
    "artifact", "artifact_manifest", "score_output", "score_manifest", "promotion_approval",
    "research_decision", "shadow_acceptance",
])
def test_right_reuse_guards_all_validated_inputs(inputs, monkeypatch, field):
    right, _ = inputs
    spies = _spy_loaders(monkeypatch)
    batch = ranking.SelectorRankingBatch(DATE)
    ranking.apply_selector_ranking_source(_rows(("right", ["B2"])), DATE, ranking_batch=batch)
    if field in ("research_decision", "shadow_acceptance"):
        path = Path(json.loads(right.paths.promotion_approval.read_text())[field]["path"])
    else:
        path = getattr(right.paths, field)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="inputs changed"):
        ranking.apply_selector_ranking_source(_rows(("right", ["B2"])), DATE, ranking_batch=batch)
    assert spies[0].call_count == 1


@pytest.mark.parametrize("field", ["score_output", "score_manifest"])
def test_left_reuse_rejects_replacement_even_with_same_size_and_mtime(inputs, field):
    _, left = inputs
    batch = ranking.SelectorRankingBatch(DATE)
    ranking.apply_selector_ranking_source(_rows(("left", ["B1"])), DATE, ranking_batch=batch)
    path = getattr(left.paths, field)
    original_stat = path.stat()
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(RuntimeError, match="inputs changed"):
        ranking.apply_selector_ranking_source(_rows(("left", ["B1"])), DATE, ranking_batch=batch)


def test_reuse_rejects_deleted_input(inputs):
    right, _ = inputs
    batch = ranking.SelectorRankingBatch(DATE)
    ranking.apply_selector_ranking_source(_rows(("right", ["B2"])), DATE, ranking_batch=batch)
    right.paths.score_output.unlink()
    with pytest.raises(RuntimeError, match="input is unavailable"):
        ranking.apply_selector_ranking_source([], DATE, ranking_batch=batch)


@pytest.mark.parametrize("side", ["right", "left"])
def test_new_batch_observes_corrected_scores(inputs, monkeypatch, side):
    right, left = inputs
    spies = _spy_loaders(monkeypatch)
    rows = _rows((side, ["B2" if side == "right" else "B1"]))
    old_batch = ranking.SelectorRankingBatch(DATE)
    old = ranking.apply_selector_ranking_source(deepcopy(rows), DATE, ranking_batch=old_batch)
    if side == "right":
        _write_scores(right, [{"symbol": "right", "date": DATE, "ranking_score": 0.4}])
    else:
        _write_left(left, score=0.4)
    with pytest.raises(RuntimeError, match="inputs changed"):
        ranking.apply_selector_ranking_source(deepcopy(rows), DATE, ranking_batch=old_batch)
    new = ranking.apply_selector_ranking_source(
        rows, DATE, ranking_batch=ranking.SelectorRankingBatch(DATE)
    )
    assert old[0]["selector_score"] != new[0]["selector_score"]
    assert new[0]["selector_score"] == 40.0
    assert spies[0 if side == "right" else 1].call_count == 2


def test_new_batch_revalidates_even_unchanged_files(inputs, monkeypatch):
    spies = _spy_loaders(monkeypatch)
    for _ in range(2):
        ranking.apply_selector_ranking_source(
            _rows(("right", ["B2"]), ("left", ["B1"])), DATE,
            ranking_batch=ranking.SelectorRankingBatch(DATE),
        )
    assert [spy.call_count for spy in spies] == [2, 2]


@pytest.mark.parametrize("change", ["generation", "writable", "exit"])
def test_batch_rejects_publication_context_change(inputs, tmp_path, change):
    view = PublicationView(tmp_path, tmp_path / "publications", "first", (), True)
    with publication_context(view):
        batch = ranking.SelectorRankingBatch(DATE, context_provider=current_publication)
        ranking.apply_selector_ranking_source(_rows(("right", ["B2"])), DATE, ranking_batch=batch)
    if change == "exit":
        with pytest.raises(RuntimeError, match="publication context mismatch"):
            ranking.apply_selector_ranking_source([], DATE, ranking_batch=batch)
    else:
        changed = replace(view, generation="second") if change == "generation" else replace(view, writable=False)
        with publication_context(changed), pytest.raises(RuntimeError, match="publication context mismatch"):
            ranking.apply_selector_ranking_source([], DATE, ranking_batch=batch)


@pytest.mark.parametrize("side", ["right", "left"])
def test_batch_still_checks_coverage_for_every_pool(inputs, side):
    batch = ranking.SelectorRankingBatch(DATE)
    strategy = "B2" if side == "right" else "B1"
    ranking.apply_selector_ranking_source(_rows((side, [strategy])), DATE, ranking_batch=batch)
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        ranking.apply_selector_ranking_source(_rows(("missing", [strategy])), DATE, ranking_batch=batch)
    with pytest.raises(RuntimeError, match="did not materialize"):
        ranking.apply_selector_ranking_source(
            _rows((side, [strategy])), DATE, ranking_batch=batch,
            require_all_ranked_candidates=True,
        )


def test_input_change_during_first_load_is_not_cached(inputs, monkeypatch):
    right, _ = inputs
    loader = ranking.load_right_side_ranking_scores

    def changing_loader(*args, **kwargs):
        result = loader(*args, **kwargs)
        right.paths.score_manifest.write_text(right.paths.score_manifest.read_text() + " ")
        return result

    spy = Mock(side_effect=changing_loader)
    monkeypatch.setattr(ranking, "load_right_side_ranking_scores", spy)
    batch = ranking.SelectorRankingBatch(DATE)
    with pytest.raises(RuntimeError, match="inputs changed during validation"):
        ranking.apply_selector_ranking_source(_rows(("right", ["B2"])), DATE, ranking_batch=batch)
    spy.side_effect = loader
    result = ranking.apply_selector_ranking_source(_rows(("right", ["B2"])), DATE, ranking_batch=batch)
    assert result[0]["selector_score"] == 60.0
    assert spy.call_count == 2


@pytest.mark.parametrize("side", ["right", "left"])
def test_first_load_checksum_validation_is_not_bypassed(inputs, side):
    config = inputs[0 if side == "right" else 1]
    path = config.paths.score_output
    original = path.read_bytes()
    path.write_bytes(original + b"corruption")
    batch = ranking.SelectorRankingBatch(DATE)
    rows = _rows((side, ["B2" if side == "right" else "B1"]))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ranking.apply_selector_ranking_source(deepcopy(rows), DATE, ranking_batch=batch)
    path.write_bytes(original)
    assert ranking.apply_selector_ranking_source(rows, DATE, ranking_batch=batch)
