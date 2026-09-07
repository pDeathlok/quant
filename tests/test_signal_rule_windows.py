from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.data.source_merge import normalize_tushare_daily
from quant.features.daily_factor_layer import attach_daily_signal_factors
from quant.features.variable_library import build_continuous_ohlc
from quant.research import b1_family_rules as family
from quant.research import strategy_signal_cache as cache
from quant.research import z_skill_rules as extended
from quant.research.rule_windows import B1_RULE_HISTORY, Z_RULE_HISTORY, rule_tail_start


def _daily(size: int = 620, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.025, size)))
    open_ = close * (1 + rng.normal(0, 0.015, size))
    pre_close = np.r_[close[0], close[:-1]]
    dates = pd.bdate_range("2023-01-02", periods=size)
    return pd.DataFrame({
        "ts_code": "000001.SZ", "symbol": "000001.SZ", "name": "Example",
        "date": dates, "trade_date": dates.strftime("%Y%m%d"),
        "open": open_, "close": close, "pre_close": pre_close,
        "high": np.maximum(close, open_) * rng.uniform(1.001, 1.045, size),
        "low": np.minimum(close, open_) * rng.uniform(0.955, 0.999, size),
        "volume": rng.lognormal(9, 0.65, size),
        "pct_chg": (close / pre_close - 1) * 100,
    })


def _changed(frame: pd.DataFrame, kind: str, position: int) -> pd.DataFrame:
    out = frame.copy()
    if kind == "correction":
        out.loc[position, ["open", "high", "low", "close"]] *= 1.06
        out.loc[position, "volume"] *= 3
    elif kind in {"corporate_action", "append_corporate_action"}:
        out.loc[position:, ["open", "high", "low", "close", "pre_close"]] *= 0.5
    elif kind in {"deletion", "suspension"}:
        stop = position + (1 if kind == "deletion" else 12)
        out = out.drop(index=out.index[position:stop]).reset_index(drop=True)
    return out


def _equal_tail(full: pd.DataFrame, tail: pd.DataFrame, start: pd.Timestamp) -> None:
    expected = full[full["date"] >= start].reset_index(drop=True)
    actual = tail.reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("kind", ["append", "append_corporate_action", "append_after_prior_action", "correction", "corporate_action", "deletion", "suspension"])
def test_full_and_bounded_rules_match_after_source_changes(tmp_path: Path, monkeypatch, kind: str) -> None:
    monkeypatch.delenv("DAILY_FACTOR_ROOT", raising=False)
    original = _daily()
    if kind == "append_after_prior_action":
        original = _changed(original, "corporate_action", 210)
    root = tmp_path / "factors"
    appending = kind in {"append", "append_corporate_action", "append_after_prior_action"}
    seed = original.iloc[:-4] if appending else original
    attach_daily_signal_factors(seed, "000001.SZ", factor_root=root)
    changed_position = len(original) - 4 if appending else 417
    changed = _changed(original, kind, changed_position)
    output_start = original.iloc[changed_position]["date"]

    incremental = attach_daily_signal_factors(changed, "000001.SZ", factor_root=root)
    oracle = attach_daily_signal_factors(changed, "000001.SZ", factor_root=tmp_path / "unused", persist_missing=False)
    pd.testing.assert_frame_equal(incremental, oracle, check_exact=False, rtol=1e-10, atol=1e-10)
    if appending:
        assert incremental.attrs["signal_factor_cache_mode"] == "incremental"

    full_family = family.compute_signal_flags(oracle, factors_attached=True, share_continuous_prices=False)
    tail_family = family.compute_signal_flags(incremental, factors_attached=True, output_start=output_start)
    _equal_tail(full_family, tail_family, output_start)
    affected = int((changed["date"] >= output_start).sum())
    assert tail_family.attrs["finite_rule_rows"] <= affected + 13
    assert tail_family.attrs["full_history_strategy_rows"] == len(changed)

    # Keep Z's existing 450-calendar-day normalization identical in both paths.
    path = Path("000001.SZ.parquet")
    start_text = output_start.strftime("%Y-%m-%d")
    z_full_input = extended._normalize_daily(path, start_text, oracle, factors_attached=True)
    z_inc_input = extended._normalize_daily(path, start_text, incremental, factors_attached=True)
    full_z = extended.compute_z_skill_flags(z_full_input)
    tail_z = extended.compute_z_skill_flags(z_inc_input, output_start=output_start)
    _equal_tail(full_z, tail_z, output_start)
    assert tail_z.attrs["finite_rule_rows"] <= affected + 64


@pytest.mark.parametrize("size", [5, 63, 129, 159, 160, 240])
def test_short_history_and_early_tail_preserve_warmup(size: int) -> None:
    frame = _daily(size)
    factored = attach_daily_signal_factors(frame, "000001.SZ", persist_missing=False)
    start = frame.iloc[max(0, size - 3)]["date"]
    full = family.compute_signal_flags(factored, factors_attached=True)
    tail = family.compute_signal_flags(factored, factors_attached=True, output_start=start)
    _equal_tail(full, tail, start)
    z_input = extended._normalize_daily(Path("000001.SZ.parquet"), str(start.date()), factored, factors_attached=True)
    if size < 130:
        assert z_input.empty
    else:
        _equal_tail(extended.compute_z_skill_flags(z_input), extended.compute_z_skill_flags(z_input, output_start=start), start)


@pytest.mark.parametrize("kind", ["plain", "corporate_action", "missing_volume", "duplicate", "no_pre_close"])
def test_shared_prices_match_independent_strategy_adjustments(kind: str) -> None:
    raw = _daily(280)
    if kind == "corporate_action":
        raw = _changed(raw, kind, 211)
    elif kind == "missing_volume":
        raw.loc[220, "volume"] = np.nan
    elif kind == "duplicate":
        raw = pd.concat([raw.iloc[:220], raw.iloc[219:]], ignore_index=True)
    elif kind == "no_pre_close":
        raw = raw.drop(columns="pre_close")
    factored = attach_daily_signal_factors(raw, "000001.SZ", persist_missing=False)
    independent = family.compute_signal_flags(factored, factors_attached=True, share_continuous_prices=False)
    shared = family.compute_signal_flags(factored, factors_attached=True, share_continuous_prices=True)
    pd.testing.assert_frame_equal(independent, shared)
    start = factored.iloc[-5]["date"]
    _equal_tail(independent, family.compute_signal_flags(factored, factors_attached=True, output_start=start), start)
    # Guard the unsafe raw inputs before the factor layer deduplicates them.
    selected = family._shared_strategy_prices(raw, build_continuous_ohlc(raw))
    if kind in {"missing_volume", "duplicate"}:
        assert selected is raw
    else:
        assert "pre_close" not in selected.columns


def test_unknown_or_unbounded_contract_keeps_full_history() -> None:
    frame = _daily(200)
    assert rule_tail_start(frame, frame.iloc[-1]["date"], ["new_rule"], Z_RULE_HISTORY) == 0
    assert rule_tail_start(frame, frame.iloc[-1]["date"], ["signal_vegas_tunnel"], B1_RULE_HISTORY) == 0
    assert rule_tail_start(frame.iloc[::-1], frame.iloc[-1]["date"], ["CHANGAN"], Z_RULE_HISTORY) == 0


def test_rule_contract_version_invalidates_cache_identity(monkeypatch) -> None:
    params = dict(rebuild_from=pd.Timestamp("2025-01-01"), start_date="2020-01-01", factor_mode="stateful")
    before = cache._semantic_params_fingerprint(**params)
    monkeypatch.setattr(cache.rule_windows, "RULE_WINDOW_VERSION", "changed-contract")
    assert cache._semantic_params_fingerprint(**params) != before


def test_raw_flag_cooldown_is_carried_into_the_output_tail() -> None:
    factored = attach_daily_signal_factors(_daily(300), "000001.SZ", persist_missing=False)
    frame = extended._normalize_daily(Path("000001.SZ.parquet"), "2023-12-01", factored, factors_attached=True)
    # Force an uninterrupted GOLDEN_BOWL raw flag across the tail boundary.
    frame["zg_white"] = 110.0
    frame["dg_yellow"] = 100.0
    frame["close"] = 102.0
    frame["kdj_j"] = 10.0
    start = frame.iloc[-5]["date"]
    full = extended.compute_z_skill_flags(frame)
    tail = extended.compute_z_skill_flags(frame, output_start=start)
    _equal_tail(full, tail, start)
    assert not tail["GOLDEN_BOWL"].any()


def test_symbol_worker_passes_requested_replacement_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DAILY_FACTOR_ROOT", raising=False)
    raw = _daily(240)
    start = raw.iloc[-2]["date"]
    result = cache._process_symbol("000001.SZ", raw, str(start.date()), factor_root=tmp_path)
    assert result["errors"] == []
    factored = attach_daily_signal_factors(normalize_tushare_daily(raw, "000001.SZ"), "000001.SZ", persist_missing=False)
    for key, full in (
        ("family", family.process_frame("000001.SZ", factored, factors_attached=True, raise_errors=True)),
        ("extended", extended.process_frame("000001.SZ", factored, str(start.date()), factors_attached=True, raise_errors=True)),
    ):
        actual = result[key]
        if full is None:
            assert actual is None or actual.empty
        elif actual is None:
            assert full[full["date"] >= start].empty
        else:
            _equal_tail(full, actual, start)
