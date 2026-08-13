from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant.routine import b1_daily_plan
from quant.routine.strategies import (
    EntryConfig,
    ExitConfig,
    StrategyConfig,
    StrategyRelease,
    load_strategy_release,
)


def _release() -> StrategyRelease:
    strategy = StrategyConfig(
        id="stable",
        name="stable",
        enabled=True,
        priority=1,
        backtest_combo="stable_combo",
        entry=EntryConfig(min_up10=0.2, max_down3=0.4),
        exit=ExitConfig(
            kind="fixed",
            hold_days=4,
            take_profit=0.08,
            stop_loss=0.015,
        ),
        description="test",
    )
    return StrategyRelease(
        id="b1-test",
        model_dir="models/production/b1",
        model_manifest="models/production/b1/manifest.json",
        model_names=("up10_es", "down3_es"),
        backtest_summary="summary.csv",
        compatibility_audit="audit.json",
        strategies=(strategy,),
    )


def test_production_b1_yaml_declares_current_release_and_five_models() -> None:
    release = load_strategy_release(
        Path("configs/strategies/b1_selected.yaml")
    )

    assert release.id == "b1-20260722"
    assert release.model_dir == "models/production/b1"
    assert release.model_names == (
        "up5_es",
        "up8_es",
        "up10_es",
        "down2_es",
        "down3_es",
    )
    assert [strategy.id for strategy in release.strategies] == [
        "b1_stable",
        "b1_aggressive",
    ]


def test_daily_plan_does_not_reuse_prior_day_when_target_has_no_signal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-23")],
            "symbol": ["000001.SZ"],
            "name": ["测试股份"],
        }
    ).to_parquet(feature_path, index=False)
    release = _release()
    placeholder = tmp_path / "placeholder"
    monkeypatch.setattr(
        b1_daily_plan,
        "_release_assets",
        lambda config_path: (
            release,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
        ),
    )
    monkeypatch.setattr(
        b1_daily_plan,
        "_oot_metrics",
        lambda summary_path, strategies: {
            "stable": {
                "trades": 1,
                "avg_return_pct": 1.0,
                "win_rate": 0.5,
                "max_drawdown_pct": -1.0,
                "profit_factor": 2.0,
            }
        },
    )
    monkeypatch.setattr(
        b1_daily_plan,
        "predict_models",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("models must not run for an empty target date")
        ),
    )

    plan = b1_daily_plan.build_daily_plan(
        signal_date="2026-07-24",
        config_path=tmp_path / "b1.yaml",
        feature_path=feature_path,
    )

    assert plan["signal_date"] == "2026-07-24"
    assert plan["plan_rows"] == []
    assert plan["unique_symbols"] == []


def test_production_daily_plan_fails_when_live_sidecar_cannot_prove_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-11")],
            "symbol": ["000001.SZ"],
        }
    ).to_parquet(training_path, index=False)
    release = _release()
    placeholder = tmp_path / "placeholder"
    monkeypatch.setattr(b1_daily_plan, "FEATURE_PATH", training_path)
    monkeypatch.setattr(b1_daily_plan, "ACTIVE_FEATURE_PATH", tmp_path / "missing-active.parquet")
    monkeypatch.setattr(
        b1_daily_plan,
        "ACTIVE_FEATURE_MANIFEST_PATH",
        tmp_path / "missing-active.json",
    )
    monkeypatch.setattr(
        b1_daily_plan,
        "_release_assets",
        lambda config_path: (
            release,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
        ),
    )
    monkeypatch.setattr(
        b1_daily_plan,
        "_oot_metrics",
        lambda summary_path, strategies: {
            "stable": {
                "trades": 1,
                "avg_return_pct": 1.0,
                "win_rate": 0.5,
                "max_drawdown_pct": -1.0,
                "profit_factor": 2.0,
            }
        },
    )

    with pytest.raises(RuntimeError, match="cannot prove the requested live date"):
        b1_daily_plan.build_daily_plan(
            signal_date="2026-08-12",
            config_path=tmp_path / "b1.yaml",
        )


def test_live_active_candidate_sidecar_is_preferred_and_filters_z_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = pd.Timestamp("2026-08-12")
    training_path = tmp_path / "training.parquet"
    active_path = tmp_path / "active.parquet"
    manifest_path = tmp_path / "active.json"
    pd.DataFrame(
        {"date": [pd.Timestamp("2026-08-11")], "symbol": ["OLD.SZ"]}
    ).to_parquet(training_path, index=False)
    pd.DataFrame(
        {
            "date": [target, target],
            "symbol": ["B1.SZ", "Z.SZ"],
            "candidate_source_b1": [True, False],
        }
    ).to_parquet(active_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-12",
                "candidate_coverage_status": "complete",
                "factor_count": 147,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(b1_daily_plan, "FEATURE_PATH", training_path)

    candidates, selected, manifest = b1_daily_plan._load_candidate_features(
        training_path,
        target,
        active_feature_path=active_path,
        active_manifest_path=manifest_path,
    )

    assert selected == active_path
    assert candidates["symbol"].tolist() == ["B1.SZ"]
    assert manifest and manifest["target_date"] == "2026-08-12"


def test_stale_active_candidate_sidecar_falls_back_to_training_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.parquet"
    active_path = tmp_path / "active.parquet"
    manifest_path = tmp_path / "active.json"
    pd.DataFrame(
        {"date": [pd.Timestamp("2026-08-12")], "symbol": ["B1.SZ"]}
    ).to_parquet(training_path, index=False)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-11")],
            "symbol": ["STALE.SZ"],
            "candidate_source_b1": [True],
        }
    ).to_parquet(active_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "success",
                "target_date": "2026-08-11",
                "candidate_coverage_status": "complete",
                "factor_count": 147,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(b1_daily_plan, "FEATURE_PATH", training_path)

    candidates, selected, _ = b1_daily_plan._load_candidate_features(
        training_path,
        pd.Timestamp("2026-08-12"),
        active_feature_path=active_path,
        active_manifest_path=manifest_path,
    )

    assert selected == training_path
    assert candidates["symbol"].tolist() == ["B1.SZ"]
