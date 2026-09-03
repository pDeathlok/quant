import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quant.research import similar_patterns as similar_patterns_module
from quant.research.similar_patterns import SimilarPatternResult, TargetSpec, summarize_forecast
from quant.webapp.app import app
from quant.webapp import services
from quant.webapp import api as webapp_api


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_long_factor_publication(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path / "long_factors")


def test_selector_score_presentation_normalizes_and_ranks_all_three_scores() -> None:
    rows = [
        {
            "symbol": "000001.SZ",
            "ranking_source": "right_side_unified",
            "ranking_score_normalized": 80.0,
            "opportunity_score": 40.0,
            "holding_score": 60.0,
        },
        {
            "symbol": "000002.SZ",
            "ranking_source": "left_side_unified",
            "ranking_score_normalized": 70.0,
            "opportunity_score": 80.0,
            "holding_score": 60.0,
        },
        {
            "symbol": "000003.SZ",
            "ranking_source": "unified_ranker_not_applicable",
            "opportunity_score": 50.0,
            "holding_score": 20.0,
        },
    ]

    result = services._apply_selector_score_presentation(rows)

    assert result[0]["model_score_normalized"] == 80.0
    assert result[0]["model_score_rank"] == 1
    assert result[0]["model_score_rank_count"] == 2
    assert result[1]["model_score_rank"] == 2
    assert result[2]["model_score_normalized"] is None
    assert result[2]["model_score_rank"] is None
    assert [row["buy_score_rank"] for row in result] == [3, 1, 2]
    assert [row["hold_score_rank"] for row in result] == [1, 1, 3]
    assert all(row["buy_score_rank_count"] == 3 for row in result)
    assert all(row["hold_score_rank_count"] == 3 for row in result)
    assert result[0]["model_score_source_label"] == "右侧统一模型"
    assert result[1]["model_score_source_label"] == "左侧统一模型"
    assert result[0]["score_normalization_schema_version"] == (
        "selector-score-presentation-v1"
    )


def test_selector_side_filter_splits_model_scales_and_reranks_pool(monkeypatch) -> None:
    monkeypatch.setattr(services, "_row_display_quality_gate", lambda _row: True)
    monkeypatch.setattr(
        services,
        "apply_selector_ranking_source",
        lambda rows, *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(services, "_diversify_default_rows", lambda rows, _limit: rows)
    common = {
        "matched_count": 1,
        "best_profit_factor": 1.5,
        "buy_score_source": "historical_return_model",
        "hold_score_source": "historical_return_model",
        "opportunity_score": 60.0,
        "holding_score": 55.0,
    }
    payload = {
        "signal_date": "2026-08-28",
        "total_stock_count": 9,
        "snapshot_scope": {"strategies": ["ALL"]},
        "stocks": [
            {
                **common,
                "symbol": "000001.SZ",
                "selector_score": 70.0,
                "ranking_source": "left_side_unified",
                "ranking_score_normalized": 82.0,
                "matched_group_sides": {"B1": "left"},
            },
            {
                **common,
                "symbol": "000002.SZ",
                "selector_score": 80.0,
                "ranking_source": "right_side_unified",
                "ranking_score_normalized": 98.0,
                "matched_group_sides": {"VEGAS": "right"},
            },
            {
                **common,
                "symbol": "000003.SZ",
                "selector_score": 75.0,
                "ranking_source": "right_side_unified",
                "ranking_score_normalized": 88.0,
                "matched_group_sides": {"SUPPORT_PULLBACK": "mixed"},
            },
        ],
    }

    left = services._display_selector_payload(payload, limit=10, side="left")
    right = services._display_selector_payload(payload, limit=10, side="right")

    assert [row["symbol"] for row in left["stocks"]] == ["000001.SZ"]
    assert {row["symbol"] for row in right["stocks"]} == {
        "000002.SZ",
        "000003.SZ",
    }
    assert right["total_stock_count"] == 2
    assert right["all_actionable_stock_count"] == 3
    assert right["complete_stock_count"] == 9
    assert right["short_side_filter"] == "right"
    assert all(row["model_score_rank_count"] == 2 for row in right["stocks"])
    assert right["score_presentation"]["rank_scope"] == (
        "current_strategy_side_actionable_candidate_pool"
    )
    assert services._selector_row_side(
        {"matched_group_sides": {"RHYTHM_PLATFORM": "mixed"}}
    ) == "right"


def test_selector_api_passes_and_validates_side_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_selector_payload(**kwargs):
        captured.update(kwargs)
        return {"signal_date": "2026-08-28", "stocks": []}

    monkeypatch.setattr(webapp_api, "get_stock_selector_payload", fake_selector_payload)

    response = client.get("/api/selector/stocks?side=right")
    invalid = client.get("/api/selector/stocks?side=middle")

    assert response.status_code == 200
    assert captured["side"] == "right"
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "未知短线策略侧: middle"


def test_selector_score_probability_bands_match_active_artifacts() -> None:
    services._selector_score_probability_bands.cache_clear()

    payload = services._selector_score_probability_bands()

    assert payload["available"] is True
    assert payload["schema_version"] == "selector-score-probability-bands-v2"
    assert {item["key"] for item in payload["calibrations"]} == {
        "model_right",
        "model_left",
        "buy",
        "hold",
    }
    by_key = {item["key"]: item for item in payload["calibrations"]}
    assert len(by_key["model_right"]["bands"]) == 20
    assert len(by_key["model_left"]["bands"]) == 20
    assert len(by_key["buy"]["bands"]) == 17
    assert len(by_key["hold"]["bands"]) == 17
    for calibration in by_key.values():
        bands = calibration["bands"]
        assert bands[0]["min_score"] == 0.0
        assert bands[-1]["max_score"] == 100.0
        assert all(
            left["max_score"] == right["min_score"]
            for left, right in zip(bands, bands[1:])
        )


def test_materialize_left_side_ranked_candidates_adds_missing_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(services, "_family_profiles_for_date", lambda *args: {})
    monkeypatch.setattr(services, "_fill_stock_profile", lambda stock, *args: stock)
    stocks = {
        "000001.SZ": {
            "symbol": "000001.SZ",
            "signals": [
                services._enrich_signal_group(
                    {
                        "strategy_key": "B1_MODEL",
                        "strategy_family": "B1",
                    }
                )
            ],
        }
    }
    candidates = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "B1": True,
                "SB1": False,
                "SUPER_B1": False,
                "LOW_PULLBACK": False,
            },
            {
                "symbol": "000002.SZ",
                "B1": False,
                "SB1": False,
                "SUPER_B1": True,
                "LOW_PULLBACK": True,
            },
        ]
    )
    latest = pd.DataFrame(
        [
            {
                "symbol": "000002.SZ",
                "date": pd.Timestamp("2026-08-26"),
                "name": "测试股票",
                "industry": "测试行业",
                "close": 12.34,
            }
        ]
    ).set_index("symbol")

    services._materialize_left_side_ranked_candidates(
        stocks,
        candidates,
        latest,
        "2026-08-26",
    )

    assert len(stocks["000001.SZ"]["signals"]) == 1
    assert {
        signal["strategy_group"]
        for signal in stocks["000002.SZ"]["signals"]
    } == {"SUPER_B1", "LOW_PULLBACK"}
    assert stocks["000002.SZ"]["name"] == "测试股票"
    assert stocks["000002.SZ"]["close"] == 12.34


def test_selector_stock_row_scores_current_snapshot_when_plan_date_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(services, "_fill_stock_profile", lambda stock, *args: stock)
    stock = {
        "symbol": "603368.SH",
        "date": "2026-08-24",
        "signals": [],
    }
    signal = services._enrich_signal_group(
        {
            "strategy_key": "B1",
            "strategy_family": "B1",
            "strategy_group": "B1",
            "strategy_name": "B1",
            "metrics": {},
        }
    )

    row = services.build_selector_stock_row(
        stock,
        [signal],
        signal_date="2026-09-03",
    )

    assert row is not None
    assert row["date"] == "2026-09-03"
    assert row["source_signal_date"] == "2026-08-24"


def test_selector_snapshot_rejects_non_current_row_dates() -> None:
    with pytest.raises(RuntimeError, match="non-current row dates"):
        services._require_exact_selector_row_dates(
            [{"symbol": "603368.SH", "date": "2026-08-24"}],
            "2026-09-03",
        )


def test_selector_snapshot_uses_exact_date_feature_close() -> None:
    rows = [{"symbol": "603368.SH", "date": "2026-09-03", "close": 15.74}]
    features = {
        "603368.SH": {
            "_score_feature_date": "2026-09-03",
            "close": 15.63,
        }
    }

    services._apply_exact_selector_market_values(rows, features, "2026-09-03")

    assert rows[0]["close"] == 15.63


def test_selector_snapshot_rejects_stale_market_values() -> None:
    with pytest.raises(RuntimeError, match="incomplete exact-date market values"):
        services._apply_exact_selector_market_values(
            [{"symbol": "603368.SH", "date": "2026-09-03", "close": 15.74}],
            {
                "603368.SH": {
                    "_score_feature_date": "2026-08-24",
                    "close": 15.74,
                }
            },
            "2026-09-03",
        )


def test_operation_plan_crud_persists_tomorrow_and_long_term(monkeypatch, tmp_path) -> None:
    plan_path = tmp_path / "operation_plans.json"
    monkeypatch.setattr(services, "OPERATION_PLANS_PATH", plan_path)

    created = client.post(
        "/api/operation-plans",
        json={
            "horizon": "tomorrow",
            "title": "次日回踩计划",
            "symbol": "002594.SZ",
            "target_date": "2026-08-13",
            "content": "缩量回踩后分批执行",
            "checklist": [
                {
                    "id": "check-volume",
                    "text": "确认成交量缩小",
                    "completed": False,
                }
            ],
            "status": "planned",
        },
    )
    assert created.status_code == 200
    plan = created.json()["plans"][0]
    assert plan["horizon"] == "tomorrow"
    assert plan["checklist"] == [
        {
            "id": "check-volume",
            "text": "确认成交量缩小",
            "completed": False,
        }
    ]
    assert plan_path.exists()

    updated = client.put(
        f"/api/operation-plans/{plan['id']}",
        json={
            **plan,
            "horizon": "long_term",
            "status": "done",
            "checklist": [{**plan["checklist"][0], "completed": True}],
        },
    )
    assert updated.status_code == 200
    assert updated.json()["plans"][0]["horizon"] == "long_term"
    assert updated.json()["plans"][0]["status"] == "done"
    assert updated.json()["plans"][0]["checklist"][0]["completed"] is True

    deleted = client.delete(f"/api/operation-plans/{plan['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["plans"] == []


def test_operation_plans_return_newest_created_plan_first(monkeypatch, tmp_path) -> None:
    plan_path = tmp_path / "operation_plans.json"
    monkeypatch.setattr(services, "OPERATION_PLANS_PATH", plan_path)
    plan_path.write_text(
        json.dumps(
            {
                "plans": [
                    {
                        "id": "older",
                        "horizon": "tomorrow",
                        "title": "较早计划",
                        "status": "planned",
                        "created_at": "2026-08-20T09:00:00",
                    },
                    {
                        "id": "newest",
                        "horizon": "long_term",
                        "title": "最新计划",
                        "status": "planned",
                        "created_at": "2026-08-24T09:00:00",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = client.get("/api/operation-plans")

    assert response.status_code == 200
    assert [plan["id"] for plan in response.json()["plans"]] == ["newest", "older"]
    assert response.json()["plans"][0]["checklist"] == []


def _stub_successful_global_refresh(
    monkeypatch,
    *,
    build_features=None,
    generate_daily_plan=None,
    generate_dashboard=None,
    score_latest_models=None,
    refresh_chan_model_scores=None,
    convertible_bond_plan=None,
    convertible_bond_allotment=None,
    byd_daily_plan=None,
    right_side_ranker=None,
    left_side_ranker=None,
):
    from quant.routine import pipeline
    from quant.routine import left_side_unified_production
    from quant.routine import right_side_unified_production

    status = {
        "status": "idle",
        "run_id": None,
        "started_at": None,
        "finished_at": None,
        "updated_at": None,
        "message": "idle",
        "percent": 0,
        "current_step": None,
        "steps": [],
        "result": None,
        "error": None,
    }
    success = lambda *args, **kwargs: {"status": "success"}
    source_success = lambda *args, **kwargs: {
        "status": "success",
        "expected_trade_date": "20260723",
        "dataset_trade_date": "20260723",
    }
    daily_basic_success = lambda *args, **kwargs: {
        "status": "success",
        "latest_trade_date": "20260723",
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services, "_write_terminal_refresh_manifest_unlocked", lambda: None)
    monkeypatch.setattr(services, "run_cache_cleanup", lambda project_root: {"status": "success"})
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260723")
    monkeypatch.setattr(pipeline, "refresh_data", source_success)
    monkeypatch.setattr(pipeline, "refresh_daily_basic_data", daily_basic_success)
    monkeypatch.setattr(pipeline, "refresh_reference_inputs", success)
    monkeypatch.setattr(pipeline, "refresh_factor_registry_snapshot", success)
    monkeypatch.setattr(
        pipeline,
        "resolve_daily_dependency_source_options",
        lambda scope: {
            "scope": scope,
            "include_market_daily": True,
            "include_daily_basic": scope in {"all", "short", "chan", "long", "cbAllotment"},
            "include_stock_basic": scope in {"all", "long", "cbAllotment", "similar"},
            "include_index": scope in {"all", "long", "similar"},
            "include_market_regime": scope in {"all", "long"},
            "include_financials": scope in {"all", "long"},
            "include_analyst": scope in {"all", "long"},
            "include_tradability": False,
            "long_factor_datasets": (),
            "active_source_nodes": (),
            "effective_feature_requirements": {},
            "model_contract_audit": {"status": "success"},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "publish_daily_dependency_contract",
        lambda target_date, scope, results, **kwargs: {
            "status": "success",
            "target_trade_date": target_date,
            "scope": scope,
            "phase": kwargs.get("phase"),
            "freshness_audit": {
                "status": "success",
                "checked_nodes": [],
                "failures": [],
            },
        },
    )
    monkeypatch.setattr(pipeline, "build_features", build_features or success)
    monkeypatch.setattr(pipeline, "refresh_strategy_signal_cache", success)
    monkeypatch.setattr(
        right_side_unified_production,
        "run_right_side_unified_production",
        right_side_ranker or (lambda target_date, **kwargs: {
            "status": "success",
            "target_date": target_date,
            "checkpoint_reused": True,
            "selector_adapter": {
                "status": "success",
                "target_date": target_date,
            },
        }),
    )
    monkeypatch.setattr(
        left_side_unified_production,
        "run_left_side_production",
        left_side_ranker or (lambda target_date, **kwargs: {
            "status": "success",
            "target_date": target_date,
            "checkpoint_reused": True,
            "adapter": {
                "status": "success",
                "target_date": target_date,
            },
        }),
    )
    monkeypatch.setattr(pipeline, "generate_daily_plan", generate_daily_plan or success)
    monkeypatch.setattr(pipeline, "generate_dashboard", generate_dashboard or success)
    monkeypatch.setattr(pipeline, "score_latest_models", score_latest_models or success)
    monkeypatch.setattr(
        pipeline,
        "refresh_chan_model_scores",
        refresh_chan_model_scores or success,
    )
    monkeypatch.setattr(services, "_clear_selector_caches", lambda: None)
    monkeypatch.setattr(
        services,
        "_ensure_selector_long_factor_snapshot",
        lambda signal_date, **kwargs: {
            "status": "success", "signal_date": signal_date, "checkpoint_reused": True,
        },
    )
    monkeypatch.setattr(
        services,
        "get_stock_selector_payload",
        lambda **kwargs: {"signal_date": "2026-07-23", "stocks": []},
    )
    monkeypatch.setattr(
        services,
        "get_convertible_bond_grid_plan",
        convertible_bond_plan
        or (lambda *args, **kwargs: {"trade_date": "20260723", "candidates": []}),
    )
    monkeypatch.setattr(
        services,
        "get_convertible_bond_allotments",
        convertible_bond_allotment
        or (lambda *args, **kwargs: {"asof": "2026-07-23", "records": []}),
    )
    monkeypatch.setattr(
        services,
        "get_byd_daily_strategy",
        byd_daily_plan
        or (lambda *args, **kwargs: {"planned_t": {"signal_date": "2026-07-23"}, "alerts": []}),
    )
    monkeypatch.setattr(
        services,
        "get_chan_model_strategy_plan",
        lambda *args, **kwargs: {
            "signal_date": "2026-07-23",
            "candidates": [],
            "primary_count": 0,
            "expanded_count": 0,
        },
    )
    monkeypatch.setattr(
        services,
        "_refresh_long_stock_pool_variants",
        lambda variants, signal_date: [
            {"variant": variant, "signal_date": signal_date, "stocks": 0}
            for variant in variants
        ],
    )
    monkeypatch.setattr(
        services,
        "get_blood_chip_long_plan",
        lambda **kwargs: {
            "signal_date": kwargs.get("signal_date") or "2026-07-23",
            "candidates": [],
            "simulated_positions": [],
        },
    )
    monkeypatch.setattr(
        services,
        "_run_similar_pattern_analysis_isolated",
        lambda: {
            "generated_at": "2026-07-23T18:00:00",
            "target_date": "2026-07-23",
            "results": [],
        },
    )
    monkeypatch.setattr(services, "_write_strategy_pool_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr(services, "_run_post_snapshot_cache_cleanup", lambda results: None)

    class DummyStoreConfig:
        @classmethod
        def from_env(cls, *args, **kwargs):
            return cls()

    class DummyStore:
        def __init__(self, config):
            self.config = type("Config", (), {"sql_url": None})()

        def latest_dataset_trade_date(self, dataset):
            return pd.Timestamp("2026-07-23")

    monkeypatch.setattr(services, "MarketDataStoreConfig", DummyStoreConfig)
    monkeypatch.setattr(services, "MarketDataStore", DummyStore)
    return status


def test_global_refresh_stub_never_calls_real_long_factor_builder(monkeypatch, tmp_path):
    calls = []

    class UnexpectedBuilder:
        def cache_clear(self):
            pass

        def __call__(self, variant, signal_date):
            calls.append(signal_date)
            raise RuntimeError("test reached an unisolated production factor builder")

    _stub_successful_global_refresh(monkeypatch)
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path / "long")
    monkeypatch.setattr(services, "_selector_active_model_features", lambda: ("roe",))
    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", UnexpectedBuilder())

    services._run_latest_refresh_job("short")

    assert calls == []


def test_historical_long_snapshot_cannot_replace_newer_latest(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path)
    row = {column: 1.0 for column in services.LONG_PRODUCTION_FACTOR_COLUMNS}
    row.update(ts_code="000001.SZ", date=pd.Timestamp("2026-08-26"))
    services._publish_long_factor_snapshot(pd.DataFrame([row]), row["date"])
    before = {name: (tmp_path / name).read_bytes() for name in ("latest.json", "latest.parquet")}
    row["date"] = pd.Timestamp("2026-07-23")

    services._publish_long_factor_snapshot(pd.DataFrame([row]), row["date"])

    assert (tmp_path / "20260723.parquet").exists()
    assert {name: (tmp_path / name).read_bytes() for name in before} == before


def test_selector_rejects_missing_required_long_factor_layer(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path / "data/features/long")
    monkeypatch.setattr(services, "_selector_active_model_features", lambda: ("roe",))
    with pytest.raises(RuntimeError, match="long_snapshot"):
        services._selector_production_snapshot_rows("2026-08-26")


def test_historical_scores_are_independent_of_current_candidate_pool(monkeypatch) -> None:
    monkeypatch.setattr(services, "_apply_return_model_scores", lambda *args: None)
    base_row = {
        "symbol": "000001.SZ",
        "historical_buy_score": 72.4,
        "historical_hold_score": 61.3,
        "buy_score_source": "historical_return_model",
        "hold_score_source": "historical_return_model",
    }
    other_row = {
        "symbol": "000002.SZ",
        "historical_buy_score": 99.0,
        "historical_hold_score": 5.0,
        "buy_score_source": "historical_return_model",
        "hold_score_source": "historical_return_model",
    }

    score_alone = services._apply_historical_score_normalization([dict(base_row)])[0]
    score_with_other = services._apply_historical_score_normalization([dict(base_row), dict(other_row)])[0]

    assert score_alone["opportunity_score"] == score_with_other["opportunity_score"] == 72.4
    assert score_alone["holding_score"] == score_with_other["holding_score"] == 61.3
    assert score_alone["selector_score"] == 72.4
    assert score_alone["score_target"] == "historical_return_model_score"


def test_return_model_scores_use_fixed_historical_reference(monkeypatch) -> None:
    class FakeImputer:
        def transform(self, frame):
            return frame[["selector_return_5d"]].to_numpy()

    class FakeModel:
        def predict(self, values):
            return values[:, 0]

    artifact = {
        "features": ["selector_return_5d"],
        "imputer": FakeImputer(),
        "model": FakeModel(),
        "score_reference": np.array([0.0, 1.0, 2.0, 3.0]),
    }
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {"buy": artifact})
    monkeypatch.setattr(
        services,
        "_selector_model_feature_rows",
        lambda signal_date: {
            "000001.SZ": {"date": signal_date, "selector_return_5d": 1.0},
            "000002.SZ": {"date": signal_date, "selector_return_5d": 3.0},
        },
    )
    rows = [
        {"symbol": "000001.SZ", "date": "2026-07-21", "matched_groups": ["B1"]},
        {"symbol": "000002.SZ", "date": "2026-07-21", "matched_groups": ["B2"]},
    ]

    services._apply_return_model_scores(rows)

    assert rows[0]["historical_buy_score"] < rows[1]["historical_buy_score"]
    assert rows[1]["historical_buy_score"] < 100.0
    assert rows[0]["buy_score_source"] == "historical_return_model"


def test_watchlist_buy_hold_scores_accepts_numpy_matched_groups(monkeypatch) -> None:
    observed_groups = []
    monkeypatch.setattr(
        services,
        "_latest_similar_pattern_target_date",
        lambda symbols: "2026-07-21",
    )
    monkeypatch.setattr(
        services,
        "_selector_model_score_date",
        lambda market_date: "2026-07-21",
    )
    monkeypatch.setattr(
        services,
        "_selector_model_feature_rows",
        lambda signal_date: {
            "000001.SZ": {
                "matched_groups": np.array(["B1", "B2"]),
                "best_profit_factor": 1.5,
                "best_avg_return_pct": 2.0,
            },
        },
    )

    def apply_scores(rows, feature_rows):
        observed_groups.extend(row["matched_groups"] for row in rows)
        for row in rows:
            row.update(
                opportunity_score=70.0,
                holding_score=60.0,
                score_target="historical_return_model_score",
                buy_score_source="historical_return_model",
                hold_score_source="historical_return_model",
                model_score_available=True,
            )

    monkeypatch.setattr(services, "_apply_historical_score_normalization", apply_scores)

    result = services._watchlist_buy_hold_scores(("000001.SZ",))

    assert observed_groups == [["B1", "B2"]]
    assert result["000001.SZ"]["opportunity_score"] == 70.0


def test_family_signal_cache_does_not_fall_back_to_prior_trade_date(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "family.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [pd.Timestamp("2026-07-23")],
            "signal_b1": [1],
        }
    ).to_parquet(cache, index=False)
    monkeypatch.setattr(services, "FAMILY_SIGNAL_CACHE", cache)
    monkeypatch.setattr(
        services,
        "_model_scored_candidates_for_date",
        lambda signal_date: {},
    )
    services._family_signals_for_date.cache_clear()

    signals = services._family_signals_for_date("2026-07-24")

    assert signals == {}
    services._family_signals_for_date.cache_clear()


@pytest.mark.parametrize("mode", ["buy", "hold"])
def test_historical_buy_and_hold_scores_preserve_return_order(monkeypatch, mode: str) -> None:
    distribution = np.linspace(-20.0, 20.0, 401)
    monkeypatch.setattr(
        services,
        "_selector_score_calibration",
        lambda: {
            "global": {mode: distribution},
            "group": {mode: {"B1": distribution}},
        },
    )
    common = {
        "strategy_group": "B1",
        "metrics": {
            "trades": 240,
            "win_rate": 0.5,
            "max_drawdown_pct": 10.0,
            "profit_factor": 1.5,
        },
    }
    lower_return = {**common, "metrics": {**common["metrics"], "avg_return_pct": 1.0}}
    higher_return = {**common, "metrics": {**common["metrics"], "avg_return_pct": 5.0}}

    assert services._normalized_signal_score(higher_return, mode=mode) > services._normalized_signal_score(
        lower_return,
        mode=mode,
    )


def test_api_refresh_runs_cleanup_before_data_refresh_even_when_refresh_fails(monkeypatch) -> None:
    from quant.routine import pipeline

    call_order: list[str] = []
    cleanup_result = {"status": "success", "reclaimed_bytes": 456, "errors": []}
    status = {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "updated_at": None,
        "message": "idle",
        "percent": 0,
        "current_step": None,
        "steps": [],
        "result": None,
        "error": None,
    }

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services, "_write_terminal_refresh_manifest_unlocked", lambda: None)
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda project_root: call_order.append("cleanup") or cleanup_result,
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_daily_dependency_source_options",
        lambda scope: {},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_data",
        lambda **kwargs: call_order.append("refresh") or {"status": "failed", "stderr_tail": "boom"},
    )

    services._run_latest_refresh_job("all")

    assert call_order == ["cleanup", "refresh"]
    assert status["status"] == "failed"
    assert status["result"]["cache_cleanup"] == cleanup_result


def test_global_refresh_starts_independent_workspaces_before_feature_build(
    monkeypatch,
) -> None:
    rendezvous = threading.Barrier(4, timeout=3)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def early_step(name: str, payload: dict):
        def run(*args, **kwargs):
            with calls_lock:
                calls.append(name)
            rendezvous.wait()
            return payload

        return run

    def build_features(**kwargs):
        rendezvous.wait()
        return {"status": "success"}

    status = _stub_successful_global_refresh(
        monkeypatch,
        build_features=build_features,
        convertible_bond_plan=early_step(
            "convertible_bond_plan",
            {"trade_date": "20260723", "candidates": []},
        ),
        convertible_bond_allotment=early_step(
            "convertible_bond_allotment",
            {
                "asof": "2026-07-23",
                "event_polled_through": "2026-07-23",
                "records": [],
            },
        ),
        byd_daily_plan=early_step(
            "byd_daily_plan",
            {"planned_t": {"signal_date": "2026-07-23"}, "alerts": []},
        ),
    )

    services._run_latest_refresh_job("all", run_id="parallel-early-workspaces")

    assert status["status"] == "success", status.get("error")
    assert sorted(calls) == [
        "byd_daily_plan",
        "convertible_bond_allotment",
        "convertible_bond_plan",
    ]
    assert status["result"]["convertible_bond_plan"]["status"] == "success"
    assert status["result"]["convertible_bond_allotment"]["status"] == "success"
    assert (
        status["result"]["convertible_bond_allotment"][
            "event_polled_through"
        ]
        == "2026-07-23"
    )
    assert status["result"]["byd_daily_plan"]["status"] == "success"


def test_allotment_quality_accepts_negative_j_values() -> None:
    payload = {
        "records": [
            {
                "stock_price_date": "2026-07-23",
                "kdj_daily_j": -6.58,
                "kdj_weekly_j": -9.08,
                "kdj_monthly_j": -3.21,
            }
        ],
        "data_sources": {
            "stock_daily": {"requested": 1, "matched": 1, "error": None},
            "daily_basic": {"matched": 1, "error": None},
        },
    }

    quality = services._convertible_bond_allotment_quality(
        payload,
        expected_trade_date="20260723",
    )

    assert quality["status"] == "success"
    assert quality["metrics"]["kdj_weekly_j"]["count"] == 1
    assert quality["metrics"]["kdj_monthly_j"]["count"] == 1


def test_allotment_quality_gate_rejects_stale_or_incomplete_payload(
    monkeypatch,
    tmp_path,
) -> None:
    payload = {
        "generated_at": "2026-07-23T18:00:00",
        "asof": "2026-07-23",
        "records": [
            {
                "stock_price_date": "2026-07-22",
                "kdj_daily_j": 12.0,
                "kdj_weekly_j": None,
                "kdj_monthly_j": None,
            }
        ],
        "data_sources": {
            "stock_daily": {"requested": 1, "matched": 1, "error": None},
            "daily_basic": {"matched": 1, "error": None},
        },
    }
    writes = []
    monkeypatch.setattr(
        services,
        "CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH",
        tmp_path / "allotments.json",
    )
    monkeypatch.setattr(
        services,
        "build_convertible_bond_allotment_payload",
        lambda **kwargs: payload,
    )
    monkeypatch.setattr(
        services,
        "_write_workspace_snapshot",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="配债股数据质量门禁失败") as exc_info:
        services.get_convertible_bond_allotments(
            refresh=True,
            expected_trade_date="2026-07-23",
            validate_quality=True,
        )

    assert "行情日期新鲜度不足" in str(exc_info.value)
    assert "周线 J 完整率不足" in str(exc_info.value)
    assert "月线 J 完整率不足" in str(exc_info.value)
    assert payload["quality"]["status"] == "failed"
    assert writes


def test_allotment_scope_refreshes_daily_basic_inputs(monkeypatch) -> None:
    from quant.routine import pipeline

    allotment_calls = []

    def refresh_allotments(*args, **kwargs):
        allotment_calls.append((args, kwargs))
        return {
            "asof": "2026-07-23",
            "event_polled_through": "2026-07-23",
            "records": [],
        }

    status = _stub_successful_global_refresh(
        monkeypatch,
        convertible_bond_allotment=refresh_allotments,
    )
    daily_basic_calls = []
    monkeypatch.setattr(
        pipeline,
        "refresh_daily_basic_data",
        lambda **kwargs: daily_basic_calls.append(kwargs)
        or {"status": "success", "latest_trade_date": "20260723"},
    )

    services._run_latest_refresh_job(
        "cbAllotment",
        run_id="allotment-daily-basic-refresh",
    )

    assert status["status"] == "success"
    assert (
        status["result"]["convertible_bond_allotment"][
            "event_polled_through"
        ]
        == "2026-07-23"
    )
    assert len(daily_basic_calls) == 1
    assert allotment_calls == [
        (
            (),
            {
                "refresh": True,
                "stage_scope": "pipeline",
                "expected_trade_date": "2026-07-23",
                "validate_quality": True,
            },
        )
    ]


def test_global_refresh_parallelizes_outputs_and_caps_cpu_workers(monkeypatch) -> None:
    rendezvous = threading.Barrier(2, timeout=3)
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    worker_args: dict[str, int] = {}

    def parallel_output(payload: dict, *, rendezvous_required: bool):
        def run(*args, **kwargs):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            if rendezvous_required:
                rendezvous.wait()
            with active_lock:
                active -= 1
            return payload

        return run

    chan_body = parallel_output(
        {"status": "success"}, rendezvous_required=True
    )
    right_body = parallel_output(
        {"status": "success", "target_date": "2026-07-23", "selector_adapter": {"status": "success"}},
        rendezvous_required=True,
    )
    left_body = parallel_output(
        {"status": "success", "target_date": "2026-07-23", "adapter": {"status": "success"}},
        rendezvous_required=False,
    )

    def chan(*, progress_callback, workers: int):
        worker_args["chan"] = workers
        return chan_body()

    def right(target_date: str, *, factor_workers: int):
        worker_args["right_side"] = factor_workers
        return right_body()

    def left(target_date: str, *, factor_workers: int):
        worker_args["left_side"] = factor_workers
        return left_body()

    monkeypatch.setenv("ROUTINE_MODEL_SCORE_WORKERS", "99")
    monkeypatch.setenv("ROUTINE_CHAN_WORKERS", "99")
    status = _stub_successful_global_refresh(
        monkeypatch,
        generate_daily_plan=lambda: {"status": "success", "output": "plan.json"},
        refresh_chan_model_scores=chan,
        right_side_ranker=right,
        left_side_ranker=left,
    )

    services._run_latest_refresh_job("all", run_id="parallel-daily-outputs")

    assert status["status"] == "success"
    assert max_active == 2
    assert worker_args == {"chan": 4, "right_side": 6, "left_side": 2}
    assert status["result"]["output_dag"]["resource_usage"][
        "max_cpu_slots"
    ] == 10
    assert status["result"]["generate_daily_plan"]["status"] == "success"
    assert status["result"]["generate_dashboard"]["status"] == "retired"
    assert status["result"]["model_score"]["status"] == "retired"
    assert status["result"]["refresh_chan_model_scores"]["status"] == "success"


def test_global_refresh_builds_selector_snapshot_once(monkeypatch) -> None:
    status = _stub_successful_global_refresh(monkeypatch)
    calls: list[dict[str, object]] = []

    def selector(**kwargs):
        calls.append(kwargs)
        return {"signal_date": "2026-07-23", "stocks": []}

    monkeypatch.setattr(services, "get_stock_selector_payload", selector)
    services._run_latest_refresh_job("short", run_id="selector-single-pass")

    assert status["status"] == "success", status.get("error")
    assert calls == [
        {
            "signal_date": "2026-07-23",
            "include_extended": True,
            "use_cache": False,
            "full_snapshot": True,
        }
    ]
    assert status["result"]["selector_core"]["execution_mode"] == (
        "covered_by_selector_extended_single_pass"
    )


def test_global_refresh_overlaps_both_unified_rankers_with_chan(
    monkeypatch,
) -> None:
    from quant.routine import right_side_unified_production

    right_side_started = threading.Event()
    chan_started = threading.Event()
    worker_args: dict[str, int] = {}

    def chan(*, progress_callback, workers: int):
        worker_args["chan"] = workers
        chan_started.set()
        assert right_side_started.wait(timeout=3)
        return {"status": "success"}

    def right_side(target_date: str, *, factor_workers: int):
        worker_args["right_side"] = factor_workers
        right_side_started.set()
        assert chan_started.wait(timeout=3)
        return {
            "status": "success",
            "target_date": target_date,
            "selector_adapter": {"status": "success"},
        }

    def left_side(target_date: str, *, factor_workers: int):
        worker_args["left_side"] = factor_workers
        assert right_side_started.is_set()
        assert chan_started.is_set()
        return {
            "status": "success",
            "target_date": target_date,
            "adapter": {"status": "success"},
        }

    monkeypatch.setenv("ROUTINE_RIGHT_SIDE_WORKERS", "99")
    status = _stub_successful_global_refresh(
        monkeypatch,
        refresh_chan_model_scores=chan,
        right_side_ranker=right_side,
        left_side_ranker=left_side,
    )
    services._run_latest_refresh_job("all", run_id="parallel-right-side")

    assert status["status"] == "success", status.get("error")
    assert worker_args == {"chan": 4, "right_side": 6, "left_side": 2}
    assert status["result"]["right_side_unified_features"] == {
        "status": "success",
        "target_date": "2026-07-23",
        "factor_workers": 6,
    }


def test_global_refresh_does_not_run_retired_legacy_model_score(
    monkeypatch,
) -> None:
    from quant.routine import right_side_unified_production

    right_side_calls = 0

    def right_side(*args, **kwargs):
        nonlocal right_side_calls
        right_side_calls += 1
        return {"status": "success"}

    legacy_score_calls = 0

    def legacy_score(**kwargs):
        nonlocal legacy_score_calls
        legacy_score_calls += 1
        return {"status": "failed", "stderr_tail": "must not run"}

    status = _stub_successful_global_refresh(
        monkeypatch,
        score_latest_models=legacy_score,
        right_side_ranker=right_side,
    )
    services._run_latest_refresh_job("all", run_id="model-failure-cancels-right")

    assert status["status"] == "success"
    assert legacy_score_calls == 0
    assert right_side_calls == 1
    assert status["result"]["model_score"]["status"] == "retired"


def test_global_refresh_propagates_early_workspace_failure(monkeypatch) -> None:
    calls = 0

    def fail_convertible_bond(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("bond upstream unavailable")

    status = _stub_successful_global_refresh(
        monkeypatch,
        convertible_bond_plan=fail_convertible_bond,
    )

    services._run_latest_refresh_job("all", run_id="parallel-early-failure")

    assert calls == 1
    assert status["status"] == "failed"
    assert "可转债策略计划刷新失败" in status["error"]
    assert "bond upstream unavailable" in status["error"]


def test_global_refresh_checkpoints_early_results_before_terminal_failure(
    monkeypatch,
) -> None:
    rendezvous = threading.Barrier(4, timeout=3)

    def early_success(payload: dict):
        def run(*args, **kwargs):
            rendezvous.wait()
            return payload

        return run

    def fail_features(**kwargs):
        rendezvous.wait()
        raise RuntimeError("feature build failed")

    status = _stub_successful_global_refresh(
        monkeypatch,
        build_features=fail_features,
        convertible_bond_plan=early_success(
            {"trade_date": "20260723", "candidates": [{"code": "123001"}]},
        ),
        convertible_bond_allotment=early_success(
            {
                "asof": "2026-07-23",
                "event_polled_through": "2026-07-23",
                "records": [{"code": "600001"}],
            },
        ),
        byd_daily_plan=early_success(
            {
                "planned_t": {"signal_date": "2026-07-23"},
                "alerts": [{"level": "info"}],
            },
        ),
    )

    services._run_latest_refresh_job("all", run_id="parallel-early-checkpoint")

    assert status["status"] == "failed"
    assert "feature build failed" in status["error"]
    assert status["result"]["convertible_bond_plan"]["candidates"] == 1
    assert status["result"]["convertible_bond_allotment"]["records"] == 1
    assert (
        status["result"]["convertible_bond_allotment"][
            "event_polled_through"
        ]
        == "2026-07-23"
    )
    assert status["result"]["byd_daily_plan"]["alerts"] == 1


def test_global_refresh_reruns_same_day_early_product_when_contract_is_dirty(
    monkeypatch,
) -> None:
    """A successful old checkpoint must not hide a same-day contract change."""

    from quant.routine import pipeline

    calls = {"cb": 0, "allotment": 0, "byd": 0}

    def cb_plan(*args, **kwargs):
        calls["cb"] += 1
        return {"trade_date": "20260723", "candidates": []}

    def allotment(*args, **kwargs):
        calls["allotment"] += 1
        return {"asof": "2026-07-23", "records": []}

    def byd(*args, **kwargs):
        calls["byd"] += 1
        return {
            "planned_t": {"signal_date": "2026-07-23"},
            "alerts": [],
        }

    status = _stub_successful_global_refresh(
        monkeypatch,
        convertible_bond_plan=cb_plan,
        convertible_bond_allotment=allotment,
        byd_daily_plan=byd,
    )

    def publish_contract(target_date, scope, results, **kwargs):
        phase = kwargs.get("phase")
        dirty = phase != "postflight"
        return {
            "status": "success",
            "target_trade_date": target_date,
            "scope": scope,
            "phase": phase,
            "refresh_node_ids": ["product.cb_grid"] if dirty else [],
            "refresh_nodes": (
                [
                    {
                        "node_id": "product.cb_grid",
                        "layer": "product",
                        "action": "refresh",
                        "ui_step": "convertible_bond_plan",
                    }
                ]
                if dirty
                else []
            ),
            "freshness_audit": {
                "status": "success",
                "checked_nodes": [],
                "failures": [],
            },
        }

    monkeypatch.setattr(
        pipeline,
        "publish_daily_dependency_contract",
        publish_contract,
    )
    started_at = pd.Timestamp.now().isoformat()
    resume_status = {
        "status": "failed",
        "run_id": "same-day-product-checkpoint",
        "attempt": 1,
        "started_at": started_at,
        "steps": services._progress_steps("all"),
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260723",
                "dataset_trade_date": "20260723",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260723",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "steps": {
                    "analyst_forecast_snapshot": {"status": "success"},
                },
            },
            "convertible_bond_plan": {
                "status": "success",
                "trade_date": "20260723",
            },
            "convertible_bond_allotment": {
                "status": "success",
                "asof": "2026-07-23",
            },
            "byd_daily_plan": {
                "status": "success",
                "signal_date": "2026-07-23",
            },
        },
    }

    services._run_latest_refresh_job(
        "all",
        resume_status=resume_status,
        run_id="same-day-product-rerun",
    )

    assert status["status"] == "success"
    assert calls == {"cb": 1, "allotment": 0, "byd": 0}


def test_successful_same_day_refresh_reuses_all_downstream_after_source_poll(
    monkeypatch,
) -> None:
    """A fresh invocation must poll sources, then reuse an unchanged DAG."""

    from quant.routine import pipeline

    status = _stub_successful_global_refresh(monkeypatch)
    services._run_latest_refresh_job("all", run_id="initial-success")
    assert status["status"] == "success"
    prior_results = dict(status["result"])
    prior_results["dependency_postflight"] = {
        "status": "success",
        "baseline_committed": True,
        "target_trade_date": "2026-07-23",
    }
    prior_status = {
        **status,
        "status": "success",
        "scope": "all",
        "run_id": "prior-success",
        "result": prior_results,
    }
    status.update({"status": "success", "run_id": None, "result": prior_results})
    calls = {"source": 0}

    def source_refresh(*args, **kwargs):
        calls["source"] += 1
        return {
            "status": "success",
            "expected_trade_date": "20260723",
            "dataset_trade_date": "20260723",
        }

    monkeypatch.setattr(pipeline, "refresh_data", source_refresh)
    monkeypatch.setattr(
        pipeline,
        "build_features",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged downstream feature node must be reused")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_strategy_signal_cache",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged downstream signal node must be reused")
        ),
    )

    services._run_latest_refresh_job(
        "all",
        resume_status=prior_status,
        run_id="same-day-idempotent-run",
    )

    assert calls["source"] == 1
    assert status["status"] == "success"
    assert status["result"]["dependency_postflight"]["status"] == "success"
    assert all(
        step["checkpoint_reused"] is True
        for step in status["steps"]
        if step["key"] != "refresh_data"
    )


def test_failed_run_retry_identity_ignores_yesterday_dirty_plan_when_stable() -> None:
    prior = {
        "schema_version": "daily_dependency_snapshot_v2",
        "identity_complete": True,
        "scope": "all",
        "target_trade_date": "2026-08-12",
        "node_contract_hashes": {"feature.strategy_signals": "contract-v1"},
        "model_contract_hashes": {"score.b1": "model-v1"},
        "node_state_fingerprints": {
            "data.market_daily": "market-v1",
            "data.daily_basic": "basic-v1",
            "feature.strategy_signals": "signals-v1",
        },
        "refresh_node_ids": [
            "feature.strategy_signals",
            "feature.project_daily",
            "product.selector_core",
        ],
    }
    current = {
        **prior,
        "node_state_fingerprints": {
            **prior["node_state_fingerprints"],
            "feature.strategy_signals": "signals-v2",
        },
    }

    assert services._retry_preflight_identity_stable(prior, current) is True
    current["node_state_fingerprints"]["data.market_daily"] = "market-v2"
    assert services._retry_preflight_identity_stable(prior, current) is False
    current["node_state_fingerprints"]["data.market_daily"] = "market-v1"
    current["node_contract_hashes"] = {"feature.strategy_signals": "contract-v2"}
    assert services._retry_preflight_identity_stable(prior, current) is False


def test_terminal_refresh_writes_immutable_manifest_with_step_timings(
    monkeypatch,
    tmp_path,
) -> None:
    status = {
        "status": "running",
        "run_id": "all-20260723170000-abcd1234",
        "started_at": "2026-07-23T17:00:00",
        "finished_at": None,
        "updated_at": "2026-07-23T17:00:00",
        "message": "running",
        "percent": 1,
        "current_step": "refresh_data",
        "steps": services._progress_steps("chan"),
        "scope": "chan",
        "scope_label": "缠论策略",
        "result": None,
        "error": None,
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "REFRESH_STATUS_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(services, "REFRESH_MANIFEST_ROOT", tmp_path / "runs")
    services._REFRESH_CONTEXT.run_id = status["run_id"]
    try:
        services._set_refresh_progress(
            step_key="refresh_data",
            message="data done",
            percent=35,
            step_status="success",
            complete_previous=False,
        )
        services._set_refresh_progress(
            status="success",
            step_key="chan_model_strategy",
            message="done",
            percent=100,
            result={"chan_model_strategy": {"status": "success"}},
        )
    finally:
        delattr(services._REFRESH_CONTEXT, "run_id")

    manifest_path = Path(status["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "web_daily_refresh"
    assert payload["status"] == "success"
    assert all(step["started_at"] for step in payload["steps"])
    assert all(step["finished_at"] for step in payload["steps"])
    assert all(step["elapsed_seconds"] is not None for step in payload["steps"])
    assert (tmp_path / "latest.json").exists()

    original = manifest_path.read_text(encoding="utf-8")
    status["message"] = "must not overwrite terminal manifest"
    services._write_terminal_refresh_manifest_unlocked()
    assert manifest_path.read_text(encoding="utf-8") == original


def test_checkpoint_reused_step_has_zero_attempt_elapsed(
    monkeypatch,
) -> None:
    status = {
        "status": "running",
        "run_id": "resume-attempt",
        "attempt": 2,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": services._progress_steps("short"),
        "percent": 35,
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    services._REFRESH_CONTEXT.run_id = status["run_id"]
    try:
        services._set_refresh_progress(
            step_key="refresh_data",
            step_status="success",
            message="checkpoint reused",
            percent=35,
            complete_previous=False,
            checkpoint_reused=True,
        )
    finally:
        delattr(services._REFRESH_CONTEXT, "run_id")

    step = next(
        item
        for item in status["steps"]
        if item["key"] == "refresh_data"
    )
    assert step["status"] == "success"
    assert step["checkpoint_reused"] is True
    assert step["elapsed_seconds"] == 0.0
    assert step["started_at"] == step["finished_at"]


def test_resumed_refresh_starts_independent_attempt_timing(
    monkeypatch,
) -> None:
    old_steps = services._progress_steps("all")
    old_steps[0].update(
        {
            "status": "success",
            "started_at": "2026-07-27T15:00:00",
            "finished_at": "2026-07-27T15:05:00",
            "elapsed_seconds": 300.0,
        }
    )
    status = {
        "status": "failed",
        "run_id": "old-run",
        "attempt": 1,
        "started_at": "2026-07-27T15:00:00",
        "finished_at": "2026-07-27T15:30:00",
        "elapsed_seconds": 1800.0,
        "updated_at": "2026-07-27T15:30:00",
        "message": "failed",
        "percent": 35,
        "current_step": "feature_cache",
        "steps": old_steps,
        "scope": "all",
        "result": {},
        "error": "old failure",
        "manifest_path": "/tmp/old-run/manifest.json",
    }

    class InlineThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(
        services,
        "_refresh_resume_ready",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(services.threading, "Thread", InlineThread)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)

    payload = services.start_latest_refresh("all")

    assert payload["status"] == "queued"
    assert payload["run_id"] != "old-run"
    assert payload["attempt"] == 2
    assert payload["started_at"] != "2026-07-27T15:00:00"
    assert payload["resumed_from"] == {
        "run_id": "old-run",
        "attempt": 1,
        "started_at": "2026-07-27T15:00:00",
        "finished_at": "2026-07-27T15:30:00",
        "elapsed_seconds": 1800.0,
        "manifest_path": "/tmp/old-run/manifest.json",
    }
    assert all(step["started_at"] is None for step in payload["steps"])
    assert all(step["elapsed_seconds"] is None for step in payload["steps"])
    assert all(
        step["checkpoint_reused"] is False
        for step in payload["steps"]
    )


def test_start_refresh_passes_successful_same_day_baseline_to_worker(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    status = {
        "status": "success",
        "run_id": "prior-success",
        "scope": "all",
        "steps": services._progress_steps("all"),
        "result": {"dependency_postflight": {"baseline_committed": True}},
    }

    class CapturingThread:
        def __init__(self, *, target, args, **kwargs) -> None:
            captured["target"] = target
            captured["args"] = args

        def start(self) -> None:
            pass

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_completed_checkpoint_ready", lambda *args: True)
    monkeypatch.setattr(services.threading, "Thread", CapturingThread)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)

    payload = services.start_latest_refresh("all")

    assert captured["args"][1]["run_id"] == "prior-success"
    assert payload["status"] == "queued"
    assert payload["attempt"] == 1
    assert payload["resumed_from"] is None
    assert "增量校验" in payload["message"]


def test_input_resume_marks_only_reused_steps_as_checkpoints(
    monkeypatch,
    tmp_path,
) -> None:
    status = _stub_successful_global_refresh(monkeypatch)
    feature_dir = tmp_path / "data/features/b1"
    feature_dir.mkdir(parents=True)
    (feature_dir / "training_xgb_project_vars.parquet").touch()
    (feature_dir / "active_candidate_project_features.parquet").touch()
    (feature_dir / "b1_family_rule_candidates.parquet").touch()
    (feature_dir.parent / "z_skill_daily_candidates.parquet").touch()
    (feature_dir / "active_candidate_project_features_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "target_date": "2026-07-23",
                "candidate_coverage_status": "complete",
                "factor_count": 147,
                "factor_schema_version": "project-v1-latest-scale-global-rank",
                "union_candidate_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "PROJECT_ROOT", tmp_path)
    source_started_at = pd.Timestamp.now().isoformat()
    resume_status = {
        "status": "failed",
        "run_id": "source-attempt",
        "attempt": 1,
        "started_at": source_started_at,
        "finished_at": source_started_at,
        "elapsed_seconds": 900.0,
        "steps": services._progress_steps("all"),
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260723",
                "dataset_trade_date": "20260723",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260723",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "steps": {
                    "analyst_forecast_snapshot": {"status": "success"}
                },
            },
            "feature_cache": {"status": "success"},
            "model_score": {"status": "success"},
        },
    }
    monkeypatch.setattr(
        services.pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame(
            {"date": [pd.Timestamp("2026-07-23")]}
        ),
    )

    services._run_latest_refresh_job(
        "all",
        resume_status=resume_status,
        run_id="resumed-run",
    )

    assert status["status"] == "success"
    assert status["attempt"] == 2
    assert status["started_at"] != source_started_at
    step_map = {
        step["key"]: step
        for step in status["steps"]
    }
    assert step_map["refresh_data"]["checkpoint_reused"] is True
    assert step_map["refresh_data"]["elapsed_seconds"] == 0.0
    assert step_map["signal_cache"]["checkpoint_reused"] is False
    assert step_map["feature_cache"]["checkpoint_reused"] is False
    assert "model_score" not in step_map
    assert step_map["daily_plan"]["checkpoint_reused"] is False


def test_input_resume_ignores_retired_z_score_dirty_hint(
    monkeypatch,
) -> None:
    from quant.routine import pipeline

    score_calls = 0

    def score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return {"status": "success"}

    status = _stub_successful_global_refresh(
        monkeypatch,
        score_latest_models=score,
    )

    def publish_contract(target_date, scope, results, **kwargs):
        phase = kwargs.get("phase")
        dirty = phase != "postflight"
        return {
            "status": "success",
            "target_trade_date": target_date,
            "scope": scope,
            "phase": phase,
            "refresh_node_ids": ["score.z_skill"] if dirty else [],
            "refresh_nodes": (
                [
                    {
                        "node_id": "score.z_skill",
                        "layer": "model_score",
                        "action": "refresh",
                        "ui_step": "model_score",
                    }
                ]
                if dirty
                else []
            ),
            "freshness_audit": {
                "status": "success",
                "checked_nodes": [],
                "failures": [],
            },
        }

    monkeypatch.setattr(
        pipeline,
        "publish_daily_dependency_contract",
        publish_contract,
    )
    resume_status = {
        "status": "failed",
        "run_id": "same-day-score-checkpoint",
        "attempt": 1,
        "started_at": pd.Timestamp.now().isoformat(),
        "steps": services._progress_steps("all"),
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260723",
                "dataset_trade_date": "20260723",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260723",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "steps": {
                    "analyst_forecast_snapshot": {"status": "success"},
                },
            },
            "model_score": {"status": "success"},
        },
    }

    services._run_latest_refresh_job(
        "all",
        resume_status=resume_status,
        run_id="same-day-score-rerun",
    )

    assert status["status"] == "success"
    assert score_calls == 0
    assert status["result"]["model_score"].get("checkpoint_reused") is not True


def test_starting_refresh_clears_stale_manifest_path(monkeypatch, tmp_path) -> None:
    stale_manifest = tmp_path / "old" / "manifest.json"
    status = {
        "status": "failed",
        "run_id": "old-run",
        "started_at": "2026-07-23T17:00:00",
        "finished_at": "2026-07-23T17:30:00",
        "updated_at": "2026-07-23T17:30:00",
        "message": "failed",
        "percent": 35,
        "current_step": "feature_cache",
        "steps": services._progress_steps("all"),
        "scope": "all",
        "scope_label": "全部工作区",
        "result": None,
        "error": "old error",
        "manifest_path": str(stale_manifest),
        "manifest_error": "old manifest error",
    }

    class InlineThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_refresh_resume_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(services.threading, "Thread", InlineThread)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)

    payload = services.start_latest_refresh("all")

    assert payload["status"] == "queued"
    assert payload["manifest_path"] is None
    assert payload["manifest_error"] is None


def test_health_endpoint_reports_service_status() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "quant-webapp"}


def test_byd_daily_plan_only_forwards_permanent_holding(monkeypatch) -> None:
    captured = {}

    def fake_strategy(**kwargs):
        captured.update(kwargs)
        return {"validation": {"execution_enabled": False}, "alerts": []}

    monkeypatch.setattr(webapp_api, "get_byd_daily_strategy", fake_strategy)

    response = client.get(
        "/api/byd/daily-plan",
        params={
            "shares": 10300,
            "cost": 108.25,
            "refresh": True,
        },
    )

    assert response.status_code == 200
    assert captured["shares"] == 10300
    assert captured["cost"] == 108.25
    assert captured["refresh"] is True
    assert set(captured) == {"shares", "cost", "refresh"}


def test_blood_chip_long_plan_endpoint_forwards_daily_iteration_parameters(monkeypatch) -> None:
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return {
            "signal_date": "2026-08-07",
            "variant": "blood_chip",
            "candidates": [],
        }

    monkeypatch.setattr(webapp_api, "get_blood_chip_long_plan", fake_plan)

    response = client.get(
        "/api/long/blood-chip",
        params={"signal_date": "2026-08-07", "refresh": True},
    )

    assert response.status_code == 200
    assert response.json()["variant"] == "blood_chip"
    assert captured == {"signal_date": "2026-08-07", "refresh": True}


def test_refresh_endpoint_rejects_unknown_scope() -> None:
    response = client.post("/api/selector/refresh-latest", json={"scope": "unknown"})

    assert response.status_code == 400
    assert "未知刷新范围" in response.json()["detail"]


def test_global_refresh_includes_daily_similar_pattern_step() -> None:
    step_keys = [step["key"] for step in services._progress_steps("all")]

    assert "convertible_bond_allotment" in step_keys
    assert "similar_patterns" in step_keys


def test_similar_refresh_scope_refreshes_shared_daily_inputs_first(monkeypatch) -> None:
    status = _stub_successful_global_refresh(monkeypatch)

    services._run_latest_refresh_job("similar", run_id="similar-run")

    assert [step["key"] for step in status["steps"]] == [
        "refresh_data",
        "similar_patterns",
    ]
    assert status["status"] == "success"
    assert status["percent"] == 100
    assert status["result"]["similar_patterns"]["target_date"] == "2026-07-23"
    assert status["result"]["dependency_postflight"]["status"] == "success"


def test_refresh_progress_ignores_stale_run_context(monkeypatch) -> None:
    status = {
        "status": "running",
        "run_id": "current-run",
        "started_at": "2026-07-22T19:00:00",
        "finished_at": None,
        "updated_at": "2026-07-22T19:00:00",
        "message": "current",
        "percent": 10,
        "current_step": "refresh_data",
        "steps": services._progress_steps("all"),
        "result": None,
        "error": None,
    }

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    services._REFRESH_CONTEXT.run_id = "old-run"
    try:
        services._set_refresh_progress(
            step_key="feature_cache",
            message="old worker should not win",
            percent=40,
        )
    finally:
        delattr(services._REFRESH_CONTEXT, "run_id")

    assert status["message"] == "current"
    assert status["percent"] == 10
    assert status["current_step"] == "refresh_data"


def test_expiring_stale_refresh_terminates_known_child_processes(monkeypatch) -> None:
    status = {
        "status": "running",
        "run_id": "stale-run",
        "started_at": "2000-01-01T00:00:00",
        "finished_at": None,
        "updated_at": "2000-01-01T00:00:00",
        "message": "stale",
        "percent": 35,
        "current_step": "feature_cache",
        "steps": [{"key": "feature_cache", "status": "running"}],
        "result": None,
        "error": None,
    }
    killed: list[tuple[int, int]] = []

    class Result:
        returncode = 0
        stdout = "\n".join(
            [
                "123 /Users/didi/miniforge3/bin/python scripts/research/refresh_b1_feature_cache.py --workers 8",
                "456 /Users/didi/miniforge3/bin/python scripts/research/rebuild_strategy_signal_cache.py --workers 8",
            ]
        )
        stderr = ""

    monkeypatch.setattr(services, "_REFRESH_STATUS", dict(status))
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(services.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    expired = services._expire_stale_refresh_status_unlocked(status)

    assert expired["status"] == "failed"
    assert expired["steps"][0]["status"] == "failed"
    assert (123, services.signal.SIGTERM) in killed
    assert (456, services.signal.SIGTERM) in killed
    assert "已终止卡住的后台节点" in expired["error"]


def test_latest_extended_signals_reuses_candidate_parquet(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "z_skill_daily_candidates.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "date": pd.Timestamp("2026-07-19"),
                "name": "平安银行",
                "close": 10.0,
                "KEY_K": True,
                "NANA": False,
            },
            {
                "symbol": "000002.SZ",
                "date": pd.Timestamp("2026-07-20"),
                "name": "万科A",
                "close": 8.0,
                "KEY_K": False,
                "NANA": True,
            },
        ]
    ).to_parquet(cache_path)

    monkeypatch.setattr(services, "EXTENDED_SIGNAL_CACHE", tmp_path / "missing.json")
    monkeypatch.setattr(services, "EXTENDED_CANDIDATE_CACHE", cache_path)
    monkeypatch.setattr(
        services,
        "_stock_basic_map",
        lambda: {
            "000001.SZ": {"name": "平安银行", "industry": "银行"},
            "000002.SZ": {"name": "万科A", "industry": "地产"},
        },
    )
    monkeypatch.setattr(
        services,
        "build_extended_daily_signals",
        lambda *args, **kwargs: pytest.fail("should reuse candidate parquet"),
    )
    services._latest_extended_signals.cache_clear()

    signals = services._latest_extended_signals("2026-07-20")

    assert set(signals) == {"000002.SZ"}
    assert signals["000002.SZ"]["industry"] == "地产"
    assert signals["000002.SZ"]["signals"][0]["strategy_key"] == "NANA"


def test_tea_variants_share_one_concurrent_live_score_build(monkeypatch) -> None:
    build_started = threading.Event()
    allow_build_to_finish = threading.Event()
    calls = {"prepare": 0, "score": 0}

    def prepare(module, signal_date):
        calls["prepare"] += 1
        build_started.set()
        assert allow_build_to_finish.wait(timeout=3)
        return pd.DataFrame({"date": [pd.Timestamp(signal_date)]}), pd.DataFrame(), {
            "source": "test"
        }

    def build_scores(frame, **kwargs):
        calls["score"] += 1
        return frame.assign(score=1.0)

    monkeypatch.setattr(
        services,
        "_tea_master_research_module",
        lambda: SimpleNamespace(build_tea_scores=build_scores),
    )
    monkeypatch.setattr(services, "_prepare_tea_master_live_data", prepare)
    services._tea_master_live_scores_cached.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                services._tea_master_live_scores,
                "2026-07-30",
            )
            assert build_started.wait(timeout=3)
            second = executor.submit(
                services._tea_master_live_scores,
                "20260730",
            )
            allow_build_to_finish.set()
            first_result = first.result(timeout=3)
            second_result = second.result(timeout=3)
    finally:
        services._tea_master_live_scores_cached.cache_clear()

    assert calls == {"prepare": 1, "score": 1}
    pd.testing.assert_frame_equal(first_result[0], second_result[0])
    assert first_result[1] == second_result[1] == {"source": "test"}


def test_long_stock_pool_variants_run_in_parallel_and_keep_order(monkeypatch) -> None:
    observed = []

    class FakeExecutor:
        def __init__(self, max_workers):
            observed.append(("max_workers", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            observed.extend(("variant", item) for item in items)
            return [fn(item) for item in reversed(list(items))]

    monkeypatch.setattr(services, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        services,
        "_refresh_long_stock_pool_variant",
        lambda variant, signal_date: {"variant": variant, "signal_date": signal_date, "stocks": len(variant)},
    )

    result = services._refresh_long_stock_pool_variants(["tea", "tea_safe", "v44"], "2026-07-20")

    assert ("max_workers", 3) in observed
    assert [item["variant"] for item in result] == ["tea", "tea_safe", "v44"]
    assert [item["signal_date"] for item in result] == ["2026-07-20"] * 3


def test_long_workspace_refresh_includes_blood_chip_daily_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_refresh_long_stock_pool_variants",
        lambda variants, signal_date: [
            {"variant": variant, "signal_date": signal_date, "stocks": 0}
            for variant in variants
        ],
    )
    monkeypatch.setattr(
        services,
        "get_blood_chip_long_plan",
        lambda **kwargs: {
            "signal_date": kwargs["signal_date"],
            "candidates": [{"ts_code": "000001.SZ"}],
            "simulated_positions": [],
        },
    )

    result = services._refresh_long_workspace(["tea"], "2026-08-07")

    assert result["variants"][0]["variant"] == "tea"
    assert result["blood_chip"] == {
        "signal_date": "2026-08-07",
        "candidates": 1,
        "simulated_positions": 0,
    }


def test_long_research_module_waits_for_concurrent_import(monkeypatch) -> None:
    module_name = "quant_long_dividend_quality_research"
    import_started = threading.Event()
    allow_import_to_finish = threading.Event()
    second_returned = threading.Event()
    import_calls = []

    class FakeLoader:
        def exec_module(self, module) -> None:
            import_calls.append(module)
            import_started.set()
            assert allow_import_to_finish.wait(timeout=2)
            module.load_stock_basic = lambda: pd.DataFrame()

    class FakeSpec:
        loader = FakeLoader()

    class FakeModule:
        pass

    monkeypatch.delitem(services.sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        services.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: FakeSpec(),
    )
    monkeypatch.setattr(
        services.importlib.util,
        "module_from_spec",
        lambda spec: FakeModule(),
    )
    services._long_research_module.cache_clear()

    def load_second():
        module = services._long_research_module()
        second_returned.set()
        return module

    try:
        with services.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(services._long_research_module)
            assert import_started.wait(timeout=2)
            second = executor.submit(load_second)
            returned_before_import_finished = second_returned.wait(timeout=0.1)
            allow_import_to_finish.set()
            first_module = first.result(timeout=2)
            second_module = second.result(timeout=2)
    finally:
        allow_import_to_finish.set()
        services._long_research_module.cache_clear()

    assert not returned_before_import_finished
    assert first_module is second_module
    assert len(import_calls) == 1
    assert first_module._quant_services_import_complete is True


def test_tea_master_research_module_discards_partial_import(monkeypatch) -> None:
    module_name = "quant_tea_master_long_research"
    imported_modules = []

    class FakeLoader:
        def exec_module(self, module) -> None:
            imported_modules.append(module)
            module.CONFIGS = []
            if len(imported_modules) == 1:
                raise RuntimeError("incomplete import")

    class FakeSpec:
        loader = FakeLoader()

    class FakeModule:
        pass

    monkeypatch.delitem(services.sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        services.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: FakeSpec(),
    )
    monkeypatch.setattr(
        services.importlib.util,
        "module_from_spec",
        lambda spec: FakeModule(),
    )
    services._tea_master_research_module.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="incomplete import"):
            services._tea_master_research_module()
        recovered = services._tea_master_research_module()
    finally:
        services._tea_master_research_module.cache_clear()

    assert len(imported_modules) == 2
    assert recovered is imported_modules[1]
    assert recovered._quant_services_import_complete is True


def test_long_stock_pool_worker_does_not_redirect_process_stdout(monkeypatch) -> None:
    original_stdout = services.sys.stdout
    observed_stdout = []
    payload = {"signal_date": "2026-07-30", "stocks": []}

    def build(variant, signal_date):
        observed_stdout.append(services.sys.stdout)
        return payload

    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", build)
    monkeypatch.setattr(services, "_write_long_stock_pool_snapshot", lambda *args, **kwargs: None)

    result = services._refresh_long_stock_pool_variant("tea", "2026-07-30")

    assert observed_stdout == [original_stdout]
    assert result == {
        "variant": "tea",
        "signal_date": "2026-07-30",
        "stocks": 0,
    }


def test_long_factor_snapshot_publishes_dated_latest_and_manifest(monkeypatch, tmp_path) -> None:
    rows = []
    for symbol, close in [("000001.SZ", 10.0), ("600000.SH", 12.0)]:
        row = {
            column: 1.0
            for column in (
                *services.LONG_FACTOR_REQUIRED_COLUMNS,
                *services.LONG_PRODUCTION_FACTOR_COLUMNS,
            )
        }
        row.update(
            {
                "date": pd.Timestamp("2026-07-30"),
                "ts_code": symbol,
                "name": symbol,
                "industry": "电子",
                "close": close,
            }
        )
        rows.append(row)
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path / "long")

    result = services._publish_long_factor_snapshot(
        pd.DataFrame(rows),
        pd.Timestamp("2026-07-30"),
    )

    latest = pd.read_parquet(tmp_path / "long/latest.parquet")
    manifest = json.loads((tmp_path / "long/latest.json").read_text(encoding="utf-8"))
    assert result["signal_date"] == "2026-07-30"
    assert result["rows"] == 2
    assert result["factor_count"] == len(services.LONG_PRODUCTION_FACTOR_COLUMNS)
    assert result["coverage_status"] == "complete"
    assert (tmp_path / "long/20260730.parquet").is_file()
    assert latest["ts_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert latest["factor_schema_version"].eq(services.LONG_FACTOR_SNAPSHOT_SCHEMA_VERSION).all()
    assert manifest["latest_path"] == str(tmp_path / "long/latest.parquet")


def test_long_factor_snapshot_rejects_incomplete_production_contract(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-30")],
            "ts_code": ["000001.SZ"],
            "good_stock_score": [80.0],
        }
    )

    with pytest.raises(RuntimeError, match="长线因子截面缺少必需字段"):
        services._publish_long_factor_snapshot(frame, pd.Timestamp("2026-07-30"))


def test_selector_long_factor_snapshot_reuses_current_and_refreshes_when_planned(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot_dir = tmp_path / "long"
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", snapshot_dir)
    row = {column: 1.0 for column in services.LONG_PRODUCTION_FACTOR_COLUMNS}
    row.update(ts_code="000001.SZ", date=pd.Timestamp("2026-07-30"))
    manifest = services._publish_long_factor_snapshot(pd.DataFrame([row]), row["date"])

    class FakeTeaBuilder:
        calls = 0
        clears = 0

        def cache_clear(self):
            self.clears += 1

        def __call__(self, variant, signal_date):
            self.calls += 1
            return {
                "factor_snapshot": {
                    **manifest,
                    "signal_date": signal_date,
                }
            }

    builder = FakeTeaBuilder()
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(services, "_selector_active_model_features", lambda: ("roe",))
    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", builder)

    reused = services._ensure_selector_long_factor_snapshot(
        "2026-07-30",
        force_refresh=False,
    )
    refreshed = services._ensure_selector_long_factor_snapshot(
        "2026-07-30",
        force_refresh=True,
    )

    assert reused["checkpoint_reused"] is True
    assert refreshed["checkpoint_reused"] is False
    assert builder.clears == 1
    assert builder.calls == 1


def test_long_refresh_publishes_factor_result_before_page_snapshot(monkeypatch) -> None:
    order = []
    factor_snapshot = {
        "status": "success",
        "signal_date": "2026-07-30",
        "latest_path": "data/features/long/latest.parquet",
    }

    def build(variant, signal_date):
        order.append("factor_snapshot")
        return {
            "signal_date": signal_date,
            "stocks": [],
            "factor_snapshot": factor_snapshot,
        }

    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", build)
    monkeypatch.setattr(
        services,
        "_write_long_stock_pool_snapshot",
        lambda *args, **kwargs: order.append("page_snapshot"),
    )

    result = services._refresh_long_stock_pool_variant("tea", "2026-07-30")

    assert order == ["factor_snapshot", "page_snapshot"]
    assert result["factor_snapshot"] == factor_snapshot


def test_live_long_base_skips_persistent_backtest_cache_and_reuses_memory(monkeypatch) -> None:
    calls = []
    features = pd.DataFrame({"date": pd.to_datetime(["2026-07-20"]), "ts_code": ["000001.SZ"]})
    daily_basic = pd.DataFrame({"date": pd.to_datetime(["2026-07-20"]), "ts_code": ["000001.SZ"]})
    stock_basic = pd.DataFrame({"ts_code": ["000001.SZ"]})

    class FakeModule:
        @staticmethod
        def parse_date(value):
            return pd.to_datetime(value)

        @staticmethod
        def load_stock_basic():
            return stock_basic

        @staticmethod
        def load_daily_basic_monthly(start, end):
            return daily_basic, {"first_trade_date": "20130101"}

        @staticmethod
        def load_daily_monthly_features(start, end, basic, candidate_symbols, **kwargs):
            calls.append(kwargs)
            return features, pd.DataFrame()

    services._load_live_long_base_cached.cache_clear()
    monkeypatch.setattr(services, "_long_research_module", lambda: FakeModule())

    first = services._load_live_long_base("2026-07-20")
    second = services._load_live_long_base("2026-07-20")

    assert len(calls) == 1
    assert calls[0] == {"use_cache": True, "include_daily_returns": False}
    assert first[0] is second[0]
    services._load_live_long_base_cached.cache_clear()


def test_live_long_base_keeps_valuation_history_sections(monkeypatch) -> None:
    observed = {}
    dates = pd.to_datetime(["2026-05-29", "2026-06-30", "2026-07-21"])
    daily_basic = pd.DataFrame(
        {
            "date": dates,
            "trade_date": dates.strftime("%Y%m%d"),
            "ts_code": ["000001.SZ"] * 3,
        }
    )
    features = pd.DataFrame({"date": dates[-2:], "ts_code": ["000001.SZ"] * 2})
    stock_basic = pd.DataFrame({"ts_code": ["000001.SZ"]})

    class FakeModule:
        @staticmethod
        def load_stock_basic():
            return stock_basic

        @staticmethod
        def load_daily_basic_monthly(start, end):
            observed["basic_start"] = start
            observed["basic_end"] = end
            return daily_basic, {"first_trade_date": "20260529"}

        @staticmethod
        def load_daily_monthly_features(start, end, basic, candidate_symbols, **kwargs):
            observed["price_start"] = start
            return features, pd.DataFrame()

    services._load_live_long_base_cached.cache_clear()
    monkeypatch.setattr(services, "_long_research_module", lambda: FakeModule())

    loaded_features, loaded_basic, _, coverage = services._load_live_long_base("2026-07-21")

    assert observed["basic_start"] == pd.Timestamp("2018-07-21")
    assert observed["price_start"] == pd.Timestamp("2026-05-29")
    assert loaded_features["date"].tolist() == list(dates[-2:])
    assert loaded_basic["date"].tolist() == list(dates)
    assert coverage["live_rebalance_dates"] == ["2026-05-29", "2026-06-30", "2026-07-21"]
    services._load_live_long_base_cached.cache_clear()


def test_long_good_price_keeps_both_pr_formulas_and_uses_history_rule() -> None:
    row = pd.Series(
        {
            "close": 10.0,
            "pe_ttm": 12.0,
            "pb": 1.5,
            "pr": 0.92,
            "pr_pe": 0.80,
            "pr_pb": 1.20,
            "pr_formula_gap": 1 / 3,
            "pe_hist_percentile": 35.0,
            "pb_hist_percentile": 30.0,
            "pr_hist_percentile": 32.0,
            "pr_pe_hist_percentile": 28.0,
            "pr_pb_hist_percentile": 40.0,
            "valuation_history_points": 36,
            "historical_value_score": 68.0,
            "valuation_profile": "earnings_based",
            "pr_pe_weight": 0.70,
            "pr_pb_weight": 0.30,
            "ma_120": 9.5,
            "ma_120_slope_20d": 0.01,
        }
    )

    result = services._long_good_price_assessment(row)

    assert result["is_good_price"] is True
    assert result["good_price_rule"] == "composite_60_guard"
    assert result["price_score"] == 68.0
    assert result["price_score_normalization"] == "per_stock_trailing_history_percentile"
    assert result["price_score_cross_date_comparable"] is True
    assert result["price_score_history_frequency"] == "month_end"
    assert result["price_score_history_window_months"] == 84
    assert result["price_score_min_history_points"] == 24
    assert result["pr_from_pe"] == 0.8
    assert result["pr_from_pb"] == 1.2
    assert result["pr_pe_hist_percentile"] == 28.0
    assert result["pr_pb_hist_percentile"] == 40.0
    assert result["valuation_profile"] == "earnings_based"
    assert result["pr_pe_weight"] == 0.7
    assert result["pr_pb_weight"] == 0.3


def test_long_good_stock_excludes_special_treatment_names() -> None:
    base = {
        "industry": "电子",
        "good_stock_score": 80.0,
        "profitability_score": 80.0,
        "fundamental_growth_score": 70.0,
        "balance_sheet_score": 70.0,
        "business_stability_score": 70.0,
        "good_stock_data_coverage": 1.0,
        "listing_years": 5.0,
        "roe": 15.0,
        "netprofit_margin": 12.0,
        "or_yoy": 10.0,
        "basic_eps_yoy": 12.0,
        "debt_to_assets": 35.0,
    }

    assert services._long_good_stock_assessment(pd.Series({**base, "name": "优质股份"}))["is_good_stock"] is True
    assert services._long_good_stock_assessment(pd.Series({**base, "name": "ST风险"}))["is_good_stock"] is False
    assert services._long_good_stock_assessment(pd.Series({**base, "name": "退市风险"}))["is_good_stock"] is False


def test_long_good_price_requires_24_months_of_complete_history() -> None:
    result = services._long_good_price_assessment(
        pd.Series(
            {
                "close": 10.0,
                "historical_value_score": 80.0,
                "valuation_history_points": 23,
                "pe_hist_percentile": 10.0,
                "pb_hist_percentile": 10.0,
                "pr_hist_percentile": 10.0,
                "ma_120": 9.5,
                "ma_120_slope_20d": 0.01,
            }
        )
    )

    assert result["is_good_price"] is False
    assert result["price_state"] == "WAIT_HISTORY"


def test_long_good_price_explains_the_failed_structure_gate() -> None:
    result = services._long_good_price_assessment(
        pd.Series(
            {
                "close": 269.58,
                "historical_value_score": 94.42,
                "valuation_history_points": 81,
                "pe_hist_percentile": 4.94,
                "pb_hist_percentile": 6.17,
                "pr_hist_percentile": 5.68,
                "ma_120": 251.57,
                "ma_120_slope_20d": -0.068084431,
            }
        )
    )

    assert result["is_good_price"] is False
    assert result["price_state"] == "WAIT_STABILITY"
    assert result["trend_price_guard_passed"] is True
    assert result["trend_slope_guard_passed"] is False
    assert result["trend_guard_passed"] is False
    assert result["price_levels"]["trend_floor_price"] == pytest.approx(226.41)
    assert result["price_state_reason"] == "MA120近20日斜率 -6.81%，低于 -6.00% 门槛"


def test_long_price_score_backtest_payload_is_versioned_and_complete() -> None:
    services._long_price_score_backtest_payload.cache_clear()
    payload = services._long_price_score_backtest_payload()

    assert payload["available"] is True
    assert payload["schema_version"] == "long_price_score_bands_v1"
    assert payload["history_window_months"] == 84
    assert payload["minimum_history_months"] == 24
    assert [item["key"] for item in payload["bands"]] == [
        "80_100",
        "60_80",
        "40_60",
        "20_40",
        "0_20",
    ]
    assert all(item["validation"]["signals"] > 0 for item in payload["bands"])
    assert all(item["test"]["signals"] > 0 for item in payload["bands"])


def test_long_pool_row_exposes_metric_percentiles_and_three_year_forecast() -> None:
    row = pd.Series(
        {
            "ts_code": "000001.SZ",
            "name": "优质股份",
            "industry": "电子",
            "close": 10.0,
            "good_stock_score": 80.0,
            "profitability_score": 80.0,
            "fundamental_growth_score": 70.0,
            "balance_sheet_score": 70.0,
            "business_stability_score": 70.0,
            "good_stock_data_coverage": 1.0,
            "listing_years": 5.0,
            "roe": 15.0,
            "roe_hist_percentile": 72.0,
            "roe_history_points": 84,
            "netprofit_margin": 12.0,
            "or_yoy": 10.0,
            "basic_eps_yoy": 12.0,
            "debt_to_assets": 35.0,
            "pe_ttm": 12.0,
            "pb": 1.5,
            "pr": 0.92,
            "pr_pe": 0.8,
            "pr_pb": 1.2,
            "pe_hist_percentile": 35.0,
            "pb_hist_percentile": 30.0,
            "pr_hist_percentile": 32.0,
            "pr_pe_hist_percentile": 28.0,
            "pr_pb_hist_percentile": 40.0,
            "valuation_history_points": 84,
            "historical_value_score": 68.0,
            "ma_120": 9.5,
            "ma_120_slope_20d": 0.01,
            "analyst_forward_eps_3y_mean_180d": 1.6,
            "analyst_forward_eps_3y_variance_180d": 0.113333,
            "analyst_forward_eps_3y_years_180d": 3,
            "analyst_forward_eps_3y_estimate_count_180d": 4,
            "analyst_forward_y0_year": 2026,
            "analyst_forward_y0_eps_mean_180d": 1.2,
            "analyst_forward_y0_eps_std_180d": None,
            "analyst_forward_y0_eps_estimate_count_180d": 1,
            "analyst_forward_y0_price_mean_180d": 12.0,
            "analyst_forward_y0_price_std_180d": None,
            "analyst_forward_y0_price_estimate_count_180d": 1,
            "analyst_forward_y1_year": 2027,
            "analyst_forward_y1_eps_mean_180d": 1.6,
            "analyst_forward_y1_eps_std_180d": 0.141421,
            "analyst_forward_y1_eps_estimate_count_180d": 2,
            "analyst_forward_y1_price_mean_180d": 17.7,
            "analyst_forward_y1_price_std_180d": 3.818377,
            "analyst_forward_y1_price_estimate_count_180d": 2,
            "analyst_forward_y2_year": 2028,
            "analyst_forward_y2_eps_mean_180d": 2.0,
            "analyst_forward_y2_eps_std_180d": None,
            "analyst_forward_y2_eps_estimate_count_180d": 1,
            "analyst_forward_y2_price_mean_180d": 20.0,
            "analyst_forward_y2_price_std_180d": None,
            "analyst_forward_y2_price_estimate_count_180d": 1,
        }
    )

    result = services._tea_good_stock_price_row(row, variant="tea")

    assert result["roe_hist_percentile"] == 72.0
    assert result["roe_history_points"] == 84
    assert result["pe_hist_percentile"] == 35.0
    assert result["pr_from_pe"] == 0.8
    assert result["pr_from_pb"] == 1.2
    assert result["analyst_forward_eps_3y_mean_180d"] == 1.6
    assert result["analyst_forward_eps_3y_variance_180d"] == 0.113333
    assert result["analyst_forward_eps_3y_years_180d"] == 3
    assert result["analyst_forward_eps_3y_estimate_count_180d"] == 4
    assert result["price_score"] == 68.0
    assert result["display_reason"] == "盈利能力较强、成长质量较好、财务安全性较好、经营稳定性较好；价格分达标"
    assert result["analyst_forecast_3y"] == [
        {
            "horizon": 0,
            "forecast_year": 2026,
            "eps_mean": 1.2,
            "eps_std": None,
            "eps_estimate_count": 1,
            "price_mean": 12.0,
            "price_std": None,
            "price_estimate_count": 1,
            "price_basis": "eps_x_forecast_pe",
        },
        {
            "horizon": 1,
            "forecast_year": 2027,
            "eps_mean": 1.6,
            "eps_std": 0.1414,
            "eps_estimate_count": 2,
            "price_mean": 17.7,
            "price_std": 3.82,
            "price_estimate_count": 2,
            "price_basis": "eps_x_forecast_pe",
        },
        {
            "horizon": 2,
            "forecast_year": 2028,
            "eps_mean": 2.0,
            "eps_std": None,
            "eps_estimate_count": 1,
            "price_mean": 20.0,
            "price_std": None,
            "price_estimate_count": 1,
            "price_basis": "eps_x_forecast_pe",
        },
    ]


def test_tea_checkpoint_recovers_previous_month_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_read_long_stock_pool_snapshot",
        lambda *args, **kwargs: {
            "signal_date": "2026-07-20",
            "stocks": [
                {"ts_code": "000001.SZ", "rank": 1, "previous_state": "CORE"},
                {"ts_code": "000002.SZ", "rank": 2, "previous_state": "WATCH"},
                {"ts_code": "000003.SZ", "rank": None, "previous_state": "CORE"},
            ],
        },
    )

    same_month = services._tea_previous_target_checkpoint("tea", pd.Timestamp("2026-07-21"))
    next_month = services._tea_previous_target_checkpoint("tea", pd.Timestamp("2026-08-03"))

    assert same_month == {"000001.SZ", "000003.SZ"}
    assert next_month == {"000001.SZ", "000002.SZ"}


def test_tea_analyst_display_recomputes_growth_score_and_preserves_missing(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-21"] * 3),
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "close": [10.0, 20.0, 30.0],
            "analyst_forward_growth_score": [0.0, 0.0, 0.0],
        }
    )

    class FakeModule:
        @staticmethod
        def load_analyst_forecast_asof(out):
            return out.assign(
                analyst_report_count_180d=[9.0, 9.0, 5.0],
                analyst_org_count_180d=[2.0, 2.0, 1.0],
                analyst_forward_years_180d=[3.0, 3.0, 3.0],
                analyst_forward_eps_growth_180d=[0.20, 0.10, float("nan")],
                analyst_forward_revenue_growth_180d=[0.15, 0.05, float("nan")],
                analyst_forward_net_profit_growth_180d=[0.25, 0.08, float("nan")],
                analyst_target_upside_180d=[0.10, 0.05, float("nan")],
            )

        @staticmethod
        def percentile_score(series, higher_is_better=True):
            rank = pd.to_numeric(series, errors="coerce").rank(pct=True)
            if not higher_is_better:
                rank = 1.0 - rank
            return (rank * 100).fillna(0.0)

    monkeypatch.setattr(services, "_long_research_module", lambda: FakeModule())

    result = services._attach_analyst_forecast_for_display(frame).set_index("ts_code")

    assert result.loc["000001.SZ", "analyst_forward_growth_score"] > result.loc["000002.SZ", "analyst_forward_growth_score"]
    assert pd.isna(result.loc["000003.SZ", "analyst_forward_growth_score"])


def test_continuous_recommendation_days_follow_latest_unbroken_streak() -> None:
    targets = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                [
                    "2026-03-31",
                    "2026-04-30",
                    "2026-05-29",
                    "2026-05-29",
                    "2026-06-30",
                    "2026-06-30",
                    "2026-07-21",
                    "2026-07-21",
                ]
            ),
            "ts_code": [
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000001.SZ",
                "000003.SZ",
                "000001.SZ",
                "000003.SZ",
            ],
        }
    )

    starts = services._continuous_recommendation_starts(targets, pd.Timestamp("2026-07-21"))

    assert starts == {
        "000001.SZ": pd.Timestamp("2026-03-31"),
        "000003.SZ": pd.Timestamp("2026-06-30"),
    }

    recommended = services._decorate_long_recommendation_display(
        {
            "state": "BUILDING",
            "close": 9.8,
            "price_levels": {
                "entry_aggressive_price": 10.0,
                "entry_target_price": 10.2,
                "entry_small_position_price": 11.0,
            },
        },
        selected=True,
        signal_ts=pd.Timestamp("2026-07-21"),
        recommendation_since=starts["000001.SZ"],
    )
    assert recommended["recommendation_level"] == "RECOMMENDED"
    assert recommended["recommendation_days"] == 113
    assert recommended["price_state"] == "AGGRESSIVE"
    assert recommended["display_reason"] == "本期进入推荐池，价格与风险条件支持分批执行"
    assert "持有" not in recommended["display_reason"]

    cautious = services._decorate_long_recommendation_display(
        {
            "state": "REDUCE",
            "reason": "价格跌破 MA60，先降仓控制风险",
            "close": 9.8,
            "price_levels": {"reduce_ma60_price": 10.0},
        },
        selected=False,
        signal_ts=pd.Timestamp("2026-07-21"),
    )
    assert cautious["recommendation_level"] == "CAUTION"
    assert "降仓" not in cautious["display_reason"]


def test_legacy_tea_snapshot_is_upgraded_without_full_pool_rebuild(monkeypatch) -> None:
    payload = {
        "variant": "tea",
        "signal_date": "2026-07-21",
        "stocks": [
            {"ts_code": "000001.SZ", "close": 10.0, "analyst_forward_growth_score": 0.0},
            {"ts_code": "000002.SZ", "close": 20.0, "analyst_forward_growth_score": 0.0},
        ],
    }
    enriched = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "analyst_report_count_180d": [9.0, 5.0],
            "analyst_org_count_180d": [2.0, 1.0],
            "analyst_forward_years_180d": [3.0, 2.0],
            "analyst_forward_growth_score": [72.5, float("nan")],
            "analyst_target_upside_180d": [0.12, float("nan")],
        }
    )
    monkeypatch.setattr(services, "_attach_analyst_forecast_for_display", lambda frame: enriched)

    upgraded, changed = services._upgrade_cached_tea_analyst_display(payload)

    assert changed is True
    assert upgraded["schema_version"] == services.LONG_STOCK_POOL_SCHEMA_VERSION
    assert upgraded["stocks"][0]["analyst_forward_growth_score"] == 72.5
    assert upgraded["stocks"][1]["analyst_forward_growth_score"] is None


def test_tail_resume_runs_pending_steps_via_executor(monkeypatch) -> None:
    submitted = []
    maintenance_order: list[str] = []

    class FakeFuture:
        def __init__(self, fn):
            self.fn = fn

        def result(self):
            return self.fn()

    class FakeExecutor:
        def __init__(self, max_workers):
            submitted.append(("max_workers", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn):
            submitted.append(("submit", fn))
            return FakeFuture(fn)

    monkeypatch.setattr(services, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(services, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(services, "_set_refresh_progress", lambda **kwargs: None)
    monkeypatch.setattr("quant.routine.pipeline._incremental_daily_start", lambda: "20260720")
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260720")
    monkeypatch.setattr(
        services,
        "get_stock_selector_payload",
        lambda **kwargs: {"signal_date": "2026-07-20", "stocks": []},
    )
    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", type("Cache", (), {"cache_clear": staticmethod(lambda: None)})())
    monkeypatch.setattr(services, "_build_long_stock_pool_cached", type("Cache", (), {"cache_clear": staticmethod(lambda: None)})())
    monkeypatch.setattr(services, "get_chan_model_strategy_plan", lambda *args, **kwargs: {"signal_date": "2026-07-20", "candidates": []})
    monkeypatch.setattr(services, "_refresh_long_stock_pool_variants", lambda *args, **kwargs: [])
    monkeypatch.setattr(services, "get_convertible_bond_grid_plan", lambda *args, **kwargs: {"trade_date": "20260720", "candidates": []})
    monkeypatch.setattr(
        services,
        "get_convertible_bond_allotments",
        lambda *args, **kwargs: {
            "asof": "2026-07-20",
            "event_polled_through": "2026-07-20",
            "records": [],
        },
    )
    monkeypatch.setattr(services, "get_byd_daily_strategy", lambda *args, **kwargs: {"planned_t": {"signal_date": "2026-07-20"}, "alerts": []})
    monkeypatch.setattr(services, "_run_similar_pattern_analysis_isolated", lambda: {"generated_at": "2026-07-20T20:00:00", "results": []})
    monkeypatch.setattr(
        services,
        "_write_strategy_pool_snapshots",
        lambda *args, **kwargs: maintenance_order.append("snapshot") or {"ALL": 0},
    )
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda project_root: maintenance_order.append("cleanup")
        or {"status": "success", "sql_snapshots": {"status": "success"}, "errors": []},
    )

    result = services._resume_tail_refresh_from_cached_selector(
        "all",
        {
            "steps": [
                {"key": "selector_extended", "status": "success"},
                {"key": "snapshot", "status": "failed"},
            ],
            "result": {
                "refresh_data": {
                    "status": "success",
                    "expected_trade_date": "20260720",
                    "dataset_trade_date": "20260720",
                }
            },
        },
    )

    assert ("max_workers", 6) in submitted
    assert len([item for item in submitted if item[0] == "submit"]) == 6
    assert result["snapshot"]["strategy_pools"] == {"ALL": 0}
    assert (
        result["convertible_bond_allotment"]["event_polled_through"]
        == "2026-07-20"
    )
    assert result["similar_patterns"]["status"] == "success"
    assert result["cache_cleanup_after_snapshot"]["status"] == "success"
    assert maintenance_order[-2:] == ["snapshot", "cleanup"]


def test_post_snapshot_cleanup_failure_is_not_silently_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda project_root: {
            "status": "partial",
            "sql_snapshots": {"status": "failed"},
            "errors": ["database unavailable"],
        },
    )
    results: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="database unavailable"):
        services._run_post_snapshot_cache_cleanup(results)

    assert results["cache_cleanup_after_snapshot"]["status"] == "partial"


def test_same_day_failed_run_can_reuse_completed_source_inputs(monkeypatch) -> None:
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260724")
    today = pd.Timestamp.now().isoformat()
    status = {
        "status": "failed",
        "started_at": today,
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260724",
                "dataset_trade_date": "20260724",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260724",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "steps": {"analyst_forecast_snapshot": {"status": "success"}},
            },
            "feature_cache": {"status": "failed"},
        },
    }

    assert services._input_resume_ready(status, "all") is True
    status["result"]["refresh_daily_basic"]["status"] = "failed"
    assert services._input_resume_ready(status, "all") is False


def test_same_day_failed_run_reuses_degraded_analyst_fallback(monkeypatch) -> None:
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260724")
    status = {
        "status": "failed",
        "started_at": pd.Timestamp.now().isoformat(),
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260724",
                "dataset_trade_date": "20260724",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260724",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "critical_errors": [],
                "steps": {"analyst_forecast_snapshot": {"status": "degraded"}},
            },
        },
    }

    assert services._input_resume_ready(status, "all") is True


def test_similar_pattern_refresh_updates_vector_caches(monkeypatch, tmp_path) -> None:
    calls = {"cache": 0}

    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: [])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-20")

    def fake_build(*args, **kwargs):
        calls["cache"] += 1
        return pd.DataFrame({"status": ["built", "cache_hit"]})

    monkeypatch.setattr(services, "build_vector_caches_parallel", fake_build)
    monkeypatch.setattr(services, "analyze_targets_by_threshold", lambda *args, **kwargs: {})

    payload = services.refresh_similar_pattern_analysis(force_vector_cache=True)

    assert calls["cache"] == 1
    assert payload["cache"]["rebuilt"] == 1
    assert payload["cache"]["reused"] == 1
    assert payload["cache"]["reference_library_policy"] == "weekly"
    assert payload["cache"]["reference_library_refreshed"] is True
    assert payload["cache"]["reference_library_reason"] == "forced"
    assert payload["cache"]["reference_library_minimum_refresh_age_days"] == 5
    assert payload["cache"]["target_vectors"] == "live_from_latest_daily_data"


def test_similar_pattern_refresh_reuses_current_weekly_library_but_analyzes_live_targets(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {"cache": 0, "analysis": 0}
    cache_root = tmp_path / "vector_cache"
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", cache_root)
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: ["002594.SZ"])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-21")

    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps(
            {
                "config_key": services.vector_cache_key(services.SIMILAR_PATTERN_CONFIG),
                "refreshed_at": datetime.now().isoformat(timespec="seconds"),
                "source_trade_date": "2026-07-20",
                "cached_files": 1,
            }
        ),
        encoding="utf-8",
    )

    def fake_build(*args, **kwargs):
        calls["cache"] += 1
        return pd.DataFrame()

    def fake_analyze(*args, **kwargs):
        calls["analysis"] += 1
        assert kwargs["target_symbols"] == ["002594.SZ"]
        return {
            "002594.SZ": SimpleNamespace(
                target=SimpleNamespace(target_date=pd.Timestamp("2026-07-21"))
            )
        }

    monkeypatch.setattr(services, "build_vector_caches_parallel", fake_build)
    monkeypatch.setattr(services, "analyze_targets_by_threshold", fake_analyze)
    monkeypatch.setattr(
        services,
        "_similar_pattern_result_payload",
        lambda *args, **kwargs: {"target": {"symbol": "002594.SZ"}},
    )
    monkeypatch.setattr(services, "_attach_watchlist_strategy_hits", lambda payload: payload)

    payload = services.refresh_similar_pattern_analysis()

    assert calls == {"cache": 0, "analysis": 1}
    assert payload["cache"]["reference_library_refreshed"] is False
    assert payload["cache"]["reference_library_reason"] in {
        "waiting_for_friday_close",
        "waiting_for_friday_trade_close",
        "friday_close_window_already_refreshed",
        "minimum_refresh_age_not_reached",
    }
    assert payload["cache"]["reused"] == 1


def test_similar_pattern_refresh_rejects_missing_watchlist_targets(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: ["002594.SZ"])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-22")
    monkeypatch.setattr(
        services,
        "_similar_pattern_vector_cache_refresh_decision",
        lambda **kwargs: {
            "due": False,
            "reason": "weekly_cache_current",
            "metadata": {},
            "inferred_legacy": False,
            "refreshed_at": None,
            "cached_files": 0,
        },
    )
    monkeypatch.setattr(services, "analyze_targets_by_threshold", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="missing=1/1"):
        services.refresh_similar_pattern_analysis()


def test_similar_pattern_refresh_coalesces_concurrent_requests(monkeypatch) -> None:
    read_count = 0
    compute_count = 0
    read_lock = threading.Lock()
    second_request_started = threading.Event()
    computation_started = threading.Event()
    release_computation = threading.Event()
    results: list[dict] = []
    errors: list[BaseException] = []

    monkeypatch.setattr(services, "_SIMILAR_PATTERN_REFRESH_INFLIGHT", None)

    def fake_symbols() -> list[str]:
        nonlocal read_count
        with read_lock:
            read_count += 1
            if read_count >= 2:
                second_request_started.set()
        return ["002594.SZ"]

    def fake_refresh_once(symbols, progress_callback=None, *, force_vector_cache=False):
        nonlocal compute_count
        compute_count += 1
        computation_started.set()
        assert release_computation.wait(timeout=3)
        return {
            "generated_at": "2026-07-28T13:21:53",
            "results": [{"target": {"symbol": symbols[0]}}],
            "cache": {"backend": "generated"},
        }

    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", fake_symbols)
    monkeypatch.setattr(services, "_refresh_similar_pattern_analysis_once", fake_refresh_once)

    def invoke_refresh() -> None:
        try:
            results.append(services.refresh_similar_pattern_analysis())
        except BaseException as exc:  # pragma: no cover - diagnostic collection
            errors.append(exc)

    first = threading.Thread(target=invoke_refresh)
    second = threading.Thread(target=invoke_refresh)
    first.start()
    assert computation_started.wait(timeout=3)
    second.start()
    assert second_request_started.wait(timeout=3)
    release_computation.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert errors == []
    assert compute_count == 1
    assert len(results) == 2
    assert sum(bool(item["cache"].get("coalesced")) for item in results) == 1


def test_similar_pattern_weekly_library_waits_for_friday_close_after_seven_days(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-14T08:00:00"}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 21, 8, 0, 0)
    )

    assert decision["due"] is False
    assert decision["reason"] == "waiting_for_friday_close"
    assert decision["next_refresh_at"] == "2026-07-24T15:00:00"


def test_similar_pattern_weekly_library_becomes_due_on_friday_after_close(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-19T15:01:00", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is True
    assert decision["reason"] == "friday_close_window"
    assert decision["refresh_age_days"] == 5
    assert decision["minimum_refresh_age_days"] == 5


def test_similar_pattern_weekly_library_waits_until_minimum_age_on_friday(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-20T20:25:57", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is False
    assert decision["reason"] == "minimum_refresh_age_not_reached"
    assert decision["refresh_age_days"] < 5


def test_similar_pattern_weekly_library_waits_when_friday_is_not_trade_date(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-18T15:01:00", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-23",
    )

    assert decision["due"] is False
    assert decision["reason"] == "waiting_for_friday_trade_close"


def test_similar_pattern_weekly_library_does_not_rebuild_twice_in_friday_window(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-24T15:05:00", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 16, 0, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is False
    assert decision["reason"] == "friday_close_window_already_refreshed"
    assert decision["next_refresh_at"] == "2026-07-31T15:00:00"


def test_similar_pattern_weekly_library_rebuilds_when_cache_count_changes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps(
            {
                "refreshed_at": "2026-07-17T15:01:00",
                "cached_files": 2,
                "errors": 0,
            }
        ),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is True
    assert decision["reason"] == "cache_file_count_changed"


def test_similar_pattern_cache_repair_waits_for_minimum_refresh_age(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps(
            {
                "refreshed_at": "2026-07-23T15:01:00",
                "cached_files": 2,
                "errors": 0,
            }
        ),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is False
    assert decision["reason"] == "minimum_refresh_age_not_reached"


def test_similar_pattern_weekly_library_does_not_advance_watermark_on_build_errors(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: [])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-23")
    monkeypatch.setattr(
        services,
        "build_vector_caches_parallel",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"symbol": "000001.SZ", "status": "built"},
                {"symbol": "000002.SZ", "status": "error", "error": "bad source"},
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="errors=1.*000002.SZ: bad source"):
        services.refresh_similar_pattern_analysis(force_vector_cache=True)

    metadata_path = (
        services._similar_pattern_vector_cache_state_dir()
        / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA
    )
    assert not metadata_path.exists()


def test_similar_pattern_payload_includes_optimized_decision_and_validation() -> None:
    cases = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "name": ["甲", "乙", "丙"],
            "industry": ["汽车整车", "汽车整车", "软件服务"],
            "date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-04"]),
            "similarity": [0.080, 0.075, 0.065],
            "fwd_1d": [0.02, 0.01, -0.01],
            "fwd_20d": [0.05, 0.02, -0.03],
            "fwd_60d": [0.10, 0.04, -0.06],
            "max_drawdown_60d": [-0.03, -0.05, -0.12],
        }
    )
    result = SimilarPatternResult(
        target=TargetSpec("002594.SZ", "比亚迪", pd.Timestamp("2025-01-02")),
        latest_snapshot={"close": 100.0, "dist_ma20": 1.0, "dist_ma60": 2.0, "drawdown_60d": -3.0, "vol_ratio20": 1.0},
        similar_cases=cases,
        forecast=summarize_forecast(cases),
        status_probs={"上升": 50.0, "震荡": 20.0, "下跌": 30.0, "高波动": 0.0},
        scan_summary={"matched_cases": 3},
    )
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-04", "2025-01-02"]), "market_regime": ["risk_on"] * 4})
    industry = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-02-02", "2025-01-02"]), "industry_regime": ["risk_on"] * 3})
    validation = {
        "calibrations": {"next_1d": {"x": [0, 100], "y": [0, 100]}},
        "model_selection": {
            "next_1d": {
                "selected": {
                    "source": "regime_industry",
                    "bearish_max": 49.0,
                    "bullish_min": 51.0,
                    "enable_risk_gate": False,
                }
            }
        },
        "summary": [{"symbol": "002594.SZ", "horizon": "next_1d", "coverage": 40.0}],
    }

    payload = services._similar_pattern_result_payload(
        result,
        profile={"industry": "汽车整车"},
        market_regime=market,
        industry_regime=industry,
        validation=validation,
    )

    assert payload["optimized_forecast"]
    assert payload["feature_freshness"] == {
        "target_date": "2025-01-02",
        "market_regime_date": "2025-01-02",
        "industry_regime_date": "2025-01-02",
        "exact_date_contract": True,
    }
    assert payload["decisions"][0]["signal"] in {"bullish", "bearish", "observe"}
    assert payload["optimization_summary"]["effective_sample_size"] > 0
    assert payload["validation_summary"][0]["coverage"] == 40.0
    assert payload["optimized_forecast"][0]["probability_source"] == "regime_industry"
    assert payload["decisions"][0]["bearish_max"] == 49.0
    top_cases = payload["top_cases"]
    ceiling = services._similarity_score_ceiling()
    assert top_cases[0]["similarity_score"] == round(
        np.log1p(services.SIMILARITY_SCORE_CONTRAST * top_cases[0]["forecast_weight"] / ceiling)
        / np.log1p(services.SIMILARITY_SCORE_CONTRAST)
        * 100,
        1,
    )
    assert top_cases[0]["similarity_score"] < 100.0
    assert [row["similarity_score"] for row in top_cases] == sorted(
        [row["similarity_score"] for row in top_cases], reverse=True
    )


def test_watchlist_strategy_hits_merge_only_strategy_workspaces(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"策略命中合并不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(services, "_latest_candidate_signal_date", lambda: "2026-07-15")
    monkeypatch.setattr(
        services,
        "_read_selector_snapshot",
        lambda *args, **kwargs: {
            "signal_date": "2026-07-15",
            "stocks": [
                {
                    "symbol": "002594.SZ",
                    "matched_families": ["B1", "强K"],
                    "matched_count": 2,
                    "selector_score": 80.0,
                    "opportunity_score": 80.0,
                    "holding_score": 70.0,
                    "best_profit_factor": 2.0,
                    "buy_score_source": "historical_return_model",
                    "hold_score_source": "historical_return_model",
                    "signals": [{"strategy_family": "B1"}],
                },
                {
                    "symbol": "002920.SZ",
                    "matched_families": ["强K/突破"],
                    "matched_count": 1,
                    "selector_score": 35.2,
                    "best_profit_factor": 1.0,
                    "signals": [
                        {
                            "strategy_family": "KEY_K",
                            "action_level": "谨慎观察",
                        }
                    ],
                },
            ],
        },
    )
    monkeypatch.setattr(
        services,
        "apply_selector_ranking_source",
        lambda rows, *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        services,
        "get_chan_model_strategy_plan",
        lambda **kwargs: {
            "signal_date": "2026-07-15",
            "candidates": [{"symbol": "002594.SZ", "rule_name": "主策略"}],
        },
    )
    monkeypatch.setattr(
        services,
        "get_long_stock_pool",
        lambda **kwargs: {
            "signal_date": "2026-07-15",
            "stocks": [{"ts_code": "002788.SZ", "state": "WATCH", "action": "观察"}],
        },
    )
    monkeypatch.setattr(
        services,
        "_read_daily_payload_cache",
        lambda path: {
            "asof": "2026-07-15",
            "records": [{"stock_code": "002788", "status": "证监会注册"}],
        },
    )

    hits = services._collect_watchlist_strategy_hits(["002594.SZ", "002788.SZ", "002920.SZ"])

    assert [item["strategy_key"] for item in hits["002594.SZ"]] == ["short", "chan"]
    assert [item["strategy_key"] for item in hits["002788.SZ"]] == ["long"]
    assert hits["002920.SZ"] == []
    assert hits["002594.SZ"][0]["detail"] == "B1 / 强K"


def test_attach_watchlist_strategy_hits_adds_badges_to_each_result(monkeypatch) -> None:
    payload = {
        "results": [
            {"target": {"symbol": "002594.SZ"}},
            {"target": {"symbol": "002788.SZ"}},
        ]
    }
    monkeypatch.setattr(
        services,
        "_collect_watchlist_strategy_hits",
        lambda symbols: {"002594.SZ": [{"strategy_key": "short"}], "002788.SZ": []},
    )

    enriched = services._attach_watchlist_strategy_hits(payload)

    assert enriched["results"][0]["strategy_hit_count"] == 1
    assert enriched["results"][1]["strategy_hit_count"] == 0


def test_watchlist_notes_are_saved_without_recalculating_scores(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["002594.SZ"],"notes":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"笔记保存不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail(f"笔记保存不应重新计算评分: {symbols}"),
    )

    payload = services.save_similar_pattern_watch_note(
        "002594.SZ",
        "操作计划：回踩 20 日线后分批关注",
    )

    stock = payload["stocks"][0]
    persisted = watchlist_path.read_text(encoding="utf-8")
    assert stock["note"] == "操作计划：回踩 20 日线后分批关注"
    assert stock["note_updated_at"]
    assert "opportunity_score" not in stock
    assert "holding_score" not in stock
    assert "回踩 20 日线" in persisted


def test_strategy_add_duplicate_preserves_existing_note(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    original = (
        '{"symbols":["002594.SZ"],"notes":{"002594.SZ":'
        '{"content":"手工计划：回踩观察","updated_at":"2026-07-16T20:00:00"}}}'
    )
    watchlist_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )

    payload = services.add_similar_pattern_watch_symbol("002594.SZ", note="7.17 触发 B1 策略")

    note = payload["stocks"][0]["note"]
    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert note == "手工计划：回踩观察"
    assert watchlist_path.read_text(encoding="utf-8") == original
    assert persisted["notes"]["002594.SZ"] == {
        "content": "手工计划：回踩观察",
        "updated_at": "2026-07-16T20:00:00",
    }


def test_watchlist_add_returns_without_calculating_scores(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["002594.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [
                {"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"},
                {"ts_code": "000001.SZ", "name": "平安银行", "industry": "银行"},
            ]
        ),
    )

    def fail_if_scores_run(symbols):
        raise AssertionError(f"add should not calculate scores synchronously: {symbols}")

    monkeypatch.setattr(services, "_watchlist_buy_hold_scores", fail_if_scores_run)

    payload = services.add_similar_pattern_watch_symbol("000001.SZ")

    assert [item["symbol"] for item in payload["stocks"]] == ["002594.SZ", "000001.SZ"]
    assert payload["stocks"][1]["name"] == "平安银行"
    assert "opportunity_score" not in payload["stocks"][1]


def test_watchlist_remove_returns_without_market_or_score_recalculation(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        json.dumps(
            {
                "symbols": ["002594.SZ", "000001.SZ"],
                "pinned": ["002594.SZ"],
                "notes": {"002594.SZ": {"content": "观察", "updated_at": "2026-07-30T10:00:00"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [
                {"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"},
                {"ts_code": "000001.SZ", "name": "平安银行", "industry": "银行"},
            ]
        ),
    )
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail("删除现有自选股不应重新校验行情"),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail("删除响应不应重新计算评分"),
    )

    payload = services.remove_similar_pattern_watch_symbol("002594.SZ")

    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert [item["symbol"] for item in payload["stocks"]] == ["000001.SZ"]
    assert persisted["symbols"] == ["000001.SZ"]
    assert persisted["pinned"] == []
    assert persisted["notes"] == {}


def test_watchlist_api_forwards_default_source_note(monkeypatch) -> None:
    captured = {}

    def fake_add(symbol, note=""):
        captured.update({"symbol": symbol, "note": note})
        return {"stocks": []}

    monkeypatch.setattr(webapp_api, "add_similar_pattern_watch_symbol", fake_add)

    response = client.post(
        "/api/similar-patterns/watchlist",
        json={"symbol": "002594.SZ", "note": "7.15 配债股"},
    )

    assert response.status_code == 200
    assert captured == {"symbol": "002594.SZ", "note": "7.15 配债股"}


def test_watchlist_api_can_return_lightweight_profiles(monkeypatch) -> None:
    captured = {}

    def fake_get(*, include_scores=True):
        captured["include_scores"] = include_scores
        return {"stocks": []}

    monkeypatch.setattr(webapp_api, "get_similar_pattern_watchlist", fake_get)

    response = client.get("/api/similar-patterns/watchlist?include_scores=false")

    assert response.status_code == 200
    assert captured == {"include_scores": False}


def test_watchlist_note_rejects_stock_outside_watchlist(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["002594.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    with pytest.raises(ValueError, match="不在自选池"):
        services.save_similar_pattern_watch_note("000001.SZ", "观察")


def test_watchlist_independent_reminders_with_notes_are_persisted(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["002594.SZ"],"notes":{},"pinned":[]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"提醒保存不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail(f"提醒保存不应重新计算评分: {symbols}"),
    )

    payload = services.save_similar_pattern_watch_alerts(
        "002594.SZ",
        {
            "enabled": True,
            "reminders": [
                {
                    "id": "reminder-1",
                    "note": "突破前高并放量",
                    "conditions": [
                        {
                            "id": "price-high",
                            "conjunction": "and",
                            "kind": "price",
                            "operator": "gt",
                            "value": 118.5,
                        },
                        {
                            "id": "volume",
                            "conjunction": "and",
                            "kind": "indicator",
                            "indicator": "vol_ratio20",
                            "operator": "gt",
                            "value": 1.8,
                        },
                    ],
                },
                {
                    "id": "reminder-2",
                    "note": "回撤或评分进入关注区",
                    "conditions": [
                        {
                            "id": "drawdown",
                            "conjunction": "and",
                            "kind": "indicator",
                            "indicator": "kdj_daily_j",
                            "operator": "lt",
                            "value": -10,
                        },
                        {
                            "id": "buy-score",
                            "conjunction": "or",
                            "kind": "indicator",
                            "indicator": "opportunity_score",
                            "operator": "gt",
                            "value": 75,
                        },
                    ],
                },
            ],
        },
    )

    alerts = payload["stocks"][0]["alerts"]
    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert alerts["enabled"] is True
    assert len(alerts["reminders"]) == 2
    assert alerts["reminders"][0]["note"] == "突破前高并放量"
    assert alerts["reminders"][1]["conditions"][1]["conjunction"] == "or"
    assert alerts["updated_at"]
    assert payload["stocks"][0]["alert_count"] == 2
    assert payload["stocks"][0]["alert_condition_count"] == 4
    assert persisted["alerts"]["002594.SZ"]["reminders"][0]["conditions"][1]["indicator"] == "vol_ratio20"
    assert persisted["alerts"]["002594.SZ"]["reminders"][1]["conditions"][0]["indicator"] == "kdj_daily_j"


def test_watchlist_alert_api_accepts_reminder_notes_and_condition_logic(monkeypatch) -> None:
    captured = {}

    def fake_save(symbol, config):
        captured.update({"symbol": symbol, "config": config})
        return {"stocks": []}

    monkeypatch.setattr(webapp_api, "save_similar_pattern_watch_alerts", fake_save)
    response = client.put(
        "/api/similar-patterns/watchlist/002594.SZ/alerts",
        json={
            "enabled": True,
            "reminders": [
                {
                    "id": "reminder-1",
                    "note": "价格或涨幅满足",
                    "conditions": [
                        {
                            "id": "p1",
                            "conjunction": "and",
                            "kind": "price",
                            "operator": "gt",
                            "value": 120,
                        },
                        {
                            "id": "i1",
                            "conjunction": "or",
                            "kind": "indicator",
                            "indicator": "kdj_daily_j",
                            "operator": "lt",
                            "value": -10,
                        },
                    ],
                }
            ],
        },
    )
    invalid = client.put(
        "/api/similar-patterns/watchlist/002594.SZ/alerts",
        json={
            "reminders": [
                {
                    "id": "bad",
                    "conditions": [
                        {
                            "id": "bad-condition",
                            "kind": "price",
                            "operator": "gte",
                            "value": 100,
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured["symbol"] == "002594.SZ"
    assert captured["config"]["reminders"][0]["note"] == "价格或涨幅满足"
    assert captured["config"]["reminders"][0]["conditions"][1]["conjunction"] == "or"
    assert captured["config"]["reminders"][0]["conditions"][1]["indicator"] == "kdj_daily_j"
    assert invalid.status_code == 422


def test_legacy_watchlist_alert_shape_is_discarded(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        json.dumps(
            {
                "symbols": ["002594.SZ"],
                "alerts": {
                    "002594.SZ": {
                        "enabled": True,
                        "logic": "and",
                        "price_rules": [{"id": "old", "operator": "gte", "value": 100}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    state = services._read_similar_pattern_watchlist_state()

    assert state["alerts"] == {}


def test_watchlist_order_and_pins_are_persisted(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["000001.SZ","000002.SZ","000003.SZ"],"notes":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"置顶或排序不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail(f"置顶或排序不应重新计算评分: {symbols}"),
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "甲", "industry": "银行"},
                {"ts_code": "000002.SZ", "name": "乙", "industry": "地产"},
                {"ts_code": "000003.SZ", "name": "丙", "industry": "制造"},
            ]
        ),
    )

    pinned_payload = services.set_similar_pattern_watch_pin("000002.SZ", True)
    reordered_payload = services.reorder_similar_pattern_watchlist(
        ["000002.SZ", "000003.SZ", "000001.SZ"]
    )

    assert [item["symbol"] for item in pinned_payload["stocks"]] == [
        "000002.SZ",
        "000001.SZ",
        "000003.SZ",
    ]
    assert pinned_payload["stocks"][0]["pinned"] is True
    assert [item["symbol"] for item in reordered_payload["stocks"]] == [
        "000002.SZ",
        "000003.SZ",
        "000001.SZ",
    ]
    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert persisted["symbols"] == ["000002.SZ", "000003.SZ", "000001.SZ"]
    assert persisted["pinned"] == ["000002.SZ"]


def test_watchlist_reorder_rejects_incomplete_symbol_set(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["000001.SZ","000002.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    with pytest.raises(ValueError, match="当前全部股票"):
        services.reorder_similar_pattern_watchlist(["000001.SZ"])


def test_daily_payload_freshness_uses_generated_date() -> None:
    today = date(2026, 7, 13)

    assert services._is_daily_payload_current({"generated_at": "2026-07-13T08:30:00"}, today=today)
    assert not services._is_daily_payload_current({"generated_at": "2026-06-28T22:52:04"}, today=today)
    assert not services._is_daily_payload_current({}, today=today)


def test_workspace_snapshot_uses_filesystem_and_nearest_prior_date(monkeypatch, tmp_path) -> None:
    class NoSqlStore:
        def __init__(self, config):
            self.config = config

    no_sql_config = type("NoSqlConfig", (), {"sql_url": None})()
    monkeypatch.setattr(services, "WEB_WORKSPACE_SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(services, "MarketDataStore", NoSqlStore)
    monkeypatch.setattr(services.MarketDataStoreConfig, "from_env", lambda **kwargs: no_sql_config)

    services._write_workspace_snapshot(
        "convertible_bond_grid_plan",
        "20260717",
        {"trade_date": "20260717", "candidates": [{"ts_code": "123001.SZ"}]},
        params={"limit": 18},
    )

    payload = services._read_workspace_snapshot(
        "convertible_bond_grid_plan",
        snapshot_date="2026-07-18",
        params={"limit": 18},
    )

    assert payload is not None
    assert payload["candidates"][0]["ts_code"] == "123001.SZ"
    assert payload["cache"]["backend"] == "filesystem"
    assert payload["cache"]["snapshot_date"] == "2026-07-17"
    assert payload["cache"]["requested_date"] == "2026-07-18"
    assert payload["cache"]["stale"] is True


def test_convertible_bond_plan_bootstraps_from_legacy_snapshot(monkeypatch, tmp_path) -> None:
    legacy_path = tmp_path / "convertible_bond_grid_plan.json"
    legacy_path.write_text(
        '{"trade_date":"20260717","generated_at":"2026-07-17T18:00:00","candidates":[]}',
        encoding="utf-8",
    )
    writes = []
    monkeypatch.setattr(services, "CONVERTIBLE_BOND_GRID_PLAN_PATH", legacy_path)
    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        services,
        "_write_filesystem_workspace_snapshot",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        services,
        "build_convertible_bond_grid_plan",
        lambda **kwargs: pytest.fail("tab read must not synchronously rebuild when a durable snapshot exists"),
    )

    payload = services.get_convertible_bond_grid_plan(trade_date="2026-07-18", limit=18)

    assert payload["cache"]["backend"] == "legacy_filesystem"
    assert payload["cache"]["snapshot_date"] == "2026-07-17"
    assert payload["cache"]["stale"] is True
    assert len(writes) == 1


def test_allotment_tab_serves_stale_daily_cache_without_blocking_refresh(monkeypatch, tmp_path) -> None:
    calls = []
    stale = {
        "generated_at": "2026-06-28T22:52:04",
        "records": [],
        "data_sources": {
            "stock_daily": {"requested": 0, "matched": 0, "error": None},
            "daily_basic": {"matched": 0, "error": None},
        },
    }

    monkeypatch.setattr(services, "CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH", tmp_path / "allotments.json")
    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *args, **kwargs: stale)
    monkeypatch.setattr(services, "_write_workspace_snapshot", lambda *args, **kwargs: None)

    def fake_build(**kwargs):
        calls.append(kwargs["refresh"])
        return {"generated_at": "2026-07-13T08:30:00", "records": [{"stock_code": "000001"}]}

    monkeypatch.setattr(services, "build_convertible_bond_allotment_payload", fake_build)

    payload = services.get_convertible_bond_allotments()

    assert calls == []
    assert payload["records"] == []
    assert payload["cache"]["stale"] is True
    assert payload["quality"]["status"] == "success"


def test_similar_pattern_tab_serves_stale_cache_without_blocking_refresh(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_analysis_cache",
        lambda: {"generated_at": "2026-06-28T22:52:04", "results": []},
    )

    def fake_refresh():
        calls.append(True)
        return {"generated_at": "2026-07-13T08:30:00", "results": []}

    monkeypatch.setattr(services, "refresh_similar_pattern_analysis", fake_refresh)

    payload = services.get_similar_pattern_analysis()

    assert calls == []
    assert payload["generated_at"].startswith("2026-06-28")
    assert payload["cache"]["stale"] is True


def test_similar_pattern_tab_without_cache_returns_immediately(monkeypatch) -> None:
    monkeypatch.setattr(services, "_read_similar_pattern_analysis_cache", lambda: None)
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_watchlist_symbols",
        lambda: ["002594.SZ"],
    )
    monkeypatch.setattr(
        services,
        "get_similar_pattern_watchlist",
        lambda *, include_scores=True: {
            "updated_at": "2026-07-30T20:00:00",
            "stocks": [{"symbol": "002594.SZ", "name": "比亚迪"}],
        },
    )
    monkeypatch.setattr(
        services,
        "refresh_similar_pattern_analysis",
        lambda: pytest.fail("打开自选池不应同步生成完整分析"),
    )

    payload = services.get_similar_pattern_analysis()

    assert payload["results"] == []
    assert payload["watchlist"][0]["symbol"] == "002594.SZ"
    assert payload["cache"]["missing"] is True


def test_similar_pattern_cache_filters_results_removed_from_watchlist(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_analysis_cache",
        lambda: {
            "generated_at": "2026-07-28T13:21:53",
            "watchlist": [
                {
                    "symbol": "002594.SZ",
                    "opportunity_score": 73.2,
                    "holding_score": 61.8,
                    "score_date": "2026-07-28",
                }
            ],
            "results": [
                {"target": {"symbol": "002594.SZ"}},
                {"target": {"symbol": "000792.SZ"}},
            ],
        },
    )
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_watchlist_symbols",
        lambda: ["002594.SZ"],
    )
    monkeypatch.setattr(
        services,
        "_similar_pattern_watchlist_profiles",
        lambda basic=None, include_scores=True: [{"symbol": "002594.SZ"}],
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        services,
        "_attach_watchlist_strategy_hits",
        lambda payload: payload,
    )

    payload = services.get_similar_pattern_analysis()

    assert [item["target"]["symbol"] for item in payload["results"]] == ["002594.SZ"]
    assert payload["watchlist"][0]["opportunity_score"] == 73.2
    assert payload["watchlist"][0]["holding_score"] == 61.8
    assert payload["cache"]["watchlist_changed"] is True


def test_refresh_watchlist_scores_updates_all_profiles_without_rebuilding_analysis(
    monkeypatch,
    tmp_path,
) -> None:
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-26T17:46:19",
                "watchlist": [
                    {
                        "symbol": "002594.SZ",
                        "opportunity_score": 70.0,
                        "holding_score": 60.0,
                    }
                ],
                "results": [
                    {"target": {"symbol": "002594.SZ"}},
                    {"target": {"symbol": "000792.SZ"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", analysis_path)
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        services,
        "_similar_pattern_watchlist_profiles",
        lambda basic=None, include_scores=True: [
            {
                "symbol": "002594.SZ",
                "opportunity_score": 73.2,
                "holding_score": 61.8,
                "score_date": "2026-08-27",
            },
            {
                "symbol": "600368.SH",
                "opportunity_score": 66.4,
                "holding_score": 58.1,
                "score_date": "2026-08-27",
            },
        ],
    )

    result = services.refresh_similar_pattern_watchlist_scores()

    saved = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert result == {
        "status": "success",
        "watchlist_count": 2,
        "scored_count": 2,
        "score_dates": ["2026-08-27"],
        "analysis_results_preserved": 1,
    }
    assert saved["generated_at"] == "2026-08-26T17:46:19"
    assert [item["symbol"] for item in saved["watchlist"]] == [
        "002594.SZ",
        "600368.SH",
    ]
    assert [item["target"]["symbol"] for item in saved["results"]] == [
        "002594.SZ"
    ]
    assert saved["watchlist_scores_generated_at"]


def test_refresh_watchlist_scores_rejects_partial_score_output(
    monkeypatch,
    tmp_path,
) -> None:
    analysis_path = tmp_path / "analysis.json"
    original = '{"generated_at":"2026-08-26T17:46:19","watchlist":[],"results":[]}'
    analysis_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", analysis_path)
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        services,
        "_similar_pattern_watchlist_profiles",
        lambda basic=None, include_scores=True: [
            {"symbol": "600368.SH", "score_date": "2026-08-27"}
        ],
    )

    with pytest.raises(RuntimeError, match="买入分或持有分缺失"):
        services.refresh_similar_pattern_watchlist_scores()

    assert analysis_path.read_text(encoding="utf-8") == original


def test_vector_cache_builder_uses_thread_pool_in_daemon_process(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        similar_patterns_module.mp,
        "current_process",
        lambda: type("Proc", (), {"daemon": True})(),
    )

    daily_dir = tmp_path / "daily"
    cache_dir = tmp_path / "cache"
    daily_dir.mkdir()
    cache_dir.mkdir()
    for symbol in ["000001.SZ", "000002.SZ"]:
        (daily_dir / f"{symbol}.parquet").write_text("", encoding="utf-8")

    worker_calls = []

    def fake_worker(task):
        worker_calls.append(task[0])
        return {"status": "cache_hit", "symbol": task[0], "vectors": 1}

    class InlineFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class InlineThreadPool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task):
            return InlineFuture(fn(task))

    monkeypatch.setattr(similar_patterns_module, "_build_stock_vector_cache_worker", fake_worker)
    monkeypatch.setattr(similar_patterns_module, "ThreadPoolExecutor", InlineThreadPool)
    monkeypatch.setattr(
        similar_patterns_module,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("process pool should not be used")),
    )
    monkeypatch.setattr(similar_patterns_module, "as_completed", lambda futures: futures)

    result = similar_patterns_module.build_vector_caches_parallel(
        daily_dir=daily_dir,
        basic=pd.DataFrame(),
        config=services.SIMILAR_PATTERN_CONFIG,
        cache_dir=cache_dir,
        workers=2,
    )

    assert len(worker_calls) == 2
    assert set(result["status"]) == {"cache_hit"}


def test_isolated_similar_pattern_process_is_not_daemon(monkeypatch) -> None:
    captured = {}

    class FakeQueue:
        def __init__(self):
            self._messages = [
                {"type": "result", "ok": True, "payload": {"generated_at": "2026-07-14T17:20:00", "results": []}}
            ]

        def get_nowait(self):
            if not self._messages:
                raise queue.Empty
            return self._messages.pop(0)

    class FakeProcess:
        pid = 12345

        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon
            self._alive = False

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

        def terminate(self):
            return None

        def kill(self):
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args, daemon):
            return FakeProcess(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(services.mp, "get_context", lambda _: FakeContext())
    monkeypatch.setattr(services, "_register_active_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "_clear_active_worker", lambda *args, **kwargs: None)

    payload = services._run_similar_pattern_analysis_isolated(timeout_seconds=1)

    assert captured["daemon"] is False
    assert payload["generated_at"] == "2026-07-14T17:20:00"


def test_isolated_similar_pattern_process_forwards_progress(monkeypatch) -> None:
    progress_messages = []

    class FakeQueue:
        def __init__(self):
            self._messages = [
                {"type": "progress", "message": "vector cache 200/5538 files usable=184 elapsed=14.4s"},
                {"type": "result", "ok": True, "payload": {"generated_at": "2026-07-16T18:40:00", "results": []}},
            ]

        def get_nowait(self):
            if not self._messages:
                raise queue.Empty
            return self._messages.pop(0)

    class FakeProcess:
        pid = 12345

        def __init__(self, *, target, args, daemon):
            self._alive = False

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

        def terminate(self):
            return None

        def kill(self):
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args, daemon):
            return FakeProcess(target=target, args=args, daemon=daemon)

    def fake_set_refresh_progress(**kwargs):
        progress_messages.append(kwargs["message"])

    monkeypatch.setattr(services.mp, "get_context", lambda _: FakeContext())
    monkeypatch.setattr(services, "_register_active_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "_clear_active_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "_set_refresh_progress", fake_set_refresh_progress)

    payload = services._run_similar_pattern_analysis_isolated(timeout_seconds=1)

    assert payload["generated_at"] == "2026-07-16T18:40:00"
    assert progress_messages == ["相似走势决策台：vector cache 200/5538 files usable=184 elapsed=14.4s"]
