from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.research.kronos_walk_forward import (
    FoldSpec,
    KlineWindowDataset,
    assert_prediction_key_parity,
    calculate_metrics,
    evaluation_origins,
    load_daily_frame,
    window_start_indices,
)


def make_daily_frame(periods: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=periods)
    close = np.linspace(10.0, 20.0, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.4,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, periods),
            "amount": np.linspace(10_000.0, 40_000.0, periods),
        }
    )


def test_load_daily_frame_maps_turnover_and_sorts_dates(tmp_path: Path) -> None:
    raw = make_daily_frame(5).rename(columns={"amount": "turnover"}).iloc[::-1]
    path = tmp_path / "daily.parquet"
    raw.to_parquet(path, index=False)

    loaded = load_daily_frame(path, "000001")

    assert loaded["date"].is_monotonic_increasing
    assert loaded["amount"].tolist() == sorted(raw["turnover"].tolist())
    assert loaded["symbol"].eq("000001").all()


def test_window_start_indices_keep_entire_training_target_inside_range() -> None:
    frame = make_daily_frame(150)
    start = frame.iloc[10]["date"]
    end = frame.iloc[99]["date"]

    starts = window_start_indices(frame, start, end, window=21, stride=7)

    assert starts[0] == 10
    assert all(frame.iloc[index]["date"] >= start for index in starts)
    assert all(frame.iloc[index + 20]["date"] <= end for index in starts)


def test_evaluation_origins_keep_decision_and_all_targets_in_test_period() -> None:
    frame = make_daily_frame(180)
    fold = FoldSpec(
        name="synthetic",
        train_start=frame.iloc[0]["date"],
        train_end=frame.iloc[69]["date"],
        validation_start=frame.iloc[70]["date"],
        validation_end=frame.iloc[99]["date"],
        test_start=frame.iloc[100]["date"],
        test_end=frame.iloc[159]["date"],
    )

    origins = evaluation_origins(frame, fold, lookback=30, horizon=10, stride=5)

    assert origins
    assert all(frame.iloc[item.origin_index]["date"] >= fold.test_start for item in origins)
    assert all(frame.iloc[item.target_end_index]["date"] <= fold.test_end for item in origins)
    assert all(item.context_start_index == item.origin_index - 29 for item in origins)


def test_kline_window_dataset_normalizes_each_feature() -> None:
    frame = make_daily_frame(40)
    dataset = KlineWindowDataset(frame, [0], window=30)

    values, stamps = dataset[0]

    assert values.shape == (30, 6)
    assert stamps.shape == (30, 5)
    assert np.allclose(values.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(values.std(axis=0), 1.0, atol=1e-3)


def make_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, multiplier in (("random_walk", 1.0), ("pretrained", 1.05), ("finetuned", 1.1)):
        for origin_number, origin_date in enumerate(pd.to_datetime(["2026-01-02", "2026-01-16"])):
            current = 10.0 + origin_number
            for step in (1, 2):
                actual = current * (1.0 + 0.05 * step)
                rows.append(
                    {
                        "fold": "fold_1",
                        "variant": variant,
                        "origin_date": origin_date,
                        "target_date": origin_date + pd.offsets.BDay(step),
                        "step": step,
                        "current_close": current,
                        "predicted_close": current * multiplier,
                        "actual_close": actual,
                    }
                )
    return pd.DataFrame(rows)


def test_metrics_use_identical_keys_and_finetuned_beats_random_walk() -> None:
    predictions = make_predictions()

    assert_prediction_key_parity(predictions)
    metrics = calculate_metrics(predictions, transaction_cost=0.002)

    all_rows = metrics.loc[metrics["fold"] == "ALL"].set_index("variant")
    assert all_rows.loc["finetuned", "path_mae_pct"] < all_rows.loc[
        "random_walk", "path_mae_pct"
    ]
    assert all_rows.loc["finetuned", "mae_skill_vs_random_walk"] > 0
    assert all_rows.loc["finetuned", "terminal_direction_accuracy"] == 1.0


def test_prediction_key_parity_rejects_missing_model_rows() -> None:
    predictions = make_predictions()
    broken = predictions.drop(predictions.index[-1])

    with pytest.raises(ValueError, match="prediction key mismatch"):
        assert_prediction_key_parity(broken)
