import numpy as np
import pandas as pd
import pytest

from quant.research.holding_event_probability import (
    add_t0_interactions,
    choose_high_confidence,
    event_labels,
    probability_metrics,
    threshold_metrics,
)


def test_t0_interactions_ignore_future_and_handle_flat_price_range():
    frame = pd.DataFrame(
        {
            "tail_return": [0.02, 0.0],
            "tail_gap": [0.01, 0.0],
            "tail_drawdown_from_high": [1.02 / 1.04 - 1, 0.0],
            "tail_rebound_from_low": [1.02 / 0.99 - 1, 0.0],
            "tail_price_to_vwap": [0.01, 0.0],
            "tail_range": [0.05, 0.0],
            "tail_turnover_est_pct": [3.0, 0.0],
            "lag1_turnover_rate": [2.0, 0.0],
            "tail_volume_vs_lag1_ma5": [1.5, 0.0],
            "tail_volume_vs_lag1_ma20": [1.2, 0.0],
            "lag1_daily_range": [0.02, 0.0],
            "lag1_daily_return_mean_20": [0.001, 0.0],
            "lag1_daily_gap_std_20": [0.01, 0.0],
            "tail_close_position": [0.6, 0.5],
            "y_high": [0.1, -0.1],
        }
    )
    actual = add_t0_interactions(frame).filter(regex="^t0x_")
    changed = frame.assign(y_high=[-0.9, 0.9])
    pd.testing.assert_frame_equal(actual, add_t0_interactions(changed).filter(regex="^t0x_"))
    assert actual.loc[0, "t0x_body_range_signed"] == pytest.approx(0.2)
    assert actual.loc[0, "t0x_range_expansion"] == pytest.approx(2.5)
    assert np.isnan(actual.loc[1, "t0x_body_range_signed"])
    assert not np.isinf(actual.to_numpy()).any()


def test_open_down_and_intraday_up_can_coexist_without_intraday_down_targets():
    frame = pd.DataFrame(
        {"y_high": [0.04, np.nan], "y_open": [-0.04, np.nan], "y_close": [0.0, np.nan]}
    )
    labels = event_labels(frame)
    assert labels.loc[0, "high_up3"] == 1 and labels.loc[0, "open_down3"] == 1
    assert not any(c.startswith("low_down") for c in labels)
    assert len(labels.columns) == 15
    assert labels.loc[0, "close_up1"] == 0 and labels.loc[0, "close_down1"] == 0
    assert labels.iloc[1].isna().all()


def test_probabilities_calibration_and_boundaries():
    metrics, bins = probability_metrics([0, 1], [0, 1])
    assert metrics["brier"] == 0 and metrics["ece"] == 0
    assert sum(b["count"] for b in bins) == 2
    with pytest.raises(ValueError):
        probability_metrics([0, 1], [0, 1.1])


def test_open_and_close_labels_keep_observation_time_and_missingness_separate():
    frame = pd.DataFrame(
        {
            "y_open": [0.012, -0.024, np.nan],
            "y_high": [0.035, 0.015, 0.035],
            "y_close": [-0.025, 0.034, -0.014],
        }
    )
    labels = event_labels(frame)
    assert labels.loc[0, "open_up1"] == 1
    assert labels.loc[0, "open_up2"] == 0
    assert labels.loc[0, "close_down2"] == 1
    assert labels.loc[1, "open_down2"] == 1
    assert labels.loc[1, "close_up3"] == 1
    assert labels.loc[2].filter(like="open_").isna().all()
    assert labels.loc[2, "high_up3"] == 1
    assert labels.loc[2, "close_down1"] == 1


def test_high_accuracy_without_lift_is_not_selected():
    curves = pd.DataFrame(
        {
            "feature_group": ["t0"],
            "threshold": [0.8],
            "signals": [500],
            "signal_dates": [50],
            "precision": [0.9],
            "precision_low90": [0.85],
            "lift": [1.05],
            "coverage": [0.5],
        }
    )
    assert choose_high_confidence(curves) is None


def test_threshold_no_daily_top_k_and_block_intervals():
    days = pd.bdate_range("2025-01-01", periods=40)
    frame = pd.DataFrame(
        {
            "date": np.repeat(days, 4),
            "outcome": np.tile([1.0, 1.0, 0.0, 0.0], 40),
            "probability": np.tile([0.95, 0.9, 0.2, 0.1], 40),
        }
    )
    curve = threshold_metrics(frame, [0.8, 0.99], bootstrap=100)
    assert curve.iloc[0].signals == 80 and curve.iloc[0].precision == 1
    assert curve.iloc[0].coverage == 0.5 and curve.iloc[0].lift == 2
    assert curve.iloc[0].precision_low90 == 1
    assert curve.iloc[1].signals == 0 and np.isnan(curve.iloc[1].precision)


def test_nested_event_probability_projection():
    from quant.research.holding_event_probability import coherent_event_probabilities

    f = pd.DataFrame(
        {
            "high_up1": [0.6, 0.9],
            "high_up2": [0.8, 0.4],
            "high_up3": [0.5, 0.8],
            "close_up1": [0.8, 0.2],
            "close_up2": [0.7, 0.1],
            "close_up3": [0.6, 0.05],
            "close_down1": [0.6, 0.3],
            "close_down2": [0.5, 0.2],
            "close_down3": [0.4, 0.1],
            "open_up1": [0.9, 0.3],
            "open_up2": [0.8, 0.4],
            "open_up3": [0.7, 0.2],
            "open_down1": [0.9, 0.7],
            "open_down2": [0.8, 0.6],
            "open_down3": [0.7, 0.5],
        }
    )
    out = coherent_event_probabilities(f)
    assert out.loc[0, ["high_up1", "high_up2", "high_up3"]].tolist() == pytest.approx(
        [0.7, 0.7, 0.5]
    )
    assert out.loc[1, ["high_up1", "high_up2", "high_up3"]].tolist() == pytest.approx(
        [0.9, 0.6, 0.6]
    )
    assert (out.close_up1 + out.close_down1 <= 1).all()
    assert (out.close_up1 <= out.high_up1).all()
    for prefix in ("high_up", "open_up", "close_up", "open_down", "close_down"):
        assert (out[f"{prefix}1"] >= out[f"{prefix}2"]).all()
        assert (out[f"{prefix}2"] >= out[f"{prefix}3"]).all()
    for observation in ("open", "close"):
        assert (out[f"{observation}_up1"] + out[f"{observation}_down1"] <= 1).all()
        for level in (1, 2, 3):
            assert (out[f"{observation}_up{level}"] <= out[f"high_up{level}"]).all()
    assert (out.high_up1 + out.open_down1 > 1).any()
    pd.testing.assert_frame_equal(out, coherent_event_probabilities(out))
