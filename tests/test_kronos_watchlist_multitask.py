from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from quant.research.kronos_watchlist_multitask import (  # noqa: E402
    FusionModel,
    bucket_labels,
    calendar_features,
    event_arrays,
    fit_scaler,
    fold_masks,
    multitask_loss,
    normalize_history_windows,
    transform_factors,
)


def test_bucket_boundaries_and_economic_events_agree() -> None:
    values = np.array([-4, -3, -1.01, -1, -.1, 0, .1, 1, 3, 4])
    np.testing.assert_array_equal(bucket_labels(values), [0, 1, 1, 2, 3, 3, 4, 5, 6, 6])
    probs = np.eye(7)[bucket_labels(values)]
    probability, outcome = event_arrays(np.stack([probs, probs], axis=1), np.stack([values, values], axis=1), "open_down1")
    np.testing.assert_array_equal(probability, outcome)
    np.testing.assert_array_equal(outcome, values < -1)


def test_target_calendar_is_known_date_only_and_encodes_holiday_gap() -> None:
    features = calendar_features(pd.to_datetime(["2025-09-30"]), pd.to_datetime(["2025-10-09"]))
    np.testing.assert_array_equal(features[:, :3], [[9, 8, 3]])
    assert features[0, 3] == 9
    with pytest.raises(ValueError):
        calendar_features(pd.to_datetime(["2025-09-30"]), pd.to_datetime(["2025-09-29"]))


def test_missing_amount_is_neutral_without_dropping_valid_price_history() -> None:
    windows = np.arange(2 * 90 * 6, dtype=np.float32).reshape(2, 90, 6)
    expected = normalize_history_windows(windows)
    windows[0, :, 5] = np.nan
    windows[1, :30, 5] = np.nan
    actual = normalize_history_windows(windows)
    assert np.isfinite(actual).all()
    assert (actual[0, :, 5] == 0).all()
    assert (actual[1, :30, 5] == 0).all()
    np.testing.assert_array_equal(expected[:, :, :5], actual[:, :, :5])


def test_scaler_admission_and_statistics_only_depend_on_training_rows() -> None:
    train = np.array([[1., 2., np.nan], [2., 2., np.nan], [3., 2., 1.], [4., 2., np.nan]])
    scaler = fit_scaler(train)
    assert scaler["indices"] == [0]
    future = np.array([[10000., 2., 5.], [np.nan, 2., 5.]])
    result = transform_factors(future, scaler)
    assert result.shape == (2, 2)
    assert result[0, 0] == 8
    assert result[1, 0] == 0 and result[1, 1] == 1
    assert fit_scaler(train) == scaler


def test_fold_masks_purge_boundaries_and_separate_calibration_selection() -> None:
    dates = pd.bdate_range("2022-12-28", "2024-01-05")
    frame = pd.DataFrame({"origin_date": dates[:-1], "target_date": dates[1:]})
    masks = fold_masks(frame, {"train_start": "2015-01-01", "train_end": "2022-12-31", "validation_year": 2023, "test_start": "2024-01-01", "test_end": "2024-12-31"})
    assert np.stack(list(masks.values())).sum(axis=0).max() <= 1
    assert all(mask.any() for mask in masks.values())
    assert frame.loc[masks["calibration"], "origin_date"].min().month == 7
    assert frame.loc[masks["selection"], "origin_date"].min().month == 10


def test_fusion_ablations_calendar_and_loss_gradients() -> None:
    model = FusionModel(8, 3, "E", np.array([0, 1, -1, 0]), np.full((2, 7), 1 / 7))
    seq, factors = torch.randn(5, 8), torch.randn(5, 6)
    dates = torch.tensor([[1, 2, 3, 1]] * 5, dtype=torch.float32)
    prices, logits = model(seq, factors, dates)
    assert prices.shape == (5, 4) and logits.shape == (5, 2, 7)
    assert (prices[:, 1] >= prices[:, [0, 3]].max(dim=1).values).all()
    loss = multitask_loss(prices, logits, torch.ones(5, 4), torch.ones(5, 2, dtype=torch.long), .3)
    loss.backward()
    assert model.bins.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    reg_only = multitask_loss(prices.detach(), logits.detach(), torch.ones(5, 4), torch.ones(5, 2, dtype=torch.long), 0)
    assert reg_only < loss
    with pytest.raises(ValueError):
        FusionModel(8, 3, "unknown", np.array([0, 1, -1, 0]), np.full((2, 7), 1 / 7))


def test_disabled_calendar_and_sequence_branches_do_not_affect_outputs(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    seq, factors = torch.randn(5, 8), torch.randn(5, 6)
    dates = torch.tensor([[1, 2, 3, 1]] * 5, dtype=torch.float32)
    for variant in ("B", "F"):
        model = FusionModel(8, 3, variant, np.array([0, 1, -1, 0]), np.full((2, 7), 1 / 7)).eval()
        torch.nn.init.normal_(model.prices.weight)
        torch.nn.init.normal_(model.bins.weight)
        reference = model(seq, factors, dates)
        actual = model(seq, factors, dates * 0 + 1) if variant == "B" else model(seq * 100, factors, dates)
        for expected, result in zip(reference, actual):
            torch.testing.assert_close(result, expected)
        path = tmp_path / f"{variant}.safetensors"
        safetensors.save_file(model.state_dict(), str(path))
        restored = FusionModel(8, 3, variant, np.array([0, 1, -1, 0]), np.full((2, 7), 1 / 7)).eval()
        restored.load_state_dict(safetensors.load_file(str(path)))
        for expected, result in zip(reference, restored(seq, factors, dates)):
            torch.testing.assert_close(result, expected)


def test_canonical_factor_history_does_not_change_when_future_is_appended() -> None:
    from quant.features.project_factor_layer import calculate_project_factor_frame
    from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

    dates = pd.bdate_range("2023-01-02", periods=280)
    close = 10 + np.arange(280) * .01 + np.sin(np.arange(280)) * .05
    daily = pd.DataFrame({"date": dates, "open": close * .99, "high": close * 1.01,
                          "low": close * .98, "close": close, "pre_close": np.r_[close[0], close[:-1]],
                          "pct_chg": np.r_[0, 100 * (close[1:] / close[:-1] - 1)],
                          "volume": 1000 + np.arange(280), "amount": 10000 + np.arange(280)})
    before = calculate_project_factor_frame(daily.iloc[:250], "002594.SZ")
    after = calculate_project_factor_frame(daily, "002594.SZ")
    pd.testing.assert_frame_equal(before[PROJECT_FACTOR_COLUMNS], after.loc[:249, PROJECT_FACTOR_COLUMNS], check_dtype=False)
