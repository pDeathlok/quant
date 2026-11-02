from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from quant.research.kronos_next_day_prices import (  # noqa: E402
    NextDayPriceModel,
    coherent_prices,
    eligible_origins,
    historical_input,
    price_loss,
    split_origins,
)


def daily_frame() -> pd.DataFrame:
    close = np.arange(10, 22, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=len(close)),
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": close * 100, "amount": close * 1000,
    })


def test_future_prices_and_volume_cannot_change_normalized_input() -> None:
    frame = daily_frame()
    values, stamps = historical_input(frame, 5, 4)
    altered = frame.copy()
    altered.loc[6:, ["open", "high", "low", "close", "volume", "amount"]] *= 100
    other_values, other_stamps = historical_input(altered, 5, 4)
    np.testing.assert_array_equal(values, other_values)
    np.testing.assert_array_equal(stamps, other_stamps)
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-5)
    assert values.shape == (4, 6)


def test_samples_use_every_eligible_day_and_skip_missing_exchange_session() -> None:
    frame = daily_frame()
    calendar = pd.DatetimeIndex(frame.date)
    assert eligible_origins(frame, calendar, 4) == list(range(3, 11))
    missing = frame.drop(index=7).reset_index(drop=True)
    assert 6 not in eligible_origins(missing, calendar, 4)
    missing.loc[3, "volume"] = 0
    assert 3 not in eligible_origins(missing, calendar, 4)


def test_split_allows_past_context_but_purges_next_day_outside_boundary() -> None:
    frame = daily_frame()
    origins = eligible_origins(frame, pd.DatetimeIndex(frame.date), 4)
    selected = split_origins(frame, origins, frame.date.iloc[5], frame.date.iloc[9])
    assert selected == [5, 6, 7, 8]
    assert split_origins(frame, origins, frame.date.iloc[5], frame.date.iloc[9], 2) == [5, 7]


def test_loss_is_equal_weighted_four_prices_only_and_rejects_volume_target() -> None:
    prediction = torch.tensor([[0., 1., 2., 3.]], requires_grad=True)
    loss = price_loss(prediction, torch.zeros_like(prediction), torch.ones(4))
    assert loss.item() == pytest.approx((0 + .5 + 1.5 + 2.5) / 4)
    loss.backward()
    assert prediction.grad is not None
    with pytest.raises(ValueError, match="exactly four"):
        price_loss(torch.zeros(1, 6), torch.zeros(1, 6), torch.ones(6))


def test_candle_constraints_and_gradients_hold_after_price_conversion() -> None:
    raw = torch.tensor([[2., -1., -.5, -2.], [-3., 4., 1., 2.]], requires_grad=True)
    log_prices = coherent_prices(raw)
    prices = torch.exp(log_prices / 100) * 12
    assert torch.all(prices[:, 1] >= prices[:, [0, 3]].max(dim=1).values)
    assert torch.all(prices[:, 2] <= prices[:, [0, 3]].min(dim=1).values)
    assert torch.all(prices > 0)
    log_prices.square().sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert (raw.grad != 0).all()


class PairEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(16, 8)

    def forward(self, pair: list[torch.Tensor]) -> torch.Tensor:
        return self.table(pair[0]) + self.table(pair[1])


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d_model = 8
        self.embedding = PairEmbedding()
        self.time_emb = nn.Linear(5, 8)
        self.token_drop = nn.Identity()
        self.transformer = nn.ModuleList([nn.Linear(8, 8)])
        self.norm = nn.LayerNorm(8)


def test_regression_head_initialization_training_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")

    prior = np.array([0.1, 1.2, -1.3, 0.2])
    model = NextDayPriceModel(TinyBackbone(), prior)
    tokens = torch.ones(2, 6, dtype=torch.long)
    stamps = torch.zeros(2, 6, 5)
    prediction = model(tokens, tokens, stamps)
    np.testing.assert_allclose(prediction.detach().numpy(), np.tile(prior, (2, 1)), atol=1e-6)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        price_loss(model(tokens, tokens, stamps), torch.zeros(2, 4), torch.ones(4)).backward()
        optimizer.step()
    assert model.embedding.table.weight.grad.abs().sum() > 0
    path = tmp_path / "prices.safetensors"
    safetensors.save_file(model.state_dict(), str(path))
    restored = NextDayPriceModel(TinyBackbone(), prior)
    restored.load_state_dict(safetensors.load_file(str(path)))
    torch.testing.assert_close(restored(tokens, tokens, stamps), model(tokens, tokens, stamps))
