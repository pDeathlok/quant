"""Price/volume-only causal Kronos encoders, with trained end-of-sequence pooling."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PRICE_VOLUME = ("open", "high", "low", "close", "volume")
SCALES = ("daily", "weekly", "monthly")


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    bars = frame.loc[:, ["date", *PRICE_VOLUME]].copy()
    bars["date"] = pd.to_datetime(bars.date).dt.normalize()
    bars = bars.sort_values("date").reset_index(drop=True)
    if bars.date.duplicated().any() or bars.date.isna().any():
        raise ValueError("bar dates must be unique and valid")
    values = bars.loc[:, PRICE_VOLUME].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("invalid price/volume")
    if (bars.high < bars[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("invalid candle high")
    if (bars.low > bars[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("invalid candle low")
    return bars


def period_bars(bars: pd.DataFrame, scale: str) -> pd.DataFrame:
    """Availability is next period start, also safe for holidays and suspensions."""
    if scale == "daily":
        return bars.assign(available_at=bars.date)
    if scale not in {"weekly", "monthly"}:
        raise ValueError(f"unknown scale: {scale}")
    periods = bars.date.dt.to_period("W-SUN" if scale == "weekly" else "M")
    grouped = bars.assign(period=periods).groupby("period", sort=True)
    result = grouped.agg(
        date=("date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    result["available_at"] = (result.index + 1).start_time
    return result.reset_index(drop=True)


def normalize_context(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Only context statistics; OHLC share an anchor, volume uses log ratios."""
    values = np.asarray(values, dtype=np.float32)
    anchor = np.array([values[-1, 3]] * 4 + [max(float(values[:, 4].mean()), 1.0)])
    relative = np.log(np.maximum(values, 1e-8) / anchor)
    center = relative.mean(axis=0)
    spread = np.maximum(relative.std(axis=0), 0.05)
    normalized = np.clip((relative - center) / spread, -8, 8)
    return normalized.astype(np.float32), anchor, np.stack([center, spread])


def context_and_target(
    period: pd.DataFrame, end: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    if end < length - 1 or end + 1 >= len(period):
        raise ValueError("insufficient pretraining context/target")
    x, anchor, stats = normalize_context(
        period.iloc[end - length + 1 : end + 1][list(PRICE_VOLUME)].to_numpy()
    )
    target = period.iloc[end + 1][list(PRICE_VOLUME)].to_numpy(dtype=np.float32)
    y = np.clip((np.log(np.maximum(target, 1e-8) / anchor) - stats[0]) / stats[1], -8, 8)
    return x, y.astype(np.float32)


def asof_windows(
    periods: dict[str, pd.DataFrame], date: pd.Timestamp, lengths: dict[str, int]
) -> list[np.ndarray] | None:
    windows = []
    for scale in SCALES:
        period = periods[scale]
        end = int(period.available_at.searchsorted(pd.Timestamp(date), side="right"))
        length = lengths[scale]
        if end < length:
            return None
        x, _, _ = normalize_context(period.iloc[end - length : end][list(PRICE_VOLUME)].to_numpy())
        windows.append(x)
    return windows


class KronosEOS(nn.Module):
    """Original Kronos blocks; continuous OHLCV projection and explicit EOS tokens.

    No tokenizer, calendar, symbol embeddings or external pretrained weights.
    Smaller dimensions reduce research cost without changing the block structure.
    """

    def __init__(
        self, source: Path, hidden: int = 128, layers: int = 4, heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        source = source.resolve()
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        module = importlib.import_module("model.module")
        if Path(module.__file__).resolve() != source / "model/module.py":
            raise ValueError("imported a different Kronos source tree")
        self.project = nn.Linear(5, hidden)
        self.scale = nn.Embedding(3, hidden)
        self.eos = nn.Parameter(torch.randn(3, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [
                module.TransformerBlock(hidden, heads, hidden * 4, dropout, dropout, dropout)
                for _ in range(layers)
            ]
        )
        self.norm = module.RMSNorm(hidden)
        self.next_bar = nn.Linear(hidden, 5)
        self.hidden = hidden

    def forward(self, values: torch.Tensor, scale: int) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != 5 or scale not in range(3):
            raise ValueError("expected batch x time x five OHLCV channels and a scale")
        hidden = self.project(values)
        eos = self.eos[scale].expand(len(values), -1, -1)
        hidden = torch.cat([hidden, eos], dim=1) + self.scale.weight[scale]
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)[:, -1]

    def concatenate(self, windows: list[torch.Tensor]) -> torch.Tensor:
        if len(windows) != 3:
            raise ValueError("daily, weekly and monthly windows required")
        return torch.cat([self(x, scale) for scale, x in enumerate(windows)], dim=-1)


class SequenceSubscores(nn.Module):
    """Three outputs: net return, profit logit and downside magnitude."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def fit_subscores(
    embeddings: np.ndarray, returns: np.ndarray, *, epochs: int = 15, seed: int = 20261021
) -> SequenceSubscores:
    if len(embeddings) != len(returns) or len(returns) < 30:
        raise ValueError("insufficient supervised samples")
    torch.manual_seed(seed)
    model = SequenceSubscores(embeddings.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    # Targets scaled to typical daily percentage points; never fit target scalers on test.
    y = torch.from_numpy(np.asarray(returns, dtype=np.float32))
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for batch in np.array_split(
            generator.permutation(len(x)), max(1, int(np.ceil(len(x) / 256)))
        ):
            out = model(x[batch])
            loss = (
                nn.functional.smooth_l1_loss(out[:, 0], y[batch] * 100)
                + nn.functional.binary_cross_entropy_with_logits(out[:, 1], (y[batch] > 0).float())
                + nn.functional.smooth_l1_loss(out[:, 2], (-y[batch]).clamp(min=0) * 100)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def predict_subscores(model: SequenceSubscores, values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        parts = [
            model(torch.from_numpy(np.asarray(part, dtype=np.float32))).numpy()
            for part in np.array_split(values, max(1, int(np.ceil(len(values) / 1024))))
            if len(part)
        ]
    result = np.concatenate(parts)
    result[:, 0] /= 100
    result[:, 1] = 1 / (1 + np.exp(-np.clip(result[:, 1], -30, 30)))
    result[:, 2] = np.maximum(result[:, 2], 0) / 100
    return result
