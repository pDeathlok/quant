"""Leakage-resistant rolling fine-tuning for the Kronos K-line foundation model."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KLINE_FEATURES = ("open", "high", "low", "close", "volume", "amount")
TIME_FEATURES = ("minute", "hour", "weekday", "day", "month")


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def validate(self) -> None:
        boundaries = (
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
            self.test_start,
            self.test_end,
        )
        if any(pd.isna(value) for value in boundaries):
            raise ValueError(f"fold {self.name} contains an invalid date")
        if not (
            self.train_start <= self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.test_end
        ):
            raise ValueError(f"fold {self.name} ranges are not strictly chronological")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    symbol: str
    data_path: Path
    output_dir: Path
    checkpoint_dir: Path
    lookback: int
    horizon: int
    transaction_cost: float


@dataclass(frozen=True)
class ModelSpec:
    source_dir: Path
    source_revision: str
    tokenizer_path: Path
    predictor_path: Path
    tokenizer_sha256: str
    predictor_sha256: str
    max_context: int
    device: str


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    batch_size: int
    train_stride: int
    validation_stride: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    clip_grad_norm: float
    early_stopping_patience: int
    seed: int


@dataclass(frozen=True)
class InferenceSpec:
    evaluation_stride: int
    batch_size: int
    temperature: float
    top_k: int
    top_p: float
    sample_count: int


@dataclass(frozen=True)
class WalkForwardConfig:
    experiment: ExperimentSpec
    model: ModelSpec
    training: TrainingSpec
    inference: InferenceSpec
    folds: tuple[FoldSpec, ...]

    def validate(self) -> None:
        if self.experiment.lookback <= 1 or self.experiment.horizon <= 0:
            raise ValueError("lookback must exceed one and horizon must be positive")
        if self.experiment.lookback + self.experiment.horizon + 1 > self.model.max_context:
            raise ValueError("training sequence exceeds the Kronos max_context")
        if not 0 <= self.experiment.transaction_cost < 1:
            raise ValueError("transaction_cost must be in [0, 1)")
        positive_values = {
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "train_stride": self.training.train_stride,
            "validation_stride": self.training.validation_stride,
            "evaluation_stride": self.inference.evaluation_stride,
            "inference_batch_size": self.inference.batch_size,
            "sample_count": self.inference.sample_count,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if not self.folds:
            raise ValueError("at least one fold is required")
        names = [fold.name for fold in self.folds]
        if len(names) != len(set(names)):
            raise ValueError("fold names must be unique")
        for fold in self.folds:
            fold.validate()


@dataclass(frozen=True)
class EvaluationOrigin:
    origin_index: int
    context_start_index: int
    target_end_index: int


class KlineWindowDataset:
    """Numpy-backed dataset; torch converts returned arrays in its default collator."""

    def __init__(
        self,
        frame: pd.DataFrame,
        start_indices: Sequence[int],
        window: int,
        clip: float = 5.0,
    ) -> None:
        self.frame = frame
        self.start_indices = tuple(int(value) for value in start_indices)
        self.window = int(window)
        self.clip = float(clip)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = self.start_indices[index]
        window = self.frame.iloc[start : start + self.window]
        values = window.loc[:, KLINE_FEATURES].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        normalized = np.clip((values - mean) / (std + 1e-5), -self.clip, self.clip)
        timestamps = pd.to_datetime(window["date"])
        stamps = np.column_stack(
            (
                timestamps.dt.minute,
                timestamps.dt.hour,
                timestamps.dt.weekday,
                timestamps.dt.day,
                timestamps.dt.month,
            )
        ).astype(np.float32)
        return normalized.astype(np.float32), stamps


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} is required")
    return value


def load_config(path: str | Path, project_root: Path = PROJECT_ROOT) -> WalkForwardConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")

    experiment = _section(raw, "experiment")
    model = _section(raw, "model")
    training = _section(raw, "training")
    inference = _section(raw, "inference")
    fold_rows = raw.get("folds")
    if not isinstance(fold_rows, list):
        raise ValueError("configuration folds must be a list")

    config = WalkForwardConfig(
        experiment=ExperimentSpec(
            name=str(experiment["name"]),
            symbol=str(experiment["symbol"]),
            data_path=_resolve_path(experiment["data_path"], project_root),
            output_dir=_resolve_path(experiment["output_dir"], project_root),
            checkpoint_dir=_resolve_path(experiment["checkpoint_dir"], project_root),
            lookback=int(experiment["lookback"]),
            horizon=int(experiment["horizon"]),
            transaction_cost=float(experiment["transaction_cost"]),
        ),
        model=ModelSpec(
            source_dir=_resolve_path(model["source_dir"], project_root),
            source_revision=str(model["source_revision"]),
            tokenizer_path=_resolve_path(model["tokenizer_path"], project_root),
            predictor_path=_resolve_path(model["predictor_path"], project_root),
            tokenizer_sha256=str(model["tokenizer_sha256"]),
            predictor_sha256=str(model["predictor_sha256"]),
            max_context=int(model["max_context"]),
            device=str(model.get("device", "auto")),
        ),
        training=TrainingSpec(
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            train_stride=int(training["train_stride"]),
            validation_stride=int(training["validation_stride"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            adam_beta1=float(training["adam_beta1"]),
            adam_beta2=float(training["adam_beta2"]),
            clip_grad_norm=float(training["clip_grad_norm"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            seed=int(training["seed"]),
        ),
        inference=InferenceSpec(
            evaluation_stride=int(inference["evaluation_stride"]),
            batch_size=int(inference["batch_size"]),
            temperature=float(inference["temperature"]),
            top_k=int(inference["top_k"]),
            top_p=float(inference["top_p"]),
            sample_count=int(inference["sample_count"]),
        ),
        folds=tuple(
            FoldSpec(
                name=str(row["name"]),
                train_start=pd.Timestamp(row["train_start"]),
                train_end=pd.Timestamp(row["train_end"]),
                validation_start=pd.Timestamp(row["validation_start"]),
                validation_end=pd.Timestamp(row["validation_end"]),
                test_start=pd.Timestamp(row["test_start"]),
                test_end=pd.Timestamp(row["test_end"]),
            )
            for row in fold_rows
        ),
    )
    config.validate()
    return config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_model_assets(config: WalkForwardConfig) -> dict[str, object]:
    source = config.model.source_dir
    predictor_file = config.model.predictor_path / "model.safetensors"
    tokenizer_file = config.model.tokenizer_path / "model.safetensors"
    for required in (source / "model" / "kronos.py", predictor_file, tokenizer_file):
        if not required.exists():
            raise FileNotFoundError(required)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    predictor_hash = file_sha256(predictor_file)
    tokenizer_hash = file_sha256(tokenizer_file)
    if revision != config.model.source_revision:
        raise ValueError(f"Kronos source revision mismatch: {revision}")
    if predictor_hash != config.model.predictor_sha256:
        raise ValueError(f"predictor SHA-256 mismatch: {predictor_hash}")
    if tokenizer_hash != config.model.tokenizer_sha256:
        raise ValueError(f"tokenizer SHA-256 mismatch: {tokenizer_hash}")
    return {
        "source_revision": revision,
        "predictor_sha256": predictor_hash,
        "tokenizer_sha256": tokenizer_hash,
        "predictor_bytes": predictor_file.stat().st_size,
        "tokenizer_bytes": tokenizer_file.stat().st_size,
    }


def load_daily_frame(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"daily data missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "turnover" in frame.columns:
        frame["amount"] = pd.to_numeric(frame["turnover"], errors="coerce")
    else:
        typical_price = frame.loc[:, ["open", "high", "low", "close"]].mean(axis=1)
        frame["amount"] = frame["volume"] * typical_price
    frame = (
        frame.loc[:, ["date", *KLINE_FEATURES]]
        .dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("daily data is empty after normalization")
    if (frame.loc[:, ["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("daily data contains non-positive prices")
    if (frame["volume"] < 0).any() or (frame["amount"] < 0).any():
        raise ValueError("daily data contains negative volume or amount")
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("daily dates must be unique and increasing")
    frame["symbol"] = symbol
    return frame


def window_start_indices(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: int,
    stride: int,
) -> list[int]:
    dates = pd.DatetimeIndex(frame["date"])
    positions = np.flatnonzero((dates >= start) & (dates <= end))
    if len(positions) < window:
        return []
    first, last = int(positions[0]), int(positions[-1])
    starts = list(range(first, last - window + 2, stride))
    return [
        value
        for value in starts
        if dates[value] >= start and dates[value + window - 1] <= end
    ]


def evaluation_origins(
    frame: pd.DataFrame,
    fold: FoldSpec,
    lookback: int,
    horizon: int,
    stride: int,
) -> list[EvaluationOrigin]:
    dates = pd.DatetimeIndex(frame["date"])
    eligible: list[EvaluationOrigin] = []
    for origin in range(lookback - 1, len(frame) - horizon):
        target_end = origin + horizon
        if dates[origin] < fold.test_start or dates[target_end] > fold.test_end:
            continue
        eligible.append(
            EvaluationOrigin(
                origin_index=origin,
                context_start_index=origin - lookback + 1,
                target_end_index=target_end,
            )
        )
    return eligible[::stride]


def _cap_evenly(values: Sequence[Any], maximum: int | None) -> list[Any]:
    items = list(values)
    if maximum is None or len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, num=maximum, dtype=int)
    return [items[index] for index in indices]


def audit_experiment(config: WalkForwardConfig) -> dict[str, object]:
    assets = audit_model_assets(config)
    frame = load_daily_frame(config.experiment.data_path, config.experiment.symbol)
    window = config.experiment.lookback + config.experiment.horizon + 1
    folds: list[dict[str, object]] = []
    for fold in config.folds:
        train = window_start_indices(
            frame,
            fold.train_start,
            fold.train_end,
            window,
            config.training.train_stride,
        )
        validation = window_start_indices(
            frame,
            fold.validation_start,
            fold.validation_end,
            window,
            config.training.validation_stride,
        )
        origins = evaluation_origins(
            frame,
            fold,
            config.experiment.lookback,
            config.experiment.horizon,
            config.inference.evaluation_stride,
        )
        if not train or not validation or not origins:
            raise ValueError(f"fold {fold.name} has insufficient data")
        folds.append(
            {
                "name": fold.name,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "evaluation_origins": len(origins),
                "last_train_target": str(frame.iloc[train[-1] + window - 1]["date"].date()),
                "first_test_origin": str(frame.iloc[origins[0].origin_index]["date"].date()),
                "last_test_target": str(frame.iloc[origins[-1].target_end_index]["date"].date()),
            }
        )
    return {
        "assets": assets,
        "data": {
            "rows": len(frame),
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "symbol": config.experiment.symbol,
        },
        "folds": folds,
    }


def _import_kronos(source_dir: Path) -> tuple[Any, Any, Any]:
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("model")
    return module.Kronos, module.KronosTokenizer, module.KronosPredictor


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_components(config: WalkForwardConfig, device: str) -> tuple[Any, Any, Any]:
    kronos_class, tokenizer_class, predictor_class = _import_kronos(config.model.source_dir)
    tokenizer = tokenizer_class.from_pretrained(str(config.model.tokenizer_path)).to(device)
    model = kronos_class.from_pretrained(str(config.model.predictor_path)).to(device)
    tokenizer.eval()
    model.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, predictor_class


def fine_tune_predictor(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    train_starts: Sequence[int],
    validation_starts: Sequence[int],
    config: WalkForwardConfig,
    device: str,
    checkpoint_path: Path,
    epochs_override: int | None = None,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    _set_seed(config.training.seed)
    window = config.experiment.lookback + config.experiment.horizon + 1
    train_dataset = KlineWindowDataset(frame, train_starts, window)
    validation_dataset = KlineWindowDataset(frame, validation_starts, window)
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.weight_decay,
    )
    epochs = epochs_override or config.training.epochs
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        train_loss = 0.0
        train_batches = 0
        for values, stamps in train_loader:
            values = values.to(device)
            stamps = stamps.to(device)
            with torch.no_grad():
                token_0, token_1 = tokenizer.encode(values, half=True)
            logits = model(token_0[:, :-1], token_1[:, :-1], stamps[:, :-1, :])
            loss, _, _ = model.head.compute_loss(
                logits[0], logits[1], token_0[:, 1:], token_1[:, 1:]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad_norm)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        with torch.no_grad():
            for values, stamps in validation_loader:
                values = values.to(device)
                stamps = stamps.to(device)
                token_0, token_1 = tokenizer.encode(values, half=True)
                logits = model(token_0[:, :-1], token_1[:, :-1], stamps[:, :-1, :])
                loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_0[:, 1:], token_1[:, 1:]
                )
                validation_loss += float(loss.detach().cpu())
                validation_batches += 1
        average_train = train_loss / max(train_batches, 1)
        average_validation = validation_loss / max(validation_batches, 1)
        improved = average_validation < best_loss
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_train,
                "validation_loss": average_validation,
                "seconds": time.monotonic() - started,
                "is_best": improved,
            }
        )
        if improved:
            best_loss = average_validation
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("fine-tuning did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_path))
    return pd.DataFrame(history)


def predict_origins(
    model: Any,
    tokenizer: Any,
    predictor_class: Any,
    frame: pd.DataFrame,
    origins: Sequence[EvaluationOrigin],
    fold_name: str,
    variant: str,
    config: WalkForwardConfig,
    device: str,
) -> pd.DataFrame:
    import torch

    predictor = predictor_class(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context=config.model.max_context,
    )
    model.eval()
    tokenizer.eval()
    records: list[dict[str, object]] = []
    batch_size = config.inference.batch_size
    for batch_number, offset in enumerate(range(0, len(origins), batch_size)):
        batch = origins[offset : offset + batch_size]
        contexts: list[pd.DataFrame] = []
        context_dates: list[pd.Series] = []
        target_dates: list[pd.Series] = []
        for origin in batch:
            context = frame.iloc[origin.context_start_index : origin.origin_index + 1]
            future = frame.iloc[origin.origin_index + 1 : origin.target_end_index + 1]
            contexts.append(context.loc[:, KLINE_FEATURES].copy())
            context_dates.append(pd.Series(context["date"].to_numpy()))
            target_dates.append(pd.Series(future["date"].to_numpy()))
        _set_seed(config.training.seed + batch_number)
        with torch.no_grad():
            forecasts = predictor.predict_batch(
                contexts,
                context_dates,
                target_dates,
                pred_len=config.experiment.horizon,
                T=config.inference.temperature,
                top_k=config.inference.top_k,
                top_p=config.inference.top_p,
                sample_count=config.inference.sample_count,
                verbose=False,
            )
        if len(forecasts) != len(batch):
            raise RuntimeError("Kronos returned an unexpected batch size")
        for origin, forecast in zip(batch, forecasts):
            current = frame.iloc[origin.origin_index]
            actual = frame.iloc[origin.origin_index + 1 : origin.target_end_index + 1]
            for step, (_, actual_row) in enumerate(actual.iterrows(), start=1):
                records.append(
                    {
                        "fold": fold_name,
                        "variant": variant,
                        "origin_date": current["date"],
                        "target_date": actual_row["date"],
                        "step": step,
                        "current_close": float(current["close"]),
                        "predicted_close": float(forecast.iloc[step - 1]["close"]),
                        "actual_close": float(actual_row["close"]),
                    }
                )
    return pd.DataFrame(records)


def random_walk_predictions(
    frame: pd.DataFrame,
    origins: Sequence[EvaluationOrigin],
    fold_name: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for origin in origins:
        current = frame.iloc[origin.origin_index]
        actual = frame.iloc[origin.origin_index + 1 : origin.target_end_index + 1]
        for step, (_, actual_row) in enumerate(actual.iterrows(), start=1):
            records.append(
                {
                    "fold": fold_name,
                    "variant": "random_walk",
                    "origin_date": current["date"],
                    "target_date": actual_row["date"],
                    "step": step,
                    "current_close": float(current["close"]),
                    "predicted_close": float(current["close"]),
                    "actual_close": float(actual_row["close"]),
                }
            )
    return pd.DataFrame(records)


def assert_prediction_key_parity(predictions: pd.DataFrame) -> None:
    keys = ["fold", "origin_date", "target_date", "step"]
    expected: set[tuple[object, ...]] | None = None
    for variant, group in predictions.groupby("variant", sort=True):
        current = set(group.loc[:, keys].itertuples(index=False, name=None))
        if group.duplicated(keys).any():
            raise ValueError(f"duplicate prediction keys for {variant}")
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(f"prediction key mismatch for {variant}")


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if (
        len(left_values) < 2
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
        or float(np.std(left_values)) <= 1e-12
        or float(np.std(right_values)) <= 1e-12
    ):
        return float("nan")
    return float(left.corr(right))


def calculate_metrics(predictions: pd.DataFrame, transaction_cost: float) -> pd.DataFrame:
    required = {
        "fold",
        "variant",
        "origin_date",
        "step",
        "current_close",
        "predicted_close",
        "actual_close",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    frames = [predictions]
    combined = predictions.copy()
    combined["fold"] = "ALL"
    frames.append(combined)
    rows: list[dict[str, object]] = []
    for sample in frames:
        for (fold, variant), group in sample.groupby(["fold", "variant"], sort=True):
            normalized_error = (group["predicted_close"] - group["actual_close"]) / group[
                "current_close"
            ]
            terminal = group.loc[group.groupby("origin_date")["step"].idxmax()].copy()
            terminal["predicted_return"] = (
                terminal["predicted_close"] / terminal["current_close"] - 1.0
            )
            terminal["actual_return"] = terminal["actual_close"] / terminal["current_close"] - 1.0
            path = (
                group.groupby("origin_date", as_index=False)
                .agg(
                    current_close=("current_close", "first"),
                    predicted_close=("predicted_close", "mean"),
                    actual_close=("actual_close", "mean"),
                )
                .assign(
                    predicted_return=lambda value: value["predicted_close"]
                    / value["current_close"]
                    - 1.0,
                    actual_return=lambda value: value["actual_close"] / value["current_close"] - 1.0,
                )
            )
            positive = terminal.loc[terminal["predicted_return"] > 0].copy()
            positive_net = positive["actual_return"] - transaction_cost
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "origins": int(terminal["origin_date"].nunique()),
                    "path_mae_pct": float(normalized_error.abs().mean() * 100),
                    "path_rmse_pct": float(np.sqrt(np.square(normalized_error).mean()) * 100),
                    "terminal_direction_accuracy": float(
                        (np.sign(terminal["predicted_return"]) == np.sign(terminal["actual_return"])).mean()
                    ),
                    "terminal_return_correlation": _safe_correlation(
                        terminal["predicted_return"], terminal["actual_return"]
                    ),
                    "mean_path_return_correlation": _safe_correlation(
                        path["predicted_return"], path["actual_return"]
                    ),
                    "positive_signals": int(len(positive)),
                    "positive_signal_mean_net_return": (
                        float(positive_net.mean()) if len(positive_net) else float("nan")
                    ),
                    "positive_signal_net_win_rate": (
                        float((positive_net > 0).mean()) if len(positive_net) else float("nan")
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    naive = metrics.loc[metrics["variant"] == "random_walk", ["fold", "path_mae_pct"]].rename(
        columns={"path_mae_pct": "naive_path_mae_pct"}
    )
    metrics = metrics.merge(naive, on="fold", how="left")
    metrics["mae_skill_vs_random_walk"] = 1.0 - (
        metrics["path_mae_pct"] / metrics["naive_path_mae_pct"]
    )
    return metrics.sort_values(["fold", "variant"]).reset_index(drop=True)


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _markdown_metrics(metrics: pd.DataFrame) -> str:
    columns = [
        "fold",
        "variant",
        "origins",
        "path_mae_pct",
        "terminal_direction_accuracy",
        "terminal_return_correlation",
        "mae_skill_vs_random_walk",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in metrics.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if pd.isna(value) else f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    config: WalkForwardConfig,
    audit: Mapping[str, object],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    training_history: pd.DataFrame,
    device: str,
    run_seconds: float,
    run_options: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    training_history.to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "config": _jsonable(asdict(config)),
        "audit": _jsonable(dict(audit)),
        "device": device,
        "run_seconds": run_seconds,
        "run_options": _jsonable(dict(run_options)),
        "prediction_rows": len(predictions),
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_metrics = metrics.loc[metrics["fold"] == "ALL"].set_index("variant")
    fine = all_metrics.loc["finetuned"]
    zero = all_metrics.loc["pretrained"]
    delta = float(fine["path_mae_pct"] - zero["path_mae_pct"])
    direction_delta = float(
        fine["terminal_direction_accuracy"] - zero["terminal_direction_accuracy"]
    )
    fine_skill = float(fine["mae_skill_vs_random_walk"])
    delta_description = "下降" if delta < 0 else "上升"
    naive_description = "优于" if fine_skill > 0 else "弱于"
    fold_rows = metrics.loc[(metrics["fold"] != "ALL") & (metrics["variant"] == "finetuned")]
    better_folds = fold_rows.loc[fold_rows["mae_skill_vs_random_walk"] > 0, "fold"].tolist()
    better_fold_text = "、".join(better_folds) if better_folds else "无"
    conclusion = (
        f"微调没有形成稳定增益：相对零样本模型，路径 MAE {delta_description} "
        f"{abs(delta):.4f} 个百分点，终点方向准确率变化 {direction_delta:+.4f}；"
        f"汇总路径 MAE {naive_description}随机游走 {abs(fine_skill):.2%}。"
        f"逐折中优于随机游走的只有：{better_fold_text}。"
    )
    report = f"""# Kronos 比亚迪日线滚动微调验证

## 结果摘要

{conclusion}

本次使用 `{config.model.source_revision[:8]}` 版官方 Kronos 源码、Kronos-small predictor 和冻结的 base tokenizer，在 `{device}` 上运行。数据覆盖 `{audit['data']['start']}` 至 `{audit['data']['end']}`，模型观察 {config.experiment.lookback} 个交易日并预测未来 {config.experiment.horizon} 个交易日。

{_markdown_metrics(metrics)}

## 口径

- `random_walk`：未来各日收盘价均等于决策日收盘价。
- `pretrained`：官方预训练权重直接零样本推理。
- `finetuned`：每折从同一官方权重开始，只用该折训练期更新 predictor，并用验证期选取最佳 epoch。
- `path_mae_pct` 和 `path_rmse_pct` 均以决策日收盘价归一化；`mae_skill_vs_random_walk` 大于零表示优于随机游走。
- 正信号收益只是在非重叠抽样原点上的诊断，扣除单次往返成本 {config.experiment.transaction_cost:.2%}，不是完整资金曲线。

## 限制

公开预训练权重的训练语料可能覆盖部分历史市场数据，因此这里的“样本外”仅针对本地滚动微调切分，不能解释为严格无预训练污染。当前也只有单标的验证；只有在扩展到固定、点时可得的股票池并完成组合级交易约束回测后，才能讨论可交易性。

运行耗时 {run_seconds / 60:.2f} 分钟。结构化结果见 `metrics.csv`、`predictions.parquet`、`training_history.csv` 和 `manifest.json`。
"""
    (output_dir / "validation_report.md").write_text(report, encoding="utf-8")


def run_experiment(
    config: WalkForwardConfig,
    fold_names: Sequence[str] | None = None,
    epochs_override: int | None = None,
    max_train_samples: int | None = None,
    max_eval_origins: int | None = None,
    output_dir_override: Path | None = None,
    device_override: str | None = None,
) -> dict[str, object]:
    import torch

    started = time.monotonic()
    audit = audit_experiment(config)
    frame = load_daily_frame(config.experiment.data_path, config.experiment.symbol)
    requested = set(fold_names or [fold.name for fold in config.folds])
    selected = [fold for fold in config.folds if fold.name in requested]
    unknown = requested - {fold.name for fold in selected}
    if unknown:
        raise ValueError(f"unknown folds: {sorted(unknown)}")
    device = resolve_device(device_override or config.model.device)
    window = config.experiment.lookback + config.experiment.horizon + 1
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []

    for fold_number, fold in enumerate(selected, start=1):
        train_starts = window_start_indices(
            frame,
            fold.train_start,
            fold.train_end,
            window,
            config.training.train_stride,
        )
        validation_starts = window_start_indices(
            frame,
            fold.validation_start,
            fold.validation_end,
            window,
            config.training.validation_stride,
        )
        origins = evaluation_origins(
            frame,
            fold,
            config.experiment.lookback,
            config.experiment.horizon,
            config.inference.evaluation_stride,
        )
        train_starts = _cap_evenly(train_starts, max_train_samples)
        origins = _cap_evenly(origins, max_eval_origins)
        if not train_starts or not validation_starts or not origins:
            raise ValueError(f"fold {fold.name} has no usable samples after limits")

        _set_seed(config.training.seed + fold_number)
        model, tokenizer, predictor_class = load_pretrained_components(config, device)
        prediction_frames.append(random_walk_predictions(frame, origins, fold.name))
        prediction_frames.append(
            predict_origins(
                model,
                tokenizer,
                predictor_class,
                frame,
                origins,
                fold.name,
                "pretrained",
                config,
                device,
            )
        )
        checkpoint = config.experiment.checkpoint_dir / fold.name / "best_model"
        history = fine_tune_predictor(
            model,
            tokenizer,
            frame,
            train_starts,
            validation_starts,
            config,
            device,
            checkpoint,
            epochs_override=epochs_override,
        )
        history.insert(0, "fold", fold.name)
        history_frames.append(history)
        prediction_frames.append(
            predict_origins(
                model,
                tokenizer,
                predictor_class,
                frame,
                origins,
                fold.name,
                "finetuned",
                config,
                device,
            )
        )
        del model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    assert_prediction_key_parity(predictions)
    metrics = calculate_metrics(predictions, config.experiment.transaction_cost)
    training_history = pd.concat(history_frames, ignore_index=True)
    output_dir = output_dir_override or config.experiment.output_dir
    run_seconds = time.monotonic() - started
    run_options = {
        "folds": [fold.name for fold in selected],
        "epochs_override": epochs_override,
        "max_train_samples": max_train_samples,
        "max_eval_origins": max_eval_origins,
    }
    write_report(
        output_dir,
        config,
        audit,
        predictions,
        metrics,
        training_history,
        device,
        run_seconds,
        run_options,
    )
    return {
        "output_dir": str(output_dir),
        "device": device,
        "metrics": metrics.to_dict(orient="records"),
        "run_seconds": run_seconds,
    }
