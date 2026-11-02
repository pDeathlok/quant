"""Frozen Kronos watchlist ablations with chronological calibration and selection."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import torch
import torch.nn.functional as functional
import yaml
from safetensors.torch import save_file
from torch import nn

from quant.features.project_factor_layer import calculate_project_factor_frame
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS, load_daily_basic_features
from quant.research.holding_event_probability import choose_high_confidence, threshold_metrics
from quant.research.kronos_next_day_prices import PRICE_COLUMNS, coherent_prices, price_loss
from quant.research.kronos_walk_forward import (
    KLINE_FEATURES,
    PROJECT_ROOT,
    _import_kronos,
    _set_seed,
    audit_model_assets,
    file_sha256,
    load_config,
    resolve_device,
)

VARIANTS = {
    "A": "Kronos regression only",
    "B": "Kronos regression + buckets",
    "C": "B + target calendar",
    "D": "B + project factors",
    "E": "B + project factors + target calendar",
    "F": "project factors + target calendar; no Kronos",
}
EDGES = np.array([-3, -1, -.1, .1, 1, 3])
EVENTS = ("open_up1", "open_down1", "close_up1", "close_down1")
THRESHOLDS = (.5, .6, .7, .8, .9, .95)


def bucket_labels(returns_pct: np.ndarray) -> np.ndarray:
    if not np.isfinite(returns_pct).all():
        raise ValueError("finite economic returns required")
    return np.searchsorted(EDGES, returns_pct, side="right").astype(np.int64)


def calendar_features(origin: pd.DatetimeIndex, target: pd.DatetimeIndex) -> np.ndarray:
    gaps = (target - origin).days.to_numpy()
    if (gaps <= 0).any():
        raise ValueError("target calendar date must follow origin")
    return np.column_stack([target.month - 1, target.day - 1, target.weekday, gaps]).astype(np.float32)


def normalize_history_windows(windows: np.ndarray) -> np.ndarray:
    """Missing auxiliary inputs become neutral standardized values, never invented amounts."""
    finite = np.isfinite(windows)
    counts = finite.sum(axis=1, keepdims=True).clip(min=1)
    observed = np.where(finite, windows, 0)
    mean = observed.sum(axis=1, keepdims=True) / counts
    centered = np.where(finite, windows - mean, 0)
    std = np.sqrt(np.square(centered).sum(axis=1, keepdims=True) / counts)
    return np.clip(centered / (std + 1e-5), -5, 5).astype(np.float32)


def fold_masks(frame: pd.DataFrame, fold: dict[str, Any]) -> dict[str, np.ndarray]:
    year = fold["validation_year"]
    intervals = {
        "train": (fold["train_start"], fold["train_end"]),
        "validation": (f"{year}-01-01", f"{year}-06-30"),
        "calibration": (f"{year}-07-01", f"{year}-09-30"),
        "selection": (f"{year}-10-01", f"{year}-12-31"),
        "test": (fold["test_start"], fold["test_end"]),
    }
    return {name: ((frame.origin_date >= start) & (frame.target_date <= end)).to_numpy()
            for name, (start, end) in intervals.items()}


def fit_scaler(values: np.ndarray) -> dict[str, Any]:
    coverage = np.isfinite(values).mean(axis=0)
    candidates = np.flatnonzero(coverage >= .5)
    mean = np.nanmean(values[:, candidates], axis=0)
    std = np.nanstd(values[:, candidates], axis=0)
    valid = np.isfinite(std) & (std > 1e-8)
    return {"indices": candidates[valid].tolist(), "mean": mean[valid].tolist(), "std": std[valid].tolist()}


def transform_factors(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    selected = values[:, scaler["indices"]]
    missing = ~np.isfinite(selected)
    normalized = (selected - np.array(scaler["mean"])) / np.array(scaler["std"])
    normalized[missing] = 0
    return np.concatenate([np.clip(normalized, -8, 8), missing], axis=1).astype(np.float32)


def event_arrays(probabilities: np.ndarray, economic_returns: np.ndarray,
                 event: str) -> tuple[np.ndarray, np.ndarray]:
    field = 0 if event.startswith("open") else 1
    if event.endswith("up1"):
        return probabilities[:, field, 5:].sum(axis=1), (economic_returns[:, field] >= 1).astype(float)
    return probabilities[:, field, :2].sum(axis=1), (economic_returns[:, field] < -1).astype(float)


class FusionModel(nn.Module):
    """Fuse the final historical token representation with optional known side inputs."""

    def __init__(self, hidden: int, factors: int, variant: str, prior: np.ndarray,
                 bucket_prior: np.ndarray) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown ablation")
        self.variant = variant
        self.sequence = nn.Sequential(nn.Linear(hidden, 128), nn.GELU())
        self.factors = nn.Sequential(nn.Linear(2 * factors, 64), nn.GELU())
        self.month = nn.Embedding(12, 8)
        self.day = nn.Embedding(31, 8)
        self.weekday = nn.Embedding(7, 4)
        self.fusion = nn.Sequential(nn.Linear(213, 128), nn.GELU(), nn.Dropout(.1))
        self.prices = nn.Linear(128, 4)
        self.bins = nn.Linear(128, 14)
        opening, high, low, closing = prior
        upper, lower = max(high - max(opening, closing), 1e-5), max(min(opening, closing) - low, 1e-5)
        bias = [opening, closing, upper + np.log(-np.expm1(-upper)), lower + np.log(-np.expm1(-lower))]
        nn.init.zeros_(self.prices.weight)
        nn.init.zeros_(self.bins.weight)
        with torch.no_grad():
            self.prices.bias.copy_(torch.tensor(bias, dtype=torch.float32))
            self.bins.bias.copy_(torch.tensor(np.log(bucket_prior).reshape(-1), dtype=torch.float32))

    def forward(self, sequence: torch.Tensor, factors: torch.Tensor,
                calendar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequence(sequence)
        fac = self.factors(factors)
        dates = torch.cat([self.month(calendar[:, 0].long()), self.day(calendar[:, 1].long()),
                           self.weekday(calendar[:, 2].long()), torch.log1p(calendar[:, 3:4])], dim=1)
        if self.variant == "F":
            seq = torch.zeros_like(seq)
        if self.variant not in {"D", "E", "F"}:
            fac = torch.zeros_like(fac)
        if self.variant not in {"C", "E", "F"}:
            dates = torch.zeros_like(dates)
        hidden = self.fusion(torch.cat([seq, fac, dates], dim=1))
        return coherent_prices(self.prices(hidden)), self.bins(hidden).reshape(-1, 2, 7)


def multitask_loss(prices: torch.Tensor, logits: torch.Tensor, target: torch.Tensor,
                   buckets: torch.Tensor, auxiliary_weight: float) -> torch.Tensor:
    regression = price_loss(prices, target, prices.new_ones(4))
    return regression + auxiliary_weight * functional.cross_entropy(logits.reshape(-1, 7), buckets.reshape(-1))


def _frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def prepare(settings: dict[str, Any], out: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    base = load_config(PROJECT_ROOT / settings["base_config"])
    assets = audit_model_assets(base)
    watch_path = PROJECT_ROOT / settings["watchlist"]
    watch = json.loads(watch_path.read_text())
    symbols = list(dict.fromkeys(watch["symbols"]))
    (out / "watchlist_snapshot.json").write_text(json.dumps(watch, ensure_ascii=False, indent=2))
    source = arrow_dataset.dataset(PROJECT_ROOT / "data/raw/daily_partitioned", format="parquet", partitioning="hive")
    predicate = arrow_dataset.field("ts_code").isin(symbols) & (arrow_dataset.field("trade_date") <= settings["data_end"].replace("-", ""))
    # Unused descriptive columns have mixed null/string schemas across old partitions.
    raw_columns = ["ts_code", "trade_date", "date", "symbol", *KLINE_FEATURES, "pre_close", "pct_chg", "change", "vol"]
    raw = source.to_table(columns=raw_columns, filter=predicate).to_pandas().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if set(raw.ts_code) != set(symbols) or raw.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("missing watchlist stocks or duplicate raw keys")
    if not np.isfinite(raw.loc[:, KLINE_FEATURES].to_numpy(float)).all():
        raise ValueError("incomplete OHLCVA: fetch and audit missing Tushare data before encoding")
    print(f"loaded {len(symbols)} stocks, {len(raw)} raw rows", flush=True)
    basic = load_daily_basic_features(PROJECT_ROOT / "data/raw/daily_basic", target_keys=raw[["ts_code", "trade_date"]])
    registry_path = PROJECT_ROOT / "data/features/factor_registry/latest.json"
    registry = json.loads(registry_path.read_text())["factors"]
    inventory = []
    for factor in registry:
        admitted = factor["name"] in PROJECT_FACTOR_COLUMNS
        reason = "daily canonical calculator; training coverage gate follows" if admitted else (
            "strategy identity excluded" if factor["role"] == "strategy_identity" else
            "historical availability / event-time / model-score provenance not established for this experiment")
        inventory.append({**factor, "candidate_this_run": admitted, "admission_reason": reason})
    pd.DataFrame(inventory).to_json(out / "factor_inventory.json", orient="records", force_ascii=False, indent=2)
    # Reconstruct market sessions independently of any single stock; the legacy
    # index file ends in May while the canonical market partitions extend to September.
    market_dates: set[str] = set()
    for fragment in source.get_fragments():
        market_dates.update(fragment.to_table(columns=["trade_date"])["trade_date"].unique().to_pylist())
    calendar = pd.DatetimeIndex(pd.to_datetime(sorted(market_dates), format="%Y%m%d"))
    if calendar.max() < raw.date.max():
        raise ValueError("market calendar does not cover raw data")
    calendar_digest = hashlib.sha256("|".join(sorted(market_dates)).encode()).hexdigest()
    pd.DataFrame({"date": calendar}).to_csv(out / "market_calendar.csv", index=False)
    next_date = dict(zip(calendar[:-1], calendar[1:]))
    klass, tokenizer_klass, _ = _import_kronos(base.model.source_dir)
    device = resolve_device("auto")
    tokenizer = tokenizer_klass.from_pretrained(str(base.model.tokenizer_path)).to(device).eval()
    backbone = klass.from_pretrained(str(base.model.predictor_path)).to(device).eval()
    tokenizer.requires_grad_(False)
    backbone.requires_grad_(False)
    cache = out / "encoded"
    cache.mkdir(exist_ok=True)
    frames, hidden_all, stock_audit = [], [], []
    earliest = pd.Timestamp(min(f["train_start"] for f in settings["folds"]))
    lookback = settings["lookback"]
    basic_by = {s: g for s, g in basic.groupby("ts_code")}
    for number, symbol in enumerate(symbols, 1):
        daily = raw.loc[raw.ts_code == symbol].copy().reset_index(drop=True)
        values = daily.loc[:, KLINE_FEATURES].to_numpy(np.float32)
        dates = pd.DatetimeIndex(daily.date)
        indices = np.arange(lookback - 1, len(daily) - 1)
        if len(indices) == 0:
            raise ValueError(f"no warm history for {symbol}")
        # Missing amount is an input-availability issue, not a missing price target.
        numeric_ok = np.isfinite(values[:, :4]).all(axis=1) & (values[:, :4] > 0).all(axis=1)
        numeric_ok &= (daily.high + 1e-6 >= daily[["open", "close"]].max(axis=1)).to_numpy()
        numeric_ok &= (daily.low - 1e-6 <= daily[["open", "close"]].min(axis=1)).to_numpy()
        history_ok = pd.Series(numeric_ok).rolling(lookback).sum().eq(lookback).to_numpy()
        expected_next = pd.DatetimeIndex([next_date.get(date, pd.NaT) for date in dates[indices]])
        valid = (dates[indices] >= earliest) & (expected_next == dates[indices + 1])
        valid &= history_ok[indices] & numeric_ok[indices + 1]
        valid &= (daily.volume.to_numpy()[indices] > 0) & (daily.volume.to_numpy()[indices + 1] > 0)
        valid &= np.isfinite(daily.pre_close.to_numpy()[indices + 1]) & (daily.pre_close.to_numpy()[indices + 1] > 0)
        indices = indices[valid]
        basic_stock = basic_by.get(symbol, pd.DataFrame())
        fingerprint = hashlib.sha256((file_sha256(Path(__file__)) + _frame_digest(daily) +
                                      _frame_digest(basic_stock) + str(lookback) + str(earliest) +
                                      calendar_digest + base.model.predictor_sha256 + base.model.tokenizer_sha256).encode()).hexdigest()
        prefix = cache / symbol.replace(".", "_")
        meta_path, table_path, vector_path = prefix.with_suffix(".json"), prefix.with_suffix(".parquet"), prefix.with_suffix(".npy")
        cached = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if cached.get("fingerprint") == fingerprint and table_path.exists() and vector_path.exists():
            sample = pd.read_parquet(table_path)
            hidden = np.load(vector_path, allow_pickle=False)
            if cached["table_sha256"] != file_sha256(table_path) or cached["vectors_sha256"] != file_sha256(vector_path):
                raise ValueError(f"cache integrity failure: {symbol}")
        else:
            factor_frame = calculate_project_factor_frame(daily, symbol, daily_basic_features=basic_stock,
                                                          factor_schema_version="project-v5-canonical-alias-free")
            if not np.array_equal(factor_frame.date.to_numpy(), daily.date.to_numpy()):
                raise ValueError("factor/date alignment failure")
            sample = pd.DataFrame({"symbol": symbol, "origin_date": dates[indices], "target_date": expected_next[valid],
                                   "context_start": dates[indices - lookback + 1],
                                   "current_close": daily.close.to_numpy()[indices],
                                   "target_pre_close": daily.pre_close.to_numpy()[indices + 1]})
            for column in PRICE_COLUMNS:
                sample[f"actual_{column}"] = daily[column].to_numpy()[indices + 1]
            factor_values = factor_frame.loc[indices, PROJECT_FACTOR_COLUMNS].reset_index(drop=True).add_prefix("factor__")
            sample = pd.concat([sample, factor_values], axis=1)
            windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0).transpose(0, 2, 1)
            stamps = np.column_stack([dates.minute, dates.hour, dates.weekday, dates.day, dates.month]).astype(np.float32)
            stamp_windows = np.lib.stride_tricks.sliding_window_view(stamps, lookback, axis=0).transpose(0, 2, 1)
            embeddings = []
            for start in range(0, len(indices), settings["encode_batch_size"]):
                positions = indices[start:start + settings["encode_batch_size"]] - lookback + 1
                batch = windows[positions].copy()
                batch = normalize_history_windows(batch)
                with torch.no_grad():
                    t0, t1 = tokenizer.encode(torch.from_numpy(batch).to(device), half=True)
                    state = backbone.embedding([t0, t1]) + backbone.time_emb(torch.from_numpy(stamp_windows[positions].copy()).to(device))
                    for layer in backbone.transformer:
                        state = layer(state)
                    embeddings.append(backbone.norm(state)[:, -1].cpu().numpy())
            hidden = np.concatenate(embeddings).astype(np.float32)
            sample.to_parquet(table_path, index=False)
            np.save(vector_path, hidden, allow_pickle=False)
            meta_path.write_text(json.dumps({"fingerprint": fingerprint, "table_sha256": file_sha256(table_path),
                                            "vectors_sha256": file_sha256(vector_path)}))
        if len(sample) != len(hidden) or len(sample) != len(indices) or not np.isfinite(hidden).all():
            raise ValueError("encoded sample mismatch")
        frames.append(sample)
        hidden_all.append(hidden)
        stock_audit.append({"symbol": symbol, "raw_rows": len(daily), "samples": len(sample),
                            "first_raw": str(dates.min().date()), "last_raw": str(dates.max().date()),
                            "last_target": str(sample.target_date.max().date()),
                            "missing_amount_rows": int(daily.amount.isna().sum()),
                            "invalid_price_rows": int((~numeric_ok).sum())})
        print(f"encoded {number}/{len(symbols)} {symbol}: {len(sample)} daily samples", flush=True)
    del tokenizer, backbone
    if device == "mps":
        torch.mps.empty_cache()
    samples = pd.concat(frames, ignore_index=True)
    pd.DataFrame(stock_audit).to_csv(out / "stock_audit.csv", index=False)
    metadata = {"assets": assets, "device": device, "raw_digest": _frame_digest(raw),
                "daily_basic_digest": _frame_digest(basic), "watchlist_sha256": file_sha256(watch_path),
                "registry_sha256": file_sha256(registry_path), "registered_factors": len(registry),
                "calendar_source": "unique dates in canonical full-market daily partitions; independent of stock-specific gaps",
                "calendar_sha256": calendar_digest,
                "factor_candidates": len(PROJECT_FACTOR_COLUMNS), "stocks": len(symbols), "samples": len(samples)}
    return samples, np.concatenate(hidden_all), metadata


def _predict(model: FusionModel, arrays: tuple[torch.Tensor, ...], positions: np.ndarray,
             batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prices, logits = [], []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            idx = positions[start:start + batch_size]
            p, z = model(*(array[idx] for array in arrays[:3]))
            prices.append(p.numpy())
            logits.append(z.numpy())
    return np.concatenate(prices), np.concatenate(logits)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.astype(np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    result = np.exp(scaled)
    return result / result.sum(axis=-1, keepdims=True)


def _temperature(logits: np.ndarray, buckets: np.ndarray) -> float:
    best = (float("inf"), 1.)
    for temperature in (.5, .75, 1., 1.5, 2., 3., 5.):
        probabilities = _softmax(logits, temperature)
        nll = -np.log(np.take_along_axis(probabilities, buckets[..., None], axis=-1).clip(1e-12)).mean()
        best = min(best, (float(nll), temperature))
    return best[1]


def _fit(arrays: tuple[torch.Tensor, ...], masks: dict[str, np.ndarray], variant: str,
         seed: int, settings: dict[str, Any], folder: Path) -> tuple[FusionModel, dict[str, Any]]:
    _set_seed(seed)
    train, validation = np.flatnonzero(masks["train"]), np.flatnonzero(masks["validation"])
    prior = np.median(arrays[3][train].numpy(), axis=0)
    labels = arrays[4][train].numpy()
    priors = np.stack([(np.bincount(labels[:, i], minlength=7) + 1) / (len(labels) + 7) for i in range(2)])
    model = FusionModel(arrays[0].shape[1], arrays[1].shape[1] // 2, variant, prior, priors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=.01)
    auxiliary_weight = 0. if variant == "A" else settings["auxiliary_weight"]

    def validation_loss() -> float:
        prices, logits = _predict(model, arrays, validation)
        return float(multitask_loss(torch.from_numpy(prices), torch.from_numpy(logits),
                                    arrays[3][validation], arrays[4][validation], auxiliary_weight))

    best_loss, best_epoch, stale = validation_loss(), 0, 0
    best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    history = [{"epoch": 0, "validation_loss": best_loss}]
    rng = np.random.default_rng(seed)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        shuffled = rng.permutation(train)
        total = 0.
        for start in range(0, len(shuffled), settings["batch_size"]):
            positions = shuffled[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prices, logits = model(*(array[positions] for array in arrays[:3]))
            loss = multitask_loss(prices, logits, arrays[3][positions], arrays[4][positions], auxiliary_weight)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(positions)
        val_loss = validation_loss()
        history.append({"epoch": epoch, "train_loss": total / len(train), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print(f"{folder.name} epoch {epoch}: val={val_loss:.5f} best={best_epoch}", flush=True)
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best)
    folder.mkdir(exist_ok=True, parents=True)
    save_file(best, str(folder / "model.safetensors"))
    metadata = {"variant": variant, "seed": seed, "hidden_dim": arrays[0].shape[1],
                "factor_count": arrays[1].shape[1] // 2, "prior": prior.tolist(), "bucket_prior": priors.tolist(),
                "best_epoch": best_epoch, "history": history, "state_sha256": file_sha256(folder / "model.safetensors")}
    (folder / "model.json").write_text(json.dumps(metadata, indent=2))
    return model, metadata


def _metrics(frame: pd.DataFrame, predicted_log: np.ndarray, probabilities: np.ndarray | None,
             bucket_prior: np.ndarray, fold: str, variant: str, seed: str) -> list[dict[str, Any]]:
    current = frame.current_close.to_numpy()
    actual = frame[[f"actual_{f}" for f in PRICE_COLUMNS]].to_numpy()
    prices = np.exp(predicted_log / 100) * current[:, None]
    errors = 100 * (prices - actual) / current[:, None]
    rows = []
    for i, field in enumerate(PRICE_COLUMNS):
        rows.append({"fold": fold, "variant": variant, "seed": seed, "metric": f"mae_{field}_pct",
                     "value": float(np.abs(errors[:, i]).mean()), "n": len(frame)})
    if probabilities is not None:
        economic = 100 * (frame[["actual_open", "actual_close"]].to_numpy() / frame.target_pre_close.to_numpy()[:, None] - 1)
        for event in EVENTS:
            p, y = event_arrays(probabilities, economic, event)
            baseline, _ = event_arrays(np.broadcast_to(bucket_prior, probabilities.shape), economic, event)
            for metric, value in [("brier", ((p - y) ** 2).mean()), ("prior_brier", ((baseline - y) ** 2).mean()),
                                  ("base_rate", y.mean())]:
                rows.append({"fold": fold, "variant": variant, "seed": seed, "metric": f"{event}_{metric}",
                             "value": float(value), "n": len(frame)})
    return rows


def _confidence(selection_frame: pd.DataFrame, selection_probs: np.ndarray, test_frame: pd.DataFrame,
                test_probs: np.ndarray, fold: str, variant: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    records, curves = [], []
    for event in EVENTS:
        by_split = {}
        for split, frame, probs in [("selection", selection_frame, selection_probs), ("test", test_frame, test_probs)]:
            economic = 100 * (frame[["actual_open", "actual_close"]].to_numpy() / frame.target_pre_close.to_numpy()[:, None] - 1)
            p, y = event_arrays(probs, economic, event)
            observed = pd.DataFrame({"date": frame.origin_date.to_numpy(), "probability": p, "outcome": y,
                                     "symbol": frame.symbol.to_numpy()})
            curve = threshold_metrics(observed, thresholds=THRESHOLDS, bootstrap=400, seed=20260912)
            curve["feature_group"] = variant
            curve["signal_symbols"] = [observed.loc[observed.probability >= t, "symbol"].nunique() for t in curve.threshold]
            curve["max_symbol_share"] = [
                observed.loc[observed.probability >= t, "symbol"].value_counts(normalize=True).max() for t in curve.threshold]
            curve["fold"], curve["variant"], curve["event"], curve["split"] = fold, variant, event, split
            curves.append(curve)
            by_split[split] = curve
        chosen = choose_high_confidence(by_split["selection"].loc[by_split["selection"].signal_symbols >= 5])
        if chosen is None:
            records.append({"fold": fold, "variant": variant, "event": event, "status": "no_validation_qualified_threshold"})
        else:
            test = by_split["test"].loc[by_split["test"].threshold == chosen["threshold"]].iloc[0].to_dict()
            records.append({**test, "status": "validation_locked_test", "validation_precision": chosen["precision"],
                            "validation_signals": chosen["signals"], "validation_precision_low90": chosen["precision_low90"]})
    return records, curves


def _report(out: Path, manifest: dict[str, Any], metrics: pd.DataFrame, confidence: pd.DataFrame) -> None:
    lines = ["# 自选池 Kronos 次日多任务消融", "",
             f"股票池：当前自选池{manifest['stocks']}股；原始行情截至{manifest['settings']['data_end']}。合格历史股票/预测日共{manifest['samples']}条。", "",
             "## 实验合同", "",
             "冻结原tokenizer和Kronos主干，训练最后历史表示之上的融合层；不是全参数微调。原始六字段90日输入仅按历史窗口归一化。",
             "原始成交额缺口已按用户要求从Tushare补拉，只填空值并保留分区备份及API原始返回。运行入口要求六字段完整，数据缺失会阻止训练，不使用中性填充替代可补拉的真实数据。缺失数量见stock_audit.csv。",
             "回归目标为100×log(次日真实OHLC/今日真实收盘)，四价等权Smooth L1。辅助目标为次日开盘、收盘相对交易所次日pre_close的经济涨跌幅七分桶，边界[-3,-1,-0.1,0.1,1,3]%，loss权重0.3。",
             "价格标签保留真实除权价差；辅助事件用经济口径避免机械除权被当成收益。两种口径有意区分，不用次日pre_close作模型输入。",
             "目标日来自交易所日历，月/日/星期嵌入和距当前日间隔在最后历史表示之后融合；不替换历史日期，也不读取未来价格。未加入真实公司公告预约日历。", ""]
    for key, description in VARIANTS.items():
        lines.append(f"- {key}: {description}")
    lines.extend(["", "每折验证年前半年选epoch、Q3温度校准、Q4锁定概率阈值，然后才评估测试年；两个种子各自训练校准并平均概率。A无辅助训练，因此不报告其分类头概率。",
                  "当前自选池回溯存在事后选股/幸存者偏差；基础模型预训练重合与已有研究使用过这些历史区间的风险仍在。", "",
                  "## 价格误差：双种子平均模型", "", "|测试折|组|开盘MAE %|最高MAE %|最低MAE %|收盘MAE %|", "|---|---|---:|---:|---:|---:|"])
    selected = metrics.loc[(metrics.seed == "ensemble") & metrics.metric.str.startswith("mae_")]
    for (fold, variant), group in selected.groupby(["fold", "variant"]):
        values = group.set_index("metric").value
        lines.append(f"|{fold}|{variant}|" + "|".join(f"{values[f'mae_{field}_pct']:.4f}" for field in PRICE_COLUMNS) + "|")
    lines.extend(["", "## 因子准入", "", f"注册{manifest['registered_factors']}条，本轮通过标准计算器尝试145列日频因子；按各训练折至少50%有效值且非恒定进一步准入。",
                  "全量逐项决定见factor_inventory.json；各折实际入模列和训练期标准化参数见factor_admission.json。缺失值用训练均值填充并显式提供缺失标记。",
                  "794条注册记录不等于794个可用于该历史任务的输入。其余包含周频财务、实时策略评分、策略身份、研究因子及需要额外历史时点审计的字段；未假装已接入。", "",
                  "## 高置信机会", "",
                  "事件固定为次日经济口径开盘/收盘上涨≥1%或下跌>1%。Q4验证门槛：至少50信号、20日期、5股票，precision≥80%，5交易日块bootstrap的90%下限≥70%，lift≥1.2。",
                  "阈值只在Q4选择，测试仅验证锁定阈值。固定阈值曲线另存threshold_curves.csv，不能从测试曲线事后挑最好结果。",
                  "块区间包含同日股票相关性，但不提供多个模型/事件/阈值搜索的同时置信保证；两个种子不是两组独立市场样本。", ""])
    locked = confidence.loc[confidence.status == "validation_locked_test"]
    if locked.empty:
        lines.append("本轮没有事件/模型通过预先规定的验证准入，不能宣布发现高确定性机会。")
    else:
        lines.extend(["|测试折|组|事件|锁定阈值|测试信号|测试命中率|90%下限|覆盖率|", "|---|---|---|---:|---:|---:|---:|---:|"])
        for row in locked.to_dict("records"):
            lines.append(f"|{row['fold']}|{row['variant']}|{row['event']}|{row['threshold']}|{row['signals']}|{row['precision']:.3f}|{row['precision_low90']:.3f}|{row['coverage']:.3f}|")
    lines.extend(["", "这是收盘后且当日行情/daily_basic已可用的概率研究，不是能按当天收盘价成交的策略。没有验证交易成本、流动性、涨跌停成交或A股交易制度下的可执行收益，不据此声称盈利机会。",
                  "详细逐折逐种子价格/Brier指标见metrics.csv，验证锁定信号见confidence.csv，训练历史及温度见模型metadata。", ""])
    (out / "report.md").write_text("\n".join(lines))


def run(config_path: Path) -> Path:
    settings = yaml.safe_load(config_path.read_text())
    if settings["lookback"] != 90 or settings["auxiliary_weight"] < 0:
        raise ValueError("unexpected experiment contract")
    out = PROJECT_ROOT / settings["output_dir"]
    if (out / "manifest.json").exists():
        raise FileExistsError("completed experiment preserved; choose another output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    samples, hidden, metadata = prepare(settings, out)
    keys = ["symbol", "origin_date", "target_date", "context_start"]
    if samples.duplicated(["symbol", "origin_date"]).any():
        raise ValueError("duplicate samples")
    samples[keys].to_parquet(out / "samples.parquet", index=False)
    all_factors = samples[[f"factor__{column}" for column in PROJECT_FACTOR_COLUMNS]].to_numpy(float)
    dates = calendar_features(pd.DatetimeIndex(samples.origin_date), pd.DatetimeIndex(samples.target_date))
    actual = samples[[f"actual_{column}" for column in PRICE_COLUMNS]].to_numpy()
    targets = (100 * np.log(actual / samples.current_close.to_numpy()[:, None])).astype(np.float32)
    economic = 100 * (actual[:, [0, 3]] / samples.target_pre_close.to_numpy()[:, None] - 1)
    buckets = bucket_labels(economic)
    all_metrics, all_confidence, all_curves, admissions, model_metadata, split_audit = [], [], [], [], [], []
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for fold in settings["folds"]:
        masks = fold_masks(samples, fold)
        if any(not mask.any() for mask in masks.values()) or np.stack(list(masks.values())).sum(axis=0).max() > 1:
            raise ValueError("invalid fold split")
        for split, mask in masks.items():
            part = samples.loc[mask]
            split_audit.append({"fold": fold["name"], "split": split, "samples": int(mask.sum()),
                                "stocks": int(part.symbol.nunique()), "dates": int(part.origin_date.nunique()),
                                "first_origin": str(part.origin_date.min().date()), "last_target": str(part.target_date.max().date())})
        scaler = fit_scaler(all_factors[masks["train"]])
        if not scaler["indices"]:
            raise ValueError("no admitted factors")
        admissions.append({"fold": fold["name"], "columns": [PROJECT_FACTOR_COLUMNS[i] for i in scaler["indices"]], "scaler": scaler})
        arrays = (torch.from_numpy(hidden), torch.from_numpy(transform_factors(all_factors, scaler)),
                  torch.from_numpy(dates), torch.from_numpy(targets), torch.from_numpy(buckets))
        parts = {name: np.flatnonzero(mask) for name, mask in masks.items()}
        prior = np.median(targets[masks["train"]], axis=0)
        priors = np.stack([(np.bincount(buckets[masks["train"], i], minlength=7) + 1) / (masks["train"].sum() + 7) for i in range(2)])
        test_frame = samples.iloc[parts["test"]]
        for name, prediction in [("previous_close", np.zeros((len(test_frame), 4))),
                                 ("train_median", np.broadcast_to(prior, (len(test_frame), 4)))]:
            all_metrics.extend(_metrics(test_frame, prediction, None, priors, fold["name"], name, "ensemble"))
        for variant in settings["variants"]:
            seed_predictions = {"test": [], "selection": []}
            for seed in settings["seeds"]:
                folder = out / "models" / f"{fold['name']}_{variant}_{seed}"
                model, model_info = _fit(arrays, masks, variant, seed, settings, folder)
                _, cal_logits = _predict(model, arrays, parts["calibration"])
                temperature = None if variant == "A" else _temperature(cal_logits, buckets[parts["calibration"]])
                model_info.update({"fold": fold["name"], "temperature": temperature, "scaler": scaler})
                (folder / "model.json").write_text(json.dumps(model_info, indent=2))
                model_metadata.append(model_info)
                for split in ("selection", "test"):
                    predicted_log, logits = _predict(model, arrays, parts[split])
                    probabilities = None if temperature is None else _softmax(logits, temperature)
                    seed_predictions[split].append((predicted_log, probabilities))
                    if split == "test":
                        all_metrics.extend(_metrics(test_frame, predicted_log, probabilities, priors, fold["name"], variant, str(seed)))
                del model
            combined = {}
            for split in ("selection", "test"):
                # Average actual prices, not logs; return to log units for shared metrics.
                predicted_log = 100 * np.log(np.mean([np.exp(pair[0] / 100) for pair in seed_predictions[split]], axis=0))
                probabilities = None if variant == "A" else np.mean([pair[1] for pair in seed_predictions[split]], axis=0)
                combined[split] = probabilities
                frame = samples.iloc[parts[split]]
                stored = frame[[*keys, "current_close", "target_pre_close", *[f"actual_{c}" for c in PRICE_COLUMNS]]].copy()
                for i, column in enumerate(PRICE_COLUMNS):
                    stored[f"pred_{column}"] = np.exp(predicted_log[:, i] / 100) * frame.current_close.to_numpy()
                if probabilities is not None:
                    for field, column in enumerate(("open", "close")):
                        for bucket in range(7):
                            stored[f"p_{column}_{bucket}"] = probabilities[:, field, bucket]
                stored.to_parquet(prediction_dir / f"{fold['name']}_{variant}_{split}.parquet", index=False)
                if split == "test":
                    all_metrics.extend(_metrics(frame, predicted_log, probabilities, priors, fold["name"], variant, "ensemble"))
            if variant != "A":
                records, curves = _confidence(samples.iloc[parts["selection"]], combined["selection"], test_frame,
                                               combined["test"], fold["name"], variant)
                all_confidence.extend(records)
                all_curves.extend(curves)
            pd.DataFrame(all_metrics).to_csv(out / "metrics.csv", index=False)
            pd.DataFrame(all_confidence).to_csv(out / "confidence.csv", index=False)
            if all_curves:
                pd.concat(all_curves, ignore_index=True).to_csv(out / "threshold_curves.csv", index=False)
            print(f"completed {fold['name']} {variant}", flush=True)
        pd.DataFrame(split_audit).to_csv(out / "split_audit.csv", index=False)
        (out / "factor_admission.json").write_text(json.dumps(admissions, indent=2))
    manifest = {**metadata, "settings": settings, "seconds": time.monotonic() - started,
                "training_source_sha256": file_sha256(Path(__file__)), "model_metadata": model_metadata,
                "torch_version": torch.__version__, "split_audit": split_audit,
                "limitations": ["current-watchlist historical selection bias", "pretraining overlap unknown", "retrospective research",
                                "frozen backbone screen, not full finetuning", "after-close availability, not executable PnL"]}
    _report(out, manifest, pd.DataFrame(all_metrics), pd.DataFrame(all_confidence))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out
