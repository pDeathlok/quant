"""Next-session OHLC regression on Kronos features, with history-only preprocessing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quant.research.kronos_walk_forward import (
    KLINE_FEATURES,
    PROJECT_ROOT,
    _import_kronos,
    _set_seed,
    audit_model_assets,
    file_sha256,
    load_config,
    load_daily_frame,
    resolve_device,
    window_start_indices,
)

PRICE_COLUMNS = ("open", "high", "low", "close")


def historical_input(frame: pd.DataFrame, origin: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Only observations through origin affect either normalization or time features."""
    if origin < lookback - 1:
        raise ValueError("insufficient history")
    context = frame.iloc[origin - lookback + 1:origin + 1]
    values = context.loc[:, KLINE_FEATURES].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite history")
    values = np.clip((values - values.mean(axis=0)) / (values.std(axis=0) + 1e-5), -5, 5)
    dates = context.date.dt
    stamps = np.column_stack([dates.minute, dates.hour, dates.weekday, dates.day, dates.month])
    return values.astype(np.float32), stamps.astype(np.float32)


def eligible_origins(frame: pd.DataFrame, calendar: pd.DatetimeIndex, lookback: int) -> list[int]:
    """Reject missing exchange sessions, non-trading bars and missing warmup/targets."""
    next_session = dict(zip(calendar[:-1], calendar[1:]))
    valid = []
    for i in range(lookback - 1, len(frame) - 1):
        if next_session.get(frame.iloc[i].date) != frame.iloc[i + 1].date:
            continue
        if frame.iloc[i].volume <= 0 or frame.iloc[i + 1].volume <= 0:
            continue
        valid.append(i)
    return valid


def split_origins(frame: pd.DataFrame, origins: list[int], start: pd.Timestamp,
                  end: pd.Timestamp, stride: int = 1) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be positive")
    # Context may predate a split; origin and fully observed next-session label may not.
    return [i for i in origins if frame.iloc[i].date >= start and frame.iloc[i + 1].date <= end][::stride]


def coherent_prices(raw: torch.Tensor) -> torch.Tensor:
    """Return OHLC in log-return percentage points, enforcing candle inequalities."""
    opening, closing = raw[..., 0], raw[..., 1]
    high = torch.maximum(opening, closing) + functional.softplus(raw[..., 2])
    low = torch.minimum(opening, closing) - functional.softplus(raw[..., 3])
    return torch.stack([opening, high, low, closing], dim=-1)


def price_loss(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
               beta: float = 1.0) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 4:
        raise ValueError("exactly four OHLC targets required")
    return (functional.smooth_l1_loss(prediction, target, reduction="none", beta=beta) * weights).sum(-1).mean() / weights.sum()


class NextDayPriceModel(nn.Module):
    """Train the pretrained trunk and a continuous OHLC head; no token or volume loss."""

    def __init__(self, backbone: nn.Module, prior: np.ndarray) -> None:
        super().__init__()
        self.embedding = backbone.embedding
        self.time_emb = backbone.time_emb
        self.token_drop = backbone.token_drop
        self.transformer = backbone.transformer
        self.norm = backbone.norm
        self.price_head = nn.Sequential(nn.Linear(backbone.d_model, 128), nn.GELU(), nn.Linear(128, 4))
        opening, high, low, closing = np.asarray(prior, dtype=float)
        upper = max(high - max(opening, closing), 1e-5)
        lower = max(min(opening, closing) - low, 1e-5)
        def inverse_softplus(x: float) -> float:
            return float(x + np.log(-np.expm1(-x)))
        bias = [opening, closing, inverse_softplus(upper), inverse_softplus(lower)]
        nn.init.zeros_(self.price_head[-1].weight)
        with torch.no_grad():
            self.price_head[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32))

    def forward(self, token0: torch.Tensor, token1: torch.Tensor, stamps: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding([token0, token1]) + self.time_emb(stamps)
        hidden = self.token_drop(hidden)
        for layer in self.transformer:
            hidden = layer(hidden)
        return coherent_prices(self.price_head(self.norm(hidden)[:, -1]))


def evaluate(model: NextDayPriceModel, loader: DataLoader, device: str,
             weights: torch.Tensor, beta: float) -> tuple[float, np.ndarray]:
    model.eval()
    outputs = []
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for token0, token1, stamps, targets in loader:
            prediction = model(token0.to(device), token1.to(device), stamps.to(device))
            loss = price_loss(prediction, targets.to(device), weights, beta)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite validation loss")
            loss_sum += float(loss) * len(targets)
            n += len(targets)
            outputs.append(prediction.cpu().numpy())
    if not n:
        raise ValueError("empty evaluation set")
    return loss_sum / n, np.concatenate(outputs)


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_name, frame in [("ALL", predictions), *list(predictions.groupby("fold"))]:
        for variant, group in frame.groupby("variant"):
            for field in PRICE_COLUMNS:
                predicted = group[f"pred_{field}"].to_numpy()
                actual = group[f"actual_{field}"].to_numpy()
                current = group.current_close.to_numpy()
                error = (predicted - actual) / current * 100
                direction = np.sign(np.round(predicted / current - 1, 10))
                actual_direction = np.sign(np.round(actual / current - 1, 10))
                rows.append({"fold": fold_name, "variant": variant, "field": field,
                             "n": len(group), "mae_pct": float(np.abs(error).mean()),
                             "rmse_pct": float(np.sqrt(np.square(error).mean())),
                             "direction_accuracy": float((direction == actual_direction).mean())})
    return pd.DataFrame(rows)


def write_report(out_dir: Path, metrics: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Kronos：次日四价格专项实验", "",
        "## 任务与 loss", "",
        "输入为截至预测日的90根真实日K线（OHLC、成交量、成交额），仅用这些历史数据计算归一化。",
        "冻结原 tokenizer，微调预训练 Transformer 主干和新增连续价格回归头。",
        "每条样本仅监督下一交易日的开、高、低、收；不生成未来 token，不计算量或额的预测 loss。",
        "目标为 y = 100 × ln(次日价格 / 今日收盘)，loss 为四项等权 Smooth L1 的平均，beta=1个百分点。",
        "通过输出参数化保证最高价≥开收、最低价≤开收，反变换后价格为正。",
        "验证集 loss 选择 checkpoint（含未训练的训练集中位数初始化），测试集不选模型。", "",
        "## 日期抽样", "",
        "|测试折|用途|新样本数|旧窗口数|旧监督日期去重数|",
        "|---|---|---:|---:|---:|",
    ]
    for row in manifest["sample_audit"]:
        lines.append(f"|{row['fold']}|{row['split']}|{row['new_samples']}|{row['old_windows_train_or_validation'] or '—'}|{row['old_unique_supervised_dates'] or '—'}|")
    lines.extend([
        "", "新版本逐日取样，同一折内每个合格预测日一次；不同滚动折允许复用更早的历史数据。",
        "排除历史不足、目标跨分区边界、次日缺少交易记录或当日/目标日成交量非正的样本。",
        "成交量筛选仅用于确认历史标签可用，不是可以提前知道次日是否停牌的交易信号。",
        "旧版本训练窗口步长5、验证步长10，但各窗口内部100个位置均有监督，不能理解为只训练了1/5日期。", "",
        "## 全测试期结果", "",
        "MAE、RMSE均按今日收盘归一化后以百分比列示。", "",
        "|模型|价格|样本数|MAE %|RMSE %|",
        "|---|---|---:|---:|---:|",
    ])
    for row in metrics.loc[metrics.fold == "ALL"].to_dict("records"):
        lines.append(f"|{row['variant']}|{row['field']}|{row['n']}|{row['mae_pct']:.4f}|{row['rmse_pct']:.4f}|")
    lines.extend(["", "price_regression=价格专项模型；previous_close=四价都预测为今日收盘；train_median_candle=训练集中位数K线形态。", "", "## 验证集选中轮次", ""])
    for checkpoint in manifest["checkpoints"]:
        lines.append(f"- {checkpoint['fold']}: epoch {checkpoint['best_epoch']}, validation loss {checkpoint['validation_loss']:.6f}")
    lines.extend(["", "## 限制", "", "仍然只有比亚迪002594，数据截至2026-05-26；不是全市场或截至今天的数据。",
                  "这些历史测试区间之前已经查看过，本次是回溯开发评估，不是全新未触碰测试。预训练重叠风险和原始复权数据时点口径未排除。",
                  "相邻90日窗口高度重叠，样本不是相互独立的。价格误差降低不等于方向正确率提高或策略盈利。", ""])
    (out_dir / "report.md").write_text("\n".join(lines))


def run(config_path: Path) -> Path:
    settings = yaml.safe_load(config_path.read_text())
    base = load_config(PROJECT_ROOT / settings["base_config"])
    if settings["loss"] != "smooth_l1_log_return_pct":
        raise ValueError("unknown loss")
    weights_array = np.asarray(settings["price_weights"], dtype=np.float32)
    if weights_array.shape != (4,) or not np.isfinite(weights_array).all() or (weights_array <= 0).any():
        raise ValueError("four finite positive price weights required")
    for name in ("epochs", "patience", "batch_size", "train_stride", "lookback", "loss_beta", "backbone_lr", "head_lr"):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    assets = audit_model_assets(base)
    out_dir = PROJECT_ROOT / settings["output_dir"]
    checkpoints = PROJECT_ROOT / settings["checkpoint_dir"]
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"Use a new output directory to preserve completed experiments: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    frame = load_daily_frame(base.experiment.data_path, base.experiment.symbol)
    frame = frame.loc[frame.date <= max(f.test_end for f in base.folds)].reset_index(drop=True)
    values = frame.loc[:, KLINE_FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite data")
    if (frame.loc[:, list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError("prices must be positive for log returns")
    if ((frame.high + 1e-6 < frame[["open", "close"]].max(axis=1)) |
        (frame.low - 1e-6 > frame[["open", "close"]].min(axis=1))).any():
        raise ValueError("invalid OHLC price ordering")
    index = pd.read_parquet(PROJECT_ROOT / "data/raw/index_000001.SH.parquet")
    calendar = pd.DatetimeIndex(pd.to_datetime(index.trade_date.astype(str))).unique().sort_values()
    lookback = int(settings["lookback"])
    origins = eligible_origins(frame, calendar, lookback)
    # Cache only history encodings, never targets or full train+future windows.
    parts = {}
    audits = []
    sample_rows = []
    for fold in base.folds:
        for split, start, end, stride in [
            ("train", fold.train_start, fold.train_end, settings["train_stride"]),
            ("validation", fold.validation_start, fold.validation_end, 1),
            ("test", fold.test_start, fold.test_end, 1),
        ]:
            selected = split_origins(frame, origins, start, end, stride)
            if not selected:
                raise ValueError(f"empty {fold.name}/{split}")
            parts[(fold.name, split)] = selected
            old_window = base.experiment.lookback + base.experiment.horizon + 1
            old_stride = base.training.train_stride if split == "train" else base.training.validation_stride
            old_starts = window_start_indices(frame, start, end, old_window, old_stride)
            old_targets = {i + j for i in old_starts for j in range(1, old_window)} if split != "test" else set()
            audits.append({"fold": fold.name, "split": split, "new_samples": len(selected),
                           "stride": stride, "first_origin": str(frame.iloc[selected[0]].date.date()),
                           "last_target": str(frame.iloc[selected[-1] + 1].date.date()),
                           "old_windows_train_or_validation": len(old_starts) if split != "test" else None,
                           "old_unique_supervised_dates": len(old_targets) if split != "test" else None})
            sample_rows.extend({"fold": fold.name, "split": split, "origin": i,
                                "context_start": frame.iloc[i - lookback + 1].date,
                                "origin_date": frame.iloc[i].date,
                                "target_date": frame.iloc[i + 1].date} for i in selected)
    pd.DataFrame(audits).to_csv(out_dir / "sample_audit.csv", index=False)
    pd.DataFrame(sample_rows).to_parquet(out_dir / "samples.parquet", index=False)
    device = resolve_device("auto")
    _set_seed(settings["seed"])
    kronos_class, tokenizer_class, _ = _import_kronos(base.model.source_dir)
    tokenizer = tokenizer_class.from_pretrained(str(base.model.tokenizer_path)).to(device).eval()
    tokenizer.requires_grad_(False)
    unique_origins = sorted({i for group in parts.values() for i in group})
    row_for = {origin: row for row, origin in enumerate(unique_origins)}
    tokens0, tokens1, stamps_all, labels_all = [], [], [], []
    started = time.monotonic()
    for start in range(0, len(unique_origins), settings["batch_size"]):
        batch = unique_origins[start:start + settings["batch_size"]]
        inputs = [historical_input(frame, i, lookback) for i in batch]
        x = torch.tensor(np.stack([pair[0] for pair in inputs]), device=device)
        with torch.no_grad():
            t0, t1 = tokenizer.encode(x, half=True)
        tokens0.append(t0.cpu())
        tokens1.append(t1.cpu())
        stamps_all.append(torch.from_numpy(np.stack([pair[1] for pair in inputs])))
        labels = np.stack([100 * np.log(frame.iloc[i + 1][list(PRICE_COLUMNS)].to_numpy(float) / frame.iloc[i].close) for i in batch])
        labels_all.append(torch.from_numpy(labels.astype(np.float32)))
        if start % (settings["batch_size"] * 20) == 0:
            print(f"tokenizing history {start}/{len(unique_origins)}", flush=True)
    cache = [torch.cat(items) for items in (tokens0, tokens1, stamps_all, labels_all)]
    del tokenizer

    def dataset(selected: list[int]) -> TensorDataset:
        positions = torch.tensor([row_for[i] for i in selected])
        return TensorDataset(*(array[positions] for array in cache))

    predictions, histories, checkpoint_info = [], [], []
    weights = torch.tensor(weights_array, device=device)
    for fold in base.folds:
        _set_seed(settings["seed"])
        train_set = dataset(parts[(fold.name, "train")])
        val_set = dataset(parts[(fold.name, "validation")])
        test_set = dataset(parts[(fold.name, "test")])
        prior = np.median(train_set.tensors[-1].numpy(), axis=0)
        model = NextDayPriceModel(kronos_class.from_pretrained(str(base.model.predictor_path)), prior).to(device)
        head_parameters = list(model.price_head.parameters())
        head_ids = {id(p) for p in head_parameters}
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": settings["backbone_lr"]},
            {"params": head_parameters, "lr": settings["head_lr"]},
        ], weight_decay=settings["weight_decay"])
        batch_size = settings["batch_size"]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(settings["seed"]))
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        best_loss, _ = evaluate(model, val_loader, device, weights, settings["loss_beta"])
        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        stale = 0
        histories.append({"fold": fold.name, "epoch": 0, "train_loss": None, "validation_loss": best_loss})
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            total = 0.0
            seen = 0
            for t0, t1, stamps, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(t0.to(device), t1.to(device), stamps.to(device))
                loss = price_loss(prediction, targets.to(device), weights, settings["loss_beta"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(targets)
                seen += len(targets)
            validation_loss, _ = evaluate(model, val_loader, device, weights, settings["loss_beta"])
            histories.append({"fold": fold.name, "epoch": epoch, "train_loss": total / seen,
                              "validation_loss": validation_loss})
            print(f"{fold.name} epoch {epoch}: train={total / seen:.5f} val={validation_loss:.5f}", flush=True)
            pd.DataFrame(histories).to_csv(out_dir / "training_history.csv", index=False)
            if validation_loss < best_loss:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= settings["patience"]:
                    break
        model.load_state_dict(best)
        from safetensors.torch import save_file

        path = checkpoints / fold.name
        path.mkdir(parents=True, exist_ok=True)
        save_file(best, str(path / "prices.safetensors"))
        metadata = {"fold": fold.name, "best_epoch": best_epoch, "validation_loss": best_loss,
                    "prior_log_return_pct": prior.tolist(), "settings": settings,
                    "state_sha256": file_sha256(path / "prices.safetensors")}
        (path / "config.json").write_text(json.dumps(metadata, indent=2))
        checkpoint_info.append(metadata)
        _, predicted_log = evaluate(model, test_loader, device, weights, settings["loss_beta"])
        test_origins = parts[(fold.name, "test")]
        current_close = np.array([frame.iloc[i].close for i in test_origins])
        predicted_price = np.exp(predicted_log / 100) * current_close[:, None]
        for variant, prices in [
            ("price_regression", predicted_price),
            ("previous_close", np.repeat(current_close[:, None], 4, axis=1)),
            ("train_median_candle", np.exp(prior[None, :] / 100) * current_close[:, None]),
        ]:
            for i, prices_row in zip(test_origins, prices):
                record = {"fold": fold.name, "variant": variant, "origin_date": frame.iloc[i].date,
                          "target_date": frame.iloc[i + 1].date, "current_close": frame.iloc[i].close}
                record.update({f"pred_{f}": float(v) for f, v in zip(PRICE_COLUMNS, prices_row)})
                record.update({f"actual_{f}": float(frame.iloc[i + 1][f]) for f in PRICE_COLUMNS})
                predictions.append(record)
        pd.DataFrame(predictions).to_parquet(out_dir / "predictions.parquet", index=False)
        del model, optimizer
        if device == "mps":
            torch.mps.empty_cache()
    prediction_frame = pd.DataFrame(predictions)
    metrics = metrics_table(prediction_frame)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    manifest: dict[str, Any] = {"assets": assets, "settings": settings, "device": device,
                                "seconds": time.monotonic() - started,
                                "data_sha256": file_sha256(base.experiment.data_path),
                                "data_end": str(frame.date.max().date()),
                                "training_source_sha256": file_sha256(Path(__file__)),
                                "unique_history_windows": len(unique_origins),
                                "sample_audit": audits, "checkpoints": checkpoint_info,
                                "limitations": ["Single stock BYD", "Original local adjusted-price provenance not newly audited", "Previously viewed historical test periods; this is retrospective development, not untouched confirmation", "Pretraining historical overlap not excluded", "Overlapping contexts are correlated, not independent samples"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    write_report(out_dir, metrics, manifest)
    print(metrics.query("fold == 'ALL'").to_string(index=False), flush=True)
    return out_dir
