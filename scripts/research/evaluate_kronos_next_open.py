"""Read-only checkpoint evaluation: next-open diagnostics on the original test origins."""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
import torch

from quant.research.kronos_walk_forward import (
    KLINE_FEATURES,
    PROJECT_ROOT,
    _set_seed,
    audit_model_assets,
    file_sha256,
    load_config,
    load_daily_frame,
    resolve_device,
)


def describe(group: pd.DataFrame) -> dict[str, object]:
    n = len(group)
    if not n:
        return {"n": 0}
    actual = group.actual_open / group.current_close - 1
    predicted = group.predicted_open / group.current_close - 1
    # Flat means an absolute return below 1e-10, to exclude floating point noise.
    actual_sign = np.where(actual.abs() < 1e-10, 0, np.sign(actual))
    predicted_sign = np.where(predicted.abs() < 1e-10, 0, np.sign(predicted))
    correct = int((actual_sign == predicted_sign).sum())
    rate = correct / n
    z = 1.96
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    error = (group.predicted_open - group.actual_open) / group.current_close
    return {
        "n": n,
        "correct": correct,
        "accuracy": rate,
        "wilson95_low": center - half,
        "wilson95_high": center + half,
        "mae_pct_of_previous_close": float(error.abs().mean() * 100),
        "rmse_pct_of_previous_close": float(np.sqrt((error ** 2).mean()) * 100),
        "within_0p5pct": float((error.abs() <= .005).mean()),
        "within_1pct": float((error.abs() <= .01).mean()),
        "actual_high_open": int((actual_sign > 0).sum()),
        "actual_low_open": int((actual_sign < 0).sum()),
        "actual_flat_open": int((actual_sign == 0).sum()),
    }


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/kronos_byd_walk_forward.yaml")
    assets = audit_model_assets(config)
    original_dir = config.experiment.output_dir
    old = pd.read_parquet(original_dir / "predictions.parquet")
    frame = load_daily_frame(config.experiment.data_path, config.experiment.symbol)
    original_manifest = json.loads((original_dir / "manifest.json").read_text())
    frame = frame.loc[frame.date <= pd.Timestamp(original_manifest["audit"]["data"]["end"])].reset_index(drop=True)
    # Use locally cached exchange-index sessions to ensure next row is next session.
    index = pd.read_parquet(PROJECT_ROOT / "data/raw/index_000001.SH.parquet")
    calendar = pd.DatetimeIndex(pd.to_datetime(index.trade_date.astype(str))).unique().sort_values()
    device = resolve_device("auto")
    sys.path.insert(0, str(config.model.source_dir))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(config.model.tokenizer_path)).to(device).eval()
    out_dir = original_dir / "next_open_diagnostic_20260912"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    checkpoint_hashes: dict[str, str] = {}
    max_close_difference = 0.0
    for fold in config.folds:
        reference = old[(old.fold == fold.name) & (old.variant == "finetuned")].sort_values(["origin_date", "step"])
        dates = reference.origin_date.drop_duplicates().tolist()
        positions = [int(frame.index[frame.date == date][0]) for date in dates]
        for position in positions:
            date = frame.iloc[position].date
            following = calendar[calendar > date]
            if not len(following) or following[0] != frame.iloc[position + 1].date:
                raise ValueError(f"Not the next exchange session: {date}")
        for variant in ("pretrained", "finetuned"):
            checkpoint = config.model.predictor_path if variant == "pretrained" else config.experiment.checkpoint_dir / fold.name / "best_model"
            checkpoint_hashes[f"{fold.name}/{variant}"] = file_sha256(checkpoint / "model.safetensors")
            model = Kronos.from_pretrained(str(checkpoint)).to(device).eval()
            predictor = KronosPredictor(model, tokenizer, device=device, max_context=config.model.max_context)
            for seed_offset in (0, 10000, 20000):
                for batch_number, start in enumerate(range(0, len(positions), config.inference.batch_size)):
                    batch = positions[start:start + config.inference.batch_size]
                    contexts = [frame.iloc[i - 89:i + 1] for i in batch]
                    futures = [frame.iloc[i + 1:i + 11] for i in batch]
                    _set_seed(config.training.seed + batch_number + seed_offset)
                    forecasts = predictor.predict_batch(
                        [c.loc[:, KLINE_FEATURES].copy() for c in contexts],
                        [pd.Series(c.date.to_numpy()) for c in contexts],
                        [pd.Series(f.date.to_numpy()) for f in futures],
                        pred_len=10, T=config.inference.temperature,
                        top_k=config.inference.top_k, top_p=config.inference.top_p,
                        sample_count=config.inference.sample_count, verbose=False,
                    )
                    for position, forecast, future in zip(batch, forecasts, futures):
                        date = frame.iloc[position].date
                        prior = old[(old.fold == fold.name) & (old.variant == variant) & (old.origin_date == date)].sort_values("step")
                        np.testing.assert_allclose(prior.actual_close, future.close, rtol=0, atol=1e-10)
                        np.testing.assert_allclose(prior.current_close, frame.iloc[position].close, rtol=0, atol=1e-10)
                        if seed_offset == 0:
                            delta = float(np.abs(prior.predicted_close.to_numpy() - forecast.close.to_numpy()).max())
                            max_close_difference = max(max_close_difference, delta)
                        records.append({
                            "fold": fold.name, "variant": variant, "seed_offset": seed_offset,
                            "origin_date": date, "target_date": future.iloc[0].date,
                            "current_close": float(frame.iloc[position].close),
                            "actual_open": float(future.iloc[0].open),
                            "predicted_open": float(forecast.iloc[0].open),
                        })
                print(f"{fold.name} {variant} seed={seed_offset} completed", flush=True)
            del predictor, model
            if device == "mps":
                torch.mps.empty_cache()

    predictions = pd.DataFrame(records)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    baseline = predictions[(predictions.variant == "finetuned") & (predictions.seed_offset == 0)].copy()
    baseline["variant"] = "previous_close"
    baseline["predicted_open"] = baseline.current_close
    summaries: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    combined = pd.concat([predictions, baseline], ignore_index=True)
    for (variant, seed_offset), subset in combined.groupby(["variant", "seed_offset"]):
        for fold_name, group in [("ALL", subset), *list(subset.groupby("fold"))]:
            summaries.append({"variant": variant, "seed_offset": seed_offset, "fold": fold_name, **describe(group)})
        if variant == "previous_close":
            continue
        gap = subset.predicted_open / subset.current_close - 1
        for direction, sign in [("high", 1), ("low", -1)]:
            for threshold in (0, .005, .01, .02):
                selected = subset[gap * sign > threshold]
                conditions.append({"variant": variant, "seed_offset": seed_offset,
                                   "direction": direction, "threshold": threshold,
                                   "coverage": len(selected) / len(subset), **describe(selected)})
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(conditions).to_csv(out_dir / "conditions.csv", index=False)
    manifest = {"assets": assets, "checkpoint_hashes": checkpoint_hashes, "device": device,
                "data_sha256": file_sha256(config.experiment.data_path),
                "original_close_max_abs_difference_seed0": max_close_difference,
                "next_session_check": "passed using Shanghai Composite session dates",
                "definition": "sign(next open / previous close - 1); prices as stored in original experiment",
                "limitations": ["Only BYD, original 57 origins through May 2026", "Subgroups are exploratory, not independently validated", "Wilson intervals approximate; market dependence and subgroup selection not included", "Three seeds reuse the same outcomes and are not independent samples", "No overnight-specific training has been performed", "Corporate action price provenance not newly audited"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summaries).query("seed_offset == 0").to_string(index=False))
    print(pd.DataFrame(conditions).query("variant == 'finetuned'").to_string(index=False))


if __name__ == "__main__":
    main()
