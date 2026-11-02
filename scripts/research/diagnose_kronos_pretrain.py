#!/usr/bin/env python
"""Compare frozen pretraining targets against persistence without fitting parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant.research.kronos_multiscale import (
    SCALES,
    KronosEOS,
    context_and_target,
    period_bars,
    validate_bars,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    config = json.loads((out / "protocol.json").read_text())["config"]
    symbols = json.loads((out / "coverage.json").read_text())["sampled_symbols"]
    paths = sorted(
        {
            p["path"]
            for p in json.loads((out / "raw_sources.json").read_text())
            if "daily_partitioned/year_month=" in p["path"]
        }
    )
    parts = []
    for path in paths:
        part = pd.read_parquet(
            path,
            columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol"],
            filters=[("ts_code", "in", symbols)],
        )
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True).rename(columns={"ts_code": "symbol"})
    raw["date"] = pd.to_datetime(raw.trade_date.astype(str), format="%Y%m%d")
    raw = raw.sort_values(["symbol", "date"])
    periods = {}
    for symbol in symbols:
        source = raw[raw.symbol.eq(symbol)].copy()
        source = source[source[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)]
        if source.empty:
            continue
        index = np.exp(np.log(source.close / source.pre_close).cumsum()) * 100
        adjusted = source[["date", "open", "high", "low", "close"]].copy()
        adjusted[["open", "high", "low", "close"]] = source[["open", "high", "low", "close"]].mul(
            index / source.close, axis=0
        )
        adjusted["volume"] = source.vol.to_numpy()
        daily = validate_bars(adjusted)
        periods[symbol] = {s: period_bars(daily, s) for s in SCALES}
    generator = np.random.default_rng(config["seed"])
    model = KronosEOS(root / config["source_dir"], **config["encoder"]).to("mps")
    model.load_state_dict(torch.load(out / "encoder.pt", map_location="mps", weights_only=True))
    model.eval()
    torch.set_num_threads(4)
    cutoff = pd.Timestamp(config["pretrain_end"])
    validation_end = pd.Timestamp(config["pretrain_validation_end"])
    records = []
    for scale_id, scale in enumerate(SCALES):
        datasets = {"train": ([], []), "validation": ([], [])}
        for symbol, series in periods.items():
            frame = series[scale]
            length = config["lengths"][scale]
            train_ends = [
                i
                for i in range(length - 1, len(frame) - 1)
                if frame.iloc[i + 1].available_at <= cutoff
            ]
            val_ends = [
                i
                for i in range(length - 1, len(frame) - 1)
                if cutoff < frame.iloc[i].available_at
                and frame.iloc[i + 1].available_at <= validation_end
            ]
            for split, ends in [("train", train_ends), ("validation", val_ends)]:
                sampled = generator.choice(
                    ends, min(len(ends), config["pretrain_windows_per_symbol_scale"]), replace=False
                )
                for end in sampled:
                    x, y = context_and_target(frame, int(end), length)
                    datasets[split][0].append(x)
                    datasets[split][1].append(y)
        for split, (xx, yy) in datasets.items():
            x = np.stack(xx)
            y = np.stack(yy)
            pred = []
            with torch.no_grad():
                for start in range(0, len(x), 32):
                    pred.append(
                        model.next_bar(
                            model(torch.from_numpy(x[start : start + 32]).to("mps"), scale_id)
                        ).cpu()
                    )
            prediction = torch.cat(pred)
            target = torch.from_numpy(y)
            for name, p in [
                ("model", prediction),
                ("last_bar", torch.from_numpy(x[:, -1])),
                ("zero", torch.zeros_like(target)),
            ]:
                losses = torch.nn.functional.smooth_l1_loss(p, target, reduction="none").numpy()
                records.append(
                    {
                        "scale": scale,
                        "split": split,
                        "predictor": name,
                        "samples": len(x),
                        "loss": float(losses.mean()),
                        "price_loss": float(losses[:, :4].mean()),
                        "volume_loss": float(losses[:, 4].mean()),
                    }
                )
        print(scale, "complete", flush=True)
    pd.DataFrame(records).to_csv(out / "diagnostic_pretrain.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
