"""Causal GRU price/event tasks and forward-prediction stacking diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from audit_overnight_right_provenance import sha256
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from torch import nn
from train_overnight_unfiltered_comparison import date_weights
from train_right_t012 import KEYS, choose, summarize

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


class PriceSequence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.GRU(5, 32, batch_first=True)
        self.head = nn.Linear(32, 12)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(values)
        raw = self.head(hidden[-1])
        open_, close = raw[:, 0], raw[:, 3]
        high = torch.maximum(open_, close) + nn.functional.softplus(raw[:, 1])
        low = torch.minimum(open_, close) - nn.functional.softplus(raw[:, 2])
        return torch.cat([torch.stack([open_, high, low, close], dim=1), raw[:, 4:]], dim=1)


def sequence_windows(bars: pd.DataFrame, events: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """32 consecutive global sessions ending on the entry date, never T+2."""
    result = np.zeros((len(events), 32, 5), dtype=np.float32)
    ready = np.zeros(len(events), dtype=bool)
    positions = pd.Series(np.arange(len(calendar)), index=calendar)
    histories = {symbol: group for symbol, group in bars.groupby("symbol", sort=False)}
    for symbol, group in events.groupby("symbol"):
        if symbol not in histories:
            continue
        b = histories[symbol].sort_values("date").reset_index(drop=True)
        if b.date.duplicated().any():
            raise ValueError("Duplicate bars")
        if len(b) < 51:
            continue
        price = b[["open", "high", "low", "close"]].div(b.pre_close, axis=0).sub(1).div(.03)
        volume = np.log(b.vol / b.vol.rolling(20, min_periods=20).mean())
        values = np.column_stack([price, volume]).astype(np.float32)
        finite = np.isfinite(values).all(axis=1) & b.valid_bar.to_numpy(bool)
        indices = pd.Index(b.date).get_indexer(group.date)
        market_positions = positions.reindex(b.date).to_numpy()
        for row, end in zip(group.index, indices, strict=True):
            start = end - 31
            if start < 19 or not finite[start:end + 1].all():
                continue
            if not np.all(np.diff(market_positions[start:end + 1]) == 1):
                continue
            result[row] = np.clip(values[start:end + 1], -10, 10)
            ready[row] = True
    return result, ready


def fit_sequence(x: np.ndarray, prices: np.ndarray, events: np.ndarray, buckets: np.ndarray,
                 weights: np.ndarray, mask: np.ndarray) -> tuple[PriceSequence, list[float]]:
    torch.manual_seed(20260913)
    model = PriceSequence()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    indices = np.flatnonzero(mask)
    generator = np.random.default_rng(20260913)
    history = []
    for epoch in range(6):
        total = 0.
        for batch in np.array_split(generator.permutation(indices), max(1, int(np.ceil(len(indices) / 512)))):
            pred = model(torch.from_numpy(x[batch]))
            loss_price = nn.functional.smooth_l1_loss(pred[:, :4], torch.from_numpy(prices[batch] / .03), reduction="none")
            loss_price = (loss_price * torch.tensor([.2, .3, .2, .3])).sum(dim=1)
            loss_event = nn.functional.binary_cross_entropy_with_logits(pred[:, 4:7], torch.from_numpy(events[batch]), reduction="none").mean(dim=1)
            loss_bucket = nn.functional.cross_entropy(pred[:, 7:], torch.from_numpy(buckets[batch]), reduction="none")
            loss = ((loss_price + .3 * loss_event + .2 * loss_bucket) * torch.from_numpy(weights[batch])).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / len(indices))
        print(f"GRU n={len(indices)} epoch={epoch + 1} loss={history[-1]:.6f}", flush=True)
    return model.eval(), history


def sequence_predict(model: PriceSequence, x: np.ndarray) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            raw = model(torch.from_numpy(x[start:start + 1024]))
            parts.append(torch.cat([raw[:, :4] * .03, raw[:, 4:7].sigmoid(), raw[:, 7:].softmax(dim=1)], dim=1).numpy())
    return np.concatenate(parts)


def main() -> None:
    torch.set_num_threads(4)
    root = PROJECT_ROOT / "reports/research"
    parent_root = root / "right_t012_baseline_v2"
    output = root / "right_t012_sequence_fusion_v1"
    parent = json.loads((parent_root / "protocol.json").read_text())
    source_hashes = dict(parent["sources"])
    bars_path = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    source_hashes[str(bars_path)] = sha256(bars_path)
    source_hashes[str(parent_root / "predictions.parquet")] = sha256(parent_root / "predictions.parquet")
    for path, digest in parent["sources"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Parent source changed: {path}")
    output.mkdir(exist_ok=False)
    seq_columns = [f"seq_return_{p}" for p in ["open", "high", "low", "close"]] + ["seq_prob_profit", "seq_prob_hit1", "seq_prob_loss2"] + [f"seq_bucket_{i}" for i in range(5)]
    protocol = dict(sources=source_hashes, sequence="32 consecutive market sessions, OHLC/pre_close-1 divided by.03; log volume/20day mean; clipinputs+-10",
                    architecture="GRU(5,32,1) +12 outputs; constrained high>=max(open,close),low<=min(open,close)",
                    loss="Price SmoothL1 weights O/H/L/C=.2/.3/.2/.3 +.3 mean BCE(profit20,hit1,loss2) +.2 CE(close return bins[-.03,-.01,.01,.03]); date weights",
                    optimization="6epochs AdamW lr.001 weight_decay.01 batch512 seed20260913; CPU4",
                    forward_folds=["July with exit<Aug1 -> August", "July+August with exit<Sep1 -> September+Q4"],
                    stacking="Both table control and fusion fit August ONLY; September calibration+selection; Q4 reused diagnostic",
                    sequence_features=seq_columns, missing="No sequence window: unavailable, kept in coverage, no zero-valued fake prediction",
                    arms=["table_matched", "sequence_only", "fusion"], targets=["net_close", "limit1"],
                    limitations=["All dates reused historical research", "Not pretrained Kronos", "Only198 admitted table inputs, not700+", "No FM yet", "Volume input only, no volume target", "14:30 and fills unverified"],
                    script_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    predictions = pd.read_parquet(parent_root / "predictions.parquet")
    base = predictions.loc[predictions.target.eq("limit1")].reset_index(drop=True)
    close = predictions.loc[predictions.target.eq("net_close"), KEYS + ["net"]].rename(columns={"net": "close_net"})
    base = base.merge(close, on=KEYS, validate="one_to_one")
    bars = pd.read_parquet(bars_path)
    calendar = pd.DatetimeIndex(bars.date.unique()).sort_values()
    bars = bars.loc[bars.date.between("2025-04-01", "2025-12-31")].copy()
    windows, ready = sequence_windows(bars, base, calendar)
    del bars
    pd.DataFrame(dict(symbol=base.symbol, date=base.date, ready=ready)).to_parquet(output / "sequence_coverage.parquet", index=False)
    prices = base[["next_open", "next_high", "next_low", "next_close"]].div(base.entry_close, axis=0).sub(1).to_numpy(np.float32)
    labels = np.column_stack([base.close_net.gt(0), base.event_hit.fillna(False), base.close_net.lt(-.02)]).astype(np.float32)
    buckets = np.digitize(prices[:, 3], [-.03, -.01, .01, .03]).astype(np.int64)
    known = ready & base.net.notna().to_numpy() & base.close_net.notna().to_numpy() & np.isfinite(prices).all(axis=1)
    folds = [("august_oof", "2025-08-01", "2025-08-01", "2025-09-01"),
             ("september_q4", "2025-09-01", "2025-09-01", "2026-01-01")]
    seq_values = np.full((len(base), 12), np.nan, dtype=np.float32)
    fold_records = []
    for name, cutoff, start, end in folds:
        fit = known & base.date.lt(cutoff).to_numpy() & base.planned_exit_date.lt(cutoff).to_numpy()
        predict = ready & base.date.ge(start).to_numpy() & base.date.lt(end).to_numpy()
        weights = np.ones(len(base), dtype=np.float32)
        weights[fit] = date_weights(base.loc[fit]).to_numpy(np.float32)
        model, losses = fit_sequence(windows, prices, labels, buckets, weights, fit)
        torch.save(model.state_dict(), output / f"{name}.pt")
        seq_values[predict] = sequence_predict(model, windows[predict])
        fold_records.append(dict(fold=name, fit=int(fit.sum()), predictions=int(predict.sum()),
                                 fit_max_exit=str(base.loc[fit, "planned_exit_date"].max()),
                                 prediction_min_entry=str(base.loc[predict, "date"].min()), losses=losses))
    seq = pd.DataFrame(seq_values, columns=seq_columns)
    seq_export = pd.concat([base[KEYS + ["entry_close"]], seq], axis=1)
    for price in ["open", "high", "low", "close"]:
        seq_export[f"seq_price_{price}"] = base.entry_close * (1 + seq[f"seq_return_{price}"])
    seq_export.to_parquet(output / "sequence_variables.parquet", index=False)
    (output / "forward_folds.json").write_text(json.dumps(fold_records, indent=2))
    frame = pd.concat([base, seq], axis=1)
    filters = [("date", ">=", pd.Timestamp("2025-07-01")), ("date", "<", pd.Timestamp("2026-01-01"))]
    aux = pd.read_parquet(root / "overnight_unfiltered_auxiliary_v1/dataset.parquet", filters=filters)
    factors = pd.read_parquet(root / "overnight_unfiltered_full_factors_v1/dataset.parquet", columns=KEYS + list(PROJECT_FACTOR_COLUMNS), filters=filters)
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(aux, on=KEYS, validate="one_to_one", how="left").merge(factors, on=KEYS, validate="one_to_one", how="left")
    pd.testing.assert_frame_equal(frame[KEYS], base[KEYS])
    available = frame[seq_columns].notna().all(axis=1)
    split_masks = {
        "fit": available & frame.date.ge("2025-08-01") & frame.date.lt("2025-09-01") & frame.planned_exit_date.lt("2025-09-01"),
        "calibration": available & frame.split.eq("calibration"),
        "q4_diagnostic": available & frame.split.eq("q4_diagnostic"),
    }
    all_summary, all_picks, all_grids = [], [], []
    for arm, columns in [("table_matched", parent["features"]), ("sequence_only", seq_columns), ("fusion", parent["features"] + seq_columns)]:
        x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
        for target in ["net_close", "limit1"]:
            net = frame.close_net if target == "net_close" else frame.net
            event = net.gt(0).astype("boolean").mask(net.isna()) if target == "net_close" else frame.event_hit
            fm, cm = split_masks["fit"] & net.notna(), split_masks["calibration"] & net.notna()
            with threadpool_limits(limits=4):
                clf = HistGradientBoostingClassifier(**parent["params"]).fit(x[fm], event[fm].astype(int), sample_weight=date_weights(frame[fm]))
                reg = HistGradientBoostingRegressor(**parent["params"]).fit(x[fm], net[fm], sample_weight=date_weights(frame[fm]))
                iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(x[cm])[:, 1], event[cm].astype(int), sample_weight=date_weights(frame[cm]))
                p = iso.predict(clf.predict_proba(x)[:, 1])
                expected = reg.predict(x)
            joblib.dump(dict(classifier=clf, regressor=reg, calibrator=iso, features=columns), output / f"{arm}_{target}.joblib")
            pred = base.copy().assign(arm=arm, target=target, net=net, event_hit=event, probability=p, expected_net=expected)
            pred.to_parquet(output / f"{arm}_{target}_predictions.parquet", index=False)
            cal = pred.loc[split_masks["calibration"]]
            days_cal = calendar[(calendar >= "2025-09-01") & (calendar < "2025-09-30")]
            grid = []
            for threshold in [0., .5, .6, .7, .8, .9]:
                for rank in ["probability", "expected_net"]:
                    selected = choose(cal, threshold, rank)
                    grid.append(dict(arm=arm, target=target, threshold=threshold, rank=rank,
                                     **summarize(selected.assign(net=selected.net - .002), days_cal)))
            grid_frame = pd.DataFrame(grid)
            all_grids.append(grid_frame)
            accepted = grid_frame.loc[grid_frame.dates.ge(10)]
            winner = None if accepted.empty else accepted.sort_values(["daily_score", "threshold", "rank"], ascending=[False, True, True]).iloc[0]
            for split in ["calibration", "q4_diagnostic"]:
                subset = pred.loc[split_masks[split]]
                picked = subset.iloc[:0].copy() if winner is None else choose(subset, float(winner.threshold), str(winner["rank"]))
                all_picks.append(picked)
                start, end = ("2025-09-01", "2025-09-30") if split == "calibration" else ("2025-10-01", "2025-12-31")
                days = calendar[(calendar >= start) & (calendar < end)]
                for cost in [20, 40]:
                    all_summary.append(dict(arm=arm, target=target, split=split, cost_bps=cost,
                                            **summarize(picked.assign(net=picked.net - (cost - 20) / 10000), days)))
            print(f"Completed {arm}/{target}", flush=True)
    picks = pd.concat(all_picks, ignore_index=True)
    picks.to_parquet(output / "picks.parquet", index=False)
    picks.loc[picks.net.isna() | picks.net.le(0) | ~picks.event_hit.fillna(False)].to_parquet(output / "badcases.parquet", index=False)
    pd.DataFrame(all_summary).to_csv(output / "summary.csv", index=False)
    pd.concat(all_grids).to_csv(output / "calibration_grid.csv", index=False)
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during run: {path}")
    (output / "complete.json").write_text(json.dumps(dict(rows=len(base), sequence_ready=int(ready.sum()),
        sequence_features=12, fusion_features=210, picks_sha256=sha256(output / "picks.parquet"),
        status="Sequence forward stacking diagnostic complete, not goal confirmation"), indent=2))
    print(pd.DataFrame(all_summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
