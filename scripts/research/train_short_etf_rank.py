"""Fit ETF residuals on existing left/right out-of-fold five-day predictions."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from quant.application.left_side_ranking import DEFAULT_LEFT_SIDE_RANKING_CONFIG
from quant.application.selector_ranking import load_selector_ranking_config
from quant.data.atomic_io import atomic_write_json
from quant.features.ownership_flow import KEYS
from quant.features.short_etf_rank import ARTIFACT, MACRO_SCALES, VERSION, apply_residual, market_matrix, stock_matrix
from quant.research.left_side_unified_features import LEFT_SIDE_SIGNALS
from quant.research.right_side_unified_features import RIGHT_SIDE_SIGNALS
from scripts.research.validate_left_side_disclosure_gain import block_interval

HISTORY = ROOT / "data/research/short_side_ownership/etf_features_v2.parquet"
REPORT = ROOT / "reports/research/short_etf_rank_production"
LEFT = ROOT / "data/research/left_side_disclosure_gain_20261029"
LEFT_REPORT = ROOT / "reports/research/left_side_disclosure_gain_20261029"
RIGHT = ROOT / "reports/research/right_side_unified_canonical_v5_rule113/test_predictions.parquet"
STRENGTHS = (.1, .25, .5)
PENALTY = 8.0


def load_side(side: str, signals: Sequence[str]) -> pd.DataFrame:
    if side == "left":
        labels = pd.read_parquet(LEFT / "labels.parquet",
                                 columns=[*KEYS, "good_path5", "terminal_return", "label_end_date"])
        flags = pd.read_parquet(LEFT / "events.parquet", columns=[*KEYS, *signals])
        pred = pd.concat([pd.read_parquet(LEFT_REPORT / f"predictions_{fold}_baseline.parquet")
                          for fold in "ABC"], ignore_index=True)
        return pred.merge(labels, on=KEYS, validate="one_to_one").merge(flags, on=KEYS, validate="one_to_one")
    frame = pd.read_parquet(RIGHT)
    mask = frame.entry_mode.eq("next_close") & frame.horizon.eq(5) & frame.label.eq("good_path5")
    return frame.loc[mask, [*KEYS, *signals, "good_path5", "terminal_return", "pred_unified_long_task_deep"]].rename(
        columns={"pred_unified_long_task_deep": "probability"},
    )


def fit(
    frame: pd.DataFrame, stock: np.ndarray, macro: np.ndarray, selected: np.ndarray,
    earlier: np.ndarray, covariance: np.ndarray, response: np.ndarray,
) -> np.ndarray:
    # Nested date sets allow previous normal equations to be reused. Fixed
    # batches bound memory independent of the number of historical candidates.
    positions = np.flatnonzero(selected & ~earlier)
    for start in range(0, len(positions), 16_384):
        indices = positions[start:start + 16_384]
        x = stock[indices]
        m = macro[indices]
        design = np.column_stack((x, (x[:, :, None] * m[:, None, :]).reshape(len(x), -1)))
        data = frame.iloc[indices]
        weight = np.sqrt(data["date_weight"].to_numpy())
        target = (data.good_path5.to_numpy(dtype=float) - data.probability.to_numpy()) * weight
        design *= weight[:, None]
        covariance += design.T @ design
        response += design.T @ target
    earlier[:] = selected
    fitted = np.linalg.solve(covariance + np.eye(len(covariance)) * PENALTY, response)
    return fitted[stock.shape[1]:].reshape(stock.shape[1], macro.shape[1])


def returns(frame: pd.DataFrame, score: np.ndarray, k: int) -> pd.Series:
    return (frame.assign(score=score).sort_values(["date", "score", "symbol"], ascending=[True, False, True])
            .groupby("date", sort=False).head(k).groupby("date").terminal_return.mean())


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    market = pd.read_parquet(HISTORY, columns=["date", *MACRO_SCALES])
    reports, models = [], {}
    specs = (
        ("left", tuple(LEFT_SIDE_SIGNALS), 20, DEFAULT_LEFT_SIDE_RANKING_CONFIG.paths.artifact),
        ("right", tuple(RIGHT_SIDE_SIGNALS), 10, load_selector_ranking_config(ROOT).paths.artifact),
    )
    with threadpool_limits(limits=4):
        for side, signals, k, artifact in specs:
            frame = load_side(side, signals).dropna(subset=["good_path5", "probability", "terminal_return"])
            frame = frame.merge(market, on="date", how="left", validate="many_to_one")
            stock = stock_matrix(frame, frame.probability.to_numpy(), signals)
            macro, ready = market_matrix(frame)
            frame["date_weight"] = 1 / frame.groupby("date").date.transform("size")
            size = stock.shape[1] * (1 + macro.shape[1])
            covariance, response = np.zeros((size, size)), np.zeros(size)
            earlier = np.zeros(len(frame), dtype=bool)
            records = []
            for year in (2025, 2026):
                boundary = pd.Timestamp(f"{year}-01-01")
                # The right OOF artifact has no label-end column. Seven
                # exchange sessions conservatively purge next-close/H5 outcomes.
                purge_date = market.loc[market.date.lt(boundary), "date"].iloc[-7]
                selected = (frame.date.lt(purge_date) & ready).to_numpy(copy=True)
                if "label_end_date" in frame:
                    selected &= frame.label_end_date.lt(boundary).to_numpy()
                coefficient = fit(frame, stock, macro, selected, earlier, covariance, response)
                test_mask = frame.date.dt.year.eq(year).to_numpy()
                test = frame.loc[test_mask, [*KEYS, "terminal_return", "good_path5", "probability"]]
                baseline = test.probability.to_numpy()
                reference = returns(test, baseline, k)
                for strength in STRENGTHS:
                    candidate = apply_residual(stock[test_mask], macro[test_mask], baseline, coefficient, strength)
                    candidate = np.where(ready[test_mask], candidate, baseline)
                    delta = (returns(test, candidate, k) - reference).to_numpy()
                    records.append({
                        "side": side, "year": year, "k": k, "strength": strength,
                        "train_rows": int(selected.sum()), "rows": len(test), "days": len(delta),
                        "coverage": float(ready[test_mask].mean()),
                        "delta_bps": float(delta.mean() * 10000),
                        "interval_bps": [bound * 10000 for bound in block_interval(delta)],
                        "ap_delta": float(average_precision_score(test.good_path5, candidate)
                                          - average_precision_score(test.good_path5, baseline)),
                    })
            # Select on 2025 only. The 2026 record is a held-out diagnostic.
            selected_record = max((row for row in records if row["year"] == 2025), key=lambda row: row["delta_bps"])
            strength = selected_record["strength"]
            # Publish only the confirmed layer. An unconfirmed side is available
            # to score for observation but does not change the ordering.
            diagnostic = next(row for row in records if row["year"] == 2026 and row["strength"] == strength)
            active = selected_record["delta_bps"] > 0 and diagnostic["delta_bps"] >= 0
            models[side] = {
                "signals": list(signals), "base_sha256": sha(artifact),
                "strength": strength, "active": active, "target": "good_path5",
                "trained_through": frame.date.max().date().isoformat(),
                "coefficients": fit(frame, stock, macro, ready, earlier, covariance, response).tolist(),
            }
            reports.extend(records)
            print(side, json.dumps(records), "active", active, flush=True)
    report = {"schema": VERSION, "history_sha256": sha(HISTORY), "results": reports,
              "selection": "select by 2025 daily Top-K; require non-negative 2026 held-out increment"}
    atomic_write_json(report, REPORT / "validation.json")
    atomic_write_json({"schema": VERSION, "market_features": list(MACRO_SCALES), "sides": models},
                      ROOT / ARTIFACT)


if __name__ == "__main__":
    main()
