"""Quarter-window walk-forward refitting of frozen Kronos embeddings."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant.research.kronos_multiscale import fit_subscores, predict_subscores
from quant.research.sequence_strategy_validation import choose_confidence_gate

FUSION_FEATURES = ["baseline_percentile", "sequence_return", "sequence_profit", "sequence_downside"]


def refit_segments(
    dates: pd.DatetimeIndex, frequency: str
) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    if frequency not in {"monthly", "weekly"}:
        raise ValueError("frequency must be monthly or weekly")
    dates = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    groups = dates.to_period("M" if frequency == "monthly" else "W-SUN")
    return [(dates[groups == period][0], dates[groups == period]) for period in groups.unique()]


def mature_window(frame: pd.DataFrame, refit_date: pd.Timestamp) -> pd.DataFrame:
    start = refit_date - pd.DateOffset(months=3)
    return frame[
        frame.date.ge(start) & frame.date.lt(refit_date) & frame.exit_date.lt(refit_date)
    ].copy()


def stacking_window(
    frame: pd.DataFrame, refit_date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = mature_window(frame, refit_date)
    boundary = refit_date - pd.DateOffset(months=1)
    return frame[frame.date.lt(boundary) & frame.exit_date.lt(boundary)].copy(), frame[
        frame.date.ge(boundary)
    ].copy()


def enough(frame: pd.DataFrame, minimum: int = 30) -> bool:
    return len(frame) >= minimum and frame.net_return.gt(0).nunique() == 2


def fusion_model(frame: pd.DataFrame):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    model.fit(frame[FUSION_FEATURES], frame.net_return.gt(0))
    return model


def attach_subscores(frame: pd.DataFrame, head, embeddings: np.ndarray) -> pd.DataFrame:
    frame = frame.copy()
    values = predict_subscores(head, embeddings[frame.embedding_row])
    frame[["sequence_return", "sequence_profit", "sequence_downside"]] = values
    return frame


def rolling_predictions(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    dates: pd.DatetimeIndex,
    frequency: str,
    model_dir: Path,
    *,
    epochs: int = 20,
) -> tuple[pd.DataFrame, list[dict]]:
    frame = frame.copy()
    frame["baseline_percentile"] = frame.groupby("date").baseline_score.rank(pct=True)
    output = frame[frame.date.isin(dates)].copy()
    columns = ["rolling_sequence", "fusion_only", "rolling_joint", "rolling_cash"]
    for column in columns:
        output[column] = np.nan
    output["refit_date"] = pd.NaT
    audit = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for refit, segment in refit_segments(dates, frequency):
        pending = output[output.date.isin(segment)]
        if pending.empty:
            continue
        train = mature_window(frame, refit)
        early, meta = stacking_window(frame, refit)
        record = {
            "refit_date": refit,
            "last_prediction_date": pending.date.max(),
            "train_start": refit - pd.DateOffset(months=3),
            "train_rows": len(train),
            "max_train_exit": train.exit_date.max(),
            "train_win_rate": float(train.net_return.gt(0).mean()) if len(train) else None,
            "early_rows": len(early),
            "meta_rows": len(meta),
            "forecast_rows": len(pending),
            "early_max_exit": early.exit_date.max(),
            "meta_min_date": meta.date.min(),
            "cash_threshold": None,
            "status": "insufficient_training_cash",
        }
        if not enough(train, 60):
            audit.append(record)
            continue
        if train.exit_date.max() >= refit:
            raise ValueError("unmatured label at refit")
        stem = refit.strftime("%Y%m%d")
        # Existing 2024 heads are frozen; their 2025/2026 scores are time-out predictions.
        recalibrated = fusion_model(train)
        output.loc[pending.index, "fusion_only"] = recalibrated.predict_proba(
            pending[FUSION_FEATURES]
        )[:, 1]
        joblib.dump(recalibrated, model_dir / f"{stem}_fusion_only.joblib")
        head = fit_subscores(
            embeddings[train.embedding_row],
            train.net_return.to_numpy(),
            epochs=epochs,
            seed=20261021,
        )
        torch.save(head.state_dict(), model_dir / f"{stem}_head.pt")
        updated = attach_subscores(pending, head, embeddings)
        output.loc[pending.index, "rolling_sequence"] = updated.sequence_profit
        output.loc[pending.index, "refit_date"] = refit
        record["status"] = "head_and_recalibration_ready"
        if enough(early) and enough(meta):
            auxiliary = fit_subscores(
                embeddings[early.embedding_row],
                early.net_return.to_numpy(),
                epochs=epochs,
                seed=20261021,
            )
            meta_scores = attach_subscores(meta, auxiliary, embeddings)
            joint = fusion_model(meta_scores)
            output.loc[pending.index, "rolling_joint"] = joint.predict_proba(
                updated[FUSION_FEATURES]
            )[:, 1]
            joblib.dump(joint, model_dir / f"{stem}_joint.joblib")
            torch.save(auxiliary.state_dict(), model_dir / f"{stem}_auxiliary_head.pt")
            # Gate selection gets a further temporal holdout inside the meta month.
            boundary = refit - pd.DateOffset(months=1) + pd.Timedelta(days=15)
            meta_fit = meta_scores[
                meta_scores.date.lt(boundary) & meta_scores.exit_date.lt(boundary)
            ]
            meta_holdout = meta_scores[meta_scores.date.ge(boundary)].copy()
            record.update(
                {
                    "gate_fit_rows": len(meta_fit),
                    "gate_holdout_rows": len(meta_holdout),
                    "gate_fit_max_exit": meta_fit.exit_date.max(),
                    "gate_holdout_min_date": meta_holdout.date.min(),
                }
            )
            if enough(meta_fit) and len(meta_holdout) >= 30:
                gate_model = fusion_model(meta_fit)
                meta_holdout["gate_probability"] = gate_model.predict_proba(
                    meta_holdout[FUSION_FEATURES]
                )[:, 1]
                threshold, search = choose_confidence_gate(meta_holdout, "gate_probability")
                record["cash_threshold"] = threshold
                joblib.dump(
                    {"model": gate_model, "threshold": threshold, "selection": search},
                    model_dir / f"{stem}_gate.joblib",
                )
                # Use exactly the gate model that produced holdout probabilities.
                probability = gate_model.predict_proba(updated[FUSION_FEATURES])[:, 1]
                if threshold is not None:
                    output.loc[pending.index, "rolling_cash"] = np.where(
                        probability >= threshold, probability, np.nan
                    )
            record["status"] = "joint_ready"
        audit.append(record)
        print(
            f"{frequency} {stem}: mature={len(train)} early={len(early)} meta={len(meta)} forecast={len(pending)} gate={record['cash_threshold']}",
            flush=True,
        )
    return output, audit
