#!/usr/bin/env python
"""Read-only model diagnosis; never tune on previously observed test outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant.research.sequence_strategy_validation import purged_split

SCORES = ["baseline_score", "sequence_profit", "fusion_score"]
FEATURES = [
    "return1",
    "return5",
    "return20",
    "return60",
    "volume5",
    "volume20",
    "close_position",
    "range",
    "ma20_distance",
    "volatility20",
]


def psi(reference, current):
    edges = np.unique(np.r_[-np.inf, np.quantile(reference, np.arange(0.1, 1, 0.1)), np.inf])
    p = np.histogram(reference, edges)[0].astype(float) + 0.5
    q = np.histogram(current, edges)[0].astype(float) + 0.5
    p /= p.sum()
    q /= q.sum()
    return float(((q - p) * np.log(q / p)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output
    metrics = []
    shifts = []
    domains = []
    fusions = []
    embeddings = np.load(output / "embeddings.npy", mmap_mode="r")
    for side in ["left", "right"]:
        frame = pd.read_parquet(output / f"{side}_scored_events.parquet")
        splits = purged_split(frame)
        train = splits["train"]
        for split, part in splits.items():
            y = part.net_return.gt(0).astype(int)
            for score in SCORES:
                p = part[score].clip(1e-7, 1 - 1e-7)
                daily = []
                normalized = part.copy()
                normalized["percentile"] = normalized.groupby("date")[score].rank(pct=True)
                top = normalized[normalized.percentile.ge(0.8)]
                bottom = normalized[normalized.percentile.le(0.2)]
                for date, group in part.groupby("date"):
                    if (
                        len(group) >= 5
                        and group[score].nunique() > 1
                        and group.net_return.nunique() > 1
                    ):
                        daily.append(spearmanr(group[score], group.net_return).statistic)
                metrics.append(
                    {
                        "side": side,
                        "split": split,
                        "score": score,
                        "rows": len(part),
                        "base_win_rate": float(y.mean()),
                        "base_return": float(part.net_return.mean()),
                        "auc": float(roc_auc_score(y, p)),
                        "brier": float(brier_score_loss(y, p)),
                        "constant_train_prior_brier": float(
                            brier_score_loss(y, np.full(len(y), train.net_return.gt(0).mean()))
                        ),
                        "logloss": float(log_loss(y, p)),
                        "mean_score": float(p.mean()),
                        "score_std": float(p.std()),
                        "daily_rank_ic": float(np.mean(daily)),
                        "top20_win_rate": float(top.net_return.gt(0).mean()),
                        "top20_return": float(top.net_return.mean()),
                        "bottom20_return": float(bottom.net_return.mean()),
                        "top_minus_bottom_return": float(
                            top.net_return.mean() - bottom.net_return.mean()
                        ),
                        "sequence_return_mse": float(
                            mean_squared_error(part.net_return, part.sequence_return)
                        ),
                        "constant_train_mean_mse": float(
                            mean_squared_error(
                                part.net_return, np.full(len(part), train.net_return.mean())
                            )
                        ),
                    }
                )
            for feature in [*SCORES, *FEATURES]:
                shifts.append(
                    {
                        "side": side,
                        "split": split,
                        "feature": feature,
                        "psi_vs_2024": psi(train[feature].to_numpy(), part[feature].to_numpy()),
                        "standardized_mean_change": float(
                            (part[feature].mean() - train[feature].mean())
                            / (train[feature].std() + 1e-12)
                        ),
                    }
                )
            if split != "train":
                # Domain detection with stock-group-held-out folds; descriptive shift only.
                reference = train.sample(min(2000, len(train)), random_state=42)
                current = part.sample(min(2000, len(part)), random_state=43)
                joined = pd.concat([reference, current], ignore_index=True)
                domain = np.r_[np.zeros(len(reference)), np.ones(len(current))]
                groups = joined.symbol.to_numpy()
                for feature_set, x in [
                    ("simple_features", joined[FEATURES].to_numpy()),
                    ("embeddings", np.asarray(embeddings[joined.embedding_row], dtype=np.float32)),
                    ("scores", joined[SCORES].to_numpy()),
                ]:
                    model = make_pipeline(
                        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
                    )
                    pred = cross_val_predict(
                        model,
                        x,
                        domain,
                        groups=groups,
                        cv=GroupKFold(3),
                        method="predict_proba",
                        n_jobs=1,
                    )[:, 1]
                    domains.append(
                        {
                            "side": side,
                            "split": split,
                            "features": feature_set,
                            "domain_auc": float(roc_auc_score(domain, pred)),
                        }
                    )
        model = joblib.load(output / f"{side}_fusion.joblib")
        for feature, weight in zip(
            ["baseline_score", "sequence_return", "sequence_profit", "sequence_downside"],
            model[-1].coef_[0],
        ):
            fusions.append(
                {"side": side, "feature": feature, "coefficient_per_training_std": float(weight)}
            )
    pd.DataFrame(metrics).to_csv(output / "diagnostic_fit.csv", index=False)
    pd.DataFrame(shifts).to_csv(output / "diagnostic_shift.csv", index=False)
    pd.DataFrame(domains).to_csv(output / "diagnostic_domain.csv", index=False)
    pd.DataFrame(fusions).to_csv(output / "diagnostic_fusion_weights.csv", index=False)
    print(
        pd.DataFrame(metrics)[
            [
                "side",
                "split",
                "score",
                "rows",
                "base_win_rate",
                "auc",
                "daily_rank_ic",
                "top_minus_bottom_return",
                "brier",
                "constant_train_prior_brier",
                "mean_score",
            ]
        ].to_string(index=False)
    )
    print(pd.DataFrame(domains).to_string(index=False))
    print(pd.DataFrame(fusions).to_string(index=False))


if __name__ == "__main__":
    main()
