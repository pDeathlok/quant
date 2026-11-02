#!/usr/bin/env python
"""Validate an absolute market-state gate around the frozen left-side v5 blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.research.left_side_regime_gate import (
    DEFAULT_ADMISSION_QUANTILE,
    DEFAULT_RIDGE_ALPHA,
    compose_absolute_display_score,
    fit_left_side_regime_gate,
    materialize_daily_regime_targets,
)
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold

from scripts.research.materialize_left_side_v5_candidate_factors import (
    ACTIVE_VALUE_FEATURE_COLUMNS,
    BREADTH_FEATURE_COLUMNS,
)
from scripts.research.optimize_left_side_model_v5 import (
    DEFAULT_BASELINE_PREDICTIONS,
    DEFAULT_CANDIDATE_FACTORS,
    DEFAULT_EVENTS,
    DEFAULT_LABELS,
    ROUND_TRIP_COST,
    TARGET_NAME,
    _baseline_probability,
    _load_training_frame,
    _profit_factor,
)
from scripts.research.select_left_side_blend_v5 import (
    DEFAULT_DECISION as DEFAULT_BLEND_DECISION,
)


DEFAULT_RETURN_PREDICTIONS = (
    PROJECT_ROOT
    / "reports/research/left_side_return_ranker_v5/experiment_predictions.parquet"
)
DEFAULT_RETURN_CONFIRMATION_PREDICTIONS = (
    PROJECT_ROOT
    / "reports/research/left_side_blend_v5/confirmation_predictions.parquet"
)
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/left_side_regime_gate_v5"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research/left_side_regime_gate_v5"
DEFAULT_METRICS = DEFAULT_REPORT_ROOT / "fold_metrics.csv"
DEFAULT_SCORE_BANDS = DEFAULT_REPORT_ROOT / "score_band_metrics.csv"
DEFAULT_PREDICTIONS = DEFAULT_REPORT_ROOT / "fold_predictions.parquet"
DEFAULT_DECISION = DEFAULT_REPORT_ROOT / "decision.json"
DEFAULT_SHADOW_ARTIFACT = (
    PROJECT_ROOT / "models/shadow/left_side_absolute_v5/bundle.joblib"
)
DEFAULT_SHADOW_MANIFEST = (
    PROJECT_ROOT / "models/shadow/left_side_absolute_v5/manifest.json"
)
DEFAULT_PRODUCTION_ARTIFACT = (
    PROJECT_ROOT
    / "models/production/left_side_unified_canonical_v4_group4/ranking.joblib"
)

REGIME_FEATURE_COLUMNS: tuple[str, ...] = (
    *BREADTH_FEATURE_COLUMNS,
    *ACTIVE_VALUE_FEATURE_COLUMNS[4:],
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _daily_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date", sort=False)[column].rank(method="average", pct=True)


def _component_probability(
    args: argparse.Namespace,
    fold: str,
    experiment: str,
    frame: pd.DataFrame,
) -> np.ndarray:
    path = (
        args.return_confirmation_predictions
        if fold == "C"
        else args.return_predictions
    )
    predictions = pd.read_parquet(path)
    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="raise"
    ).dt.normalize()
    if "fold" in predictions:
        predictions = predictions[predictions["fold"].eq(fold)]
    predictions = predictions[predictions["experiment"].eq(experiment)]
    merged = frame[["symbol", "date"]].merge(
        predictions[["symbol", "date", "probability"]],
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    values = pd.to_numeric(merged["probability"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"return-ranker predictions are incomplete for fold {fold}")
    return values


def _ranked_frame(
    frame: pd.DataFrame,
    *,
    production_probability: np.ndarray,
    component_probability: np.ndarray,
    production_weight: float,
    regime_by_date: pd.DataFrame,
) -> pd.DataFrame:
    ranked = frame[["symbol", "date", "terminal_return", TARGET_NAME]].copy()
    ranked["production_probability"] = np.asarray(
        production_probability, dtype=float
    )
    ranked["component_probability"] = np.asarray(component_probability, dtype=float)
    ranked["production_percentile"] = _daily_percentile(
        ranked, "production_probability"
    )
    ranked["component_percentile"] = _daily_percentile(
        ranked, "component_probability"
    )
    ranked["blended_percentile"] = (
        float(production_weight) * ranked["production_percentile"]
        + (1.0 - float(production_weight)) * ranked["component_percentile"]
    )
    ranked = ranked.merge(
        regime_by_date[["date", "regime_score", "trade_gate_open"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    if ranked[["regime_score", "trade_gate_open"]].isna().any().any():
        raise RuntimeError("regime prediction coverage is incomplete")
    ranked["absolute_display_score"] = compose_absolute_display_score(
        ranked["blended_percentile"].to_numpy(float),
        ranked["trade_gate_open"].to_numpy(bool),
    )
    return ranked


def _top_k_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_k: int,
    apply_gate: bool,
) -> dict[str, float | int]:
    ranked = (
        frame.sort_values(
            ["date", score_column, "symbol"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("date", sort=False)
        .head(top_k)
    )
    if apply_gate:
        ranked = ranked[ranked["trade_gate_open"]]
    net = pd.to_numeric(ranked["terminal_return"], errors="coerce") - ROUND_TRIP_COST
    return {
        f"top{top_k}_trades": len(net),
        f"top{top_k}_days": int(ranked["date"].nunique()),
        f"top{top_k}_average_net_return": float(net.mean()),
        f"top{top_k}_median_net_return": float(net.median()),
        f"top{top_k}_win_rate": float(net.gt(0.0).mean()),
        f"top{top_k}_profit_factor": _profit_factor(net),
    }


def _evaluate(
    frame: pd.DataFrame,
    *,
    score_column: str,
    apply_gate: bool,
) -> dict[str, float | int]:
    dates = frame[["date", "trade_gate_open"]].drop_duplicates("date")
    return {
        "rows": len(frame),
        "dates": len(dates),
        "gate_open_dates": int(dates["trade_gate_open"].sum()),
        "gate_open_coverage": float(dates["trade_gate_open"].mean()),
        **_top_k_metrics(
            frame, score_column=score_column, top_k=10, apply_gate=apply_gate
        ),
        **_top_k_metrics(
            frame, score_column=score_column, top_k=20, apply_gate=apply_gate
        ),
    }


def _score_band_metrics(frame: pd.DataFrame, fold: str) -> pd.DataFrame:
    bands = pd.cut(
        frame["absolute_display_score"],
        bins=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.000001],
        labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"],
        include_lowest=True,
        right=False,
    )
    work = frame[["terminal_return", TARGET_NAME]].copy()
    work["score_band"] = bands.astype("string")
    work["net_return"] = pd.to_numeric(work["terminal_return"], errors="coerce") - ROUND_TRIP_COST
    rows = []
    for band, current in work.groupby("score_band", sort=False, observed=True):
        rows.append(
            {
                "fold": fold,
                "score_band": str(band),
                "samples": len(current),
                "positive_probability": float(current[TARGET_NAME].mean()),
                "average_net_return": float(current["net_return"].mean()),
                "profit_factor": _profit_factor(current["net_return"]),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    blend = json.loads(args.blend_decision.read_text(encoding="utf-8"))
    experiment = str(blend.get("selected_component_experiment") or "")
    production_weight = float(blend.get("production_weight"))
    if not experiment or not 0.0 < production_weight < 1.0:
        raise RuntimeError("left-side blend decision is not frozen")

    selected = _load_training_frame(args)
    fold_rows: list[dict[str, Any]] = []
    band_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_artifacts: dict[str, str] = {}
    folds = {fold.name: fold for fold in DEFAULT_YEAR_FOLDS}
    for fold_name in ("A", "B", "C"):
        splits = split_by_year_fold(selected, folds[fold_name])
        train_daily = materialize_daily_regime_targets(
            splits.train, features=REGIME_FEATURE_COLUMNS
        )
        validation_daily = materialize_daily_regime_targets(
            splits.validation, features=REGIME_FEATURE_COLUMNS
        )
        test_daily = materialize_daily_regime_targets(
            splits.test, features=REGIME_FEATURE_COLUMNS
        )
        gate = fit_left_side_regime_gate(
            train_daily,
            validation_daily,
            features=REGIME_FEATURE_COLUMNS,
            admission_quantile=args.admission_quantile,
            ridge_alpha=args.ridge_alpha,
        )
        test_daily["regime_score"] = gate.predict_score(test_daily)
        test_daily["trade_gate_open"] = gate.predict_open(test_daily)

        model_path = args.model_root / fold_name / "regime_gate.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gate, model_path)
        model_artifacts[fold_name] = str(model_path.relative_to(PROJECT_ROOT))

        production = _baseline_probability(args, fold_name, splits.test)
        component = _component_probability(
            args, fold_name, experiment, splits.test
        )
        ranked = _ranked_frame(
            splits.test,
            production_probability=production,
            component_probability=component,
            production_weight=production_weight,
            regime_by_date=test_daily,
        )
        configurations = (
            ("production_v4_ungated", "production_probability", False),
            ("blend_v5_ungated", "blended_percentile", False),
            ("production_v4_regime_gated", "production_probability", True),
            ("blend_v5_regime_gated", "blended_percentile", True),
        )
        for configuration, score_column, apply_gate in configurations:
            fold_rows.append(
                {
                    "fold": fold_name,
                    "configuration": configuration,
                    "regime_threshold": gate.threshold,
                    "admission_quantile": gate.admission_quantile,
                    **_evaluate(
                        ranked,
                        score_column=score_column,
                        apply_gate=apply_gate,
                    ),
                }
            )
        band_frames.append(_score_band_metrics(ranked, fold_name))
        output = ranked.copy()
        output.insert(0, "fold", fold_name)
        prediction_frames.append(output)

    metrics = pd.DataFrame(fold_rows)
    score_bands = pd.concat(band_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_write_csv(metrics, args.metrics_out, index=False)
    atomic_write_csv(score_bands, args.score_bands_out, index=False)
    atomic_write_parquet(predictions, args.predictions_out, index=False)

    indexed = metrics.set_index(["fold", "configuration"])
    fold_gate = {}
    for fold_name in ("A", "B", "C"):
        baseline = indexed.loc[(fold_name, "production_v4_ungated")]
        candidate = indexed.loc[(fold_name, "blend_v5_regime_gated")]
        fold_gate[fold_name] = bool(
            candidate["top20_average_net_return"] > 0.0
            and candidate["top20_profit_factor"] > 1.0
            and candidate["top20_average_net_return"]
            > baseline["top20_average_net_return"]
            and candidate["top20_profit_factor"] > baseline["top20_profit_factor"]
            and candidate["gate_open_dates"] >= 20
        )
    coverage = metrics[
        metrics["configuration"].eq("blend_v5_regime_gated")
    ].set_index("fold")["gate_open_coverage"]
    coverage_ratio = float(coverage.max() / coverage.min())
    c_candidate = indexed.loc[("C", "blend_v5_regime_gated")]
    component_path = (
        PROJECT_ROOT
        / "models/research/left_side_return_ranker_v5/C"
        / f"{experiment}.joblib"
    )
    gate_path = args.model_root / "C" / "regime_gate.joblib"
    shadow_bundle = {
        "schema_version": "left-side-absolute-shadow-v5-bundle-v1",
        "lifecycle": "shadow",
        "production_action": "observe_only",
        "component_model": joblib.load(component_path),
        "regime_gate": joblib.load(gate_path),
        "production_weight": production_weight,
        "component_weight": 1.0 - production_weight,
        "component_experiment": experiment,
        "regime_feature_columns": list(REGIME_FEATURE_COLUMNS),
        "score_field": "shadow_v5_absolute_score",
        "trade_gate_field": "shadow_v5_trade_gate_open",
        "production_artifact_sha256": _sha256(args.production_artifact),
        "component_artifact_sha256": _sha256(component_path),
        "regime_artifact_sha256": _sha256(gate_path),
    }
    args.shadow_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(shadow_bundle, args.shadow_artifact)
    shadow_manifest = {
        "status": "success",
        "schema_version": "left-side-absolute-shadow-v5-manifest-v1",
        "lifecycle": "shadow",
        "production_action": "observe_only",
        "artifact_path": str(args.shadow_artifact.relative_to(PROJECT_ROOT)),
        "artifact_sha256": _sha256(args.shadow_artifact),
        "production_artifact_path": str(args.production_artifact.relative_to(PROJECT_ROOT)),
        "production_artifact_sha256": _sha256(args.production_artifact),
        "component_artifact_path": str(component_path.relative_to(PROJECT_ROOT)),
        "component_artifact_sha256": _sha256(component_path),
        "regime_artifact_path": str(gate_path.relative_to(PROJECT_ROOT)),
        "regime_artifact_sha256": _sha256(gate_path),
        "score_field": "shadow_v5_absolute_score",
        "trade_gate_field": "shadow_v5_trade_gate_open",
        "selected_component_experiment": experiment,
        "production_weight": production_weight,
        "component_weight": 1.0 - production_weight,
        "regime_feature_columns": list(REGIME_FEATURE_COLUMNS),
        "source_decision_path": str(args.decision_out.relative_to(PROJECT_ROOT)),
    }
    atomic_write_json(shadow_manifest, args.shadow_manifest, indent=2)

    payload: dict[str, Any] = {
        "status": "success",
        "schema_version": "left-side-regime-gate-v5-decision-v1",
        "selected_component_experiment": experiment,
        "production_weight": production_weight,
        "component_weight": 1.0 - production_weight,
        "regime_features": list(REGIME_FEATURE_COLUMNS),
        "regime_model": {
            "type": "median_imputer_standard_scaler_ridge",
            "ridge_alpha": float(args.ridge_alpha),
            "validation_admission_quantile": float(args.admission_quantile),
            "daily_target": "average_eligible_terminal_return_minus_20bps",
        },
        "fold_gate_passed": fold_gate,
        "all_folds_positive_after_gate": all(fold_gate.values()),
        "gate_coverage_by_fold": coverage.to_dict(),
        "gate_coverage_max_min_ratio": coverage_ratio,
        "confirmation_disclosure": (
            "Fold C had already been opened by the preceding blend confirmation; "
            "this gate result is rolling out-of-time evidence, not a pristine second confirmation."
        ),
        "production_action": "shadow_only",
        "promote_recommended": False,
        "shadow_recommended": bool(
            all(fold_gate.values())
            and c_candidate["top20_average_net_return"] > 0.0
            and c_candidate["top20_profit_factor"] > 1.0
        ),
        "promotion_blockers": [
            "confirmation_fold_reused_after_prior_model_selection",
            "gate_coverage_is_not_stable_across_folds",
        ],
        "model_artifacts": model_artifacts,
        "shadow_artifact": str(args.shadow_artifact.relative_to(PROJECT_ROOT)),
        "shadow_artifact_sha256": _sha256(args.shadow_artifact),
        "shadow_manifest": str(args.shadow_manifest.relative_to(PROJECT_ROOT)),
        "metrics_sha256": _sha256(args.metrics_out),
        "score_bands_sha256": _sha256(args.score_bands_out),
        "predictions_sha256": _sha256(args.predictions_out),
        "input_sha256": {
            "events": _sha256(args.events),
            "labels": _sha256(args.labels),
            "candidate_factors": _sha256(args.candidate_factors),
            "blend_decision": _sha256(args.blend_decision),
        },
    }
    atomic_write_json(payload, args.decision_out, indent=2)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--candidate-factors", type=Path, default=DEFAULT_CANDIDATE_FACTORS
    )
    parser.add_argument(
        "--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS
    )
    parser.add_argument("--return-predictions", type=Path, default=DEFAULT_RETURN_PREDICTIONS)
    parser.add_argument(
        "--return-confirmation-predictions",
        type=Path,
        default=DEFAULT_RETURN_CONFIRMATION_PREDICTIONS,
    )
    parser.add_argument("--blend-decision", type=Path, default=DEFAULT_BLEND_DECISION)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--score-bands-out", type=Path, default=DEFAULT_SCORE_BANDS)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--decision-out", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--shadow-artifact", type=Path, default=DEFAULT_SHADOW_ARTIFACT)
    parser.add_argument("--shadow-manifest", type=Path, default=DEFAULT_SHADOW_MANIFEST)
    parser.add_argument(
        "--production-artifact", type=Path, default=DEFAULT_PRODUCTION_ARTIFACT
    )
    parser.add_argument("--admission-quantile", type=float, default=DEFAULT_ADMISSION_QUANTILE)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    return parser


def main() -> None:
    payload = run(_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
