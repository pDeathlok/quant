from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from quant.data.atomic_io import atomic_write_json
from quant.routine.left_side_unified_production import normalize_daily_percentile
from quant.routine.paths import PROJECT_ROOT
from quant.routine.right_side_unified_production import _normalize_ranking_scores


OUTPUT_PATH = PROJECT_ROOT / "config/selector_score_probability_bands.json"
RIGHT_ARTIFACT = (
    PROJECT_ROOT / "models/production/right_side_unified_canonical_v2/ranking.joblib"
)
RIGHT_SAMPLE = (
    PROJECT_ROOT
    / "reports/research/right_side_unified_canonical_v5_rule113/test_predictions.parquet"
)
LEFT_ARTIFACT = (
    PROJECT_ROOT
    / "models/production/left_side_unified_canonical_v4_group4/ranking.joblib"
)
LEFT_SAMPLE = (
    PROJECT_ROOT
    / "reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet"
)
BUY_ARTIFACT = (
    PROJECT_ROOT / "models/production/selector_buy_hold_registry_v3/buy.joblib"
)
HOLD_ARTIFACT = (
    PROJECT_ROOT / "models/production/selector_buy_hold_registry_v3/hold.joblib"
)
BUY_HOLD_SAMPLE = (
    PROJECT_ROOT
    / "data/research/selector_buy_hold_registry_v2/selector_buy_hold_registry_dataset.parquet"
)

MODEL_EDGES = np.arange(0.0, 105.0, 5.0)
RETURN_SCORE_EDGES = np.asarray(
    [0.0, 40.0, *np.arange(42.0, 72.0, 2.0), 100.0],
    dtype=float,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _format_edge(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _historical_scores(
    predictions: np.ndarray,
    reference: np.ndarray,
    normalization_width: float,
) -> np.ndarray:
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), 1e-6)
    z_score = (np.asarray(predictions, dtype=float) - median) / scale
    width = max(float(normalization_width), 0.1)
    return np.clip(
        50.0 + 100.0 / np.pi * np.arctan(z_score / width),
        0.0,
        100.0,
    )


def _band_rows(
    scores: np.ndarray,
    events: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, object]]:
    numeric_scores = np.asarray(scores, dtype=float)
    numeric_events = np.asarray(events, dtype=bool)
    if len(numeric_scores) != len(numeric_events) or not np.isfinite(numeric_scores).all():
        raise ValueError("score/event samples are invalid")
    rows: list[dict[str, object]] = []
    for index, (minimum, maximum) in enumerate(zip(edges[:-1], edges[1:])):
        mask = numeric_scores >= minimum
        mask &= numeric_scores <= maximum if index == len(edges) - 2 else numeric_scores < maximum
        sample_count = int(mask.sum())
        if sample_count <= 0:
            raise ValueError(f"empty score band: {minimum}-{maximum}")
        event_count = int(numeric_events[mask].sum())
        rows.append(
            {
                "label": f"{_format_edge(minimum)}-{_format_edge(maximum)}",
                "min_score": float(minimum),
                "max_score": float(maximum),
                "sample_count": sample_count,
                "event_count": event_count,
                "probability_pct": round(100.0 * event_count / sample_count, 2),
            }
        )
    if sum(int(row["sample_count"]) for row in rows) != len(numeric_scores):
        raise ValueError("score bands do not cover every sample exactly once")
    return rows


def _source(artifact_path: Path, sample_path: Path) -> dict[str, str]:
    return {
        "artifact_path": _relative(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "sample_path": _relative(sample_path),
        "sample_sha256": _sha256(sample_path),
    }


def _right_model_calibration() -> dict[str, object]:
    columns = [
        "fold",
        "entry_mode",
        "horizon",
        "label",
        "good_path5",
        "pred_unified_long_task_deep",
    ]
    frame = pd.read_parquet(RIGHT_SAMPLE, columns=columns)
    frame = frame[
        frame["fold"].astype(str).eq("B")
        & frame["entry_mode"].astype(str).eq("next_close")
        & pd.to_numeric(frame["horizon"], errors="coerce").eq(5)
        & frame["label"].astype(str).eq("good_path5")
    ]
    artifact = joblib.load(RIGHT_ARTIFACT)
    scores = _normalize_ranking_scores(
        frame["pred_unified_long_task_deep"].to_numpy(dtype=float),
        artifact["score_normalization"],
    )
    return {
        "key": "model_right",
        "label": "右侧模型分",
        "score_field": "model_score_normalized",
        "target_label": "T+5 好路径概率",
        "target_definition": "次日收盘买入后，未来 5 个交易日达到 +5% 冲高且未先触发 -3% 回撤",
        "normalization": "frozen_oot_b_quantile_cdf_v1",
        "evaluation_scope": "Fold B 样本外，next_close，horizon=5",
        "source": _source(RIGHT_ARTIFACT, RIGHT_SAMPLE),
        "bands": _band_rows(
            scores,
            frame["good_path5"].to_numpy(dtype=bool),
            MODEL_EDGES,
        ),
    }


def _left_model_calibration() -> dict[str, object]:
    columns = ["fold", "date", "good_path5", "pred_unified_left_long_task_deep"]
    frame = pd.read_parquet(LEFT_SAMPLE, columns=columns)
    frame = frame[frame["fold"].astype(str).eq("C")].reset_index(drop=True)
    raw_scores = frame["pred_unified_left_long_task_deep"].to_numpy(dtype=float)
    scores = np.empty(len(frame), dtype=float)
    for positions in frame.groupby("date", sort=False).indices.values():
        scores[positions] = normalize_daily_percentile(raw_scores[positions])
    return {
        "key": "model_left",
        "label": "左侧模型分",
        "score_field": "model_score_normalized",
        "target_label": "T+5 好路径概率",
        "target_definition": "次日收盘买入后，未来 5 个交易日达到 +5% 冲高且未先触发 -3% 回撤",
        "normalization": "daily_cross_section_percentile_v1",
        "evaluation_scope": "Fold C 样本外，next_close，horizon=5",
        "source": _source(LEFT_ARTIFACT, LEFT_SAMPLE),
        "bands": _band_rows(
            scores,
            frame["good_path5"].to_numpy(dtype=bool),
            MODEL_EDGES,
        ),
    }


def _buy_hold_calibrations() -> list[dict[str, object]]:
    buy_artifact = joblib.load(BUY_ARTIFACT)
    hold_artifact = joblib.load(HOLD_ARTIFACT)
    features = sorted(
        set(buy_artifact["features"])
        | set(hold_artifact["features"])
    )
    columns = [
        "date",
        "future_max_high_t5_pct",
        "future_return_t5_pct",
        *features,
    ]
    frame = pd.read_parquet(BUY_HOLD_SAMPLE, columns=columns)
    frame = frame[pd.to_datetime(frame["date"], errors="raise").dt.year.eq(2026)]
    buy_scores = _historical_scores(
        buy_artifact["model"].predict(frame[buy_artifact["features"]]),
        buy_artifact["score_reference"],
        buy_artifact["normalization_width"],
    )
    hold_components = {
        key: _historical_scores(
            model.predict(frame[hold_artifact["features"]]),
            hold_artifact["score_references"][key],
            hold_artifact["normalization_widths"][key],
        )
        for key, model in hold_artifact["models"].items()
    }
    buy_weight = float(hold_artifact["buy_weight"])
    hold_scores = (
        buy_weight * hold_components["buy"]
        + (1.0 - buy_weight) * hold_components["hold"]
    )
    scope = "Fold C：2026-01-05 至 2026-08-14 严格样本外"
    source = _source(BUY_ARTIFACT, BUY_HOLD_SAMPLE)
    return [
        {
            "key": "buy",
            "label": "买入分",
            "score_field": "buy_score_normalized",
            "target_label": "T+5 冲高 ≥ 5% 概率",
            "target_definition": "次日收盘买入后，未来 5 个交易日最高价相对买入价涨幅达到 5%",
            "normalization": "fixed_robust_historical_transform_v1",
            "evaluation_scope": scope,
            "source": source,
            "bands": _band_rows(
                buy_scores,
                frame["future_max_high_t5_pct"].to_numpy(dtype=float) >= 5.0,
                RETURN_SCORE_EDGES,
            ),
        },
        {
            "key": "hold",
            "label": "持有分",
            "score_field": "hold_score_normalized",
            "target_label": "T+5 正收益概率",
            "target_definition": "次日收盘买入后，第 5 个交易日收盘收益大于 0",
            "normalization": "fixed_robust_historical_transform_v1",
            "evaluation_scope": scope,
            "source": _source(HOLD_ARTIFACT, BUY_HOLD_SAMPLE),
            "bands": _band_rows(
                hold_scores,
                frame["future_return_t5_pct"].to_numpy(dtype=float) > 0.0,
                RETURN_SCORE_EDGES,
            ),
        },
    ]


def build_payload() -> dict[str, object]:
    return {
        "schema_version": "selector-score-probability-bands-v2",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "method": "strict_out_of_sample_observed_frequency_by_display_score_band",
        "score_range": [0.0, 100.0],
        "band_policy": {
            "model": "5_point_bins",
            "buy_hold": "2_point_bins_from_40_to_70_with_aggregated_tails",
        },
        "calibrations": [
            _right_model_calibration(),
            _left_model_calibration(),
            *_buy_hold_calibrations(),
        ],
        "disclaimer": "概率为严格样本外历史分档频率，只说明同分档历史表现，不是对单只股票的收益承诺。",
    }


def main() -> None:
    payload = build_payload()
    atomic_write_json(payload, OUTPUT_PATH, indent=2)
    print(f"wrote {OUTPUT_PATH}")
    for calibration in payload["calibrations"]:
        print(
            calibration["key"],
            len(calibration["bands"]),
            sum(int(row["sample_count"]) for row in calibration["bands"]),
        )


if __name__ == "__main__":
    main()
