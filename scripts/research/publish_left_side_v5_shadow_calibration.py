#!/usr/bin/env python
"""Publish honest Fold-C probability bands for the left-side v5 shadow score."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_json


DEFAULT_CONFIG = PROJECT_ROOT / "config/selector_score_probability_bands.json"
DEFAULT_SAMPLE = (
    PROJECT_ROOT / "reports/research/left_side_regime_gate_v5/fold_predictions.parquet"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "models/shadow/left_side_absolute_v5/bundle.joblib"
)
KEY = "model_left_v5_shadow"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def publish(args: argparse.Namespace) -> dict[str, object]:
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "selector-score-probability-bands-v2":
        raise RuntimeError("selector score probability schema mismatch")
    sample = pd.read_parquet(
        args.sample,
        columns=[
            "fold",
            "absolute_display_score",
            "terminal_net_positive_20bps",
        ],
    )
    sample = sample[sample["fold"].eq("C")].copy()
    if sample.empty:
        raise RuntimeError("left-side v5 calibration has no Fold C rows")
    score = pd.to_numeric(sample["absolute_display_score"], errors="coerce")
    target = pd.to_numeric(
        sample["terminal_net_positive_20bps"], errors="coerce"
    )
    if score.isna().any() or target.isna().any() or not score.between(0.0, 100.0).all():
        raise RuntimeError("left-side v5 calibration sample is invalid")

    bands = []
    for minimum in np.arange(0.0, 100.0, 5.0):
        maximum = minimum + 5.0
        mask = score.ge(minimum) & (
            score.le(maximum) if maximum == 100.0 else score.lt(maximum)
        )
        current = target[mask]
        if current.empty:
            raise RuntimeError(f"left-side v5 calibration band is empty: {minimum}-{maximum}")
        events = int(current.sum())
        bands.append(
            {
                "label": f"{int(minimum)}-{int(maximum)}",
                "min_score": float(minimum),
                "max_score": float(maximum),
                "sample_count": len(current),
                "event_count": events,
                "probability_pct": round(100.0 * events / len(current), 2),
            }
        )
    calibration = {
        "key": KEY,
        "label": "左侧 v5 绝对分（影子）",
        "score_field": "model_score_shadow_v5",
        "target_label": "T+5 扣费后正收益概率",
        "target_definition": "次日收盘买入后持有 5 个交易日，终值收益扣除 20bp 往返成本后大于 0",
        "normalization": "absolute_regime_gate_plus_blended_cross_section_v1",
        "evaluation_scope": "Fold C 滚动样本外，next_close，horizon=5；影子观察，不参与生产排序",
        "lifecycle": "shadow",
        "source": {
            "artifact_path": _relative(args.artifact),
            "artifact_sha256": _sha256(args.artifact),
            "sample_path": _relative(args.sample),
            "sample_sha256": _sha256(args.sample),
        },
        "bands": bands,
    }
    calibrations = [
        item for item in payload.get("calibrations", []) if item.get("key") != KEY
    ]
    left_index = next(
        (
            index + 1
            for index, item in enumerate(calibrations)
            if item.get("key") == "model_left"
        ),
        len(calibrations),
    )
    calibrations.insert(left_index, calibration)
    payload["calibrations"] = calibrations
    payload["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_write_json(payload, args.config, indent=2)
    return calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser


def main() -> None:
    print(json.dumps(publish(_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
