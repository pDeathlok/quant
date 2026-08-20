from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant.research.right_side_targets import (
    SUPPORTED_TRAINING_TARGETS,
    TERMINAL_NET_POSITIVE_15BPS,
    TERMINAL_NET_POSITIVE_THRESHOLD_RETURN,
    materialize_training_target,
    target_metadata,
    target_source_columns,
    validate_persisted_target_metadata,
    validate_target_cost,
)


def test_terminal_net_positive_15bps_has_strict_cost_aligned_boundary() -> None:
    frame = pd.DataFrame(
        {
            "terminal_return": [
                0.001499,
                TERMINAL_NET_POSITIVE_THRESHOLD_RETURN,
                0.001501,
                None,
                float("inf"),
            ]
        }
    )

    actual = materialize_training_target(
        frame,
        TERMINAL_NET_POSITIVE_15BPS,
    )[TERMINAL_NET_POSITIVE_15BPS]

    assert actual.dtype == "boolean"
    assert actual.iloc[:3].tolist() == [False, False, True]
    assert actual.iloc[3:].isna().all()
    assert target_source_columns(TERMINAL_NET_POSITIVE_15BPS) == (
        "terminal_return",
    )


def test_terminal_net_positive_target_rejects_cost_metric_drift() -> None:
    validate_target_cost(TERMINAL_NET_POSITIVE_15BPS, 15.0)

    with pytest.raises(ValueError, match="fixed to 15 bps"):
        validate_target_cost(TERMINAL_NET_POSITIVE_15BPS, 10.0)


def test_derived_target_metadata_is_complete_and_fail_closed() -> None:
    metadata = target_metadata(TERMINAL_NET_POSITIVE_15BPS)
    assert metadata == {
        "target_kind": "derived_binary",
        "target_source": "terminal_return",
        "target_operator": ">",
        "target_threshold_return": 0.0015,
        "target_cost_bps": 15.0,
        "target_definition": (
            "terminal_return > 0.0015 (gross terminal return strictly exceeds "
            "the fixed 15 bps round-trip cost; equality is negative)"
        ),
    }
    artifact = pd.DataFrame([metadata, metadata])
    validate_persisted_target_metadata(artifact, TERMINAL_NET_POSITIVE_15BPS)

    artifact.loc[0, "target_cost_bps"] = 10.0
    with pytest.raises(ValueError, match="target_cost_bps"):
        validate_persisted_target_metadata(artifact, TERMINAL_NET_POSITIVE_15BPS)


def test_training_cli_accepts_derived_target_without_changing_fold_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "research"
        / "validate_unified_right_side_models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_unified_right_side_models_target_cli_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "train",
            "--entry-mode",
            "next_close",
            "--horizon",
            "5",
            "--label",
            TERMINAL_NET_POSITIVE_15BPS,
            "--folds",
            "A",
            "B",
            "--round-trip-cost-bps",
            "15",
        ],
    )

    args = module.parse_args()

    assert TERMINAL_NET_POSITIVE_15BPS in SUPPORTED_TRAINING_TARGETS
    assert args.label == TERMINAL_NET_POSITIVE_15BPS
    assert args.entry_mode == "next_close"
    assert args.horizon == 5
    assert args.folds == ["A", "B"]
    assert args.round_trip_cost_bps == pytest.approx(15.0)
