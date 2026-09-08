import argparse
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import refresh_chan_model_live_scores as refresh


def test_live_top_list_default_matches_daily_refresh_output():
    assert refresh.DEFAULT_TOP_LIST_DIR == refresh.PROJECT_ROOT / "data/raw/top_list"


@pytest.fixture
def setup_refresh(monkeypatch, tmp_path):
    reference = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-06-01"]),
        "symbol": ["000001.SZ", "000002.SZ"],
        "split": ["train", "oot"],
        "hold_10d_close": [3.0, 2.0],
        "factor": [0.3, 0.5],
    })
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    args = argparse.Namespace(
        output_dir=tmp_path / "output", scored_path=tmp_path / "scored.parquet",
        report_dir=tmp_path, model_dir=tmp_path / "models", reference_dataset=reference_path,
        daily_dir=tmp_path / "daily", daily_basic_dir=tmp_path / "basic",
        top_list_dir=tmp_path / "top", start="19900101", end="2026-09-07",
        rebuild_candidates=True, candidate_start_date="1990-01-01",
        max_workers=1, executor="threads", batch_size=16, top_n=20,
        backfill_snapshots=False,
    )
    calls = {}
    monkeypatch.setattr(refresh, "_load_env", lambda _: None)
    monkeypatch.setattr(refresh, "_load_models", lambda _: {"target_good": {"features": ["factor"]}})

    def predict(frame, _models):
        out = frame.copy()
        out["pred_target_good"] = out["factor"]
        return out

    def candidates(_daily, start, _workers, **kwargs):
        calls["candidate_start"] = start
        return pd.DataFrame({"date": [pd.Timestamp(args.end)], "symbol": ["000003.SZ"]})

    def features(**kwargs):
        calls["feature_start"] = kwargs["start"]
        return pd.DataFrame({
            "date": [pd.Timestamp(args.end)], "symbol": ["000003.SZ"],
            "split": ["live"], "factor": [0.9],
        })

    def strategy(frame, *_args):
        calls["strategy_frame"] = frame.copy()
        assert frame["split"].eq("train").any()
        return {"latest_signal_date": args.end}

    monkeypatch.setattr(refresh, "_add_predictions", predict)
    monkeypatch.setattr(refresh, "build_candidates", candidates)
    monkeypatch.setattr(refresh, "_build_recent_feature_dataset", features)
    monkeypatch.setattr(refresh, "_write_strategy_outputs", strategy)
    return args, reference, calls


def test_full_rebuild_preserves_training_reference_and_bounds_live_compute(setup_refresh):
    args, reference, calls = setup_refresh
    result = refresh.refresh_live_scores(args)
    saved = pd.read_parquet(args.scored_path)

    assert calls["candidate_start"] == calls["feature_start"] == "2026-06-02"
    assert saved["split"].tolist() == ["train", "oot", "live"]
    assert saved.loc[0, "hold_10d_close"] == 3.0
    assert saved.loc[0, "pred_target_good"] == 0.3
    pd.testing.assert_frame_equal(pd.read_parquet(args.reference_dataset), reference)
    assert result["requested_start"] == "19900101"
    assert result["training_reference"]["train_rows"] == 1


def test_reference_repairs_score_cache_whose_training_rows_were_overwritten(setup_refresh):
    args, reference, _ = setup_refresh
    corrupted = reference.assign(split="live", pred_target_good=0.999)
    corrupted.to_parquet(args.scored_path, index=False)

    refresh.refresh_live_scores(args)
    saved = pd.read_parquet(args.scored_path)

    assert len(saved) == 3
    assert saved.loc[0, "split"] == "train"
    assert saved.loc[0, "pred_target_good"] == 0.3


@pytest.mark.parametrize("invalid", ["missing_train", "live", "duplicate", "future"])
def test_invalid_reference_fails_before_heavy_work(setup_refresh, invalid):
    args, reference, calls = setup_refresh
    if invalid == "missing_train":
        reference["split"] = "oot"
    elif invalid == "live":
        reference.loc[0, "split"] = "live"
    elif invalid == "duplicate":
        reference = pd.concat([reference, reference.iloc[[0]]], ignore_index=True)
    else:
        reference.loc[0, "date"] = pd.Timestamp(args.end)
    reference.to_parquet(args.reference_dataset, index=False)

    with pytest.raises(ValueError, match="Chan"):
        refresh.refresh_live_scores(args)
    assert not calls
    assert not args.scored_path.exists()


def test_strategy_failure_does_not_overwrite_scored_cache(monkeypatch, setup_refresh):
    args, reference, _ = setup_refresh
    reference.to_parquet(args.scored_path, index=False)
    before = args.scored_path.read_bytes()

    def fail(*_args):
        raise ValueError("strategy failure")

    monkeypatch.setattr(refresh, "_write_strategy_outputs", fail)
    with pytest.raises(ValueError, match="strategy failure"):
        refresh.refresh_live_scores(args)
    assert args.scored_path.read_bytes() == before
