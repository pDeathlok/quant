from pathlib import Path

import pandas as pd

from scripts.research import score_latest_strategy_models as scoring


def test_build_latest_dataset_allows_explicitly_excluded_candidates(monkeypatch, tmp_path):
    target_date = pd.Timestamp("2026-07-27")
    signal_rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "600228.SH"],
            "date": [target_date, target_date],
            "signal": [True, True],
        }
    )
    paths = [tmp_path / "000001.SZ.parquet", tmp_path / "600228.SH.parquet"]
    scored_frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [target_date],
            "signal": [True],
        }
    )

    monkeypatch.setattr(scoring, "_load_signal_cache", lambda signals, start_date: signal_rows)
    monkeypatch.setattr(scoring, "list_partitioned_symbol_paths", lambda daily_dir: paths)

    def fake_process(args):
        path = Path(args[0])
        if path.stem == "600228.SH":
            return None, None, "600228.SH: target signal belongs to ST/delisting stock"
        return scored_frame, None, None

    monkeypatch.setattr(scoring, "_process_symbol", fake_process)

    result = scoring.build_latest_dataset(
        tmp_path,
        ["signal"],
        "2020-01-01",
        "2026-07-27",
        workers=1,
    )

    assert result["symbol"].tolist() == ["000001.SZ"]
    assert result.attrs["excluded_candidates"] == [
        "600228.SH: target signal belongs to ST/delisting stock"
    ]
