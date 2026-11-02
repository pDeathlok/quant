import importlib
from pathlib import Path

import pandas as pd


def test_rolling_splits_purge_unobserved_labels_without_losing_quarter_end_test(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("train_overnight_market_rolling")
    f = pd.DataFrame({"date": pd.to_datetime(["2025-09-29", "2025-09-30", "2025-10-09", "2025-12-31", "2026-01-05", "2026-03-31", "2026-04-01"]),
                      "exit_date": pd.to_datetime(["2025-09-30", "2025-10-09", "2025-10-10", "2026-01-05", "2026-01-06", "2026-04-01", "2026-04-02"])})
    train, cal, test = module.split_masks(f, *module.FOLDS[0])
    assert train.tolist() == [True, False, False, False, False, False, False]
    assert cal.tolist() == [False, False, True, False, False, False, False]
    assert test.tolist() == [False, False, False, False, True, True, False]
    _, cal2, test2 = module.split_masks(f, *module.FOLDS[1])
    assert not cal2.iloc[5]  # March31 label is not observable before April1.
    assert test2.iloc[6]
    assert not (test & test2).any()
