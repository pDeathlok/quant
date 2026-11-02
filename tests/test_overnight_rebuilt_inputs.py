import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


@pytest.fixture
def rebuilt_case(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/research"))
    module = importlib.import_module("audit_overnight_lag_only")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    frame = pd.DataFrame({"symbol": ["A", "B"], "date": pd.to_datetime(["2025-10-09"] * 2),
                          "y_close": [0.1, -0.1], **{f"lag1_daily_{i}": [1., 2.] for i in range(13)}})
    factors = frame[["symbol", "date"]].copy()
    factors = pd.concat([factors, pd.DataFrame({c: [1., np.nan] for c in PROJECT_FACTOR_COLUMNS})], axis=1)
    root = tmp_path / "reports/research/overnight_current_full_factors_v1"
    root.mkdir(parents=True)
    path = root / "dataset.parquet"

    def write(data):
        data.to_parquet(path, index=False)
        (root / "complete.json").write_text(json.dumps({"dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))

    write(factors)
    bars = pd.DataFrame({"ts_code": ["A", "B"], "trade_date": ["20251009"] * 2,
                         "open": [10., 20.], "close": [11., 19.], "pre_close": [10., 20.],
                         "high": [12., 21.], "low": [9., 18.], "vol": [100., np.nan],
                         "volume": [100., 200.], "amount": [1000., 4000.]})

    class DailySource:
        def to_table(self, **kwargs):
            return pa.Table.from_pandas(bars[kwargs["columns"]])

    monkeypatch.setattr(module.ds, "dataset", lambda *args, **kwargs: DailySource())
    output = tmp_path / "output"
    output.mkdir()
    return module, frame, factors, write, path, output


def test_rebuilt_join_preserves_missing_values_and_excludes_labels(rebuilt_case):
    module, frame, _, _, _, output = rebuilt_case
    result, columns = module.same_day_features(frame, output, rebuilt=True)
    pd.testing.assert_frame_equal(frame, result[frame.columns])
    assert len(columns) == 169
    assert "y_close" not in columns
    assert result.loc[1, "today_factor_close"] != result.loc[1, "today_factor_close"]
    assert result.loc[0, "today_raw_close"] == 11.
    assert result.loc[1, "today_raw_vol"] == 200.
    changed = frame.assign(y_close=[-100., 100.])
    result2, columns2 = module.same_day_features(changed, output, rebuilt=True)
    assert columns2 == columns
    pd.testing.assert_frame_equal(result[columns], result2[columns])


def test_rebuilt_join_rejects_missing_candidate_key(rebuilt_case):
    module, frame, factors, write, _, output = rebuilt_case
    write(factors.iloc[:1])
    with pytest.raises(AssertionError):
        module.same_day_features(frame, output, rebuilt=True)


def test_rebuilt_join_rejects_changed_artifact(rebuilt_case):
    module, frame, factors, _, path, output = rebuilt_case
    factors.assign(close=999.).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="changed"):
        module.same_day_features(frame, output, rebuilt=True)
