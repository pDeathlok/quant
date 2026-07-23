from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.data.atomic_io import atomic_write_csv, atomic_write_parquet


def test_atomic_parquet_failure_preserves_previous_file(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "features.parquet"
    target.write_bytes(b"last-known-good")

    def fail_after_partial_write(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated writer crash")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="writer crash"):
        atomic_write_parquet(pd.DataFrame({"value": [1]}), target, index=False)

    assert target.read_bytes() == b"last-known-good"
    assert list(tmp_path.glob(".*.tmp.parquet")) == []


def test_atomic_csv_publishes_complete_replacement(tmp_path: Path) -> None:
    target = tmp_path / "scores.csv"
    target.write_text("old\n", encoding="utf-8")

    atomic_write_csv(pd.DataFrame({"score": [0.9, 0.8]}), target, index=False)

    assert pd.read_csv(target)["score"].tolist() == [0.9, 0.8]
    assert list(tmp_path.glob(".*.tmp.csv")) == []
