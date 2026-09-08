import json
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from quant.data import market_snapshot as snapshots


def test_date_normalization_parses_distinct_values_once(monkeypatch):
    original = pd.to_datetime
    calls = []

    def tracked(values, **kwargs):
        calls.append(len(values))
        return original(values, **kwargs)

    values = pd.Series(["20260908", "2026-9-7"] * 1000, index=range(10, 2010))
    monkeypatch.setattr(pd, "to_datetime", tracked)
    actual = snapshots._canonical_dates(values)
    assert calls == [2]
    assert actual.tolist() == ["20260908", "20260907"] * 1000
    assert actual.index.equals(values.index)


@pytest.mark.parametrize("date", ["20260230", "not-a-date", "20261301"])
def test_date_normalization_rejects_invalid_values(date):
    with pytest.raises(ValueError):
        snapshots._canonical_dates(pd.Series(["20260908", date]))


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "raw-basic"
    directory.mkdir()
    frame = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ"],
        "trade_date": ["20260907", "20260908"],
        "turnover_rate": [1.25, 2.5],
    })
    frame.to_parquet(directory / "sample.parquet", index=False)
    return directory, frame


def capture(source, destination, **kwargs):
    store = SimpleNamespace(config=SimpleNamespace(backend="file", root=destination.parent))
    return snapshots.export_market_snapshot(
        store, destination, datasets=("daily_basic",),
        supplemental_sources={"daily_basic": source}, **kwargs,
    )


def test_canonical_files_are_copied_without_decoding_value_columns(source, tmp_path, monkeypatch):
    directory, expected = source
    with monkeypatch.context() as patch:
        patch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("Do not decode all columns"))
        patch.setattr(pd.DataFrame, "to_parquet", lambda *a, **k: pytest.fail("Do not re-encode canonical bytes"))
        sealed = capture(directory, tmp_path / "sealed")

    copied = sealed.root / "daily_basic/sample.parquet"
    assert copied.read_bytes() == (directory / "sample.parquet").read_bytes()
    assert copied.stat().st_ino != (directory / "sample.parquet").stat().st_ino
    pd.testing.assert_frame_equal(snapshots.SnapshotReader(sealed.manifest_path).available("daily_basic"), expected)
    assert sealed.metrics["supplemental"]["daily_basic"]["copied"] == 1


@pytest.mark.parametrize("variant", ["date", "timestamp", "numeric", "index", "filtered"])
def test_noncanonical_and_filtered_sources_retain_normalization(source, tmp_path, variant):
    directory, expected = source
    frame = expected.copy()
    options = {}
    if variant == "date":
        frame["trade_date"] = ["2026-9-7", "2026/09/08"]
    elif variant == "timestamp":
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    elif variant == "numeric":
        frame["trade_date"] = [20260907, 20260908]
    elif variant == "index":
        frame.index = pd.Index([10, 20], name="source_index")
    else:
        options = {"start_date": "2026-09-08", "symbols": ["000002.SZ"]}
        expected = expected.iloc[[1]].reset_index(drop=True)
    frame.to_parquet(directory / "sample.parquet", index=(variant == "index"))

    sealed = capture(directory, tmp_path / "sealed", **options)
    actual = snapshots.SnapshotReader(sealed.manifest_path).read(
        "daily_basic", start_date=options.get("start_date"), symbols=options.get("symbols"),
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert sealed.metrics["supplemental"]["daily_basic"]["normalized"] == 1


def test_filtered_empty_file_does_not_leave_undeclared_chunks(source, tmp_path):
    directory, _ = source
    sealed = capture(directory, tmp_path / "sealed", start_date="2027-01-01")
    with snapshots.pinned_market_snapshot(sealed.manifest_path):
        assert list(snapshots.pinned_dataset_path("daily_basic").glob("*.parquet")) == []


def test_changed_raw_source_never_changes_already_sealed_copy(source, tmp_path):
    directory, expected = source
    sealed = capture(directory, tmp_path / "sealed")
    expected.assign(turnover_rate=999.0).to_parquet(directory / "sample.parquet", index=False)
    pd.testing.assert_frame_equal(snapshots.SnapshotReader(sealed.manifest_path).available("daily_basic"), expected)


@pytest.mark.parametrize("mutation", ["overwrite", "add", "delete", "symlink"])
def test_source_set_mutations_during_capture_abort_generation(source, tmp_path, monkeypatch, mutation):
    directory, frame = source
    original = snapshots._capture_supplemental_file

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        path = directory / "sample.parquet"
        if mutation == "overwrite":
            frame.assign(turnover_rate=999.0).to_parquet(path, index=False)
        elif mutation == "add":
            frame.to_parquet(directory / "extra.parquet", index=False)
        elif mutation == "delete":
            path.unlink()
        else:
            saved = tmp_path / "replacement.parquet"
            path.rename(saved)
            path.symlink_to(saved)
        return result

    monkeypatch.setattr(snapshots, "_capture_supplemental_file", mutate)
    with pytest.raises((snapshots.MarketSnapshotError, OSError)):
        capture(directory, tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()
    assert not list(tmp_path.glob(".*.building"))


def test_copy_corruption_is_rejected(source, tmp_path, monkeypatch):
    directory, _ = source
    original = snapshots.shutil.copyfileobj

    def corrupt(source_handle, output, **kwargs):
        original(source_handle, output, **kwargs)
        output.write(b"corrupt")

    monkeypatch.setattr(snapshots.shutil, "copyfileobj", corrupt)
    with pytest.raises(snapshots.MarketSnapshotError, match="copy differs"):
        capture(directory, tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()


def test_bounded_workers_preserve_manifest_identity(source, tmp_path, monkeypatch):
    directory, frame = source
    for number in range(3):
        frame.to_parquet(directory / f"{number}.parquet", index=False)
    serial = capture(directory, tmp_path / "serial", capture_workers=1)
    original = snapshots._capture_supplemental_file
    lock = threading.Lock()
    barrier = threading.Barrier(2)
    active = maximum = 0

    def counted(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait(timeout=10)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(snapshots, "_capture_supplemental_file", counted)
    parallel = capture(directory, tmp_path / "parallel", capture_workers=2)

    assert maximum == 2
    assert serial.fingerprint == parallel.fingerprint
    assert json.loads(serial.manifest_path.read_text()) == json.loads(parallel.manifest_path.read_text())
    assert "metrics" not in json.loads(parallel.manifest_path.read_text())


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_invalid_worker_budget_rejected_before_capture(source, tmp_path, workers):
    with pytest.raises(ValueError, match="capture_workers"):
        capture(source[0], tmp_path / "sealed", capture_workers=workers)
    assert not (tmp_path / "sealed").exists()
