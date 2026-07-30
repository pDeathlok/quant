from __future__ import annotations

import json

from quant.infrastructure.workspace_snapshots import (
    WorkspaceSnapshotRepository,
    canonical_snapshot_date,
    workspace_params_key,
)


def test_snapshot_date_and_params_key_are_canonical() -> None:
    assert canonical_snapshot_date("20260718") == "2026-07-18"
    assert canonical_snapshot_date("2026-07-18") == "2026-07-18"
    assert canonical_snapshot_date(None) == "latest"
    assert workspace_params_key({"limit": 18, "enabled": True}) == workspace_params_key(
        {"enabled": True, "limit": 18}
    )


def test_repository_round_trip_reads_nearest_prior_snapshot(tmp_path) -> None:
    repository = WorkspaceSnapshotRepository(
        directory=tmp_path / "snapshots",
        table_name="web_workspace_snapshots",
    )
    repository.write(
        "convertible_bond_grid_plan",
        "20260717",
        {"trade_date": "20260717", "candidates": [{"ts_code": "123001.SZ"}]},
        params={"limit": 18},
        write_sql=False,
    )

    payload = repository.read(
        "convertible_bond_grid_plan",
        snapshot_date="2026-07-18",
        params={"limit": 18},
        allow_sql=False,
    )

    assert payload is not None
    assert payload["candidates"][0]["ts_code"] == "123001.SZ"
    assert payload["cache"] == {
        "hit": True,
        "backend": "filesystem",
        "workspace": "convertible_bond_grid_plan",
        "snapshot_date": "2026-07-17",
        "requested_date": "2026-07-18",
        "stale": True,
    }


def test_repository_skips_corrupt_snapshot_and_does_not_open_sql_on_file_hit(tmp_path) -> None:
    store_calls: list[str] = []

    def store_factory():
        store_calls.append("called")
        raise AssertionError("filesystem hit must not initialize SQL storage")

    repository = WorkspaceSnapshotRepository(
        directory=tmp_path / "snapshots",
        table_name="web_workspace_snapshots",
        store_factory=store_factory,
    )
    params_key = workspace_params_key({"limit": 5})
    directory = repository.file_path("selector", params_key, "latest").parent
    directory.mkdir(parents=True)
    repository.file_path("selector", params_key, "2026-07-16").write_text(
        json.dumps({"signal_date": "2026-07-16", "stocks": [{"ts_code": "000001.SZ"}]}),
        encoding="utf-8",
    )
    repository.file_path("selector", params_key, "2026-07-17").write_text(
        "{not-json",
        encoding="utf-8",
    )

    payload = repository.read("selector", snapshot_date="2026-07-18", params={"limit": 5})

    assert payload is not None
    assert payload["signal_date"] == "2026-07-16"
    assert store_calls == []


def test_write_records_generated_cache_metadata_without_mutating_input(tmp_path) -> None:
    repository = WorkspaceSnapshotRepository(
        directory=tmp_path / "snapshots",
        table_name="web_workspace_snapshots",
    )
    source = {"trade_date": "2026-07-18", "candidates": []}

    repository.write("workspace", "2026-07-18", source, write_sql=False)

    params_key = workspace_params_key()
    stored = json.loads(
        repository.file_path("workspace", params_key, "2026-07-18").read_text(encoding="utf-8")
    )
    assert "cache" not in source
    assert stored["cache"]["backend"] == "generated"
    assert stored["cache"]["hit"] is False
    assert stored["cache"]["snapshot_date"] == "2026-07-18"
    assert repository.file_path("workspace", params_key, "latest").exists()
