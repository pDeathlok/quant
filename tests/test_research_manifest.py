import json

import pytest

from quant.research import build_research_manifest, write_research_manifest


def test_research_manifest_records_data_hash_parameters_and_seed(tmp_path) -> None:
    data_path = tmp_path / "daily.parquet"
    data_path.write_bytes(b"stable-data")

    manifest = build_research_manifest(
        strategy_name="dual_ma",
        parameters={"fast": 5, "slow": 20},
        data_paths=[data_path],
        start_date="20250101",
        end_date="20251231",
        random_seed=42,
        project_root=tmp_path,
        run_id="fixed-run",
    )

    assert manifest["schema_version"] == "research-manifest/v1"
    assert manifest["run_id"] == "fixed-run"
    assert manifest["parameters"] == {"fast": 5, "slow": 20}
    assert manifest["random_seed"] == 42
    assert len(manifest["inputs"][0]["sha256"]) == 64
    assert manifest["inputs"][0]["bytes"] == len(b"stable-data")


def test_research_manifest_is_written_atomically(tmp_path) -> None:
    data_path = tmp_path / "input.csv"
    data_path.write_text("a\n1\n", encoding="utf-8")

    path = write_research_manifest(
        tmp_path / "run",
        strategy_name="test",
        parameters={},
        data_paths=[data_path],
        start_date="20260101",
        end_date="20260131",
        random_seed=7,
        project_root=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "research_manifest.json"
    assert payload["strategy_name"] == "test"
    assert not list(path.parent.glob("*.tmp*"))


def test_research_manifest_fails_closed_for_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        build_research_manifest(
            strategy_name="test",
            parameters={},
            data_paths=[tmp_path / "missing.parquet"],
            start_date="20260101",
            end_date="20260131",
            random_seed=1,
            project_root=tmp_path,
        )
