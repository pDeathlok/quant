from __future__ import annotations

from quant.config import Settings
from quant.core.paths import PROJECT_ROOT, ProjectPaths
from quant.routine import paths as routine_paths


def test_project_paths_derive_all_runtime_locations_without_writing(tmp_path) -> None:
    paths = ProjectPaths.from_root(tmp_path)

    assert paths.data == tmp_path / "data"
    assert paths.cache == tmp_path / "data/cache"
    assert paths.logs == tmp_path / "logs"
    assert paths.reports == tmp_path / "reports"
    assert paths.models == tmp_path / "models"
    assert paths.configs == tmp_path / "configs"
    assert paths.web == tmp_path / "web"
    assert not paths.data.exists()
    assert not paths.logs.exists()


def test_project_paths_create_runtime_directories_only_when_requested(tmp_path) -> None:
    paths = ProjectPaths.from_root(tmp_path)

    created = paths.ensure_runtime_directories()

    assert created == (paths.data, paths.cache, paths.logs, paths.reports)
    assert all(path.is_dir() for path in created)


def test_settings_use_canonical_root_without_import_time_writes(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)

    assert settings.data_dir == tmp_path / "data"
    assert settings.cache_dir == tmp_path / "data/cache"
    assert settings.log_dir == tmp_path / "logs"
    assert settings.reports_dir == tmp_path / "reports"
    assert not settings.data_dir.exists()
    assert not settings.log_dir.exists()


def test_routine_paths_reexport_canonical_project_root() -> None:
    assert routine_paths.PROJECT_ROOT == PROJECT_ROOT
