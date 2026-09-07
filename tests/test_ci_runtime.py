from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_snapshot_uses_installed_dependency_closure(monkeypatch, tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("Foo>=1\n", encoding="utf-8")
    distributions = {
        "foo": SimpleNamespace(
            version="1.2", metadata={"Name": "Foo"},
            requires=["Bar>=2", 'missing-package; extra == "test"'],
        ),
        "bar": SimpleNamespace(version="2.1", metadata={"Name": "Bar"}, requires=[]),
    }
    monkeypatch.setattr(module["metadata"], "distribution", lambda name: distributions[name.lower()])

    snapshot = module["capture_runtime"](requirements)

    assert snapshot["packages"] == {"bar": "2.1", "foo": "1.2"}
    assert "foo==1.2" in module["format_constraints"](snapshot)
    assert "missing-package" not in module["format_constraints"](snapshot)


def test_runtime_snapshot_rejects_missing_packages_without_guessing_versions(monkeypatch, tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("missing-package>=1\n", encoding="utf-8")

    def missing(name):
        raise module["metadata"].PackageNotFoundError(name)

    monkeypatch.setattr(module["metadata"], "distribution", missing)

    with pytest.raises(RuntimeError, match="not installed: missing-package"):
        module["capture_runtime"](requirements)


def test_runtime_snapshot_rejects_unsatisfied_installed_versions(monkeypatch, tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("foo>=2\n", encoding="utf-8")
    monkeypatch.setattr(
        module["metadata"], "distribution",
        lambda name: SimpleNamespace(version="1.0", metadata={"Name": "foo"}, requires=[]),
    )

    with pytest.raises(RuntimeError, match="does not satisfy"):
        module["capture_runtime"](requirements)


def test_runtime_snapshot_does_not_emit_direct_urls(tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("foo @ https://user:secret@example.test/foo.whl\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        module["capture_runtime"](requirements)

    assert "secret" not in str(caught.value)
    assert "example.test" not in str(caught.value)


def test_runtime_snapshot_tracks_extras_and_cycles(monkeypatch, tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("foo[fast]>=1\nfoo>=1\n", encoding="utf-8")
    distributions = {
        "foo": SimpleNamespace(version="1.2", requires=[
            'bar>=2; extra == "fast"',
            'missing[unused]; extra == "unused"',
            'private @ https://secret@example.test/a.whl ; extra == "unused"',
        ]),
        "bar": SimpleNamespace(version="2.1", requires=["foo[fast]>=1"]),
    }
    monkeypatch.setattr(module["metadata"], "distribution", distributions.__getitem__)

    assert module["capture_runtime"](requirements)["packages"] == {"foo": "1.2", "bar": "2.1"}


def test_runtime_snapshot_checks_constraints_even_after_visiting_package(monkeypatch, tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("foo>=2\nfoo>=1\n", encoding="utf-8")
    monkeypatch.setattr(module["metadata"], "distribution",
                        lambda name: SimpleNamespace(version="1.2", requires=[]))

    with pytest.raises(RuntimeError, match="does not satisfy"):
        module["capture_runtime"](requirements)


@pytest.mark.parametrize("profile", ["py39", "py313"])
def test_checked_in_constraints_match_observed_runtime_snapshot(profile: str) -> None:
    module = runpy.run_path(str(ROOT / "scripts/ci/runtime_snapshot.py"))
    snapshot = json.loads((ROOT / f"requirements/runtime-ci-{profile}.json").read_text(encoding="utf-8"))
    constraints = (ROOT / f"requirements/constraints-ci-{profile}.txt").read_text(encoding="utf-8")

    assert constraints == module["format_constraints"](snapshot)
    assert snapshot["schema_version"] == 1
    assert snapshot["scope"] == "ci-regression-dependency-closure"
    assert snapshot["runtime"]["python"].startswith({"py39": "3.9.", "py313": "3.13."}[profile])
    source = (ROOT / "requirements/ci.txt").read_bytes()
    assert snapshot["requirements_sha256"] == hashlib.sha256(source).hexdigest()
    for line in source.decode("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        version = snapshot["runtime"]["python"]
        if requirement.marker and not requirement.marker.evaluate({
            "python_version": ".".join(version.split(".")[:2]), "python_full_version": version,
        }):
            continue
        assert snapshot["packages"][requirement.name] in requirement.specifier
    for line in constraints.splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            assert str(requirement.specifier) == f"=={snapshot['packages'][requirement.name]}"


def test_ci_is_read_only_and_never_uses_production_sql() -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/source-quality.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )

    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["MARKET_DATA_BACKEND"] == "file"
    assert workflow["env"]["MARKET_DATA_SQL_URL"] == ""
    assert workflow["env"]["TUSHARE_TOKEN"] == ""
    assert workflow["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    matrix = workflow["jobs"]["regression"]["strategy"]["matrix"]["include"]
    assert {item["python"] for item in matrix} == {"3.9", "3.13"}
    for step in workflow["jobs"]["regression"]["steps"]:
        if "uses" in step:
            reference = step["uses"].split("@", 1)[1]
            assert len(reference) == 40 and all(char in "0123456789abcdef" for char in reference)
