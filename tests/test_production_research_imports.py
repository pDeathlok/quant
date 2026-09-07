from __future__ import annotations

import importlib
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "backtest_long_dividend_quality": "long_dividend_quality",
    "backtest_tea_master_long": "tea_master_long",
    "analyze_b1_family_rule_backtest": "b1_family_rules",
    "analyze_z_skill_entry_exit_backtest": "z_skill_rules",
    "rebuild_strategy_signal_cache": "strategy_signal_cache",
}


@pytest.mark.parametrize("script,implementation", ENTRYPOINTS.items())
def test_legacy_import_aliases_the_production_module(script: str, implementation: str, monkeypatch) -> None:
    module = importlib.import_module(f"quant.research.{implementation}")
    legacy = importlib.import_module(f"scripts.research.{script}")
    assert legacy is module
    monkeypatch.setattr(legacy, "PROJECT_ROOT", Path("temporary-test-root"))
    assert module.PROJECT_ROOT == Path("temporary-test-root")


@pytest.mark.parametrize("script,implementation", ENTRYPOINTS.items())
def test_original_cli_dispatches_with_arguments_unchanged(script: str, implementation: str, monkeypatch) -> None:
    module = importlib.import_module(f"quant.research.{implementation}")
    path = PROJECT_ROOT / "scripts/research" / f"{script}.py"
    calls = []
    monkeypatch.setattr(module, "main", lambda: calls.append(sys.argv.copy()))
    monkeypatch.setattr(sys, "argv", [str(path), "--original-argument", "value"])
    runpy.run_path(str(path), run_name="__main__")
    assert calls == [[str(path), "--original-argument", "value"]]


@pytest.mark.parametrize("script", [
    "backtest_long_dividend_quality", "analyze_z_skill_entry_exit_backtest",
    "rebuild_strategy_signal_cache",
])
def test_argparse_cli_help_preserves_src_environment(script: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/research" / f"{script}.py"), "--help"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_package_only_imports_without_repository_scripts(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    shutil.copytree(PROJECT_ROOT / "src/quant", site / "quant", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Governance remains an explicit runtime config, not executable research.
    governance = tmp_path / "governance.json"
    shutil.copyfile(PROJECT_ROOT / "configs/factors/governance.json", governance)
    strategies = tmp_path / "configs/strategies"
    strategies.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "configs/strategies/triple_volume_breakout.yaml", strategies / "triple_volume_breakout.yaml")
    code = """
import importlib
import sys
from pathlib import Path
before = list(sys.path)
for name in %r:
    module = importlib.import_module('quant.research.' + name)
    assert callable(module.main)
    assert Path(module.__file__).is_relative_to(Path.cwd() / 'site-packages')
assert before == sys.path
assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)
assert not any(name.startswith(('analyze_b1_', 'backtest_long_', 'build_training_data')) for name in sys.modules)
""" % list(ENTRYPOINTS.values())
    result = subprocess.run(
        [sys.executable, "-s", "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(site), "FACTOR_GOVERNANCE_CONFIG": str(governance)}, capture_output=True,
        text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_services_use_identical_static_modules() -> None:
    from quant.research import long_dividend_quality, tea_master_long
    from quant.webapp import services

    assert services._long_research_module() is long_dividend_quality
    assert services._tea_master_research_module() is tea_master_long
