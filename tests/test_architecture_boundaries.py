from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "quant"
PRODUCTION_RESEARCH_MODULES = (
    "long_dividend_quality", "tea_master_long", "b1_family_rules",
    "z_skill_rules", "strategy_signal_cache",
)


def _python_imports(root: Path) -> dict[Path, set[str]]:
    imports: dict[Path, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        imports[path] = modules
    return imports


def _layer_violations(root: Path, forbidden_prefixes: tuple[str, ...]) -> dict[str, list[str]]:
    def is_forbidden(module: str) -> bool:
        return any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )

    return {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            module for module in modules if is_forbidden(module)
        )
        for path, modules in _python_imports(root).items()
        if any(is_forbidden(module) for module in modules)
    }


def test_routine_layer_does_not_depend_on_web_interface() -> None:
    assert _layer_violations(PACKAGE_ROOT / "routine", ("quant.webapp",)) == {}


def test_application_layer_does_not_depend_on_interfaces_or_routine() -> None:
    assert _layer_violations(
        PACKAGE_ROOT / "application",
        ("quant.webapp", "quant.routine", "quant.infrastructure"),
    ) == {}


def test_core_and_infrastructure_do_not_depend_on_delivery_layers() -> None:
    assert _layer_violations(
        PACKAGE_ROOT / "core",
        ("quant.application", "quant.infrastructure", "quant.routine", "quant.webapp"),
    ) == {}
    assert _layer_violations(
        PACKAGE_ROOT / "infrastructure",
        ("quant.application", "quant.routine", "quant.webapp"),
    ) == {}


def test_package_does_not_import_research_scripts_as_python_modules() -> None:
    violations = {
        str(path.relative_to(PROJECT_ROOT)): sorted(
            module for module in modules if module == "scripts" or module.startswith("scripts.")
        )
        for path, modules in _python_imports(PACKAGE_ROOT).items()
        if any(module == "scripts" or module.startswith("scripts.") for module in modules)
    }

    assert violations == {}


def _runtime_boundary_violations(source: str) -> list[str]:
    tree = ast.parse(source)
    # Standalone examples guarded by __main__ are not import-time production
    # dependencies. Function bodies remain checked because callers execute them.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.If)
        and ast.dump(node.test) == ast.dump(ast.parse("__name__ == '__main__'", mode="eval").body)
    )]
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({alias.asname or alias.name: alias.name for alias in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update({alias.asname or alias.name: f"{node.module}.{alias.name}" for alias in node.names})

    def qualified(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified(node.value)}.{node.attr}"
        if isinstance(node, ast.Subscript):
            return qualified(node.value)
        return ""

    forbidden_loaders = {
        "spec_from_file_location", "SourceFileLoader", "SourcelessFileLoader",
        "exec_module", "load_module", "run_path",
    }
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = qualified(node.func)
            if target.rsplit(".", 1)[-1] in forbidden_loaders:
                violations.append(f"{node.lineno}: file-based execution {target}")
            if target.startswith("sys.path.") and target.rsplit(".", 1)[-1] in {
                "insert", "append", "extend", "remove", "pop", "clear",
                "sort", "reverse", "__setitem__", "__delitem__",
            }:
                violations.append(f"{node.lineno}: search-path mutation {target}")
            if target in {"importlib.import_module", "__import__"} and node.args:
                module = node.args[0]
                if isinstance(module, ast.Constant) and isinstance(module.value, str):
                    if module.value == "scripts" or module.value.startswith("scripts."):
                        violations.append(f"{node.lineno}: dynamic script import {module.value}")
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        for target in targets:
            if qualified(target) == "sys.path":
                violations.append(f"{node.lineno}: search-path assignment")
    return violations


def _package_dependency_closure(roots: list[Path]) -> set[Path]:
    """Follow local imports, including from-package aliases and relative imports."""
    pending = list(roots)
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
        package = parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package) - node.level + 1]) if node.level else []
                if node.module:
                    prefix += node.module.split(".")
                module = ".".join(prefix)
                modules.add(module)
                modules.update(f"{module}.{alias.name}" for alias in node.names)
        for module in modules:
            if not module.startswith("quant.") and module != "quant":
                continue
            candidate = PACKAGE_ROOT.parent.joinpath(*module.split("."))
            initializers = [PACKAGE_ROOT.parent.joinpath(*module.split(".")[:depth]) / "__init__.py" for depth in range(1, len(module.split(".")))]
            for dependency in (candidate.with_suffix(".py"), candidate / "__init__.py", *initializers):
                if dependency.is_file() and dependency not in visited:
                    pending.append(dependency)
    return visited


def test_production_research_closure_has_no_runtime_script_loading() -> None:
    roots = [PACKAGE_ROOT / "research" / f"{name}.py" for name in PRODUCTION_RESEARCH_MODULES]
    closure = _package_dependency_closure(roots)
    assert PACKAGE_ROOT / "research/b1_backtest.py" in closure
    assert PACKAGE_ROOT / "research/rule_backtest_support.py" in closure
    assert PACKAGE_ROOT / "__init__.py" in closure
    violations = {
        str(path.relative_to(PROJECT_ROOT)): found
        for path in sorted(closure | {PACKAGE_ROOT / "webapp/services.py"})
        if (found := _runtime_boundary_violations(path.read_text(encoding="utf-8")))
    }
    assert violations == {}


@pytest.mark.parametrize("source", [
    "import importlib.util as util\nutil.spec_from_file_location('x', path)",
    "from importlib.util import spec_from_file_location as load\nload('x', path)",
    "spec.loader.exec_module(module)",
    "from importlib.machinery import SourceFileLoader as Loader\nLoader('x', path)",
    "import runpy\nrunpy.run_path(path)",
    "import sys as system\nsystem.path.insert(0, script_dir)",
    "import sys\nsys.path[:] = paths",
    "import sys\nsys.path += [script_dir]",
    "from importlib import import_module as load\nload('scripts.research.model')",
    "__import__('scripts.research.model')",
])
def test_runtime_boundary_guard_detects_indirect_loading(source: str) -> None:
    assert _runtime_boundary_violations(source)


def test_runtime_boundary_guard_allows_static_package_imports() -> None:
    assert _runtime_boundary_violations("from quant.research import long_dividend_quality") == []
