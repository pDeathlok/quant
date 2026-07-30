from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "quant"


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
