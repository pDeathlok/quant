"""Print observed CI dependency versions, without URLs, paths, or guessed pins."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def capture_runtime(requirements_path: Path) -> dict[str, Any]:
    """Capture the installed dependency closure active for this interpreter/OS."""
    source = requirements_path.read_text(encoding="utf-8")
    pending = [
        (Requirement(line.strip()), frozenset({""})) for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    packages: dict[str, str] = {}
    expanded: set[tuple[str, frozenset[str]]] = set()
    while pending:
        requirement, parent_extras = pending.pop()
        name = canonicalize_name(requirement.name)
        if requirement.marker and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in parent_extras
        ):
            continue
        if requirement.url:
            raise ValueError(f"Snapshot requires a plain package requirement: {name}")
        try:
            installed = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            raise RuntimeError(f"Required runtime dependency is not installed: {name}") from None
        if installed.version not in requirement.specifier:
            raise RuntimeError(f"Installed runtime does not satisfy requirement for {name}")
        extras = frozenset({"", *requirement.extras})
        key = (name, extras)
        if key in expanded:
            continue
        expanded.add(key)
        packages[name] = installed.version
        pending.extend((Requirement(value), extras) for value in installed.requires or [])
    return {
        "schema_version": 1,
        "scope": "ci-regression-dependency-closure",
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "requirements_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "packages": dict(sorted(packages.items())),
    }


def format_constraints(snapshot: dict[str, Any]) -> str:
    """Format exact versions from a capture; this is not a portable wheel lock."""
    runtime = snapshot["runtime"]
    header = (
        f"# Observed {runtime['implementation']} {runtime['python']} "
        f"on {runtime['system']} {runtime['machine']}.\n"
        "# CI dependency snapshot, not a full production or cross-platform lock.\n"
        "# See docs/ci-runtime.md for capture and validation scope.\n"
    )
    return header + "".join(
        f"{name}=={version}\n" for name, version in sorted(snapshot["packages"].items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements", type=Path,
        default=Path(__file__).resolve().parents[2] / "requirements/ci.txt",
    )
    parser.add_argument("--format", choices=("json", "constraints"), default="json")
    arguments = parser.parse_args()
    snapshot = capture_runtime(arguments.requirements)
    if arguments.format == "constraints":
        print(format_constraints(snapshot), end="")
    else:
        print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
