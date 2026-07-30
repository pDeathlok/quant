#!/usr/bin/env python3
"""Validate an Agent Skill's structure, local references, scripts, and tests."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SKILL_PATH_PATTERN = re.compile(
    r"<skill-dir>/((?:references|scripts)/[A-Za-z0-9._/-]+)"
)
REQUIRED_INTERFACE_FIELDS = ("display_name", "short_description", "default_prompt")


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    location: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a skill's frontmatter, UI metadata, local links, declared "
            "resources, script CLIs, and unittest suite using only the standard library."
        )
    )
    parser.add_argument(
        "skill_dir",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1]),
        help="Skill directory (default: the parent skill containing this script).",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Skip script --help smoke checks and unittest execution.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return parser.parse_args()


def issue(level: str, location: Path | str, message: str) -> ValidationIssue:
    return ValidationIssue(level=level, location=str(location), message=message)


def parse_frontmatter(skill_file: Path) -> tuple[dict[str, str], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    try:
        lines = skill_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [issue("error", skill_file, f"cannot read SKILL.md: {exc}")]

    if not lines or lines[0].strip() != "---":
        return {}, [issue("error", skill_file, "SKILL.md must start with YAML frontmatter.")]
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, [issue("error", skill_file, "SKILL.md frontmatter is not closed.")]

    fields: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:end], start=2):
        if not line.strip():
            continue
        if ":" not in line:
            issues.append(
                issue("error", f"{skill_file}:{line_number}", "invalid frontmatter field.")
            )
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key in fields:
            issues.append(
                issue("error", f"{skill_file}:{line_number}", f"duplicate field {key!r}.")
            )
        fields[key] = value

    expected = {"name", "description"}
    missing = expected - set(fields)
    extra = set(fields) - expected
    if missing:
        issues.append(issue("error", skill_file, f"missing frontmatter fields: {sorted(missing)}."))
    if extra:
        issues.append(
            issue("error", skill_file, f"unsupported frontmatter fields: {sorted(extra)}.")
        )
    if not fields.get("description", "").strip():
        issues.append(issue("error", skill_file, "description must be non-empty."))
    return fields, issues


def validate_frontmatter(skill_dir: Path) -> list[ValidationIssue]:
    skill_file = skill_dir / "SKILL.md"
    fields, issues = parse_frontmatter(skill_file)
    name = fields.get("name", "")
    if name and not NAME_PATTERN.fullmatch(name):
        issues.append(
            issue("error", skill_file, "name must contain only lowercase letters, digits, and hyphens.")
        )
    if name and skill_dir.name != name:
        issues.append(
            issue(
                "error",
                skill_file,
                f"folder name {skill_dir.name!r} must match skill name {name!r}.",
            )
        )
    return issues


def read_simple_quoted_yaml_fields(path: Path) -> tuple[dict[str, str], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [issue("error", path, f"cannot read agents/openai.yaml: {exc}")]

    fields: dict[str, str] = {}
    pattern = re.compile(r"^\s+([a-z_]+):\s+\"(.*)\"\s*$")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.strip().endswith(":"):
            continue
        match = pattern.match(line)
        if not match:
            issues.append(
                issue(
                    "error",
                    f"{path}:{line_number}",
                    "interface string values must be double-quoted.",
                )
            )
            continue
        fields[match.group(1)] = match.group(2)
    return fields, issues


def validate_openai_yaml(skill_dir: Path) -> list[ValidationIssue]:
    metadata_file = skill_dir / "agents" / "openai.yaml"
    if not metadata_file.is_file():
        return [issue("error", metadata_file, "agents/openai.yaml is required for this skill.")]

    fields, issues = read_simple_quoted_yaml_fields(metadata_file)
    for field in REQUIRED_INTERFACE_FIELDS:
        if not fields.get(field, "").strip():
            issues.append(issue("error", metadata_file, f"missing interface.{field}."))

    short_description = fields.get("short_description", "")
    if short_description and not 25 <= len(short_description) <= 64:
        issues.append(
            issue(
                "error",
                metadata_file,
                "interface.short_description must contain 25–64 characters.",
            )
        )

    skill_name = skill_dir.name
    default_prompt = fields.get("default_prompt", "")
    if default_prompt and f"${skill_name}" not in default_prompt:
        issues.append(
            issue(
                "error",
                metadata_file,
                f"interface.default_prompt must explicitly mention ${skill_name}.",
            )
        )
    return issues


def is_within(candidate: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(candidate), str(root))) == str(root)
    except ValueError:
        return False


def iter_declared_paths(markdown_file: Path, skill_dir: Path) -> Iterable[tuple[str, Path]]:
    text = markdown_file.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
        target = raw_target.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#", 1)[0]
        if not target or "*" in target:
            continue
        yield raw_target, (markdown_file.parent / target).resolve()
    for relative in SKILL_PATH_PATTERN.findall(text):
        yield f"<skill-dir>/{relative}", (skill_dir / relative).resolve()


def validate_local_paths(skill_dir: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    resolved_root = skill_dir.resolve()
    for markdown_file in sorted(skill_dir.rglob("*.md")):
        try:
            declared_paths = list(iter_declared_paths(markdown_file, resolved_root))
        except OSError as exc:
            issues.append(issue("error", markdown_file, f"cannot read Markdown: {exc}"))
            continue
        for declared, target in declared_paths:
            if not is_within(target, resolved_root):
                issues.append(
                    issue("error", markdown_file, f"local path escapes the skill: {declared}.")
                )
            elif not target.exists():
                issues.append(
                    issue("error", markdown_file, f"declared local path does not exist: {declared}.")
                )
    return issues


def run_command(
    command: list[str], *, cwd: Path, label: str, timeout: int = 30
) -> list[ValidationIssue]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [issue("error", label, f"could not run check: {exc}")]
    if result.returncode == 0:
        return []
    output = (result.stderr or result.stdout).strip()
    if len(output) > 1200:
        output = output[-1200:]
    return [
        issue(
            "error",
            label,
            f"check exited with {result.returncode}: {output or 'no output'}",
        )
    ]


def validate_runtime_checks(skill_dir: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    scripts_dir = skill_dir / "scripts"
    for script in sorted(scripts_dir.glob("*.py")):
        issues.extend(
            run_command(
                [sys.executable, str(script), "--help"],
                cwd=skill_dir,
                label=f"{script} --help",
            )
        )

    tests_dir = skill_dir / "tests"
    if not tests_dir.is_dir():
        if any(scripts_dir.glob("*.py")):
            issues.append(issue("warning", tests_dir, "scripts exist but no tests directory was found."))
        return issues
    issues.extend(
        run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(tests_dir),
                "-p",
                "test_*.py",
            ],
            cwd=skill_dir,
            label="unittest suite",
            timeout=60,
        )
    )
    return issues


def validate_skill(skill_dir: Path, *, run_checks: bool = True) -> list[ValidationIssue]:
    skill_dir = skill_dir.expanduser().resolve()
    if not skill_dir.is_dir():
        return [issue("error", skill_dir, "skill directory does not exist.")]

    issues = [
        *validate_frontmatter(skill_dir),
        *validate_openai_yaml(skill_dir),
        *validate_local_paths(skill_dir),
    ]
    if run_checks and not any(item.level == "error" for item in issues):
        issues.extend(validate_runtime_checks(skill_dir))
    return issues


def render_text(skill_dir: Path, issues: list[ValidationIssue]) -> str:
    errors = sum(item.level == "error" for item in issues)
    warnings = sum(item.level == "warning" for item in issues)
    lines = [f"Skill validation: {skill_dir}", f"errors={errors} warnings={warnings}"]
    lines.extend(
        f"- {item.level.upper()} [{item.location}] {item.message}" for item in issues
    )
    if not issues:
        lines.append("- OK: no validation issues found.")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    skill_dir = Path(args.skill_dir).expanduser().resolve()
    issues = validate_skill(skill_dir, run_checks=not args.static_only)
    if args.format == "json":
        payload = {
            "skill_dir": str(skill_dir),
            "valid": not any(item.level == "error" for item in issues),
            "issues": [asdict(item) for item in issues],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(skill_dir, issues))
    return 1 if any(item.level == "error" for item in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
