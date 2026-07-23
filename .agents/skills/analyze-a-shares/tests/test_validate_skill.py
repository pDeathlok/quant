"""Regression tests for semantic skill validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import validate_skill as validator  # noqa: E402


def write_minimal_skill(root: Path, *, link_target: str = "references/policy.md") -> Path:
    skill_dir = root / "sample-skill"
    (skill_dir / "agents").mkdir(parents=True)
    (skill_dir / "references").mkdir()
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: sample-skill",
                "description: Validate a sample skill.",
                "---",
                "",
                f"Read [policy]({link_target}).",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "agents" / "openai.yaml").write_text(
        "\n".join(
            [
                "interface:",
                '  display_name: "Sample Skill"',
                '  short_description: "Validate a complete sample Agent Skill"',
                '  default_prompt: "Use $sample-skill to validate this sample."',
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


class ValidateSkillTests(unittest.TestCase):
    def test_current_skill_passes_static_validation(self) -> None:
        issues = validator.validate_skill(SKILL_DIR, run_checks=False)

        self.assertEqual(issues, [])

    def test_missing_markdown_target_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            skill_dir = write_minimal_skill(Path(temporary_dir))

            issues = validator.validate_skill(skill_dir, run_checks=False)

        self.assertTrue(
            any("declared local path does not exist" in item.message for item in issues)
        )

    def test_missing_declared_skill_script_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            skill_dir = write_minimal_skill(Path(temporary_dir))
            (skill_dir / "references" / "policy.md").write_text(
                "# Policy\n", encoding="utf-8"
            )
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                skill_file.read_text(encoding="utf-8")
                + "\nRun `<skill-dir>/scripts/missing.py`.\n",
                encoding="utf-8",
            )

            issues = validator.validate_skill(skill_dir, run_checks=False)

        self.assertTrue(
            any("scripts/missing.py" in item.message for item in issues)
        )

    def test_existing_markdown_target_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            skill_dir = write_minimal_skill(Path(temporary_dir))
            (skill_dir / "references" / "policy.md").write_text(
                "# Policy\n", encoding="utf-8"
            )

            issues = validator.validate_skill(skill_dir, run_checks=False)

        self.assertEqual(issues, [])

    def test_default_prompt_must_name_the_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            skill_dir = write_minimal_skill(Path(temporary_dir))
            (skill_dir / "references" / "policy.md").write_text(
                "# Policy\n", encoding="utf-8"
            )
            metadata = skill_dir / "agents" / "openai.yaml"
            metadata.write_text(
                metadata.read_text(encoding="utf-8").replace("$sample-skill", "the skill"),
                encoding="utf-8",
            )

            issues = validator.validate_skill(skill_dir, run_checks=False)

        self.assertTrue(
            any("default_prompt must explicitly mention" in item.message for item in issues)
        )


if __name__ == "__main__":
    unittest.main()
