from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical repository and runtime artifact locations."""

    root: Path
    data: Path
    cache: Path
    logs: Path
    reports: Path
    models: Path
    configs: Path
    web: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        canonical_root = Path(root).expanduser().resolve()
        data = canonical_root / "data"
        return cls(
            root=canonical_root,
            data=data,
            cache=data / "cache",
            logs=canonical_root / "logs",
            reports=canonical_root / "reports",
            models=canonical_root / "models",
            configs=canonical_root / "configs",
            web=canonical_root / "web",
        )

    def ensure_runtime_directories(self) -> tuple[Path, Path, Path, Path]:
        """Create only shared runtime roots through an explicit operation."""

        directories = (self.data, self.cache, self.logs, self.reports)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories


PROJECT_PATHS = ProjectPaths.from_root(Path(__file__).resolve().parents[3])
PROJECT_ROOT = PROJECT_PATHS.root
