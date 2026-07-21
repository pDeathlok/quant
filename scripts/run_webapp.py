from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args(project_root: Path, arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Quant web application.")
    parser.add_argument("--log-file", default=str(project_root / ".run" / "webapp.log"))
    parser.add_argument("--log-max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=2)
    return parser.parse_args(arguments)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from quant.routine.rotating_logs import configure_rotating_logging

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--log-file", default=str(project_root / ".run" / "webapp.log"))
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    configure_rotating_logging(
        Path(bootstrap_args.log_file).expanduser().resolve(),
        redirect_streams=False,
    )

    args = _parse_args(project_root)
    configure_rotating_logging(
        Path(args.log_file).expanduser().resolve(),
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
    )

    try:
        import uvicorn

        uvicorn.run(
            "quant.webapp.app:create_app",
            factory=True,
            host="127.0.0.1",
            port=8088,
            reload=False,
            log_config=None,
        )
    except Exception:
        logging.getLogger(__name__).exception("Quant web application terminated unexpectedly")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
