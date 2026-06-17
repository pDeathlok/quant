from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    uvicorn.run(
        "quant.webapp.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8088,
        reload=False,
    )


if __name__ == "__main__":
    main()
