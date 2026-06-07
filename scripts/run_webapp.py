from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "quant.webapp.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8088,
        reload=False,
    )


if __name__ == "__main__":
    main()
