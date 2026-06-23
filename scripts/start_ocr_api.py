"""Launch the standalone dw-compatible OCR service."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    uvicorn.run("app.ocr_compat_app:app", host="0.0.0.0", port=11169, reload=False)


if __name__ == "__main__":
    main()
