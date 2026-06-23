from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts._launch_api import run_api


if __name__ == "__main__":
    print("启动问答生成API服务...")
    run_api()
