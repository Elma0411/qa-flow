import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.storage import cleanup_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="清理 outputs 目录中早于指定秒数的文件（默认 60 秒）。"
    )
    parser.add_argument(
        "--age-seconds",
        type=int,
        default=60,
        help="删除早于该秒数的文件，默认 60。",
    )
    args = parser.parse_args()
    removed = cleanup_outputs(max(1, args.age_seconds))
    print(f"Removed {len(removed)} file(s).")
    for path in removed:
        print(path)


if __name__ == "__main__":
    main()
