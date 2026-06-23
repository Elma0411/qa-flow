import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import CONFIG  # noqa: E402
from app.services.milvus import (  # noqa: E402
    MILVUS_AVAILABLE,
    create_milvus_collection,
    init_milvus,
    utility,
)


def main() -> int:
    if not MILVUS_AVAILABLE:
        print("Milvus 依赖未安装，无法重建集合。请先安装 pymilvus。")
        return 1

    ok, msg = init_milvus()
    if not ok:
        print(f"初始化 Milvus 失败: {msg}")
        return 1

    collection_name = CONFIG["milvus"]["collection_name"]
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"已删除旧集合: {collection_name}")
    except Exception as exc:  # pragma: no cover - admin helper
        print(f"删除集合失败: {exc}")
        return 1

    ok, msg = create_milvus_collection()
    if not ok:
        print(f"重建集合失败: {msg}")
        return 1

    print(f"Milvus 集合 {collection_name} 已按当前 schema 重建完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
