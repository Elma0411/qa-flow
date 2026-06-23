# 文件作用：维护 Milvus 客户端、集合名、嵌入模型等运行态对象。
# 关联说明：被 collection、embedding、store_search 共享，保存客户端和模型运行态。

from typing import Any, Optional

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    MILVUS_AVAILABLE = True
except ImportError as import_exc:  # pragma: no cover - optional dependency
    Collection = Any  # type: ignore
    CollectionSchema = Any  # type: ignore
    DataType = Any  # type: ignore
    FieldSchema = Any  # type: ignore
    connections = None  # type: ignore
    utility = None  # type: ignore
    MILVUS_AVAILABLE = False
    print(f" Milvus相关库导入失败: {import_exc}")
    print("向量搜索功能将不可用，请安装pymilvus和sentence-transformers")

try:
    from milvus import default_server as MILVUS_DEFAULT_SERVER

    MILVUS_LITE_AVAILABLE = True
except Exception:
    MILVUS_DEFAULT_SERVER = None  # type: ignore
    MILVUS_LITE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = Any  # type: ignore


class MilvusRuntime:
    """Container for Milvus runtime state and optional dependencies."""

    def __init__(self) -> None:
        self.Collection = Collection
        self.CollectionSchema = CollectionSchema
        self.DataType = DataType
        self.FieldSchema = FieldSchema
        self.SentenceTransformer = SentenceTransformer
        self.connections = connections
        self.utility = utility
        self.MILVUS_AVAILABLE = MILVUS_AVAILABLE
        self.MILVUS_DEFAULT_SERVER = MILVUS_DEFAULT_SERVER
        self.MILVUS_LITE_AVAILABLE = MILVUS_LITE_AVAILABLE
        self.milvus_client: Optional[Collection] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.json_dumps_separators = (",", ":")


MILVUS_RUNTIME = MilvusRuntime()


def __getattr__(name: str) -> Any:
    if hasattr(MILVUS_RUNTIME, name):
        return getattr(MILVUS_RUNTIME, name)
    raise AttributeError(name)


__all__ = [
    "Collection",
    "CollectionSchema",
    "DataType",
    "FieldSchema",
    "MILVUS_AVAILABLE",
    "MILVUS_DEFAULT_SERVER",
    "MILVUS_LITE_AVAILABLE",
    "MILVUS_RUNTIME",
    "SentenceTransformer",
    "connections",
    "utility",
    "MilvusRuntime",
]
