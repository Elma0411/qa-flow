# 文件作用：作为 Milvus 运行时、集合、向量和搜索服务的公共 facade。
# 关联说明：聚合 runtime、collection、embedding、store_search、service 给外部调用。

"""Public facade for Milvus runtime, collection, embedding, and search services."""

from .runtime import (
    MILVUS_RUNTIME,
    MilvusRuntime,
)
from .service import (
    _resolve_category_field_names,
    _resolve_source_field_name,
    create_milvus_collection,
    ensure_milvus_initialized,
    generate_embeddings,
    init_milvus,
    load_embedding_model,
    search_qa_pairs_in_milvus,
    store_qa_pairs_to_milvus,
)
def __getattr__(name):
    if hasattr(MILVUS_RUNTIME, name):
        return getattr(MILVUS_RUNTIME, name)
    raise AttributeError(name)


__all__ = [
    "MILVUS_AVAILABLE",
    "MILVUS_LITE_AVAILABLE",
    "MILVUS_RUNTIME",
    "_resolve_category_field_names",
    "_resolve_source_field_name",
    "create_milvus_collection",
    "embedding_model",
    "ensure_milvus_initialized",
    "generate_embeddings",
    "init_milvus",
    "load_embedding_model",
    "milvus_client",
    "search_qa_pairs_in_milvus",
    "store_qa_pairs_to_milvus",
    "MilvusRuntime",
]
