# 文件作用：聚合 Milvus 初始化、存储、搜索和状态接口。
# 关联说明：聚合同目录底层模块，是 __init__.py facade 的主要来源。

from .runtime import MILVUS_RUNTIME as _runtime
from .collection import (
    _resolve_category_field_names,
    _resolve_source_field_name,
    create_milvus_collection,
    ensure_milvus_initialized,
    init_milvus,
)
from .embedding import generate_embeddings, load_embedding_model
from .store_search import search_qa_pairs_in_milvus, store_qa_pairs_to_milvus

MILVUS_RUNTIME = _runtime

_DYNAMIC_ATTRS = {
    'Collection',
    'CollectionSchema',
    'DataType',
    'FieldSchema',
    'MILVUS_AVAILABLE',
    'MILVUS_DEFAULT_SERVER',
    'MILVUS_LITE_AVAILABLE',
    'SentenceTransformer',
    'connections',
    'embedding_model',
    'milvus_client',
    'utility',
}

def __getattr__(name):
    if name in _DYNAMIC_ATTRS:
        return getattr(_runtime, name)
    raise AttributeError(name)

__all__ = [
    'MILVUS_RUNTIME',
    '_resolve_category_field_names',
    '_resolve_source_field_name',
    'create_milvus_collection',
    'ensure_milvus_initialized',
    'generate_embeddings',
    'init_milvus',
    'load_embedding_model',
    'search_qa_pairs_in_milvus',
    'store_qa_pairs_to_milvus',
]
