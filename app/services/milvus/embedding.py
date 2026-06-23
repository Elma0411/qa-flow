# 文件作用：加载嵌入模型并生成文本向量。
# 关联说明：依赖 runtime/config 加载模型，供 store_search 写入和查询时生成向量。

from typing import List

from app.core.config import CONFIG
from .runtime import MILVUS_RUNTIME as _rt


def load_embedding_model() -> _rt.SentenceTransformer:
    """Lazy load the embedding model."""
    if _rt.embedding_model is None:
        try:
            print(f"加载嵌入模型: {CONFIG['milvus']['embedding_model']} ...")
            _rt.embedding_model = _rt.SentenceTransformer(CONFIG['milvus']['embedding_model'])
            print("嵌入模型加载完成")
        except Exception as exc:
            print(f"嵌入模型加载失败: {exc}")
            raise ValueError(f"嵌入模型加载失败: {str(exc)}")
    return _rt.embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    model = load_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


__all__ = ['generate_embeddings', 'load_embedding_model']
