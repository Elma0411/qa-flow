# 文件作用：初始化、重建和检查 Milvus 问答集合结构。
# 关联说明：依赖 runtime，负责集合结构；embedding 和 store_search 负责向量生成与数据读写。

import os
import time
from typing import Optional

from app.core.config import CONFIG
from .runtime import MILVUS_RUNTIME as _rt


def _resolve_source_field_name(allowed_fields: Optional[set] = None) -> str:
    """Return the active source field name for the connected collection."""
    if allowed_fields is not None:
        names = set(allowed_fields)
    else:
        try:
            names = {f.name for f in _rt.milvus_client.schema.fields} if _rt.milvus_client else set()
        except Exception:
            names = set()
    if "source" in names:
        return "source"
    if "source_id" in names:
        return "source_id"
    return "source"


def _resolve_category_field_names(allowed_fields: Optional[set] = None) -> tuple[str, str, str]:
    """Return the active knowledge-category field triplet for the collection."""
    if allowed_fields is not None:
        names = set(allowed_fields)
    else:
        try:
            names = {f.name for f in _rt.milvus_client.schema.fields} if _rt.milvus_client else set()
        except Exception:
            names = set()

    if "knowledge_category" in names:
        return (
            "knowledge_category",
            "knowledge_category_reason",
            "knowledge_category_confidence",
        )
    if "theme" in names:
        return ("theme", "theme_reason", "theme_confidence")
    return (
        "knowledge_category",
        "knowledge_category_reason",
        "knowledge_category_confidence",
    )


def init_milvus() -> tuple[bool, str]:
    """Initialize Milvus or Milvus Lite and connect the main collection."""
    if not _rt.MILVUS_AVAILABLE:
        return False, "Milvus相关库未安装"

    try:
        use_lite = bool(CONFIG.get("milvus", {}).get("enable_milvus_lite", False))
        use_lite = use_lite and _rt.MILVUS_LITE_AVAILABLE

        if use_lite:
            try:
                print("启动 Milvus Lite 服务...")
                base_dir = CONFIG["milvus"].get("lite_base_dir", "milvus_lite_data")
                base_dir_abs = os.path.abspath(base_dir)
                os.makedirs(base_dir_abs, exist_ok=True)
                _rt.MILVUS_DEFAULT_SERVER.set_base_dir(base_dir_abs)
                print(f"Milvus Lite 数据目录: {base_dir_abs}")

                lite_config = CONFIG["milvus"].get("lite_config", {})
                memory_limit_mb = int(lite_config.get("memory_limit_mb", 4096))
                os.environ["MILVUS_LITE_MEMORY_LIMIT"] = str(memory_limit_mb * 1024 * 1024)
                print(f"Milvus Lite 内存限制: {memory_limit_mb}MB")

                try:
                    _rt.MILVUS_DEFAULT_SERVER.stop()
                except Exception:
                    pass

                _rt.MILVUS_DEFAULT_SERVER.start()
                print(f"Milvus Lite 已启动，端口: {_rt.MILVUS_DEFAULT_SERVER.listen_port}")
            except Exception as exc:
                print(f"Milvus Lite 启动失败，尝试连接外部 Milvus: {exc}")
                try:
                    _rt.connections.connect(
                        alias="default",
                        host=CONFIG["milvus"]["host"],
                        port=CONFIG["milvus"]["port"],
                    )
                    _rt.utility.get_server_version()
                    print("已连接到配置中的外部 Milvus")
                except Exception as connect_exc:
                    return False, f"Milvus Lite 启动失败且无法连接现有服务: {connect_exc}"

            max_retries = 15
            retry_interval = 3
            for attempt in range(max_retries):
                try:
                    _rt.connections.connect(
                        alias="default",
                        host="127.0.0.1",
                        port=_rt.MILVUS_DEFAULT_SERVER.listen_port,
                    )
                    server_version = _rt.utility.get_server_version()
                    print(f"Milvus Lite 连接成功，版本: {server_version}")
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        print(f"等待 Milvus Lite 就绪... ({attempt + 1}/{max_retries})")
                        time.sleep(retry_interval)
                    else:
                        try:
                            _rt.connections.connect(
                                alias="default",
                                host=CONFIG["milvus"]["host"],
                                port=CONFIG["milvus"]["port"],
                            )
                            print("Milvus Lite 端口不可用，已退回外部 Milvus 配置")
                            break
                        except Exception:
                            return False, f"Milvus Lite 连接失败: {exc}"
        else:
            ext_host = CONFIG["milvus"]["host"]
            ext_port = int(CONFIG["milvus"]["port"])
            print(f"连接外部 Milvus: {ext_host}:{ext_port}")
            last_err = None
            for attempt in range(1, 31):
                try:
                    _rt.connections.connect(alias="default", host=ext_host, port=ext_port)
                    _rt.utility.get_server_version()
                    print("已连接到外部 Milvus，并通过版本校验")
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt < 30:
                        print(f"等待 Milvus 就绪重试({attempt}/30): {exc}")
                        time.sleep(2)
                    else:
                        return False, f"Milvus连接失败: {last_err}"

        collection_name = CONFIG["milvus"]["collection_name"]
        if _rt.utility.has_collection(collection_name):
            _rt.milvus_client = _rt.Collection(collection_name)
            print(f"Milvus Collection '{collection_name}' 已连接")
            return True, "连接成功"

        success, message = create_milvus_collection()
        if success:
            _rt.milvus_client = _rt.Collection(collection_name)
            print(f"Milvus Collection '{collection_name}' 已创建并连接")
            return True, "创建并连接成功"
        return False, message
    except Exception as exc:
        return False, f"Milvus连接失败: {exc}"


def create_milvus_collection() -> tuple[bool, str]:
    """Create the primary QA collection if it does not exist."""
    try:
        collection_name = CONFIG["milvus"]["collection_name"]
        vector_dim = CONFIG["milvus"]["vector_dim"]
        fields = [
            _rt.FieldSchema(name="id", dtype=_rt.DataType.VARCHAR, max_length=128, is_primary=True),
            _rt.FieldSchema(name="task_id", dtype=_rt.DataType.VARCHAR, max_length=128),
            _rt.FieldSchema(name="original_filename", dtype=_rt.DataType.VARCHAR, max_length=512),
            _rt.FieldSchema(name="source", dtype=_rt.DataType.VARCHAR, max_length=512),
            _rt.FieldSchema(name="source_fact_text", dtype=_rt.DataType.VARCHAR, max_length=4096),
            _rt.FieldSchema(name="question", dtype=_rt.DataType.VARCHAR, max_length=4096),
            _rt.FieldSchema(name="answer", dtype=_rt.DataType.VARCHAR, max_length=8192),
            _rt.FieldSchema(name="question_type", dtype=_rt.DataType.VARCHAR, max_length=64),
            _rt.FieldSchema(name="question_type_reason", dtype=_rt.DataType.VARCHAR, max_length=1024),
            _rt.FieldSchema(name="answer_explanation", dtype=_rt.DataType.VARCHAR, max_length=8192),
            _rt.FieldSchema(name="knowledge_category", dtype=_rt.DataType.VARCHAR, max_length=256),
            _rt.FieldSchema(name="knowledge_category_reason", dtype=_rt.DataType.VARCHAR, max_length=1024),
            _rt.FieldSchema(name="knowledge_category_confidence", dtype=_rt.DataType.FLOAT),
            _rt.FieldSchema(name="difficulty_level", dtype=_rt.DataType.VARCHAR, max_length=64),
            _rt.FieldSchema(name="difficulty_score", dtype=_rt.DataType.FLOAT),
            _rt.FieldSchema(name="llm_model", dtype=_rt.DataType.VARCHAR, max_length=256),
            _rt.FieldSchema(name="embed_model", dtype=_rt.DataType.VARCHAR, max_length=256),
            _rt.FieldSchema(name="embed_dim", dtype=_rt.DataType.INT64),
            _rt.FieldSchema(name="filtered", dtype=_rt.DataType.BOOL),
            _rt.FieldSchema(name="average_score", dtype=_rt.DataType.FLOAT),
            _rt.FieldSchema(name="faithfulness", dtype=_rt.DataType.FLOAT),
            _rt.FieldSchema(name="evaluation_method", dtype=_rt.DataType.VARCHAR, max_length=64),
            _rt.FieldSchema(name="llm_scores", dtype=_rt.DataType.VARCHAR, max_length=2048),
            _rt.FieldSchema(name="llm_reasons", dtype=_rt.DataType.VARCHAR, max_length=8192),
            _rt.FieldSchema(name="local_scores", dtype=_rt.DataType.VARCHAR, max_length=2048),
            _rt.FieldSchema(name="unsupervised_method", dtype=_rt.DataType.VARCHAR, max_length=64),
            _rt.FieldSchema(name="unsupervised_scores", dtype=_rt.DataType.VARCHAR, max_length=2048),
            _rt.FieldSchema(name="unsupervised_meta", dtype=_rt.DataType.VARCHAR, max_length=8192),
            _rt.FieldSchema(name="is_primary", dtype=_rt.DataType.BOOL),
            _rt.FieldSchema(name="is_augmented", dtype=_rt.DataType.BOOL),
            _rt.FieldSchema(name="variant_of", dtype=_rt.DataType.VARCHAR, max_length=256),
            _rt.FieldSchema(name="embedding_vector", dtype=_rt.DataType.FLOAT_VECTOR, dim=vector_dim),
            _rt.FieldSchema(name="created_at", dtype=_rt.DataType.INT64),
            _rt.FieldSchema(name="filter_basis", dtype=_rt.DataType.VARCHAR, max_length=64),
        ]
        schema = _rt.CollectionSchema(fields, "QA pairs collection for semantic search")
        collection = _rt.Collection(collection_name, schema)

        metric_type = str(CONFIG["milvus"].get("metric_type") or "IP").upper()
        index_params = {
            "index_type": CONFIG["milvus"]["index_type"],
            "metric_type": metric_type,
            "params": CONFIG["milvus"]["index_params"],
        }
        try:
            collection.create_index(field_name="embedding_vector", index_params=index_params)
        except Exception as exc:
            message = str(exc)
            if metric_type == "COSINE" and ("metric type not found" in message or "not supported" in message):
                fallback_params = dict(index_params)
                fallback_params["metric_type"] = "IP"
                collection.create_index(field_name="embedding_vector", index_params=fallback_params)
                print("COSINE 不受支持，已回退为 IP（向量已做归一化）。")
            else:
                raise

        for scalar_field in (
            "task_id",
            "knowledge_category",
            "question_type",
            "difficulty_level",
            "filtered",
            "average_score",
            "faithfulness",
        ):
            try:
                collection.create_index(field_name=scalar_field)
            except Exception:
                pass

        print(f"Collection '{collection_name}' 创建成功")
        return True, "创建成功"
    except Exception as exc:
        return False, f"创建Collection失败: {exc}"


def ensure_milvus_initialized() -> None:
    """Initialize Milvus on startup without blocking application boot on failure."""
    if not _rt.MILVUS_AVAILABLE:
        return

    try:
        _rt.embedding_model = None
        print("嵌入模型将在首次使用时加载")
        print("自动初始化 Milvus 连接...")
        success, message = init_milvus()
        if success:
            print(f"Milvus 自动初始化成功: {message}")
        else:
            print(f"Milvus 自动初始化失败: {message}")
            print("服务将继续启动，可稍后手动调用 POST /init-milvus")
    except Exception as exc:
        print(f"Milvus 初始化异常: {exc}")
        print("服务将继续启动，向量功能可能不可用")

__all__ = [
    '_resolve_category_field_names',
    '_resolve_source_field_name',
    'create_milvus_collection',
    'ensure_milvus_initialized',
    'init_milvus',
]
