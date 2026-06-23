# 文件作用：解析容器与本地环境下的运行时资源路径。
# 关联说明：为 config、storage、模型缓存等模块统一解释本地和容器路径。

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(raw_value: str, *, base: Path = REPO_ROOT) -> str:
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("path value cannot be empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def _resolve_env_path(env_name: str, default_value: str) -> str:
    raw_value = str(os.environ.get(env_name) or "").strip()
    return _resolve_path(raw_value or default_value)


RUNTIME_ROOT = _resolve_env_path("APP_RUNTIME_ROOT", "runtime_assets")
CACHE_ROOT = _resolve_env_path("APP_CACHE_ROOT", os.path.join(RUNTIME_ROOT, "cache"))
MODELS_DIR = _resolve_env_path("APP_MODELS_DIR", os.path.join(RUNTIME_ROOT, "models"))
OUTPUTS_DIR = _resolve_env_path("APP_OUTPUTS_DIR", os.path.join(RUNTIME_ROOT, "outputs"))
UPLOADS_DIR = _resolve_env_path("APP_UPLOADS_DIR", os.path.join(RUNTIME_ROOT, "uploads"))
MILVUS_LITE_DIR = _resolve_env_path("APP_MILVUS_LITE_DIR", os.path.join(RUNTIME_ROOT, "milvus_lite_data"))
KNOWLEDGE_TAGGING_ROOT = _resolve_env_path(
    "APP_KNOWLEDGE_TAGGING_ROOT",
    os.path.join(RUNTIME_ROOT, "knowledge_tagging_3lvl"),
)
KNOWLEDGE_TAGGING_OUTPUTS_DIR = _resolve_env_path(
    "APP_KNOWLEDGE_TAGGING_OUTPUTS_DIR",
    os.path.join(KNOWLEDGE_TAGGING_ROOT, "outputs"),
)
TORCH_CACHE_DIR = _resolve_env_path("TORCH_HOME", os.path.join(CACHE_ROOT, "torch"))
TRANSFORMERS_CACHE_DIR = _resolve_env_path("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
HF_HOME_DIR = _resolve_env_path("HF_HOME", os.path.join(CACHE_ROOT, "huggingface"))

DEFAULT_UNSUPERVISED_NLI_MODEL_NAME = "mdeberta_v3_base_xnli_nli_2mil7"
ERLANGSHEN_NLI_MODEL_NAME = "erlangshen_roberta_110m_nli"
DEFAULT_UNSUPERVISED_QA_MODEL_NAME = "deepset_xlm_roberta_base_squad2"
DEFAULT_COVERAGE_EMBED_MODEL_NAME = "bge-m3"
DEFAULT_FLUENCY_MODEL_NAME = "chinese_bert_wwm_ext_pytorch"
DEFAULT_KNOWLEDGE_TAGGER_MODEL_NAME = "model_rbt3_webhq_v1"
DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR = _resolve_env_path(
    "KNOWLEDGE_TAGGER_MODEL_DIR",
    os.path.join(KNOWLEDGE_TAGGING_OUTPUTS_DIR, DEFAULT_KNOWLEDGE_TAGGER_MODEL_NAME),
)


def model_path(model_name: str) -> str:
    name = str(model_name or "").strip()
    if not name:
        raise ValueError("model name cannot be empty")
    return str((Path(MODELS_DIR) / name).resolve())


def resolve_model_reference(model_ref: Optional[str], *, default_name: Optional[str] = None) -> str:
    raw_value = str(model_ref or default_name or "").strip()
    if not raw_value:
        raise ValueError("model reference cannot be empty")
    if "/" not in raw_value and "\\" not in raw_value:
        return model_path(raw_value)
    return _resolve_path(raw_value)


os.environ["APP_RUNTIME_ROOT"] = RUNTIME_ROOT
os.environ["APP_CACHE_ROOT"] = CACHE_ROOT
os.environ["APP_MODELS_DIR"] = MODELS_DIR
os.environ["APP_OUTPUTS_DIR"] = OUTPUTS_DIR
os.environ["APP_UPLOADS_DIR"] = UPLOADS_DIR
os.environ["APP_MILVUS_LITE_DIR"] = MILVUS_LITE_DIR
os.environ["APP_KNOWLEDGE_TAGGING_ROOT"] = KNOWLEDGE_TAGGING_ROOT
os.environ["APP_KNOWLEDGE_TAGGING_OUTPUTS_DIR"] = KNOWLEDGE_TAGGING_OUTPUTS_DIR
os.environ["KNOWLEDGE_TAGGER_MODEL_DIR"] = DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR
os.environ["TORCH_HOME"] = TORCH_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE_DIR
os.environ["HF_HOME"] = HF_HOME_DIR

for path in (
    RUNTIME_ROOT,
    CACHE_ROOT,
    MODELS_DIR,
    OUTPUTS_DIR,
    UPLOADS_DIR,
    MILVUS_LITE_DIR,
    KNOWLEDGE_TAGGING_ROOT,
    KNOWLEDGE_TAGGING_OUTPUTS_DIR,
    TORCH_CACHE_DIR,
    TRANSFORMERS_CACHE_DIR,
    HF_HOME_DIR,
):
    os.makedirs(path, exist_ok=True)


__all__ = [
    "CACHE_ROOT",
    "DEFAULT_COVERAGE_EMBED_MODEL_NAME",
    "DEFAULT_FLUENCY_MODEL_NAME",
    "DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR",
    "DEFAULT_KNOWLEDGE_TAGGER_MODEL_NAME",
    "DEFAULT_UNSUPERVISED_NLI_MODEL_NAME",
    "DEFAULT_UNSUPERVISED_QA_MODEL_NAME",
    "ERLANGSHEN_NLI_MODEL_NAME",
    "HF_HOME_DIR",
    "KNOWLEDGE_TAGGING_OUTPUTS_DIR",
    "KNOWLEDGE_TAGGING_ROOT",
    "MILVUS_LITE_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "REPO_ROOT",
    "RUNTIME_ROOT",
    "TORCH_CACHE_DIR",
    "TRANSFORMERS_CACHE_DIR",
    "UPLOADS_DIR",
    "model_path",
    "resolve_model_reference",
]
