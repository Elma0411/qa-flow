# 文件作用：集中定义环境变量配置、运行目录和共享运行状态。
# 关联说明：为 routers 和 services 提供全局配置；runtime_paths 处理路径，logger 处理日志。

import asyncio
import os
from typing import Dict

from app.core.runtime_paths import (
    DEFAULT_COVERAGE_EMBED_MODEL_NAME,
    DEFAULT_FLUENCY_MODEL_NAME,
    DEFAULT_UNSUPERVISED_NLI_MODEL_NAME,
    DEFAULT_UNSUPERVISED_QA_MODEL_NAME,
    MILVUS_LITE_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    RUNTIME_ROOT,
    UPLOADS_DIR,
    model_path,
    resolve_model_reference,
)

# Central configuration shared across the FastAPI app. This mirrors the original
# CONFIG dictionary so route handlers can keep their existing behavior.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # dotenv 为可选依赖，容错处理以便在缺失时继续运行
    pass

API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"
MODEL = os.environ.get("LLM_MODEL") or "glm-4-flash"
API_TYPE = os.environ.get("LLM_API_TYPE") or os.environ.get("VLM_API_TYPE") or "openai"
MODEL_VERSION = os.environ.get("LLM_MODEL_VERSION") or os.environ.get("VLM_MODEL_VERSION") or ""

_UNSUPERVISED_HYPOTHESIS_TIMEOUT_ENV = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_TIMEOUT") or "").strip()
UNSUPERVISED_HYPOTHESIS_TIMEOUT = (
    int(_UNSUPERVISED_HYPOTHESIS_TIMEOUT_ENV) if _UNSUPERVISED_HYPOTHESIS_TIMEOUT_ENV else None
)
_UNSUPERVISED_HYPOTHESIS_MAX_RETRIES_ENV = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_MAX_RETRIES") or "").strip()
UNSUPERVISED_HYPOTHESIS_MAX_RETRIES = (
    int(_UNSUPERVISED_HYPOTHESIS_MAX_RETRIES_ENV) if _UNSUPERVISED_HYPOTHESIS_MAX_RETRIES_ENV else None
)
_UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY_ENV = str(
    os.environ.get("UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY") or ""
).strip()
UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY = (
    int(_UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY_ENV) if _UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY_ENV else None
)
_UNSUPERVISED_HYPOTHESIS_API_KEY_ENV = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_API_KEY") or "").strip() or None
_UNSUPERVISED_HYPOTHESIS_BASE_URL_ENV = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_BASE_URL") or "").strip() or None
_UNSUPERVISED_HYPOTHESIS_MODEL_ENV = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_MODEL") or "").strip() or None

# By default, hypothesis LLM follows the same (api_key/base_url/model) as the main LLM config.
# If any UNSUPERVISED_HYPOTHESIS_* env var is set, we treat it as an explicit override and do
# not auto-sync it when `/llm-configs/{name}/activate` is called.
UNSUPERVISED_HYPOTHESIS_LLM_LOCKED = bool(
    _UNSUPERVISED_HYPOTHESIS_API_KEY_ENV
    or _UNSUPERVISED_HYPOTHESIS_BASE_URL_ENV
    or _UNSUPERVISED_HYPOTHESIS_MODEL_ENV
)
UNSUPERVISED_HYPOTHESIS_API_KEY = _UNSUPERVISED_HYPOTHESIS_API_KEY_ENV or (API_KEY.strip() or None)
UNSUPERVISED_HYPOTHESIS_BASE_URL = _UNSUPERVISED_HYPOTHESIS_BASE_URL_ENV or (BASE_URL.strip() or None)
UNSUPERVISED_HYPOTHESIS_MODEL = _UNSUPERVISED_HYPOTHESIS_MODEL_ENV or (MODEL.strip() or None)
UNSUPERVISED_HYPOTHESIS_MODE = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_MODE") or "").strip().lower() or None

UNSUPERVISED_NLI_MODEL_PATH = os.environ.get(
    "UNSUPERVISED_NLI_MODEL_PATH",
    model_path(DEFAULT_UNSUPERVISED_NLI_MODEL_NAME),
)
UNSUPERVISED_NLI_MODEL_PATH = resolve_model_reference(
    UNSUPERVISED_NLI_MODEL_PATH,
    default_name=DEFAULT_UNSUPERVISED_NLI_MODEL_NAME,
)
UNSUPERVISED_NLI_DEVICE = os.environ.get("UNSUPERVISED_NLI_DEVICE", "auto")
UNSUPERVISED_NLI_MAX_LENGTH = int(os.environ.get("UNSUPERVISED_NLI_MAX_LENGTH", "512") or 512)
UNSUPERVISED_NLI_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_NLI_BATCH_SIZE", "16") or 16)

UNSUPERVISED_QA_MODEL_PATH = os.environ.get(
    "UNSUPERVISED_QA_MODEL_PATH",
    model_path(DEFAULT_UNSUPERVISED_QA_MODEL_NAME),
)
UNSUPERVISED_QA_MODEL_PATH = resolve_model_reference(
    UNSUPERVISED_QA_MODEL_PATH,
    default_name=DEFAULT_UNSUPERVISED_QA_MODEL_NAME,
)
UNSUPERVISED_QA_DEVICE = os.environ.get("UNSUPERVISED_QA_DEVICE", "auto")
UNSUPERVISED_QA_MAX_LENGTH = int(os.environ.get("UNSUPERVISED_QA_MAX_LENGTH", "384") or 384)
UNSUPERVISED_QA_DOC_STRIDE = int(os.environ.get("UNSUPERVISED_QA_DOC_STRIDE", "128") or 128)
UNSUPERVISED_QA_MAX_ANSWER_LENGTH = int(os.environ.get("UNSUPERVISED_QA_MAX_ANSWER_LENGTH", "64") or 64)
UNSUPERVISED_QA_N_BEST = int(os.environ.get("UNSUPERVISED_QA_N_BEST", "20") or 20)
UNSUPERVISED_QA_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_QA_BATCH_SIZE", "8") or 8)
UNSUPERVISED_QA_SCORE_MODE = str(os.environ.get("UNSUPERVISED_QA_SCORE_MODE", "auto") or "auto").strip().lower()
UNSUPERVISED_QA_TEMPERATURE = float(os.environ.get("UNSUPERVISED_QA_TEMPERATURE", "1.0") or 1.0)
UNSUPERVISED_QA_SOFTMAX_TOPK = int(os.environ.get("UNSUPERVISED_QA_SOFTMAX_TOPK", "8") or 8)
UNSUPERVISED_QA_USE_FAST_TOKENIZER = str(
    os.environ.get("UNSUPERVISED_QA_USE_FAST_TOKENIZER", "true") or "true"
).strip().lower() in {"1", "true", "yes", "y"}

UNSUPERVISED_COVERAGE_EMBED_MODEL_PATH = os.environ.get(
    "UNSUPERVISED_COVERAGE_EMBED_MODEL_PATH",
    model_path(DEFAULT_COVERAGE_EMBED_MODEL_NAME),
)
UNSUPERVISED_COVERAGE_EMBED_MODEL_PATH = resolve_model_reference(
    UNSUPERVISED_COVERAGE_EMBED_MODEL_PATH,
    default_name=DEFAULT_COVERAGE_EMBED_MODEL_NAME,
)
UNSUPERVISED_COVERAGE_DEVICE = os.environ.get("UNSUPERVISED_COVERAGE_DEVICE", "auto")
UNSUPERVISED_COVERAGE_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_COVERAGE_BATCH_SIZE", "32") or 32)
UNSUPERVISED_COVERAGE_UNIT_TYPE = str(os.environ.get("UNSUPERVISED_COVERAGE_UNIT_TYPE", "clause_sentence") or "clause_sentence")
UNSUPERVISED_COVERAGE_MIN_UNIT_CHARS = int(os.environ.get("UNSUPERVISED_COVERAGE_MIN_UNIT_CHARS", "10") or 10)
UNSUPERVISED_COVERAGE_MAX_UNITS = int(os.environ.get("UNSUPERVISED_COVERAGE_MAX_UNITS", "256") or 256)
UNSUPERVISED_COVERAGE_SIM_MAPPING = str(
    os.environ.get("UNSUPERVISED_COVERAGE_SIM_MAPPING", "neg_cdf") or "neg_cdf"
).strip().lower()
if UNSUPERVISED_COVERAGE_SIM_MAPPING not in {"clip0", "linear01", "sigmoid_auto_tau", "neg_cdf"}:
    UNSUPERVISED_COVERAGE_SIM_MAPPING = "clip0"
UNSUPERVISED_COVERAGE_TAU = float(os.environ.get("UNSUPERVISED_COVERAGE_TAU", "0.42") or 0.42)
UNSUPERVISED_COVERAGE_AUTO_TAU = str(os.environ.get("UNSUPERVISED_COVERAGE_AUTO_TAU", "true") or "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
UNSUPERVISED_COVERAGE_NEG_QUANTILE = float(os.environ.get("UNSUPERVISED_COVERAGE_NEG_QUANTILE", "0.95") or 0.95)
UNSUPERVISED_COVERAGE_NEG_SAMPLES_PER_GROUP = int(
    os.environ.get("UNSUPERVISED_COVERAGE_NEG_SAMPLES_PER_GROUP", "24") or 24
)
UNSUPERVISED_COVERAGE_RANDOM_SEED = int(os.environ.get("UNSUPERVISED_COVERAGE_RANDOM_SEED", "13") or 13)
UNSUPERVISED_COVERAGE_SIGMOID_TEMPERATURE = float(os.environ.get("UNSUPERVISED_COVERAGE_SIGMOID_TEMPERATURE", "0.08") or 0.08)

UNSUPERVISED_ENABLE_FLUENCY_PPL = str(os.environ.get("UNSUPERVISED_ENABLE_FLUENCY_PPL", "false") or "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
UNSUPERVISED_FLUENCY_MODEL_PATH = os.environ.get(
    "UNSUPERVISED_FLUENCY_MODEL_PATH",
    model_path(DEFAULT_FLUENCY_MODEL_NAME),
)
UNSUPERVISED_FLUENCY_MODEL_PATH = resolve_model_reference(
    UNSUPERVISED_FLUENCY_MODEL_PATH,
    default_name=DEFAULT_FLUENCY_MODEL_NAME,
)
UNSUPERVISED_FLUENCY_DEVICE = os.environ.get("UNSUPERVISED_FLUENCY_DEVICE", "auto")
UNSUPERVISED_FLUENCY_SENTENCE_LENGTH = int(os.environ.get("UNSUPERVISED_FLUENCY_SENTENCE_LENGTH", "100") or 100)
UNSUPERVISED_FLUENCY_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_FLUENCY_BATCH_SIZE", "100") or 100)
UNSUPERVISED_FLUENCY_TEMPERATURE = float(os.environ.get("UNSUPERVISED_FLUENCY_TEMPERATURE", "1.0") or 1.0)
UNSUPERVISED_FLUENCY_TEXT_MODE = str(os.environ.get("UNSUPERVISED_FLUENCY_TEXT_MODE", "qa") or "qa").strip().lower()
UNSUPERVISED_FLUENCY_NORM_ALPHA = float(os.environ.get("UNSUPERVISED_FLUENCY_NORM_ALPHA", "0.01") or 0.01)
UNSUPERVISED_FLUENCY_NORM_BETA = float(os.environ.get("UNSUPERVISED_FLUENCY_NORM_BETA", "0.8") or 0.8)

CONFIG = {
    "api_key": API_KEY,
    "base_url": BASE_URL,
    "model": MODEL,
    "api_type": API_TYPE,
    "model_version": MODEL_VERSION,
    "request_timeout": int(os.environ.get("LLM_REQUEST_TIMEOUT", "120") or 120),
    # 文件路径
    "runtime_root": RUNTIME_ROOT,
    "models_dir": MODELS_DIR,
    "outputs_dir": OUTPUTS_DIR,
    "uploads_dir": UPLOADS_DIR,
    "input_file": "qa/1.1.txt",
    "atomic_facts_file": "qa/atomic_facts.json",
    "categorized_facts_file": "qa/categorized_facts.json",
    "qa_pairs_file": "qa/qa_pairs.json",
    "qa_csv_file": "qa/input.csv",
    # 处理参数
    "chunk_size": 600,
    "max_retries": 2,
    "qa_per_fact": 2,
    "fact_detail_level": "fine",
    # Milvus配置
    "milvus": {
        "host": os.environ.get("MILVUS_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MILVUS_PORT", "19530")),
        "collection_name": "qa_pairs_collection",
        "embedding_model": model_path(DEFAULT_COVERAGE_EMBED_MODEL_NAME),
        "vector_dim": 1024,
        "index_type": "HNSW",
        "metric_type": "IP",
        "index_params": {
            "M": 16,
            "efConstruction": 200
        },
        "search_params": {
            "ef": 64
        },
        "enable_milvus_lite": False,
        "lite_base_dir": MILVUS_LITE_DIR,
        "lite_config": {
            "memory_limit_mb": 1024,
            "simple_mode": True,
        }
    },
    "unsupervised": {
        "enable_fluency_ppl": UNSUPERVISED_ENABLE_FLUENCY_PPL,
        "hypothesis_timeout": UNSUPERVISED_HYPOTHESIS_TIMEOUT,
        "hypothesis_max_retries": UNSUPERVISED_HYPOTHESIS_MAX_RETRIES,
        "hypothesis_max_concurrency": UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY,
        "hypothesis_llm_locked": UNSUPERVISED_HYPOTHESIS_LLM_LOCKED,
        "hypothesis_api_key": UNSUPERVISED_HYPOTHESIS_API_KEY,
        "hypothesis_base_url": UNSUPERVISED_HYPOTHESIS_BASE_URL,
        "hypothesis_model": UNSUPERVISED_HYPOTHESIS_MODEL,
        "hypothesis_mode": UNSUPERVISED_HYPOTHESIS_MODE,
        "nli_model_path": UNSUPERVISED_NLI_MODEL_PATH,
        "nli_device": UNSUPERVISED_NLI_DEVICE,
        "nli_max_length": UNSUPERVISED_NLI_MAX_LENGTH,
        "nli_batch_size": UNSUPERVISED_NLI_BATCH_SIZE,
        "qa_model_path": UNSUPERVISED_QA_MODEL_PATH,
        "qa_device": UNSUPERVISED_QA_DEVICE,
        "qa_max_length": UNSUPERVISED_QA_MAX_LENGTH,
        "qa_doc_stride": UNSUPERVISED_QA_DOC_STRIDE,
        "qa_max_answer_length": UNSUPERVISED_QA_MAX_ANSWER_LENGTH,
        "qa_n_best": UNSUPERVISED_QA_N_BEST,
        "qa_batch_size": UNSUPERVISED_QA_BATCH_SIZE,
        "qa_score_mode": UNSUPERVISED_QA_SCORE_MODE,
        "qa_temperature": UNSUPERVISED_QA_TEMPERATURE,
        "qa_softmax_topk": UNSUPERVISED_QA_SOFTMAX_TOPK,
        "qa_use_fast_tokenizer": UNSUPERVISED_QA_USE_FAST_TOKENIZER,
        "coverage_embed_model_path": UNSUPERVISED_COVERAGE_EMBED_MODEL_PATH,
        "coverage_device": UNSUPERVISED_COVERAGE_DEVICE,
        "coverage_embed_batch_size": UNSUPERVISED_COVERAGE_BATCH_SIZE,
        "coverage_unit_type": UNSUPERVISED_COVERAGE_UNIT_TYPE,
        "coverage_min_unit_chars": UNSUPERVISED_COVERAGE_MIN_UNIT_CHARS,
        "coverage_max_units": UNSUPERVISED_COVERAGE_MAX_UNITS,
        "coverage_sim_mapping": UNSUPERVISED_COVERAGE_SIM_MAPPING,
        "coverage_tau": UNSUPERVISED_COVERAGE_TAU,
        "coverage_auto_tau": UNSUPERVISED_COVERAGE_AUTO_TAU,
        "coverage_neg_quantile": UNSUPERVISED_COVERAGE_NEG_QUANTILE,
        "coverage_neg_samples_per_group": UNSUPERVISED_COVERAGE_NEG_SAMPLES_PER_GROUP,
        "coverage_random_seed": UNSUPERVISED_COVERAGE_RANDOM_SEED,
        "coverage_sigmoid_temperature": UNSUPERVISED_COVERAGE_SIGMOID_TEMPERATURE,
        "fluency_model_path": UNSUPERVISED_FLUENCY_MODEL_PATH,
        "fluency_device": UNSUPERVISED_FLUENCY_DEVICE,
        "fluency_sentence_length": UNSUPERVISED_FLUENCY_SENTENCE_LENGTH,
        "fluency_batch_size": UNSUPERVISED_FLUENCY_BATCH_SIZE,
        "fluency_temperature": UNSUPERVISED_FLUENCY_TEMPERATURE,
        "fluency_text_mode": UNSUPERVISED_FLUENCY_TEXT_MODE,
        "fluency_norm_alpha": UNSUPERVISED_FLUENCY_NORM_ALPHA,
        "fluency_norm_beta": UNSUPERVISED_FLUENCY_NORM_BETA,
    },
}

DEFAULT_BATCH_CONCURRENCY = max(1, int(os.environ.get("BATCH_PIPELINE_CONCURRENCY", "3")))
MAX_BATCH_CONCURRENCY = max(
    DEFAULT_BATCH_CONCURRENCY,
    int(os.environ.get("BATCH_PIPELINE_MAX_CONCURRENCY", "8"))
)
LLM_EVALUATION_METRICS = ["relevance", "completeness", "accuracy", "reasonableness", "agnosticism"]
LOCAL_EVALUATION_METRICS = [
    "em",
    "token_f1",
    "rouge_l_f1",
    "bleu_100",
    "bleu",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "missing_reference",
]
LOCAL_EVALUATION_AVG_METRICS = ["em", "token_f1", "rouge_l_f1", "bleu", "bertscore_f1"]
EVAL_BATCH_SIZE = max(1, int(os.environ.get("LLM_EVAL_BATCH_SIZE", "12")))
AUTO_EVAL_MAX_ITEMS_PER_REQUEST = max(
    1, int(os.environ.get("AUTO_EVAL_MAX_ITEMS_PER_REQUEST", "500"))
)
ACTIVE_BATCH_JOBS: Dict[str, asyncio.Task] = {}

__all__ = [
    "CONFIG",
    "DEFAULT_BATCH_CONCURRENCY",
    "MAX_BATCH_CONCURRENCY",
    "LLM_EVALUATION_METRICS",
    "LOCAL_EVALUATION_METRICS",
    "LOCAL_EVALUATION_AVG_METRICS",
    "EVAL_BATCH_SIZE",
    "AUTO_EVAL_MAX_ITEMS_PER_REQUEST",
    "ACTIVE_BATCH_JOBS",
]
