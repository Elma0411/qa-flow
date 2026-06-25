# 文件作用：提供 baseline 评测脚本共享的 LLM-as-a-judge 客户端与评审函数。
# 关联说明：由 benchmark_synthetic_qa 和各 baseline 脚本复用，统一评审配置与输出结构。

from __future__ import annotations

import json
import os
from typing import Any, Dict

from app.core.config import CONFIG as APP_CONFIG
from app.services.llm import VLMClientConfig, create_vlm_client

try:
    from app.services import llm_config as llm_config_service  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    llm_config_service = None  # type: ignore


DEFAULT_CONFIG: Dict[str, Any] = {
    "api_key": APP_CONFIG.get("api_key", ""),
    "base_url": APP_CONFIG.get("base_url", "https://open.bigmodel.cn/api/paas/v4/"),
    "model": APP_CONFIG.get("model", "glm-4.5-flash"),
    "max_retries": 2,
    "request_timeout": 90,
}


def _load_config_from_env() -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
    env_judge_key = os.environ.get("BENCH_JUDGE_API_KEY")
    env_judge_base_url = os.environ.get("BENCH_JUDGE_BASE_URL")
    env_judge_model = os.environ.get("BENCH_JUDGE_MODEL")

    if llm_config_service:
        try:
            store = llm_config_service.list_profiles()
            active_name = store.get("active")
            profiles = store.get("profiles") or {}
            active_profile = profiles.get(active_name) if isinstance(profiles, dict) else None
            if isinstance(active_profile, dict):
                cfg["api_key"] = active_profile.get("api_key") or cfg["api_key"]
                cfg["base_url"] = active_profile.get("base_url") or cfg["base_url"]
                cfg["model"] = active_profile.get("model") or cfg["model"]
        except Exception:
            pass

    api_key = env_judge_key or os.environ.get("LLM_API_KEY") or cfg.get("api_key")
    if not api_key:
        raise ValueError(
            "评审 LLM API Key 未配置，请在 app.core.config.CONFIG['api_key'] 中设置，或通过环境变量 LLM_API_KEY 提供。"
        )
    cfg["api_key"] = api_key
    if env_judge_base_url or os.environ.get("LLM_BASE_URL"):
        cfg["base_url"] = env_judge_base_url or os.environ.get("LLM_BASE_URL") or cfg["base_url"]
    if env_judge_model or os.environ.get("LLM_MODEL"):
        cfg["model"] = env_judge_model or os.environ.get("LLM_MODEL") or cfg["model"]
    return cfg


def build_judge_client(config: Dict[str, Any]):
    return create_vlm_client(
        VLMClientConfig.from_values(
            api_base=config.get("base_url", ""),
            model_name=config.get("model", ""),
            api_key=config.get("api_key", ""),
            api_type=config.get("api_type"),
            model_version=config.get("model_version"),
            timeout_seconds=float(config.get("request_timeout", 90)),
        )
    )


def build_judge_system_prompt() -> str:
    return (
        "你是一个严谨的中文阅读理解评审专家，任务是比较【参考问答】与【合成问答】的质量差异，"
        "并在三个维度上给出 0–1 的评分：correctness, semantic_equivalence, style_similarity。\n\n"
        "要求：\n"
        "1. correctness：在给定上下文下，合成答案是否正确、是否存在事实性错误或严重遗漏。\n"
        "2. semantic_equivalence：参考问答与合成问答在语义和信息点上的接近程度。\n"
        "3. style_similarity：两者在出题方式、措辞风格、答案长度等方面的相似程度。\n"
        "4. 分数范围为 0–1，可以有小数；请尽量使用两位小数，避免总是 0/0.5/1。\n"
        "5. 所有 reasons 必须使用简体中文，给出简短但具体的解释。\n"
        "6. 最终只返回一个 JSON 对象，键为 correctness / semantic_equivalence / style_similarity，"
        "每个键对应一个形如 {\"score\": float, \"reasons\": str} 的对象，不要包含额外字段。"
    )


def build_judge_user_prompt(
    context: str,
    ref_question: str,
    ref_answer: str,
    gen_question: str,
    gen_answer: str,
) -> str:
    return (
        "请根据下面提供的阅读理解上下文、参考问答和合成问答进行比较打分。\n\n"
        "【上下文】\n"
        f"{context}\n\n"
        "【参考问答（来自人工标注的公开数据集）】\n"
        f"参考问题：{ref_question}\n"
        f"参考答案：{ref_answer}\n\n"
        "【合成问答（由模型自动生成）】\n"
        f"合成问题：{gen_question}\n"
        f"合成答案：{gen_answer}\n\n"
        "请综合判断，在 correctness、semantic_equivalence、style_similarity 三个维度分别给出 0–1 的评分，"
        "并用简体中文简要说明理由。"
    )


def _normalize_judge_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    metrics = ["correctness", "semantic_equivalence", "style_similarity"]
    result: Dict[str, Any] = {}
    for m in metrics:
        entry = raw.get(m, {})
        if not isinstance(entry, dict):
            entry = {}
        score = entry.get("score", 0.0)
        reasons = entry.get("reasons") or entry.get("reason") or ""
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            score_val = 0.0
        result[m] = {
            "score": max(0.0, min(1.0, score_val)),
            "reasons": str(reasons)[:500],
        }
    return result


def judge_pair(
    client: OpenAI,
    context: str,
    ref_question: str,
    ref_answer: str,
    gen_question: str,
    gen_answer: str,
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = DEFAULT_CONFIG

    system_prompt = build_judge_system_prompt()
    user_prompt = build_judge_user_prompt(
        context=context,
        ref_question=ref_question,
        ref_answer=ref_answer,
        gen_question=gen_question,
        gen_answer=gen_answer,
    )

    last_error: Exception | None = None
    for attempt in range(int(config.get("max_retries", 2)) + 1):
        try:
            content = client.create_chat_completion_text(
                model=config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=float(config.get("request_timeout", 90)),
            ) or ""
            data = json.loads(content)
            return _normalize_judge_result(data)
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover
            last_error = exc

    fallback_reason = f"评审失败: {last_error}" if last_error else "评审失败: 未知错误"
    return {
        "correctness": {"score": 0.0, "reasons": fallback_reason},
        "semantic_equivalence": {"score": 0.0, "reasons": fallback_reason},
        "style_similarity": {"score": 0.0, "reasons": fallback_reason},
    }
