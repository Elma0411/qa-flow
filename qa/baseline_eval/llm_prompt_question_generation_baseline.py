# 文件作用：运行 LLM 提示词式问题生成基线评测。
# 关联说明：复用 benchmark_synthetic_qa 的样本工具和 synthetic_qa_judge 的评审，替换生成策略为 LLM 提示词。

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI

from qa.common import extract_first_choice_content

# 确保可以从项目根目录导入 qa / app 等包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qa.baseline_eval.common import ALIGN_THRESHOLD, ALIGN_WEIGHTS, _iter_triples
from qa.qa_evaluation.qa_quality_evaluator import QAEvaluator
from qa.baseline_eval.judge import _load_config_from_env, build_judge_client, judge_pair


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ChatGPT/LLM 直接 QG 基线（参考 Chan et al., 2023 'A Case Study on ChatGPT Question Generation'）："
            "使用论文 Fig.1/Fig.2 的提示词生成问题/问答对，再复用本项目的对齐与评审流程输出可对比的 JSONL。"
        ),
    )
    parser.add_argument(
        "--triples-file",
        "-t",
        type=str,
        required=True,
        help="输入三元组 JSONL，需包含 context/question/answer 字段。",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 JSONL（包含对齐与评审结果）。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="最多读取多少条样本（默认 200）。",
    )
    parser.add_argument(
        "--mode",
        choices=["fig1_squad", "fig1_answer_only", "fig2_inago"],
        default="fig1_squad",
        help=(
            "生成模式："
            "fig1_squad=论文 Fig.1（context+answer→question）；"
            "fig1_answer_only=论文 Fig.1 去掉 context（answer→question）；"
            "fig2_inago=论文 Fig.2（context→问答集合，再对齐到参考 QA）。"
        ),
    )
    parser.add_argument(
        "--prompt-language",
        choices=["zh", "en"],
        default="zh",
        help="提示词语言：zh=中文（适合中文数据集），en=英文（贴近论文原文）。",
    )
    parser.add_argument(
        "--qa-per-context",
        type=int,
        default=6,
        help="仅对 fig2_inago 生效：期望生成的问答对数量（默认 6）。",
    )
    parser.add_argument(
        "--max-candidates-per-context",
        type=int,
        default=50,
        help="仅对 fig2_inago 生效：候选问答对上限（默认 50）。",
    )
    parser.add_argument(
        "--keep-below-threshold",
        action="store_true",
        help="保留对齐未达阈值的记录到输出（仍会跳过评审以节省成本）。默认不写入输出。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="并发线程数（默认 20，受 BENCH_JUDGE_MAX_CONCURRENCY 截断）。",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="跳过 LLM 评审（judge_pair），只输出生成与对齐信息。",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=40,
        help="单条评审超时时间（秒，默认 40）。该超时会映射为评审侧 request_timeout，并关闭重试以避免卡死；0 表示不额外超时控制。",
    )
    parser.add_argument(
        "--gen-api-key",
        type=str,
        help="生成侧 API Key（默认读取 BENCH_GEN_API_KEY，否则可回退 judge 配置）。",
    )
    parser.add_argument(
        "--gen-base-url",
        type=str,
        help="生成侧 Base URL（默认读取 BENCH_GEN_BASE_URL，否则可回退 judge 配置）。",
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        help="生成侧模型名称（默认读取 BENCH_GEN_MODEL，否则可回退 judge 配置）。",
    )
    parser.add_argument(
        "--gen-max-retries",
        type=int,
        default=2,
        help="生成侧失败重试次数（默认 2）。",
    )
    parser.add_argument(
        "--gen-timeout",
        type=int,
        default=60,
        help="生成侧请求超时（秒，默认 60）。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="生成响应的最大新 token 数（默认 8192）。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="生成温度（默认 0.0）。论文建议初期使用 0 或较低值以获得稳定输出。",
    )
    parser.add_argument(
        "--save-generated",
        type=str,
        help="可选：保存每个 context 的生成候选与原始输出到 JSONL，便于排查。",
    )
    parser.add_argument(
        "--include-raw-generation",
        action="store_true",
        help="可选：在主输出 JSONL 中额外保留模型原始响应 raw_generation，便于逐条排查解析问题。",
    )
    return parser.parse_args()


def _build_generation_config(args: argparse.Namespace, judge_cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_api_key = os.environ.get("BENCH_GEN_API_KEY")
    env_base_url = os.environ.get("BENCH_GEN_BASE_URL")
    env_model = os.environ.get("BENCH_GEN_MODEL")

    api_key = env_api_key or args.gen_api_key or judge_cfg.get("api_key") or ""
    base_url = env_base_url or args.gen_base_url or judge_cfg.get("base_url") or "https://api.deepseek.com/v1"
    model = env_model or args.gen_model or judge_cfg.get("model") or "deepseek-chat"
    if not api_key:
        raise SystemExit("生成侧 API Key 未配置，请设置 BENCH_GEN_API_KEY 或 --gen-api-key")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": max(0.0, float(args.temperature)),
        "max_new_tokens": max(32, int(args.max_new_tokens)),
        "timeout": max(1, int(args.gen_timeout)),
        "max_retries": max(0, int(args.gen_max_retries)),
    }


def _build_fig1_messages(
    context: str,
    answer: str,
    include_context: bool,
    language: str,
) -> List[Dict[str, str]]:
    """
    论文 Fig.1：
    Given the following context paragraph and answer, generate a question that can be answered by the provided answer:
    Context: {put context here}
    Answer: {put answer here}
    """
    if language == "en":
        system_prompt = (
            "You are a question generation assistant. Generate exactly ONE question. "
            "Return ONLY the question text without quotes or extra words."
        )
        user_prompt = (
            "Given the following context paragraph and answer, generate a question that can be answered by the provided answer:\n"
        )
        if include_context:
            user_prompt += f"Context: {context}\n"
        user_prompt += f"Answer: {answer}\n"
        user_prompt += "Question:"
    else:
        system_prompt = (
            "你是问题生成助手。请只生成且只输出 1 个中文问题，不要输出解释、不要加引号、不要输出多余文本。"
        )
        user_prompt = "给定以下上下文段落和答案，生成一个可以用所给答案回答的问题：\n"
        if include_context:
            user_prompt += f"Context: {context}\n"
        user_prompt += f"Answer: {answer}\n"
        user_prompt += "请只输出问题："
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _build_fig2_messages(context: str, qa_per_context: int, language: str) -> List[Dict[str, str]]:
    """
    论文 Fig.2（iNAGO 主提示词，面向“手册段落→问答集合”的生成）。
    """
    if language == "en":
        prompt = (
            "Given a passage from the user manual of a specific machinery, imagine you are operating said machinery, "
            "and generate a set of question-answer pairs that a user operating it might ask.\n"
            f"Passage: {context}\n"
            f"Generate {max(1, int(qa_per_context))} question-answer pairs.\n"
            "Requirements:\n"
            "1) Each question should be closely related to the passage.\n"
            "2) The set should sufficiently cover the major information in the passage (no more, no less).\n"
            "3) Questions must not be about the passage location.\n"
            "4) Each question should not be a combination of two or more questions.\n"
            "5) Each question should focus on practical application, problem-solving, or inquiry for clarification/definition, "
            "rather than being a traditional quiz question.\n"
            "6) Each question must be answerable by the information in the passage.\n"
            "7) Each question should be clear and brief, and not overly-specific.\n"
            "8) Each answer must be grounded in the passage and concise.\n"
            "Return ONLY a JSON array. Each item must be an object with fields: \"question\" and \"answer\"."
        )
        return [{"role": "user", "content": prompt}]

    prompt = (
        "给定一段文字（Passage），请你设想自己是读者/使用者，基于段落内容生成一组“问答对”。\n"
        f"Passage: {context}\n"
        f"请生成 {max(1, int(qa_per_context))} 条问答对。\n"
        "要求：\n"
        "1. 问题必须与段落内容紧密相关（relatedness）。\n"
        "2. 问题集合应尽量覆盖段落中的主要信息点（completeness），不要扩展到段落之外。\n"
        "3. 问题不要询问段落位置/页码等定位信息。\n"
        "4. 问题应偏向实际理解、问题解决、或对概念/定义的澄清，而不是传统测验题。\n"
        "5. 每个问题必须能仅凭段落信息作答；答案必须来自段落事实（不得编造）。\n"
        "6. 问题表达要清晰、简短（conciseness）；答案也要尽量简短（一个句子或短语）。\n"
        "7. 问题不要过度具体。\n"
        "输出格式：只返回 JSON 数组，每个元素是一个对象，包含字段：\"question\" 和 \"answer\"。不要输出任何额外文字。"
    )
    return [{"role": "user", "content": prompt}]


def _call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
) -> str:
    last_error: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                timeout=timeout_seconds,
            )
            content = extract_first_choice_content(resp) or ""
            if content.strip():
                return content
        except Exception as exc:  # pragma: no cover
            last_error = exc
            continue
    return f"" if last_error is None else ""


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_single_question(raw: str) -> str:
    text = _strip_code_fence(raw)
    if not text:
        return ""

    parsed = _extract_qa_list(raw)
    if parsed:
        first_q = str(parsed[0].get("question") or "").strip()
        if first_q:
            return first_q

    # 兼容模型返回 JSON 数组/对象的情况
    if text.lstrip().startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, str):
                    return first.strip()
                if isinstance(first, dict):
                    q = first.get("question")
                    if q:
                        return str(q).strip()
        except json.JSONDecodeError:
            pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    first = re.sub(r"^(问题|Question)\s*[:：]\s*", "", first, flags=re.IGNORECASE)
    first = re.sub(r"^\s*\d+[\.\)、]\s*", "", first)
    return first.strip().strip('"').strip("'").strip()


def _extract_qa_list(raw: str) -> List[Dict[str, str]]:
    def _coerce_to_qa_list(obj: Any) -> List[Dict[str, str]]:
        items: List[Any] = []
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            items = [obj]
        else:
            return []

        out: List[Dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                q = str(item.get("question") or item.get("Question") or item.get("q") or "").strip()
                a = str(item.get("answer") or item.get("Answer") or item.get("a") or "").strip()
                if q or a:
                    out.append({"question": q, "answer": a})
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                q = str(item[0] or "").strip()
                a = str(item[1] or "").strip()
                if q or a:
                    out.append({"question": q, "answer": a})
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append({"question": s, "answer": ""})
        return out

    def _try_parse_structured(text: str) -> List[Dict[str, str]]:
        if not text:
            return []

        normalized = text.strip()
        if normalized.endswith(","):
            normalized = normalized[:-1].rstrip()

        # 1) 尝试严格 JSON
        try:
            obj = json.loads(normalized)
            if isinstance(obj, str):
                # 兼容：模型返回了“JSON 字符串”，再 parse 一次
                try:
                    obj2 = json.loads(obj)
                    parsed = _coerce_to_qa_list(obj2)
                    if parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
            parsed = _coerce_to_qa_list(obj)
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # 2) 兼容：输出中带有大量反斜杠转义（如 {\"question\": \"...\"}）
        if "\\\"" in normalized:
            fixed = re.sub(r"\\+\"", "\"", normalized)
            try:
                obj = json.loads(fixed)
                if isinstance(obj, str):
                    try:
                        obj2 = json.loads(obj)
                        parsed = _coerce_to_qa_list(obj2)
                        if parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass
                parsed = _coerce_to_qa_list(obj)
                if parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # 3) 兼容：Python 字面量（单引号 / 列表字典）
        try:
            obj = ast.literal_eval(text)
            parsed = _coerce_to_qa_list(obj)
            if parsed:
                return parsed
        except Exception:
            pass

        # 4) 容错：逐个扫描 JSON 对象（适合 JSON 数组被截断、缺少末尾 ] 的情况）
        decoder = json.JSONDecoder()
        idx = 0
        scanned: List[Dict[str, str]] = []
        while True:
            start = normalized.find("{", idx)
            if start == -1:
                break
            try:
                obj, end = decoder.raw_decode(normalized, start)
            except json.JSONDecodeError:
                idx = start + 1
                continue
            scanned.extend(_coerce_to_qa_list(obj))
            idx = end
        if scanned:
            return scanned

        return []

    text = _strip_code_fence(raw)
    if not text:
        return []

    # 尽量从输出中截取 JSON 主体（有些模型会在前后加说明文字）
    candidates = [text.strip()]
    first_bracket = text.find("[")
    first_brace = text.find("{")
    starts = [i for i in (first_bracket, first_brace) if i != -1]
    if starts:
        start = min(starts)
        end = max(text.rfind("]"), text.rfind("}"))
        if end != -1 and end > start:
            candidates.append(text[start : end + 1].strip())

    for cand in candidates:
        parsed = _try_parse_structured(cand)
        if parsed:
            return parsed

    pairs: List[Dict[str, str]] = []
    pattern = (
        r"(?:^|\n)\s*(?:\d+[\.\)、]?\s*)?(?:问题|Q)\s*[:：]\s*(.+?)\s*"
        r"(?:\n|$)\s*(?:答案|A)\s*[:：]\s*(.+?)(?=\n\s*(?:\d+[\.\)、]?\s*)?(?:问题|Q)\s*[:：]|\Z)"
    )
    for q, a in re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL):
        qq = " ".join((q or "").split()).strip()
        aa = " ".join((a or "").split()).strip()
        if qq or aa:
            pairs.append({"question": qq, "answer": aa})
    if pairs:
        return pairs

    # 最后回退：将每行当作“问题”，答案留空
    out: List[Dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\s*[-*]\s*", "", stripped)
        stripped = re.sub(r"^\s*\d+[\.\)、]\s*", "", stripped)
        stripped = stripped.strip().strip('"').strip("'").strip()
        if stripped:
            out.append({"question": stripped, "answer": ""})
    return out


def _looks_like_json_qa_object(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return bool(re.search(r'"question"\s*:\s*"', s) and re.search(r'"answer"\s*:\s*"', s))


def _pick_best_qa_by_alignment(
    aligner: QAEvaluator,
    ref_question: str,
    ref_answer: str,
    candidate_qas: List[Dict[str, str]],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    if not candidate_qas:
        return {}, {
            "question_similarity": 0.0,
            "answer_similarity": 0.0,
            "alignment_score": 0.0,
            "below_threshold": True,
        }

    questions = [ref_question] + [str(qa.get("question", "") or "") for qa in candidate_qas]
    q_embs = aligner.st_model.encode(questions, convert_to_tensor=False)
    q_embs = list(q_embs)
    ref_q_emb = q_embs[0]
    cand_q_embs = q_embs[1:]

    has_ref_answer = bool(ref_answer and str(ref_answer).strip())
    if has_ref_answer:
        answers = [ref_answer] + [str(qa.get("answer", "") or "") for qa in candidate_qas]
        a_embs = aligner.st_model.encode(answers, convert_to_tensor=False)
        a_embs = list(a_embs)
        ref_a_emb = a_embs[0]
        cand_a_embs = a_embs[1:]
    else:
        cand_a_embs = []

    best_idx = 0
    best_score = -1.0
    best_q_sim = 0.0
    best_a_sim = 0.0

    wq = float(ALIGN_WEIGHTS.get("question", 0.65))
    wa = float(ALIGN_WEIGHTS.get("answer", 0.35))

    for idx, _qa in enumerate(candidate_qas):
        q_sim = _cosine_similarity(ref_q_emb, cand_q_embs[idx])
        if has_ref_answer and cand_a_embs:
            a_sim = _cosine_similarity(ref_a_emb, cand_a_embs[idx])
            score = wq * q_sim + wa * a_sim
        else:
            a_sim = 0.0
            score = q_sim
        if score > best_score:
            best_score = float(score)
            best_idx = idx
            best_q_sim = float(q_sim)
            best_a_sim = float(a_sim)

    return candidate_qas[best_idx], {
        "question_similarity": float(best_q_sim),
        "answer_similarity": float(best_a_sim),
        "alignment_score": float(best_score),
        "below_threshold": bool(best_score < float(ALIGN_THRESHOLD)),
    }


def _cosine_similarity(vec1: Any, vec2: Any) -> float:
    v1 = np.asarray(vec1, dtype=float).reshape(-1)
    v2 = np.asarray(vec2, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _compute_alignment_info(
    aligner: QAEvaluator,
    ref_question: str,
    ref_answer: str,
    gen_question: str,
    gen_answer: str,
) -> Dict[str, Any]:
    if not gen_question:
        return {
            "question_similarity": 0.0,
            "answer_similarity": 0.0,
            "alignment_score": 0.0,
            "below_threshold": True,
        }

    q_embs = aligner.st_model.encode([ref_question, gen_question], convert_to_tensor=False)
    q_sim = _cosine_similarity(q_embs[0], q_embs[1])

    ref_a = str(ref_answer or "").strip()
    gen_a = str(gen_answer or "").strip()
    if ref_a and gen_a and ref_a == gen_a:
        a_sim = 1.0
    elif ref_a and gen_a:
        a_embs = aligner.st_model.encode([ref_a, gen_a], convert_to_tensor=False)
        a_sim = _cosine_similarity(a_embs[0], a_embs[1])
    else:
        a_sim = 0.0

    score = float(ALIGN_WEIGHTS.get("question", 0.65)) * float(q_sim) + float(
        ALIGN_WEIGHTS.get("answer", 0.35)
    ) * float(a_sim)
    return {
        "question_similarity": float(q_sim),
        "answer_similarity": float(a_sim),
        "alignment_score": float(score),
        "below_threshold": bool(score < float(ALIGN_THRESHOLD)),
    }


def _placeholder_scores(reason: str) -> Dict[str, Any]:
    return {
        "correctness": {"score": 0.0, "reasons": reason},
        "semantic_equivalence": {"score": 0.0, "reasons": reason},
        "style_similarity": {"score": 0.0, "reasons": reason},
    }


def _run_judge_with_timeout(
    timeout_seconds: int,
    judge_client: OpenAI,
    judge_cfg: Dict[str, Any],
    context: str,
    ref_question: str,
    ref_answer: str,
    gen_question: str,
    gen_answer: str,
) -> Dict[str, Any]:
    # 说明：
    # 这里不使用 ThreadPoolExecutor+future.result(timeout=...) 方案做“硬超时”，
    # 因为退出 ThreadPoolExecutor 的上下文管理器时会等待线程结束，反而可能“越等越久”。
    # 改为直接把 timeout 映射到 judge_pair 使用的 request_timeout，并关闭重试，
    # 以确保单条评审有明确的时间上限。
    cfg = judge_cfg
    if timeout_seconds and timeout_seconds > 0:
        cfg = dict(judge_cfg)
        cfg["request_timeout"] = int(timeout_seconds)
        cfg["max_retries"] = 0

    return judge_pair(
        judge_client,
        context=context,
        ref_question=ref_question,
        ref_answer=ref_answer,
        gen_question=gen_question,
        gen_answer=gen_answer,
        config=cfg,
    )


def _evaluate_context_group(
    mode: str,
    prompt_language: str,
    judge_client: OpenAI | None,
    judge_cfg: Dict[str, Any],
    gen_client: OpenAI,
    gen_cfg: Dict[str, Any],
    aligner: QAEvaluator,
    context: str,
    triples: List[Dict[str, Any]],
    qa_per_context: int,
    max_candidates: int,
    skip_judge: bool,
    keep_below_threshold: bool,
    judge_timeout: int,
    wall_start: float,
    save_generated_path: str | None,
    save_lock: threading.Lock | None,
    include_raw_generation: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    generation_log: Dict[str, Any] = {
        "context_id": triples[0].get("id") if triples else None,
        "dataset": triples[0].get("dataset") if triples else None,
        "mode": mode,
        "context": context,
        "generated": [],
    }

    candidate_qas: List[Dict[str, str]] = []
    raw_generation: str | None = None
    gen_duration_for_context: float | None = None

    if mode == "fig2_inago":
        gen_start = time.perf_counter()
        messages = _build_fig2_messages(context, qa_per_context, prompt_language)
        raw_generation = _call_llm(
            gen_client,
            model=gen_cfg["model"],
            messages=messages,
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
            timeout_seconds=int(gen_cfg["timeout"]),
            max_retries=int(gen_cfg["max_retries"]),
        )
        gen_duration_for_context = time.perf_counter() - gen_start
        candidate_qas = _extract_qa_list(raw_generation)
        if qa_per_context > 0:
            candidate_qas = candidate_qas[: max(1, qa_per_context)]
        if max_candidates > 0:
            candidate_qas = candidate_qas[: max_candidates]
        generation_log["candidate_qas"] = candidate_qas
        generation_log["raw_generation"] = raw_generation

    for triple in triples:
        ref_question = str(triple.get("question", "") or "")
        ref_answer = str(triple.get("answer", "") or "")

        gen_question = ""
        gen_answer = ""
        alignment: Dict[str, Any] = {}

        gen_duration = 0.0
        raw_for_record: str | None = None
        if mode == "fig2_inago":
            gen_duration = float(gen_duration_for_context or 0.0)
            best_qa, alignment = _pick_best_qa_by_alignment(
                aligner,
                ref_question=ref_question,
                ref_answer=ref_answer,
                candidate_qas=candidate_qas,
            )
            gen_question = str(best_qa.get("question", "") or "")
            gen_answer = str(best_qa.get("answer", "") or "")
            raw_for_record = raw_generation

            # 兼容：若解析失败导致把整段 JSON 行当作 question（answer 为空），尝试二次解包
            if (not gen_answer.strip()) and _looks_like_json_qa_object(gen_question):
                parsed_one = _extract_qa_list(gen_question)
                if parsed_one:
                    gen_question = str(parsed_one[0].get("question", "") or "").strip()
                    gen_answer = str(parsed_one[0].get("answer", "") or "").strip()
        else:
            gen_answer = ref_answer
            gen_start = time.perf_counter()
            include_context = mode == "fig1_squad"
            messages = _build_fig1_messages(
                context=context,
                answer=ref_answer,
                include_context=include_context,
                language=prompt_language,
            )
            raw = _call_llm(
                gen_client,
                model=gen_cfg["model"],
                messages=messages,
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                temperature=float(gen_cfg["temperature"]),
                timeout_seconds=int(gen_cfg["timeout"]),
                max_retries=int(gen_cfg["max_retries"]),
            )
            gen_duration = time.perf_counter() - gen_start
            gen_question = _extract_single_question(raw)
            alignment = _compute_alignment_info(
                aligner,
                ref_question=ref_question,
                ref_answer=ref_answer,
                gen_question=gen_question,
                gen_answer=gen_answer,
            )
            raw_for_record = raw
            generation_log["generated"].append(
                {
                    "id": triple.get("id"),
                    "ref_answer": ref_answer,
                    "raw_generation": raw,
                    "gen_question": gen_question,
                }
            )

        wall_seconds = time.perf_counter() - wall_start
        if not alignment:
            alignment = _compute_alignment_info(
                aligner,
                ref_question=ref_question,
                ref_answer=ref_answer,
                gen_question=gen_question,
                gen_answer=gen_answer,
            )

        # 若未能生成有效问题，则不写入输出（避免产生空记录）
        if not str(gen_question).strip():
            continue

        if alignment.get("below_threshold") and gen_question:
            if not keep_below_threshold:
                continue
            scores = _placeholder_scores("对齐未达阈值，已跳过评审")
        elif skip_judge or judge_client is None:
            scores = _placeholder_scores("LLM judge disabled (--skip-judge)")
        else:
            scores = _run_judge_with_timeout(
                timeout_seconds=judge_timeout,
                judge_client=judge_client,
                judge_cfg=judge_cfg,
                context=context,
                ref_question=ref_question,
                ref_answer=ref_answer,
                gen_question=gen_question,
                gen_answer=gen_answer,
            )

        result_entry: Dict[str, Any] = {
            "id": triple.get("id"),
            "dataset": triple.get("dataset"),
            "context": context,
            "ref_question": ref_question,
            "ref_answer": ref_answer,
            "gen_question": gen_question,
            "gen_answer": gen_answer,
            "timing": {
                "generation_seconds": float(gen_duration),
                "wall_seconds": float(wall_seconds),
            },
            "alignment": alignment,
            "scores": scores,
        }
        if include_raw_generation:
            result_entry["raw_generation"] = raw_for_record
        results.append(result_entry)

    if save_generated_path:
        os.makedirs(os.path.dirname(save_generated_path) or ".", exist_ok=True)
        if save_lock:
            with save_lock:
                with open(save_generated_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(generation_log, ensure_ascii=False) + "\n")
        else:
            with open(save_generated_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(generation_log, ensure_ascii=False) + "\n")

    return results


def main() -> None:
    args = _parse_args()

    judge_cfg: Dict[str, Any] = {}
    judge_client: OpenAI | None = None
    if not args.skip_judge:
        judge_cfg = _load_config_from_env()
        judge_client = build_judge_client(judge_cfg)

    gen_cfg = _build_generation_config(args, judge_cfg)
    gen_client = OpenAI(api_key=gen_cfg["api_key"], base_url=gen_cfg["base_url"])

    try:
        aligner = QAEvaluator(use_local_models=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"对齐所需本地模型未就绪，请先下载 qa_evaluation 所需模型: {exc}")

    if not os.path.exists(args.triples_file):
        raise SystemExit(f"未找到三元组文件: {args.triples_file}")
    triples_iter = list(_iter_triples(args.triples_file, args.max_samples))
    if not triples_iter:
        raise SystemExit(f"未在 {args.triples_file} 中读取到有效样本")

    context_groups: Dict[str, List[Dict[str, Any]]] = {}
    for triple in triples_iter:
        ctx = str(triple.get("context", "") or "")
        context_groups.setdefault(ctx, []).append(triple)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = len(context_groups)
    processed = 0
    wall_start = time.perf_counter()
    judge_max_concurrency = max(
        1, int(os.environ.get("BENCH_JUDGE_MAX_CONCURRENCY", str(args.max_workers)))
    )
    worker_count = max(1, min(args.max_workers, judge_max_concurrency))
    save_lock = threading.Lock() if args.save_generated else None

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_context_group,
                    args.mode,
                    args.prompt_language,
                    judge_client,
                    judge_cfg,
                    gen_client,
                    gen_cfg,
                    aligner,
                    context,
                    triples,
                    args.qa_per_context,
                    args.max_candidates_per_context,
                    args.skip_judge,
                    args.keep_below_threshold,
                    args.judge_timeout,
                    time.perf_counter(),
                    args.save_generated,
                    save_lock,
                    args.include_raw_generation,
                )
                for context, triples in context_groups.items()
            ]
            for fut in as_completed(futures):
                scored_list = fut.result()
                processed += 1
                print(f"[{processed}/{total}] context done, scored {len(scored_list)} samples")
                for scored in scored_list:
                    out_f.write(json.dumps(scored, ensure_ascii=False) + "\n")

    wall_elapsed = time.perf_counter() - wall_start
    print(
        f"Total wall-clock time: {wall_elapsed:.2f}s for {total} contexts "
        f"(max_workers={worker_count}, cap={judge_max_concurrency})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
