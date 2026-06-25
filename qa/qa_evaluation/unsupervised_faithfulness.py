# 文件作用：计算无监督忠实性指标。
# 关联说明：与其他 unsupervised_* 文件并列，提供单项忠实性指标。

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from app.core.runtime_paths import (
    DEFAULT_UNSUPERVISED_NLI_MODEL_NAME,
    ERLANGSHEN_NLI_MODEL_NAME,
    resolve_model_reference,
)
from qa.common import detect_language
from qa.qa_evaluation.unsupervised_runtime import (
    get_or_create_infer_lock,
    release_cached_models_for_device,
    resolve_first_existing_model_path,
    select_torch_device,
)

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

try:
    from app.services.llm import VLMClientConfig, create_vlm_client as _create_vlm_client

    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _create_vlm_client = None  # type: ignore
    VLMClientConfig = None  # type: ignore
    OPENAI_AVAILABLE = False


NliLabel = Literal["entailment", "contradiction", "neutral"]
HypothesisMode = Literal["llm"]


DEFAULT_NLI_MODEL_PATHS = (
    resolve_model_reference(
        os.environ.get("UNSUPERVISED_NLI_MODEL_PATH"),
        default_name=DEFAULT_UNSUPERVISED_NLI_MODEL_NAME,
    ),
    resolve_model_reference(ERLANGSHEN_NLI_MODEL_NAME),
)
DEFAULT_NLI_MAX_LENGTH = int(os.environ.get("UNSUPERVISED_NLI_MAX_LENGTH", "512"))
DEFAULT_NLI_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_NLI_BATCH_SIZE", "16"))
DEFAULT_NLI_DEVICE = os.environ.get("UNSUPERVISED_NLI_DEVICE", "auto").strip().lower()

DEFAULT_NLI_PREMISE_MODE = str(
    os.environ.get("UNSUPERVISED_NLI_PREMISE_MODE", "select_units") or "select_units"
).strip().lower()
if DEFAULT_NLI_PREMISE_MODE not in {"full", "select_units"}:
    DEFAULT_NLI_PREMISE_MODE = "select_units"
DEFAULT_NLI_PREMISE_MAX_UNITS = int(os.environ.get("UNSUPERVISED_NLI_PREMISE_MAX_UNITS", "2") or 2)
DEFAULT_NLI_PREMISE_NEIGHBOR = int(os.environ.get("UNSUPERVISED_NLI_PREMISE_NEIGHBOR", "1") or 1)
DEFAULT_NLI_PREMISE_MAX_CHARS = int(os.environ.get("UNSUPERVISED_NLI_PREMISE_MAX_CHARS", "220") or 220)
DEFAULT_NLI_PREMISE_MIN_UNIT_CHARS = int(os.environ.get("UNSUPERVISED_NLI_PREMISE_MIN_UNIT_CHARS", "8") or 8)

DEFAULT_HYPOTHESIS_MODE = str(os.environ.get("UNSUPERVISED_HYPOTHESIS_MODE", "llm") or "llm").strip().lower()
if DEFAULT_HYPOTHESIS_MODE != "llm":
    DEFAULT_HYPOTHESIS_MODE = "llm"
DEFAULT_HYPOTHESIS_TIMEOUT = int(os.environ.get("UNSUPERVISED_HYPOTHESIS_TIMEOUT", "60") or 60)
DEFAULT_HYPOTHESIS_MAX_RETRIES = int(os.environ.get("UNSUPERVISED_HYPOTHESIS_MAX_RETRIES", "2") or 2)
DEFAULT_HYPOTHESIS_MAX_CONCURRENCY = int(
    os.environ.get("UNSUPERVISED_HYPOTHESIS_MAX_CONCURRENCY", "8") or 8
)
DEFAULT_HYPOTHESIS_API_KEY = str(
    os.environ.get("UNSUPERVISED_HYPOTHESIS_API_KEY")
    or os.environ.get("LLM_API_KEY")
    or ""
).strip()
DEFAULT_HYPOTHESIS_BASE_URL = str(
    os.environ.get("UNSUPERVISED_HYPOTHESIS_BASE_URL")
    or os.environ.get("LLM_BASE_URL")
    or ""
).strip()
DEFAULT_HYPOTHESIS_MODEL = str(
    os.environ.get("UNSUPERVISED_HYPOTHESIS_MODEL")
    or os.environ.get("LLM_MODEL")
    or ""
).strip()

_RE_AMBIGUOUS_ZH = re.compile(
    r"(这份|上述|其中|它们|他们|她们|该表|本表|此表|该通知|本通知|此通知|该文件|本文件|此文件|该办法|本办法|此办法|该制度|本制度|此制度|该规定|本规定|此规定|该附件|本附件|此附件)"
)
_RE_AMBIGUOUS_ZH_QI = re.compile(r"其(?!他|它|余)")
_RE_AMBIGUOUS_EN = re.compile(
    r"\b(this|that|the above|above|aforementioned|herein|thereof)\b",
    flags=re.IGNORECASE,
)
_RE_WS = re.compile(r"\s+")
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
_RE_FALLBACK_SPLIT = re.compile(r"(?<=[，,；;])")
_RE_SENT_SPLIT_EN = re.compile(r"(?<=[.!?;])")

_LLM_HYPOTHESIS_SYSTEM_ZH = """你是一个“QA→可检验陈述句(Hypothesis)”改写器，用于做 NLI 忠实度评估。

你会收到：premise（证据/来源事实）、question、answer、question_type。

你的任务：把 question+answer 改写成 1 条或多条“可被 premise 蕴含/反驳”的陈述句 hypotheses。

硬性要求：
1) 只输出 JSON，禁止输出 markdown/代码块。
2) hypotheses 必须是字符串数组，每条是完整陈述句，尽量 1 句表达清楚。
3) strategy 只能是 "single" 或 "list_avg"。
   - single：hypotheses 长度必须为 1
   - list_avg：hypotheses 长度必须 >= 2（用于答案是多条并列项/清单的情况）
4) 严禁“指代不明”：不得出现“这份/该表/本通知/上述/其中/其…/this/that/above”等；必须写明具体对象（可从 premise 中抽取更具体的名称/标题/附件名来消解指代）。
5) 禁止引入新事实：只能重述 question/answer 的含义；premise 仅用于消解指代与补全指称（不允许借 premise 修正/改写 answer 的事实内容）。
6) 对判断题/是非题：
   - hypotheses 应只表达“题干要判断的命题”（不要把答案 True/False 写进句子里）。
   - 后续系统会根据 answer 判定应该走 entailment 或 contradiction。
7) 保持“断言强度”一致：优先复用 answer 的原词/原句式，避免把较弱表述改写成更强断言。
   - 例如：不要把“能写一手好字/写得不错/比较擅长/可能/大概/有一定”改成“书法技艺高超/精通/必然/顶尖”。
   - 如需用 premise 消解指代，也尽量直接引用 premise 中的原词组，不要“美化/拔高”描述。

输出 JSON 结构：
{"strategy":"single","hypotheses":["..."]}"""

_LLM_HYPOTHESIS_SYSTEM_EN = """You rewrite QA into verifiable declarative hypothesis statements for NLI faithfulness.

Input: premise (evidence), question, answer, question_type.

Rules:
1) Output JSON only (no markdown/code fences).
2) hypotheses must be an array of complete declarative sentences; keep each concise (prefer 1 sentence).
3) strategy must be "single" or "list_avg".
   - single: hypotheses length must be 1
   - list_avg: hypotheses length must be >= 2 (for list-style answers)
4) No ambiguous references: avoid "this/that/the above/above" etc; name the specific document/section explicitly (you may use premise to resolve references).
5) Do not add new facts: preserve the meaning of question+answer; premise is only for reference resolution, not for correcting the answer.
6) For True/False questions: hypotheses should only state the proposition being judged; do NOT include the answer True/False in the sentence.
7) Keep claim strength consistent: prefer reusing the answer wording; don't strengthen hedged/weaker statements into stronger claims.

Output JSON:
{"strategy":"single","hypotheses":["..."]}"""

_LLM_CLIENT_LOCAL = threading.local()


def _get_llm_client(*, api_key: str, base_url: str, model_name: str = "", timeout_seconds: float = 120.0) -> Any:
    if not OPENAI_AVAILABLE or _create_vlm_client is None:
        raise RuntimeError("LLM client is unavailable for hypothesis generation")
    key = str(api_key or "").strip()
    url = str(base_url or "").strip()
    if not key:
        raise RuntimeError("Missing llm_api_key for hypothesis generation")

    cached = getattr(_LLM_CLIENT_LOCAL, "client", None)
    cached_key = getattr(_LLM_CLIENT_LOCAL, "api_key", None)
    cached_url = getattr(_LLM_CLIENT_LOCAL, "base_url", None)
    if cached is None or cached_key != key or cached_url != url:
        client = _create_vlm_client(
            VLMClientConfig.from_values(
                api_base=url,
                model_name=model_name or "default",
                api_key=key,
                timeout_seconds=timeout_seconds,
            )
        )
        _LLM_CLIENT_LOCAL.client = client
        _LLM_CLIENT_LOCAL.api_key = key
        _LLM_CLIENT_LOCAL.base_url = url
        return client
    return cached


def _contains_ambiguous_reference(text: str, *, language_code: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if language_code == "zh":
        compact = re.sub(r"\s+", "", raw)
        if _RE_AMBIGUOUS_ZH.search(compact):
            return True
        if "其" in compact and _RE_AMBIGUOUS_ZH_QI.search(compact):
            return True
        return False
    return bool(_RE_AMBIGUOUS_EN.search(raw))


def _normalize_premise(text: str) -> str:
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _RE_WS.sub(" ", s)
    return s.strip()


def _split_premise_units(text: str, *, min_chars: int, max_units: int) -> List[str]:
    raw = _normalize_premise(text)
    if not raw:
        return []

    # Avoid extremely long single blocks by chunking consecutive lines.
    blocks: List[str] = []
    cur: List[str] = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        cur.append(s)
        if len(cur) >= 4:
            blocks.append(" ".join(cur).strip())
            cur = []
    if cur:
        blocks.append(" ".join(cur).strip())

    min_chars = max(1, min(200, int(min_chars or 1)))
    max_units = max(1, min(1024, int(max_units or 1)))

    units: List[str] = []
    for blk in blocks:
        if not blk:
            continue
        parts = [p.strip() for p in _RE_SENT_SPLIT.split(blk) if str(p or "").strip()]
        if len(parts) <= 1 and len(blk) >= 160:
            parts = [p.strip() for p in _RE_FALLBACK_SPLIT.split(blk) if str(p or "").strip()] or parts
        if not parts:
            parts = [blk]
        for p in parts:
            u = str(p or "").strip()
            if not u or len(u) < min_chars:
                continue
            units.append(u)
            if len(units) >= max_units:
                break
        if len(units) >= max_units:
            break

    dedup: List[str] = []
    seen: set = set()
    for u in units:
        key = _RE_WS.sub(" ", u).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(u)
    return dedup


def _sig_tokens(text: str, *, language_code: str) -> List[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []

    if language_code == "zh":
        compact = re.sub(r"\s+", "", raw)
        compact = re.sub(r"[，,。\.！？!?；;：:\u3001“”\"'‘’（）()《》【】\[\]…\-—_]+", "", compact)
        if not compact:
            return []
        if len(compact) <= 2:
            return [compact]
        return [compact[i : i + 2] for i in range(0, len(compact) - 1)]

    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    parts = [p for p in raw.split(" ") if p]
    return parts[:256]


def _lex_overlap_score(unit: str, hyp_sig: set, *, language_code: str) -> float:
    if not hyp_sig:
        return 0.0
    u_tokens = set(_sig_tokens(unit, language_code=language_code))
    if not u_tokens:
        return 0.0
    inter = hyp_sig.intersection(u_tokens)
    return float(len(inter) / max(1.0, float(len(hyp_sig))))


def _select_premise_excerpt(
    premise: str,
    hypotheses: List[str],
    *,
    language_code: str,
    max_rank_units: int,
    neighbor: int,
    max_chars: int,
    min_unit_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    raw = _normalize_premise(premise)
    if not raw:
        return "", {"mode": "empty", "units_total": 0, "units_selected": 0, "evidence_units": []}

    max_rank_units = max(1, min(12, int(max_rank_units or 1)))
    neighbor = max(0, min(3, int(neighbor or 0)))
    max_chars = max(120, min(1600, int(max_chars or 360)))
    min_unit_chars = max(1, min(200, int(min_unit_chars or 8)))

    hyp_joined = " ".join([str(h or "").strip() for h in hypotheses if str(h or "").strip()])
    hyp_sig = set(_sig_tokens(hyp_joined, language_code=language_code))

    units = _split_premise_units(raw, min_chars=min_unit_chars, max_units=256)
    if not units:
        excerpt = raw[:max_chars]
        return excerpt, {
            "mode": "head_clip",
            "units_total": 0,
            "units_selected": 0,
            "max_chars": max_chars,
            "evidence_units": [],
        }

    scored: List[Tuple[float, int]] = []
    for idx, u in enumerate(units):
        scored.append((_lex_overlap_score(u, hyp_sig, language_code=language_code), idx))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    top = [idx for _s, idx in scored[:max_rank_units] if idx >= 0]
    picked: set = set()
    for idx in top:
        for j in range(max(0, idx - neighbor), min(len(units), idx + neighbor + 1)):
            picked.add(j)

    selected_idxs = sorted(picked)
    excerpt_parts: List[str] = []
    evidence_units: List[Dict[str, Any]] = []
    char_count = 0
    score_map = {idx: s for s, idx in scored}
    for idx in selected_idxs:
        u = str(units[idx] or "").strip()
        if not u:
            continue
        add_len = len(u) + (1 if excerpt_parts else 0)
        if excerpt_parts and (char_count + add_len) > max_chars:
            break
        excerpt_parts.append(u)
        char_count += add_len
        if len(evidence_units) < 8:
            evidence_units.append(
                {
                    "text": u[:220],
                    "lex_score": float(score_map.get(idx, 0.0)),
                    "idx": int(idx),
                }
            )

    excerpt = "\n".join(excerpt_parts).strip()
    if not excerpt:
        excerpt = raw[:max_chars]

    evidence_units.sort(
        key=lambda x: (float(x.get("lex_score") or 0.0), -int(x.get("idx") or 0)),
        reverse=True,
    )

    return excerpt, {
        "mode": "select_units_v1",
        "units_total": len(units),
        "units_selected": len(excerpt_parts),
        "max_chars": max_chars,
        "rank_units": max_rank_units,
        "neighbor": neighbor,
        "min_unit_chars": min_unit_chars,
        "evidence_units": evidence_units,
    }


def _llm_build_hypotheses(
    *,
    premise: str,
    question: str,
    answer: str,
    question_type: Any,
    llm_client: Any,
    llm_model: str,
    request_timeout: int,
    max_retries: int,
) -> Tuple[List[str], str]:
    lang = detect_language(f"{question}\n{answer}\n{premise}")
    language_code = "en" if lang == "en" else "zh"
    system_prompt = _LLM_HYPOTHESIS_SYSTEM_EN if language_code == "en" else _LLM_HYPOTHESIS_SYSTEM_ZH
    user_payload = {
        "premise": str(premise or ""),
        "question": str(question or ""),
        "answer": str(answer or ""),
        "question_type": str(question_type or ""),
    }

    last_error: Optional[str] = None
    for _ in range(max(1, int(max_retries or 1))):
        try:
            raw = llm_client.create_chat_completion_text(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=float(request_timeout or 60),
            ).strip()
            parsed = None
            try:
                parsed = json.loads(raw) if raw else None
            except Exception as exc:
                last_error = f"hypothesis_json_parse_error: {exc}"
                continue

            if not isinstance(parsed, dict):
                last_error = "hypothesis_not_object"
                continue

            strategy = str(parsed.get("strategy") or "single").strip().lower()
            if strategy not in {"single", "list_avg"}:
                last_error = "hypothesis_invalid_strategy"
                continue

            hypotheses_raw = parsed.get("hypotheses")
            if not isinstance(hypotheses_raw, list):
                last_error = "hypothesis_missing_hypotheses"
                continue

            hypotheses: List[str] = []
            for h in hypotheses_raw:
                s = str(h or "").strip()
                if not s:
                    continue
                if _contains_ambiguous_reference(s, language_code=language_code):
                    last_error = "hypothesis_ambiguous_reference"
                    hypotheses = []
                    break
                hypotheses.append(s)
                if len(hypotheses) >= 12:
                    break

            if not hypotheses:
                if not last_error:
                    last_error = "hypothesis_empty"
                continue

            if strategy == "single" and len(hypotheses) != 1:
                last_error = "hypothesis_single_len_not_1"
                continue
            if strategy == "list_avg" and len(hypotheses) < 2:
                last_error = "hypothesis_list_len_lt_2"
                continue

            return hypotheses, strategy
        except Exception as exc:  # pragma: no cover - network/provider
            last_error = f"hypothesis_llm_error: {exc}"
            continue

    raise RuntimeError(last_error or "hypothesis_llm_failed")

@dataclass(frozen=True)
class NliPrediction:
    pred_label: NliLabel
    probs: Dict[NliLabel, float]


@dataclass(frozen=True)
class FaithfulnessResult:
    faithfulness: float
    expected_label: NliLabel
    pred_label: NliLabel
    probs: Dict[NliLabel, float]


@dataclass(frozen=True)
class _FaithfulnessItemMeta:
    item: Dict[str, Any]
    expected: NliLabel
    start: int
    end: int
    strategy: str
    hypothesis_mode: str
    hypothesis_error: str
    hypotheses: List[str]
    clause_premises: List[str]
    clause_premise_select: List[Dict[str, Any]]


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any, Dict[int, NliLabel]]] = {}
_INFER_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}
_INFER_LOCKS_GUARD = threading.Lock()


def _get_infer_lock(cache_key: Tuple[str, str]) -> threading.Lock:
    return get_or_create_infer_lock(_INFER_LOCKS, _INFER_LOCKS_GUARD, cache_key)


def _resolve_default_model_path() -> str:
    return resolve_first_existing_model_path(DEFAULT_NLI_MODEL_PATHS)


def _select_device(device: Optional[str]) -> str:
    return select_torch_device(device, default_device=DEFAULT_NLI_DEVICE, torch_module=torch)


def _normalize_id2label(id2label: Any) -> Dict[int, NliLabel]:
    mapping: Dict[int, NliLabel] = {}
    if not isinstance(id2label, dict):
        return mapping
    for k, v in id2label.items():
        try:
            idx = int(k)
        except Exception:
            continue
        label = str(v or "").strip().lower()
        if "entail" in label:
            mapping[idx] = "entailment"
        elif "contrad" in label:
            mapping[idx] = "contradiction"
        elif "neutral" in label:
            mapping[idx] = "neutral"
    return mapping


def _get_nli_bundle(
    model_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
) -> Tuple[Any, Any, Dict[int, NliLabel]]:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers/torch 未安装，无法运行 NLI 忠实度评估")

    resolved_path = (model_path or "").strip() or _resolve_default_model_path()
    resolved_device = _select_device(device)
    cache_key = (resolved_path, resolved_device)

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path, local_files_only=True, fix_mistral_regex=True
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=True)
        torch_dtype = None
        if resolved_device == "cpu" and torch is not None:
            torch_dtype = torch.float32
        model_kwargs = {"local_files_only": True}
        if torch_dtype is None:
            model = AutoModelForSequenceClassification.from_pretrained(resolved_path, **model_kwargs)
        else:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    resolved_path, dtype=torch_dtype, **model_kwargs
                )
            except TypeError:
                model = AutoModelForSequenceClassification.from_pretrained(
                    resolved_path, torch_dtype=torch_dtype, **model_kwargs
                )
        model.eval()
        model.to(resolved_device)

        id2label = _normalize_id2label(getattr(model.config, "id2label", None))
        if len(id2label) != 3:
            raise RuntimeError(
                f"NLI 模型 label 映射异常，无法识别 entailment/neutral/contradiction: {resolved_path}"
            )

        bundle = (tokenizer, model, id2label)
        _MODEL_CACHE[cache_key] = bundle
        return bundle


def release_nli_device_cache(device: Optional[str]) -> None:
    resolved_device = _select_device(device)
    release_cached_models_for_device(
        _MODEL_CACHE,
        _INFER_LOCKS,
        _MODEL_LOCK,
        resolved_device,
        torch_module=torch,
    )


def _softmax_probs(logits: Any) -> List[List[float]]:
    if torch is None:
        raise RuntimeError("torch 未安装")
    probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().tolist()


def predict_nli(
    pairs: List[Tuple[str, str]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    batch_size: int = DEFAULT_NLI_BATCH_SIZE,
) -> List[NliPrediction]:
    """
    Run NLI 3-way classification for (premise, hypothesis) pairs.

    Returns:
    - pred_label: argmax label
    - probs: probability per label
    """
    if not pairs:
        return []

    resolved_path = (model_path or "").strip() or _resolve_default_model_path()
    resolved_device = _select_device(device)
    cache_key = (resolved_path, resolved_device)
    tokenizer, model, id2label = _get_nli_bundle(resolved_path, device=resolved_device)

    out: List[NliPrediction] = []
    infer_lock = _get_infer_lock(cache_key)
    with infer_lock:
        for i in range(0, len(pairs), max(1, int(batch_size or 1))):
            batch = pairs[i : i + max(1, int(batch_size or 1))]
            premises = [p[0] or "" for p in batch]
            hypotheses = [p[1] or "" for p in batch]

            inputs = tokenizer(
                premises,
                hypotheses,
                truncation=True,
                max_length=max(8, int(max_length or 512)),
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs_rows = _softmax_probs(logits)

            for row in probs_rows:
                label_probs: Dict[NliLabel, float] = {
                    id2label[idx]: float(row[idx]) for idx in id2label.keys()
                }
                pred_idx = max(id2label.keys(), key=lambda idx: row[idx])
                out.append(
                    NliPrediction(
                        pred_label=id2label[pred_idx],
                        probs=label_probs,
                    )
                )
    return out

def parse_judge_answer(answer: str) -> Optional[bool]:
    """
    Parse boolean-like answers for 判断题.

    Returns:
    - True  -> "对/是/正确/YES/True/√"
    - False -> "错/否/错误/NO/False/×"
    - None  -> cannot parse
    """
    raw = (answer or "").strip()
    if not raw:
        return None

    head = raw.splitlines()[0].strip()
    head = re.split(r"[。；;，,\s]+", head, maxsplit=1)[0].strip()
    lowered = head.lower()

    negative = ("不是", "不对", "不正确", "否", "错", "错误", "no", "false", "×", "0")
    positive = ("是", "对", "正确", "yes", "true", "√", "1")

    for tok in negative:
        if lowered == tok or lowered.startswith(tok):
            return False
    for tok in positive:
        if lowered == tok or lowered.startswith(tok):
            return True
    return None


def is_judge_question(question_type: Any, question: str) -> bool:
    qt = str(question_type or "").strip()
    if qt:
        return ("判断" in qt) or (qt in {"是非题"})
    q = str(question or "").strip()
    return q.startswith("是否") or q.endswith("吗") or q.endswith("么")


def _fallback_build_hypotheses_from_answer(
    *,
    question: str,
    answer: str,
    question_type: Any,
    language_code: str,
    max_hypotheses: int = 12,
) -> Tuple[List[str], str]:
    """
    Fallback hypothesis builder when LLM rewrite fails.

    Default behavior:
    - Non-judge: split answer into sentences (or coarse clauses for long lines).
    - Judge questions: prefer using the question proposition (avoid "是/否" as hypothesis).
    """
    max_hypotheses = max(1, min(64, int(max_hypotheses or 12)))
    q = str(question or "").strip()
    a = str(answer or "").strip()

    if is_judge_question(question_type, q):
        q2 = q.replace("\r\n", "\n").replace("\r", "\n").strip()
        q2 = q2.rstrip("？?").strip()
        if q2.startswith("是否"):
            q2 = q2[2:].strip()
        if q2.endswith("吗") or q2.endswith("么"):
            q2 = q2[:-1].strip()
        if q2:
            return [q2], "single"

    raw = a.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw and q:
        q2 = q.rstrip("？?").strip()
        if q2:
            return [q2], "single"
        return [], "single"
    if not raw:
        return [], "single"

    splitter = _RE_SENT_SPLIT_EN if language_code == "en" else _RE_SENT_SPLIT
    blocks = [s.strip() for s in raw.split("\n") if s.strip()] or [raw]

    out: List[str] = []
    for blk in blocks:
        parts = [p.strip() for p in splitter.split(blk) if str(p or "").strip()]
        if len(parts) <= 1 and len(blk) >= 160:
            parts = [p.strip() for p in _RE_FALLBACK_SPLIT.split(blk) if str(p or "").strip()] or parts
        if not parts:
            parts = [blk]
        for p in parts:
            s = str(p or "").strip()
            if not s:
                continue
            out.append(s)
            if len(out) >= max_hypotheses:
                break
        if len(out) >= max_hypotheses:
            break

    dedup: List[str] = []
    seen: set = set()
    for h in out:
        key = _RE_WS.sub(" ", str(h or "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(str(h))

    if not dedup:
        return [], "single"
    if len(dedup) == 1:
        return dedup[:1], "single"
    return dedup, "list_avg"


def compute_faithfulness_for_item(
    *,
    source_fact_text: str,
    question: str,
    answer: str,
    question_type: Any = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
) -> Optional[FaithfulnessResult]:
    premise = str(source_fact_text or "").strip()
    if not premise:
        return None

    q = str(question or "").strip()
    a = str(answer or "").strip()
    if not q and not a:
        return None

    item: Dict[str, Any] = {
        "source_fact_text": premise,
        "question": q,
        "answer": a,
        "question_type": question_type,
    }
    attach_faithfulness(
        [item],
        model_path=model_path,
        device=device,
        max_length=max_length,
        batch_size=1,
        only_primary=False,
    )
    ue = item.get("unsupervised_evaluation")
    if not isinstance(ue, dict):
        return None
    scores = ue.get("scores") or {}
    meta = ue.get("meta") or {}
    if not isinstance(scores, dict) or not isinstance(meta, dict):
        return None

    faithfulness = float(scores.get("faithfulness") or 0.0)
    expected = str(meta.get("expected_label") or "entailment").strip().lower()
    pred_label = str(meta.get("pred_label") or "neutral").strip().lower()
    probs = meta.get("probs") or {}

    if expected not in {"entailment", "contradiction", "neutral"}:
        expected = "entailment"
    if pred_label not in {"entailment", "contradiction", "neutral"}:
        pred_label = "neutral"
    if not isinstance(probs, dict):
        probs = {}
    probs_out: Dict[NliLabel, float] = {
        "entailment": float(probs.get("entailment") or 0.0),
        "contradiction": float(probs.get("contradiction") or 0.0),
        "neutral": float(probs.get("neutral") or 0.0),
    }
    return FaithfulnessResult(
        faithfulness=faithfulness,
        expected_label=expected,  # type: ignore[arg-type]
        pred_label=pred_label,  # type: ignore[arg-type]
        probs=probs_out,
    )


def attach_faithfulness(
    qa_items: List[Dict[str, Any]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = DEFAULT_NLI_MAX_LENGTH,
    batch_size: int = DEFAULT_NLI_BATCH_SIZE,
    only_primary: bool = True,
    hypothesis_mode: str = DEFAULT_HYPOTHESIS_MODE,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_request_timeout: int = DEFAULT_HYPOTHESIS_TIMEOUT,
    llm_max_retries: int = DEFAULT_HYPOTHESIS_MAX_RETRIES,
    llm_max_concurrency: int = DEFAULT_HYPOTHESIS_MAX_CONCURRENCY,
) -> Dict[str, Any]:
    """
    Mutate qa_items in-place and attach:
      item["unsupervised_evaluation"] = {method, scores, meta}

    Returns a small summary dict for logging.
    """
    if not qa_items:
        return {"computed": 0, "skipped": 0, "method": "nli_faithfulness_v1"}

    mode = str(hypothesis_mode or DEFAULT_HYPOTHESIS_MODE).strip().lower()
    if mode != "llm":
        mode = "llm"

    if not OPENAI_AVAILABLE or _create_vlm_client is None:
        raise RuntimeError("LLM client is unavailable for hypothesis generation")

    api_key = str(llm_api_key or DEFAULT_HYPOTHESIS_API_KEY or "").strip()
    base_url = str(llm_base_url or DEFAULT_HYPOTHESIS_BASE_URL or "").strip()
    llm_model_resolved = str(llm_model or DEFAULT_HYPOTHESIS_MODEL or "").strip()
    if not api_key or not llm_model_resolved:
        raise RuntimeError("Missing llm_api_key/llm_model for hypothesis generation")

    def _extract_premise(item: Dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        for key in ("qa_generation_unit_text", "source_fact_text", "context", "source"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        meta = item.get("meta")
        if isinstance(meta, dict):
            ctx = meta.get("context")
            if isinstance(ctx, str) and ctx.strip():
                return ctx.strip()
        return ""

    targets: List[Tuple[Dict[str, Any], str, str, str, Any]] = []
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        premise = _extract_premise(item)
        if not premise:
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question and not answer:
            continue
        targets.append((item, premise, question, answer, item.get("question_type")))

    if not targets:
        return {"computed": 0, "skipped": len(qa_items), "method": "nli_faithfulness_v1"}

    max_workers = max(1, min(int(llm_max_concurrency or 1), len(targets)))

    def _run_one(
        premise: str, question: str, answer: str, question_type: Any
    ) -> Tuple[List[str], str]:
        client = _get_llm_client(api_key=api_key, base_url=base_url)
        return _llm_build_hypotheses(
            premise=premise,
            question=question,
            answer=answer,
            question_type=question_type,
            llm_client=client,
            llm_model=llm_model_resolved,
            request_timeout=int(llm_request_timeout or DEFAULT_HYPOTHESIS_TIMEOUT),
            max_retries=int(llm_max_retries or DEFAULT_HYPOTHESIS_MAX_RETRIES),
        )

    results_by_idx: Dict[int, Tuple[List[str], str]] = {}
    errors_by_idx: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_one, premise, question, answer, question_type): idx
            for idx, (_item, premise, question, answer, question_type) in enumerate(targets)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results_by_idx[idx] = future.result()
            except Exception as exc:  # pragma: no cover - network/provider
                errors_by_idx[idx] = str(exc)

    pairs: List[Tuple[str, str]] = []
    metas: List[_FaithfulnessItemMeta] = []
    eligible_items = 0
    fallback_used = 0
    for idx, (item, premise, question, answer, question_type) in enumerate(targets):
        eligible_items += 1
        lang = detect_language(f"{question}\n{answer}\n{premise}")
        language_code = "en" if lang == "en" else "zh"
        got = results_by_idx.get(idx)
        hypothesis_mode_used = "llm"
        hypothesis_error = ""
        if got and got[0]:
            hypotheses, strategy = got
        else:
            hypotheses, strategy = _fallback_build_hypotheses_from_answer(
                question=question,
                answer=answer,
                question_type=question_type,
                language_code=language_code,
            )
            hypothesis_mode_used = "fallback_answer_sentences"
            hypothesis_error = str(errors_by_idx.get(idx) or "").strip()
            if hypotheses:
                fallback_used += 1
        if not hypotheses:
            continue

        # Reduce long-context noise for NLI by selecting relevant evidence units.
        # NOTE: we select evidence per hypothesis/clause to better cover long,
        # summarized answers that span multiple scattered evidence sentences.
        full_premise = str(premise or "").strip()
        if not full_premise:
            continue

        expected: NliLabel = "entailment"
        if is_judge_question(question_type, question):
            bool_ans = parse_judge_answer(answer)
            if bool_ans is False:
                expected = "contradiction"
        start = len(pairs)
        clause_premises: List[str] = []
        clause_selects: List[Dict[str, Any]] = []
        for h in hypotheses:
            premise_used = full_premise
            premise_select_meta: Dict[str, Any] = {"mode": "full"}
            if DEFAULT_NLI_PREMISE_MODE == "select_units":
                premise_used, premise_select_meta = _select_premise_excerpt(
                    full_premise,
                    [h],
                    language_code=language_code,
                    max_rank_units=DEFAULT_NLI_PREMISE_MAX_UNITS,
                    neighbor=DEFAULT_NLI_PREMISE_NEIGHBOR,
                    max_chars=DEFAULT_NLI_PREMISE_MAX_CHARS,
                    min_unit_chars=DEFAULT_NLI_PREMISE_MIN_UNIT_CHARS,
                )
                premise_used = str(premise_used or "").strip() or full_premise
            if not premise_used:
                premise_used = full_premise
            pairs.append((premise_used, h))
            clause_premises.append(premise_used)
            clause_selects.append(premise_select_meta)
        end = len(pairs)
        metas.append(
            _FaithfulnessItemMeta(
                item=item,
                expected=expected,
                start=start,
                end=end,
                strategy=strategy,
                hypothesis_mode=hypothesis_mode_used,
                hypothesis_error=hypothesis_error[:300],
                hypotheses=hypotheses,
                clause_premises=clause_premises,
                clause_premise_select=clause_selects,
            )
        )

    if not pairs:
        return {
            "computed": 0,
            "skipped": max(0, eligible_items),
            "eligible": eligible_items,
            "hypothesis_failed": len(errors_by_idx),
            "method": "nli_faithfulness_v1",
        }

    preds = predict_nli(
        pairs,
        model_path=model_path,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    computed = 0
    for m in metas:
        item_preds = preds[m.start : m.end]
        if not item_preds:
            continue

        if m.strategy == "single" or len(item_preds) == 1:
            pred = item_preds[0]
            faithfulness = float(pred.probs.get(m.expected, 0.0))
            premise_used = m.clause_premises[0] if m.clause_premises else ""
            premise_select_meta = m.clause_premise_select[0] if m.clause_premise_select else {"mode": "full"}
            m.item["unsupervised_evaluation"] = {
                "method": "nli_faithfulness_v1",
                "scores": {"faithfulness": faithfulness},
                "meta": {
                    "expected_label": m.expected,
                    "pred_label": pred.pred_label,
                    "probs": pred.probs,
                    "strategy": "single",
                    "hypothesis_mode": m.hypothesis_mode,
                    "hypothesis_error": m.hypothesis_error,
                    "hypothesis": (m.hypotheses[0] if m.hypotheses else "")[:300],
                    "premise_mode": str(premise_select_meta.get("mode") or "full"),
                    "premise_excerpt": str(premise_used or "")[:600],
                    "premise_select": premise_select_meta,
                },
            }
            computed += 1
            continue

        sums: Dict[NliLabel, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        expected_vals: List[Tuple[float, int]] = []
        clauses: List[Dict[str, Any]] = []
        for idx, pred in enumerate(item_preds):
            for k in sums.keys():
                sums[k] += float(pred.probs.get(k, 0.0))
            p_expected = float(pred.probs.get(m.expected, 0.0))
            expected_vals.append((p_expected, idx))
            clause_text = m.hypotheses[idx] if idx < len(m.hypotheses) else ""
            premise_used = m.clause_premises[idx] if idx < len(m.clause_premises) else ""
            premise_select_meta = (
                m.clause_premise_select[idx] if idx < len(m.clause_premise_select) else {"mode": "full"}
            )
            clauses.append(
                {
                    "text": clause_text[:300],
                    "p_expected": float(p_expected),
                    "pred_label": pred.pred_label,
                    "probs": pred.probs,
                    "premise_mode": str(premise_select_meta.get("mode") or "full"),
                    "premise_excerpt": str(premise_used or "")[:600],
                    "premise_select": premise_select_meta,
                }
            )

        denom = float(len(item_preds))
        avg_probs: Dict[NliLabel, float] = {k: (v / denom) for k, v in sums.items()}
        pred_label = max(avg_probs.keys(), key=lambda k: avg_probs[k])
        faithfulness = float(avg_probs.get(m.expected, 0.0))

        expected_vals.sort(key=lambda x: x[0])
        worst = []
        for val, idx in expected_vals[:3]:
            c = clauses[idx] if idx < len(clauses) else {}
            worst.append(
                {
                    "text": str(c.get("text") or "")[:200],
                    "p_expected": float(c.get("p_expected") or val),
                    "pred_label": str(c.get("pred_label") or ""),
                    "probs": c.get("probs") or {},
                    "premise_mode": str(c.get("premise_mode") or "full"),
                    "premise_excerpt": str(c.get("premise_excerpt") or "")[:600],
                    "premise_select": c.get("premise_select") or {},
                }
            )

        worst_idx = int(expected_vals[0][1]) if expected_vals else 0
        primary_clause = clauses[worst_idx] if 0 <= worst_idx < len(clauses) else (clauses[0] if clauses else {})

        m.item["unsupervised_evaluation"] = {
            "method": "nli_faithfulness_v1",
            "scores": {"faithfulness": faithfulness},
            "meta": {
                "expected_label": m.expected,
                "pred_label": pred_label,
                "probs": avg_probs,
                "strategy": "list_avg",
                "hypothesis_mode": m.hypothesis_mode,
                "hypothesis_error": m.hypothesis_error,
                "clauses_used": len(item_preds),
                "worst_clauses": worst,
                "clauses": clauses,
                "premise_mode": str(primary_clause.get("premise_mode") or "full"),
                "premise_excerpt": str(primary_clause.get("premise_excerpt") or "")[:600],
                "premise_select": primary_clause.get("premise_select") or {},
            },
        }
        computed += 1

    skipped = max(0, eligible_items - computed)
    return {
        "computed": computed,
        "skipped": max(0, skipped),
        "eligible": eligible_items,
        "hypothesis_failed": len(errors_by_idx),
        "hypothesis_fallback_used": fallback_used,
        "method": "nli_faithfulness_v1",
        "model_path": (model_path or "").strip() or _resolve_default_model_path(),
        "hypothesis_mode": "llm",
        "hypothesis_concurrency": max_workers,
    }


__all__ = [
    "TRANSFORMERS_AVAILABLE",
    "attach_faithfulness",
    "compute_faithfulness_for_item",
    "is_judge_question",
    "parse_judge_answer",
    "predict_nli",
    "release_nli_device_cache",
]
