# 文件作用：调用大模型生成候选问题和最终问答条目。
# 关联说明：依赖 prompts、evidence_units 和 validation，是生成阶段的 LLM 调用层。

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.common import (
    build_language_instruction,
    detect_language,
    safe_response_dump,
)
from qa.prompts.qa_generation_prompts import (
    build_candidate_question_system_prompt,
    build_evidence_answer_system_prompt,
    build_planner_category_profile,
    build_question_editor_system_prompt,
    build_scenario_planner_system_prompt,
)
from qa.validation import normalize_question_type

ALLOWED_QUESTION_TYPES = {"简答题", "单选题", "判断题", "计算题"}


class _ScenarioPlannerResult(list):
    """List-compatible planner result carrying non-prompt audit metadata."""

    def __init__(self, items: List[Dict[str, Any]], debug_metadata: Optional[Dict[str, Any]] = None):
        super().__init__(items)
        self.debug_metadata = dict(debug_metadata or {})


def normalize_question_types(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        cleaned = [str(x).strip() for x in raw if str(x).strip()]
    else:
        cleaned = [s.strip() for s in str(raw).split(",") if s.strip()]
    return cleaned or None


def normalize_question_type_mode(raw: Any) -> str:
    mode = (str(raw or "mixed")).strip().lower()
    return mode if mode in {"fixed", "mixed"} else "mixed"


def normalize_question_type_weights(raw: Any) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        weights = raw
    elif isinstance(raw, str):
        try:
            weights = json.loads(raw)
        except Exception:
            return None
    else:
        return None
    if not isinstance(weights, dict):
        return None
    normalized: Dict[str, float] = {}
    for key, val in weights.items():
        k = str(key).strip()
        try:
            f = float(val)
        except Exception:
            continue
        if not k:
            continue
        normalized[k] = f
    return normalized or None


def build_question_type_plan(
    *,
    question_type_mode: str,
    question_types: Optional[List[str]],
    question_type_weights: Optional[Dict[str, float]],
    qa_per_chunk: int,
    seed_text: str,
) -> List[str]:
    """
    Build a deterministic per-item question_type plan to make the model follow
    the requested types.

    - If question_types is empty -> all "简答题"
    - fixed -> all first type
    - mixed -> weighted allocation (if provided) else round-robin cycling
    """
    desired = max(1, int(qa_per_chunk or 1))
    types = [t for t in (question_types or []) if t in ALLOWED_QUESTION_TYPES]
    if not types:
        return ["简答题"] * desired

    mode = (str(question_type_mode or "mixed")).strip().lower()
    if mode not in {"fixed", "mixed"}:
        mode = "mixed"

    if mode == "fixed":
        return [types[0]] * desired

    weights = question_type_weights or None
    if isinstance(weights, dict) and weights:
        filtered: List[Tuple[str, float]] = []
        for question_type in types:
            try:
                weight = float(weights.get(question_type, 0.0))
            except Exception:
                weight = 0.0
            if weight > 0:
                filtered.append((question_type, weight))
        if filtered:
            total_weight = sum(weight for _, weight in filtered)
            if total_weight > 0:
                exact = [
                    (question_type, (weight / total_weight) * desired)
                    for question_type, weight in filtered
                ]
                counts: Dict[str, int] = {
                    question_type: int(math.floor(value))
                    for question_type, value in exact
                }
                remaining = desired - sum(counts.values())
                remainders = sorted(
                    [
                        (question_type, value - counts[question_type])
                        for question_type, value in exact
                    ],
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                for index in range(max(0, remaining)):
                    question_type = remainders[index % len(remainders)][0]
                    counts[question_type] = counts.get(question_type, 0) + 1

                plan: List[str] = []
                for question_type, _ in filtered:
                    plan.extend([question_type] * counts.get(question_type, 0))
                if len(plan) < desired:
                    plan.extend([filtered[0][0]] * (desired - len(plan)))
                plan = plan[:desired]

                seed = hashlib.sha1((seed_text or "").encode("utf-8")).hexdigest()
                rnd = random.Random(seed)
                rnd.shuffle(plan)
                return plan

    plan: List[str] = []
    while len(plan) < desired:
        plan.extend(types)
    return plan[:desired]


def apply_question_type_plan(
    items: List[Dict[str, Any]],
    plan: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Reorder/select items to follow the planned question types.
    - Greedy matching: for each planned type, pick the first remaining item with that type.
    - If a planned type cannot be satisfied, skip it.
    """
    if not plan:
        return items
    remaining = [it for it in items if isinstance(it, dict)]
    planned: List[Dict[str, Any]] = []
    for question_type in plan:
        matched_index = None
        for index, item in enumerate(remaining):
            if item.get("question_type") == question_type:
                matched_index = index
                break
        if matched_index is None:
            continue
        planned.append(remaining.pop(matched_index))
    return planned


def _parse_json_items(raw: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads((raw or "").strip()) if raw else None
    except Exception:
        return []
    if isinstance(parsed, dict):
        if "question" in parsed:
            return [parsed]
        items = parsed.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        item = parsed.get("item")
        if isinstance(item, dict):
            return [item]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _normalize_candidate_question(
    item: Dict[str, Any],
    *,
    expected_question_type: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    question = str(item.get("question") or item.get("q") or "").strip()
    if not question:
        return None, "missing_question"

    question_type = normalize_question_type(
        expected_question_type,
        expected=expected_question_type,
    )
    return (
        {
            "question": question,
            "question_type": question_type,
        },
        "ok",
    )


def _restore_evidence_usage_ids(
    raw_entries: Any,
    ref_map: Dict[str, Dict[str, Any]],
    valid_hop_refs: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert readable evidence labels into audited material/image pointers."""
    if not isinstance(raw_entries, list):
        return []
    restored: List[Dict[str, Any]] = []
    restored_by_ref: Dict[str, Dict[str, Any]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        ref = str(raw_entry.get("evidence_ref") or "").strip()
        mapped = ref_map.get(ref) if ref else None
        if not isinstance(mapped, dict):
            continue
        restored_entry = restored_by_ref.get(ref)
        if restored_entry is None:
            restored_entry = {
                "evidence_ref": ref,
                "role": str(mapped.get("role") or "evidence"),
                "material_id": mapped.get("material_id"),
                "chunk_id": mapped.get("chunk_id"),
                "chunk_index": mapped.get("chunk_index"),
                "title_path": mapped.get("title_path"),
            }
            if mapped.get("image_id"):
                restored_entry["image_id"] = mapped.get("image_id")
            if valid_hop_refs:
                restored_entry["hop_refs"] = []
            restored_by_ref[ref] = restored_entry
            restored.append(restored_entry)
        if valid_hop_refs:
            raw_hop_refs = raw_entry.get("hop_refs")
            raw_hop_refs = raw_hop_refs if isinstance(raw_hop_refs, list) else []
            for hop_ref in raw_hop_refs:
                normalized_hop_ref = str(hop_ref or "").strip()
                if (
                    normalized_hop_ref in valid_hop_refs
                    and normalized_hop_ref not in restored_entry["hop_refs"]
                ):
                    restored_entry["hop_refs"].append(normalized_hop_ref)
    return restored


def _primary_usage_covers_bound_materials(
    evidence_usage: List[Dict[str, Any]],
    *,
    source_unit: Dict[str, Any],
) -> bool:
    required_material_ids = [
        str(value)
        for value in source_unit.get("required_material_ids")
        or []
        if str(value)
    ]
    cited_material_ids = {
        str(entry.get("material_id") or "")
        for entry in evidence_usage
        if isinstance(entry, dict)
        and str(entry.get("role") or "") in {"primary_source", "primary_visual"}
    }
    return all(material_id in cited_material_ids for material_id in required_material_ids)


def _evidence_usage_covers_contract(
    evidence_usage: List[Dict[str, Any]],
    *,
    source_unit: Dict[str, Any],
) -> Tuple[bool, str]:
    summary_hops = source_unit.get("summary_hops")
    summary_hops = summary_hops if isinstance(summary_hops, list) else []
    if summary_hops:
        if not 2 <= len(summary_hops) <= 3:
            return False, "incomplete_summary_hop_coverage"
        for index, hop in enumerate(summary_hops, start=1):
            if not isinstance(hop, dict):
                return False, "incomplete_summary_hop_coverage"
            hop_ref = f"HOP-{index}"
            material_id = str(hop.get("material_id") or "")
            hop_entries = [
                entry
                for entry in evidence_usage
                if isinstance(entry, dict)
                and hop_ref in (entry.get("hop_refs") or [])
                and str(entry.get("material_id") or "") == material_id
            ]
            hop_mode = str(hop.get("evidence_mode") or "text").strip().lower()
            has_text = any(
                str(entry.get("role") or "") == "primary_source"
                for entry in hop_entries
            )
            cited_images = {
                str(entry.get("image_id") or "")
                for entry in hop_entries
                if str(entry.get("role") or "") == "primary_visual"
            }
            required_hop_images = {
                str(value)
                for value in hop.get("required_image_ids") or []
                if str(value)
            }
            if hop_mode in {"text", "mixed"} and not has_text:
                return False, "incomplete_summary_hop_coverage"
            if hop_mode in {"visual", "mixed"} and not required_hop_images.issubset(
                cited_images
            ):
                return False, "incomplete_summary_hop_coverage"
    elif not _primary_usage_covers_bound_materials(
        evidence_usage, source_unit=source_unit
    ):
        return False, "incomplete_primary_material_coverage"
    evidence_mode = str(source_unit.get("evidence_mode") or "text").strip().lower()
    required_image_ids = {
        str(value)
        for value in source_unit.get("required_image_ids") or []
        if str(value)
    }
    cited_image_ids = {
        str(entry.get("image_id") or "")
        for entry in evidence_usage
        if isinstance(entry, dict) and str(entry.get("role") or "") == "primary_visual"
    }
    if evidence_mode in {"visual", "mixed"} and not required_image_ids.issubset(cited_image_ids):
        return False, "missing_required_visual_evidence"
    if evidence_mode == "mixed" and not any(
        isinstance(entry, dict) and str(entry.get("role") or "") == "primary_source"
        for entry in evidence_usage
    ):
        return False, "missing_required_text_evidence"
    return True, "ok"


def _answer_retry_instruction(
    reason: str,
    evidence_ref_map: Dict[str, Dict[str, Any]],
    llm_summary_hops: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if reason == "missing_items":
        return "请按既定 JSON 结构返回一条完整答案。"
    if reason == "missing_required_visual_evidence":
        labels = [
            label for label, item in evidence_ref_map.items()
            if isinstance(item, dict) and item.get("role") == "primary_visual"
        ]
        return "必须使用并在 evidence_usage 中引用图片证据：" + "、".join(labels)
    if reason in {"incomplete_primary_material_coverage", "missing_required_text_evidence"}:
        labels = [
            label for label, item in evidence_ref_map.items()
            if isinstance(item, dict) and item.get("role") in {"primary_source", "primary_visual"}
        ]
        return "必须覆盖并在 evidence_usage 中引用所有必需证据：" + "、".join(labels)
    if reason == "incomplete_summary_hop_coverage":
        hop_requirements = []
        for hop in llm_summary_hops or []:
            if not isinstance(hop, dict):
                continue
            hop_ref = str(hop.get("hop_ref") or "").strip()
            evidence_refs = [
                str(value)
                for value in hop.get("evidence_refs") or []
                if str(value)
            ]
            if hop_ref:
                hop_requirements.append(
                    f"{hop_ref} 使用 {'、'.join(evidence_refs) or '其绑定证据'}"
                )
        return (
            "必须逐一回答所有原子子问题，并在 evidence_usage.hop_refs 中标明支撑关系："
            + "；".join(hop_requirements)
        )
    return ""


def _build_qa_evaluation_evidence_text(
    evidence_usage: Any,
    evidence_text_by_ref: Any,
    *,
    fallback: str,
) -> str:
    """Return the exact readable evidence blocks actually cited by an answer.

    The answer model sees a complete unit so it can answer safely, while the
    evaluator should judge the answer against only the blocks the model says it
    used.  This keeps long OCR sections and unrelated retrieved windows from
    overwhelming a visual or mixed-evidence premise.
    """
    if not isinstance(evidence_usage, list) or not isinstance(evidence_text_by_ref, dict):
        return str(fallback or "").strip()
    blocks: List[str] = []
    seen_refs: set[str] = set()
    for entry in evidence_usage:
        if not isinstance(entry, dict):
            continue
        evidence_ref = str(entry.get("evidence_ref") or "").strip()
        if not evidence_ref or evidence_ref in seen_refs:
            continue
        block = evidence_text_by_ref.get(evidence_ref)
        if not isinstance(block, str) or not block.strip():
            continue
        seen_refs.add(evidence_ref)
        blocks.append(block.strip())
    return "\n\n".join(blocks).strip() or str(fallback or "").strip()


def _resolve_generation_language(prompt_language: str, text: str) -> Tuple[str, str]:
    lang = (prompt_language or "auto").strip().lower()
    if lang == "auto":
        detected = detect_language(text)
        lang = detected if detected in {"zh", "en"} else "zh"
    if lang not in {"zh", "en"}:
        lang = "zh"
    return lang, build_language_instruction(lang)


def _prompt_title_path(value: Any) -> str:
    """Return only a human-readable node path; never expose internal IDs."""
    return str(value or "").strip()


def _planning_material_ref(position: int) -> str:
    number = max(1, int(position))
    letters: List[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "主材料-" + "".join(reversed(letters))


def _planning_image_ref(position: int) -> str:
    number = max(1, int(position))
    letters: List[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "图片-" + "".join(reversed(letters))


def _prompt_paths_from_meta(meta: Optional[Dict[str, Any]]) -> List[str]:
    source_meta = meta if isinstance(meta, dict) else {}
    values: List[Any] = []
    for key in ("qa_generation_unit_material_paths", "source_material_paths"):
        raw = source_meta.get(key)
        if isinstance(raw, list):
            values.extend(raw)
    values.extend(
        [
            source_meta.get("qa_generation_unit_title_path"),
            source_meta.get("source_material_path"),
            source_meta.get("title_path"),
        ]
    )
    paths: List[str] = []
    for value in values:
        path = _prompt_title_path(value)
        if path and path not in paths:
            paths.append(path)
    return paths


def _format_source_material_for_prompt(
    text: str,
    meta: Optional[Dict[str, Any]],
) -> str:
    paths = _prompt_paths_from_meta(meta)
    path_lines = "\n".join(f"- {path}" for path in paths) or "- 未标注章节"
    return (
        "主材料节点路径：\n"
        f"{path_lines}\n\n"
        "主材料正文：\n"
        f"{str(text or '').strip()}"
    )


def _select_style_example(
    examples: Optional[List[Dict[str, Any]]],
    *,
    qa_detail_mode: str,
) -> str:
    """Render at most one reviewed wording example without exposing its schema."""
    for example in examples or []:
        if not isinstance(example, dict):
            continue
        mode = str(example.get("qa_detail_mode") or example.get("mode") or "").strip().lower()
        if mode and mode != qa_detail_mode:
            continue
        question = str(example.get("question") or "").strip()
        if not question:
            continue
        if len(question) > 120:
            continue
        return f"\n风格示例：{question}\n"
    return ""


_DOCUMENT_DEICTIC_RE = re.compile(
    r"(?:本|这份|该)(?:操作说明|说明|通知|文件|手册|指南|材料|文档)"
)
_SOURCE_ATTRIBUTION_PREFIX_RE = re.compile(
    r"^(?:请问[，,\s]*)?(?:(?:根据|依据|按照|在)\s*)"
    r"《[^》]{3,120}》(?:中|里)?[，,：:\s]+"
)
_EN_SOURCE_ATTRIBUTION_PREFIX_RE = re.compile(
    r"^(?:(?:according to|in)\s+[\"“][^\"”]{3,120}[\"”][,:]?\s*)",
    re.IGNORECASE,
)
_GENERIC_SECTION_LABEL_RE = re.compile(
    r"^(?:[（(]?[一二三四五六七八九十0-9]+[）).、．\s]*)?"
    r"(?:适用范围|功能说明|政策依据|注意事项|操作步骤|业务操作要点说明|"
    r"业务办理流程说明|业务办理渠道说明|更多详情.*)$"
)
_BINARY_CONSTRAINT_RE = re.compile(
    r"(?:是否|能否|可否|可不可以|能不能|还(?:能|可以)|"
    r"(?:可以|能够|能|会|不得|不能).{0,18}(?:吗|？|\?))",
    re.IGNORECASE,
)
_PROCEDURAL_QUESTION_RE = re.compile(
    r"(?:如何|怎么(?:样)?|怎样|怎么办|应当如何|该如何)",
    re.IGNORECASE,
)


def _clean_question_subject_part(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"^(?:[（(]?[一二三四五六七八九十0-9]+[）).、．\s]*)+", "", text)
    return text.strip(" -_：:")


def _question_object_label(meta: Optional[Dict[str, Any]]) -> str:
    """Choose a short business/object name for repairing document deictics."""
    candidates: List[str] = []
    for path in _prompt_paths_from_meta(meta):
        parts = [
            _clean_question_subject_part(part)
            for part in path.replace("＞", ">").split(">")
        ]
        for part in reversed(parts):
            if not part or _GENERIC_SECTION_LABEL_RE.fullmatch(part):
                continue
            if 4 <= len(part) <= 72:
                candidates.append(part)
                if re.search(r"(操作说明|使用说明|指南|手册|申报|办理|服务)", part):
                    return part
    for value in candidates:
        if not _GENERIC_SECTION_LABEL_RE.fullmatch(value):
            return value
    source_meta = meta if isinstance(meta, dict) else {}
    subject = _clean_question_subject_part(source_meta.get("qa_generation_unit_subject_label"))
    return subject if 4 <= len(subject) <= 72 else ""


def _repair_document_deictic(question: str, *, question_object: str) -> str:
    text = str(question or "").strip()
    if not text or not question_object:
        return text
    return _DOCUMENT_DEICTIC_RE.sub(question_object, text).strip()


def _clean_standalone_question(question: str, *, question_object: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    without_source = _SOURCE_ATTRIBUTION_PREFIX_RE.sub("", text).strip()
    without_source = _EN_SOURCE_ATTRIBUTION_PREFIX_RE.sub("", without_source).strip()
    if len(without_source) >= 6:
        text = without_source
    return _repair_document_deictic(text, question_object=question_object)


def _preserve_binary_question_form(original_question: str, edited_question: str) -> bool:
    """Keep a valid permission/prohibition question from becoming a how-to question."""
    original = str(original_question or "").strip()
    edited = str(edited_question or "").strip()
    return bool(
        original
        and edited
        and _BINARY_CONSTRAINT_RE.search(original)
        and _PROCEDURAL_QUESTION_RE.search(edited)
    )


def _build_question_writing_brief(
    *,
    source_material: str,
    source_meta: Optional[Dict[str, Any]],
    scenario_intent: str,
    reader_need: str,
    qa_detail_mode: str,
) -> str:
    """Render only semantic facts needed by a wording model, never aliases/IDs."""
    meta = source_meta if isinstance(source_meta, dict) else {}
    prompt_materials = meta.get("qa_generation_unit_prompt_materials")
    material_ref_map = meta.get("qa_generation_unit_material_ref_map")
    image_ref_map = meta.get("qa_generation_unit_image_ref_map")
    material_ref_map = material_ref_map if isinstance(material_ref_map, dict) else {}
    image_ref_map = image_ref_map if isinstance(image_ref_map, dict) else {}
    required_material_ids = {
        str(value)
        for value in meta.get("qa_generation_unit_required_material_ids") or []
        if str(value)
    }
    required_image_ids = {
        str(value)
        for value in meta.get("qa_generation_unit_required_image_ids") or []
        if str(value)
    }
    evidence_mode = str(meta.get("qa_generation_unit_evidence_mode") or "text").strip().lower()
    raw_summary_hops = meta.get("qa_generation_unit_summary_hops")
    raw_summary_hops = raw_summary_hops if isinstance(raw_summary_hops, list) else []
    summary_sub_questions: List[str] = []
    for raw_hop in raw_summary_hops:
        if not isinstance(raw_hop, dict):
            continue
        sub_question = " ".join(str(raw_hop.get("sub_question") or "").split()).strip()
        if sub_question and sub_question not in summary_sub_questions:
            summary_sub_questions.append(sub_question)
    subject = str(meta.get("qa_generation_unit_subject_label") or "").strip()
    question_object = _question_object_label(meta)
    text_facts: List[str] = []
    image_facts: List[str] = []
    if isinstance(prompt_materials, list) and prompt_materials:
        for material in prompt_materials:
            if not isinstance(material, dict):
                continue
            material_ref = str(material.get("material_ref") or "")
            if required_material_ids and str(material_ref_map.get(material_ref) or "") not in required_material_ids:
                continue
            text_content = str(material.get("text_content") or "").strip()
            if text_content:
                text_facts.append(text_content)
            if not required_image_ids:
                continue
            for image in material.get("image_materials") or []:
                if not isinstance(image, dict):
                    continue
                image_ref = str(image.get("image_ref") or "")
                if required_image_ids and str(image_ref_map.get(image_ref) or "") not in required_image_ids:
                    continue
                description = str(image.get("description") or "").strip()
                if description:
                    image_facts.append(description)
    if not text_facts and source_material:
        text_facts.append(_format_source_material_for_prompt(source_material, meta))

    lines = []
    if subject:
        lines.append(f"主体：{subject}")
    if question_object and question_object != subject:
        lines.append(
            "问题对象（题干出现“本/该/这份说明、通知或文件”时，用此名称替换）："
            f"{question_object}"
        )
    lines.append(f"提问焦点：{str(scenario_intent or '').strip()}")
    lines.append(
        "读者情境（只用于选择口吻，不要逐项写入题干）："
        f"{str(reader_need or '').strip()}"
    )
    lines.append("题目粒度：总结" if qa_detail_mode == "summary" else "题目粒度：单点")
    if qa_detail_mode == "summary" and summary_sub_questions:
        lines.append(
            "总括问题必须同时覆盖的信息缺口（合并成一句自然问题，不要逐项照抄）：\n"
            + "\n".join(
                f"{index}. {sub_question}"
                for index, sub_question in enumerate(summary_sub_questions, start=1)
            )
        )
    if text_facts:
        lines.append(
            "回答依据（只用于限定答案范围，不要逐项复述数值、名单、步骤或日期）：\n"
            + "\n\n".join(text_facts)
        )
    if image_facts:
        label = "视觉回答依据（问题应围绕可观察视觉焦点，不要照搬图片数值）"
        if evidence_mode == "mixed":
            label = "视觉回答依据（本题必须保留视觉焦点，但不要照搬图片数值）"
        lines.append(label + "：\n" + "\n\n".join(image_facts))
    return "\n\n".join(line for line in lines if line.strip())


def _ensure_evidence_paths_for_prompt(
    unit_text: str,
    generation_unit: Dict[str, Any],
) -> str:
    """Add readable paths for hand-built/legacy units missing the new markers."""
    text = str(unit_text or "").strip()
    if "节点路径：" in text or "主材料节点路径：" in text:
        return text
    chunks = generation_unit.get("source_chunks")
    if not isinstance(chunks, list):
        chunks = []
    paths: List[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        path = _prompt_title_path(chunk.get("title_path"))
        if path and path not in paths:
            paths.append(path)
    source_chunk = generation_unit.get("source_chunk")
    if isinstance(source_chunk, dict):
        path = _prompt_title_path(source_chunk.get("title_path"))
        if path and path not in paths:
            paths.append(path)
    path_lines = "\n".join(f"- {path}" for path in paths) or "- 未标注章节"
    if not text:
        text = str(
            generation_unit.get("source_unit_text")
            or (source_chunk.get("text") if isinstance(source_chunk, dict) else "")
            or ""
        ).strip()
    return f"【主材料节点路径】\n{path_lines}\n\n{text}".strip()


def call_scenario_planner_llm(
    *,
    client: Any,
    model: str,
    section_materials: List[Any],
    requested_count: int,
    qa_detail_mode: str,
    prompt_language: str,
    request_timeout: int,
    knowledge_category: Optional[str] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    planning_batch_index: Optional[int] = None,
    planning_batch_count: Optional[int] = None,
    _allow_scenario_type_retry: bool = True,
    _planner_retry_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Plan typed scenarios using opaque aliases mapped back by the backend."""
    if requested_count <= 0 or not section_materials:
        return _ScenarioPlannerResult([], {"items_raw_count": 0, "items_validated_count": 0})
    image_refs_by_id: Dict[str, str] = {}
    for material in section_materials:
        for image in getattr(material, "image_materials", []) or []:
            image_id = str(getattr(image, "image_id", "") or "").strip()
            if image_id and image_id not in image_refs_by_id:
                image_refs_by_id[image_id] = _planning_image_ref(len(image_refs_by_id) + 1)
    readable_materials = [
        material.to_prompt_dict(
            material_ref=_planning_material_ref(index),
            image_refs_by_id=image_refs_by_id,
        )
        for index, material in enumerate(section_materials, start=1)
    ]
    material_by_ref = {
        str(readable["material_ref"]): material
        for readable, material in zip(readable_materials, section_materials)
    }
    joined_text = "\n\n".join(
        "\n".join(
            [
                str(item.get("text_content") or ""),
                *[
                    str(image.get("description") or "")
                    for image in item.get("image_materials") or []
                    if isinstance(image, dict)
                ],
            ]
        )
        for item in readable_materials
    )
    image_id_by_ref = {ref: image_id for image_id, ref in image_refs_by_id.items()}
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        joined_text,
    )
    system_prompt = build_scenario_planner_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        requested_count=requested_count,
        qa_detail_mode=qa_detail_mode,
        category_profile=build_planner_category_profile(
            knowledge_category=knowledge_category,
            language_code=language_code,
        ),
        material_count=len(section_materials),
    )
    user_content = json.dumps({"materials": readable_materials}, ensure_ascii=False)
    retry_context = dict(_planner_retry_context or {})
    expected_scenario_type = str(qa_detail_mode or "").strip().lower()
    retry_instruction = ""
    if retry_context and expected_scenario_type in {"point", "summary"}:
        if language_code == "en":
            retry_instruction = (
                f"Correction: this batch accepts only `{expected_scenario_type}` items. "
                f"Set every `scenario_type` to `{expected_scenario_type}` and return the JSON again."
            )
        else:
            retry_instruction = (
                f"纠正：本批次只接受 `{expected_scenario_type}` 场景。"
                f"每一条 `scenario_type` 都必须填写 `{expected_scenario_type}`，请重新返回 JSON。"
            )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if retry_instruction:
        messages.append({"role": "user", "content": retry_instruction})
    try:
        raw = client.create_chat_completion_text(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=float(request_timeout),
        ).strip()
    except Exception as exc:
        if debug_writer:
            debug_writer(
                {
                    "event": "scenario_planner_llm_call",
                    "model": model,
                    "qa_detail_mode": qa_detail_mode,
                    "requested_count": requested_count,
                    "material_count": len(readable_materials),
                    "planning_batch_index": planning_batch_index,
                    "planning_batch_count": planning_batch_count,
                    "planning_materials": [
                        {
                            "material_ref": item.get("material_ref"),
                            "node_path": item.get("node_path"),
                            "parent_node_path": item.get("parent_node_path"),
                        }
                        for item in readable_materials
                    ],
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "planner_retry_instruction": retry_instruction or None,
                    "planning_attempt_count": 2 if retry_context else 1,
                    "initial_raw_response": retry_context.get("initial_raw_response"),
                    "parse_error": str(exc),
                }
            )
        raise
    raw_items = _parse_json_items(raw)
    normalized: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    for item in raw_items:
        scenario_type = str(item.get("scenario_type") or item.get("type") or "").strip().lower()
        scenario_type = {
            "point only": "point",
            "summary only": "summary",
        }.get(scenario_type, scenario_type)
        if scenario_type not in {"point", "summary"}:
            dropped["invalid_scenario_type"] = dropped.get("invalid_scenario_type", 0) + 1
            continue
        if expected_scenario_type in {"point", "summary"} and scenario_type != expected_scenario_type:
            dropped["scenario_type_mismatch"] = dropped.get("scenario_type_mismatch", 0) + 1
            continue
        intent = str(item.get("intent") or "").strip()
        reader_need = str(item.get("reader_need") or "").strip()
        def resolve_refs(raw_values: Any) -> Tuple[List[str], int]:
            values = raw_values if isinstance(raw_values, list) else []
            resolved: List[str] = []
            unknown = 0
            for value in values:
                token = str(value or "").strip()
                material = material_by_ref.get(token)
                if material is None:
                    unknown += 1
                    continue
                material_id = str(material.material_id)
                if material_id not in resolved:
                    resolved.append(material_id)
            return resolved, unknown

        summary_hops: List[Dict[str, Any]] = []
        required_material_ids: List[str] = []
        required_image_ids: List[str] = []
        if scenario_type == "summary":
            raw_hops = item.get("summary_hops")
            raw_hops = raw_hops if isinstance(raw_hops, list) else []
            if not 2 <= len(raw_hops) <= 3:
                dropped["summary_requires_2_to_3_hops"] = (
                    dropped.get("summary_requires_2_to_3_hops", 0) + 1
                )
                continue
            seen_sub_questions: set[str] = set()
            invalid_hop_reason = ""
            for hop_index, raw_hop in enumerate(raw_hops, start=1):
                if not isinstance(raw_hop, dict):
                    invalid_hop_reason = "invalid_summary_hop"
                    break
                sub_question = " ".join(
                    str(raw_hop.get("sub_question") or "").split()
                ).strip()
                sub_key = sub_question.casefold()
                material_ref = str(raw_hop.get("material_ref") or "").strip()
                material = material_by_ref.get(material_ref)
                if not sub_question or material is None:
                    invalid_hop_reason = (
                        "unknown_summary_hop_material_ref"
                        if material is None
                        else "invalid_summary_hop"
                    )
                    break
                if sub_key in seen_sub_questions:
                    invalid_hop_reason = "duplicate_summary_hop"
                    break
                seen_sub_questions.add(sub_key)
                material_id = str(material.material_id)
                material_image_ids = {
                    str(getattr(image, "image_id", "") or "").strip()
                    for image in getattr(material, "image_materials", []) or []
                }
                hop_image_ids: List[str] = []
                hop_image_refs: List[str] = []
                for value in raw_hop.get("image_refs") or []:
                    image_ref = str(value or "").strip()
                    image_id = image_id_by_ref.get(image_ref)
                    if not image_id or image_id not in material_image_ids:
                        invalid_hop_reason = "unknown_summary_hop_image_ref"
                        break
                    if image_id not in hop_image_ids:
                        hop_image_ids.append(image_id)
                        hop_image_refs.append(image_ref)
                if invalid_hop_reason:
                    break
                hop_mode = str(raw_hop.get("evidence_mode") or "").strip().lower()
                if hop_mode not in {"text", "visual", "mixed"}:
                    hop_mode = "mixed" if hop_image_ids else "text"
                if hop_image_ids and hop_mode == "text":
                    hop_mode = "mixed"
                if not hop_image_ids and hop_mode in {"visual", "mixed"}:
                    invalid_hop_reason = "summary_hop_visual_requires_image"
                    break
                summary_hops.append(
                    {
                        "hop_id": f"hop-{hop_index}",
                        "sub_question": sub_question,
                        "material_id": material_id,
                        "material_ref": material_ref,
                        "evidence_mode": hop_mode,
                        "required_image_ids": hop_image_ids,
                        "image_refs": hop_image_refs,
                    }
                )
                if material_id not in required_material_ids:
                    required_material_ids.append(material_id)
                for image_id in hop_image_ids:
                    if image_id not in required_image_ids:
                        required_image_ids.append(image_id)
            if invalid_hop_reason:
                dropped[invalid_hop_reason] = dropped.get(invalid_hop_reason, 0) + 1
                continue
        else:
            required_material_ids, unknown_required = resolve_refs(
                item.get("required_material_refs"),
            )
            if unknown_required:
                dropped["unknown_material_ref"] = (
                    dropped.get("unknown_material_ref", 0) + unknown_required
                )
            selected_material_ids = set(required_material_ids)
            available_image_ids = {
                str(getattr(image, "image_id", "") or "").strip()
                for material in section_materials
                if str(material.material_id) in selected_material_ids
                for image in getattr(material, "image_materials", []) or []
            }
            unknown_image_refs = 0
            for value in item.get("required_image_refs") or []:
                image_id = image_id_by_ref.get(str(value or "").strip())
                if not image_id or image_id not in available_image_ids:
                    unknown_image_refs += 1
                    continue
                if image_id not in required_image_ids:
                    required_image_ids.append(image_id)
            if unknown_image_refs:
                dropped["unknown_image_ref"] = (
                    dropped.get("unknown_image_ref", 0) + unknown_image_refs
                )
        optional_material_ids, unknown_optional = resolve_refs(
            item.get("optional_material_refs"),
        )
        if unknown_optional:
            dropped["unknown_material_ref"] = (
                dropped.get("unknown_material_ref", 0) + unknown_optional
            )
        optional_material_ids = [
            material_id
            for material_id in optional_material_ids
            if material_id not in required_material_ids
        ]
        material_ids: List[str] = []
        for material_id in [*required_material_ids, *optional_material_ids]:
            if material_id not in material_ids:
                material_ids.append(material_id)
        if not intent or not reader_need or not required_material_ids:
            dropped["missing_required_field"] = dropped.get("missing_required_field", 0) + 1
            continue
        if scenario_type == "point" and (
            len(required_material_ids) != 1 or optional_material_ids
        ):
            dropped["point_requires_one_material"] = dropped.get("point_requires_one_material", 0) + 1
            continue
        if len(required_material_ids) > 3:
            dropped["summary_requires_at_most_three_materials"] = (
                dropped.get("summary_requires_at_most_three_materials", 0) + 1
            )
            continue
        if scenario_type == "summary":
            hop_modes = {str(hop.get("evidence_mode") or "text") for hop in summary_hops}
            evidence_mode = (
                "text"
                if hop_modes == {"text"}
                else "visual"
                if hop_modes == {"visual"}
                else "mixed"
            )
        else:
            evidence_mode = str(item.get("evidence_mode") or "").strip().lower()
            if evidence_mode not in {"text", "visual", "mixed"}:
                evidence_mode = "mixed" if required_image_ids else "text"
            if required_image_ids and evidence_mode == "text":
                evidence_mode = "mixed"
            elif not required_image_ids and evidence_mode in {"visual", "mixed"}:
                evidence_mode = "text"
        normalized.append(
            {
                "scenario_type": scenario_type,
                "intent": intent,
                "reader_need": reader_need,
                "material_ids": material_ids,
                "required_material_ids": required_material_ids,
                "optional_material_ids": optional_material_ids,
                "required_material_refs": [
                    _planning_material_ref(
                        next(
                            index
                            for index, material in enumerate(section_materials, start=1)
                            if str(material.material_id) == material_id
                        )
                    )
                    for material_id in required_material_ids
                ],
                "optional_material_refs": [
                    _planning_material_ref(
                        next(
                            index
                            for index, material in enumerate(section_materials, start=1)
                            if str(material.material_id) == material_id
                        )
                    )
                    for material_id in optional_material_ids
                ],
                "evidence_mode": evidence_mode,
                "required_image_ids": required_image_ids,
                "required_image_refs": [
                    image_refs_by_id[image_id]
                    for image_id in required_image_ids
                    if image_id in image_refs_by_id
                ],
                "summary_hops": summary_hops,
            }
        )
    only_scenario_type_mismatch = (
        _allow_scenario_type_retry
        and expected_scenario_type in {"point", "summary"}
        and bool(raw_items)
        and not normalized
        and set(dropped) == {"scenario_type_mismatch"}
    )
    if only_scenario_type_mismatch:
        return call_scenario_planner_llm(
            client=client,
            model=model,
            section_materials=section_materials,
            requested_count=requested_count,
            qa_detail_mode=qa_detail_mode,
            prompt_language=prompt_language,
            request_timeout=request_timeout,
            knowledge_category=knowledge_category,
            debug_writer=debug_writer,
            planning_batch_index=planning_batch_index,
            planning_batch_count=planning_batch_count,
            _allow_scenario_type_retry=False,
            _planner_retry_context={
                "initial_raw_response": raw,
                "initial_items_raw_count": len(raw_items),
                "initial_dropped_validation_reasons": dict(dropped),
            },
        )
    if debug_writer:
        debug_writer(
            {
                "event": "scenario_planner_llm_call",
                "model": model,
                "qa_detail_mode": qa_detail_mode,
                "requested_count": requested_count,
                "material_count": len(readable_materials),
                "planning_batch_index": planning_batch_index,
                "planning_batch_count": planning_batch_count,
                "planning_materials": [
                    {
                        "material_ref": item.get("material_ref"),
                        "node_path": item.get("node_path"),
                        "parent_node_path": item.get("parent_node_path"),
                    }
                    for item in readable_materials
                ],
                "system_prompt": system_prompt,
                "user_content": user_content,
                "planner_retry_reason": "scenario_type_mismatch" if retry_context else None,
                "planner_retry_instruction": retry_instruction or None,
                "planning_attempt_count": 2 if retry_context else 1,
                "initial_raw_response": retry_context.get("initial_raw_response"),
                "initial_items_raw_count": retry_context.get("initial_items_raw_count"),
                "initial_dropped_validation_reasons": retry_context.get(
                    "initial_dropped_validation_reasons"
                ),
                "raw_response": raw,
                "items_raw_count": len(raw_items),
                "items_validated_count": len(normalized),
                "dropped_validation_reasons": dropped,
            }
        )
    return _ScenarioPlannerResult(
        normalized,
        {
            "raw_response": raw,
            "items_raw_count": len(raw_items),
            "items_validated_count": len(normalized),
            "dropped_validation_reasons": dropped,
            "planning_attempt_count": 2 if retry_context else 1,
            "planner_retry_reason": "scenario_type_mismatch" if retry_context else "",
            "initial_raw_response": retry_context.get("initial_raw_response"),
            "initial_items_raw_count": retry_context.get("initial_items_raw_count"),
            "initial_dropped_validation_reasons": retry_context.get(
                "initial_dropped_validation_reasons"
            ),
        },
    )


def call_question_editor_llm(
    *,
    client: Any,
    model: str,
    candidate: Dict[str, Any],
    source_material: str,
    scenario_intent: str,
    reader_need: str,
    qa_detail_mode: str,
    prompt_language: str,
    request_timeout: int,
    chunk_index: Optional[int] = None,
    source_material_path: Optional[str] = None,
    source_material_paths: Optional[List[str]] = None,
    source_chunk_meta: Optional[Dict[str, Any]] = None,
    style_examples: Optional[List[Dict[str, Any]]] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Produce the final wording without changing the frozen scenario contract."""
    original_question = str(candidate.get("question") or "").strip()
    if not original_question:
        return None, "missing_question"
    editor_meta = dict(source_chunk_meta or {})
    editor_meta.setdefault("source_material_path", source_material_path)
    editor_meta.setdefault("source_material_paths", source_material_paths or [])
    writing_brief = _build_question_writing_brief(
        source_material=source_material,
        source_meta=editor_meta,
        scenario_intent=scenario_intent,
        reader_need=reader_need,
        qa_detail_mode=qa_detail_mode,
    )
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        writing_brief or original_question,
    )
    system_prompt = build_question_editor_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        qa_detail_mode=qa_detail_mode,
        style_example=_select_style_example(style_examples, qa_detail_mode=qa_detail_mode),
    )
    user_content = f"原问题：{original_question}\n\n写作 brief：\n{writing_brief}"
    try:
        raw = client.create_chat_completion_text(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=float(request_timeout),
        ).strip()
    except Exception as exc:
        if debug_writer:
            debug_writer(
                {
                    "event": "question_editor_llm_call",
                    "chunk_index": chunk_index,
                    "model": model,
                    "qa_detail_mode": qa_detail_mode,
                    "scenario_intent": scenario_intent,
                    "reader_need": reader_need,
                    "original_question": original_question,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "parse_error": str(exc),
                }
            )
        raise
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}
    raw_edited_question = (
        str(parsed.get("question") or "").strip() if isinstance(parsed, dict) else ""
    )
    question_object = _question_object_label(editor_meta)
    semantic_guard = ""
    edited_question = ""
    if raw_edited_question:
        edited_question = _clean_standalone_question(
            raw_edited_question,
            question_object=question_object,
        )
        if _preserve_binary_question_form(original_question, edited_question):
            edited_question = _clean_standalone_question(
                original_question,
                question_object=question_object,
            )
            semantic_guard = "preserved_binary_question_form"
        elif edited_question != raw_edited_question:
            semantic_guard = "repaired_source_or_document_reference"
    result = dict(candidate) if edited_question else None
    if result is not None:
        result["question"] = edited_question
    status = "edited" if result is not None else "invalid_editor_response"
    if debug_writer:
        debug_writer(
            {
                "event": "question_editor_llm_call",
                "chunk_index": chunk_index,
                "model": model,
                "qa_detail_mode": qa_detail_mode,
                "scenario_intent": scenario_intent,
                "reader_need": reader_need,
                "original_question": original_question,
                "editor_decision": semantic_guard or ("edited" if result is not None else "invalid"),
                "edited_question": edited_question,
                "question_object": question_object or None,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "raw_response": raw,
                "result_status": status,
            }
        )
    return result, status


def call_candidate_question_llm(
    *,
    client: Any,
    model: str,
    source_chunk_text: str,
    source_chunk_meta: Dict[str, Any],
    candidate_count: int,
    prompt_language: str,
    question_type_plan: List[str],
    few_shot_examples: Optional[List[Dict[str, Any]]],
    request_timeout: int,
    qa_detail_mode: str = "point",
    chunk_index: Optional[int] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        source_chunk_text,
    )
    system_prompt = build_candidate_question_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        qa_detail_mode=qa_detail_mode,
        style_example=_select_style_example(few_shot_examples, qa_detail_mode=qa_detail_mode),
    )
    scenario_intent = str(source_chunk_meta.get("qa_generation_unit_scenario_intent") or "").strip()
    reader_need = str(source_chunk_meta.get("qa_generation_unit_reader_need") or "").strip()
    user_content = _build_question_writing_brief(
        source_material=source_chunk_text,
        source_meta=source_chunk_meta,
        scenario_intent=scenario_intent,
        reader_need=reader_need,
        qa_detail_mode=qa_detail_mode,
    )

    response_type: Optional[str] = None
    response_dump: Any = None
    raw = ""
    parse_error: Optional[str] = None
    try:
        raw = client.create_chat_completion_text(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=float(request_timeout),
        ).strip()
        response_type = "str"
        response_dump = safe_response_dump(raw)
        raw_items = _parse_json_items(raw)
    except Exception as exc:
        parse_error = str(exc)
        raw_items = []
        if debug_writer:
            debug_writer(
                {
                    "event": "candidate_question_llm_call",
                    "chunk_index": chunk_index,
                    "model": model,
                    "qa_detail_mode": qa_detail_mode,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "response_type": response_type,
                    "response_dump": response_dump,
                    "raw_response": raw,
                    "parse_error": parse_error,
                }
            )
        raise

    normalized: List[Dict[str, Any]] = []
    dropped_reasons: Dict[str, int] = {}
    seen_questions: set[str] = set()
    for index, item in enumerate(raw_items):
        expected = question_type_plan[index] if index < len(question_type_plan) else None
        candidate, reason = _normalize_candidate_question(
            item,
            expected_question_type=expected,
        )
        if not candidate:
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            continue
        key = " ".join(str(candidate.get("question") or "").split()).casefold()
        if key in seen_questions:
            dropped_reasons["duplicate_question"] = dropped_reasons.get("duplicate_question", 0) + 1
            continue
        seen_questions.add(key)
        normalized.append(candidate)

    if debug_writer:
        debug_writer(
            {
                "event": "candidate_question_llm_call",
                "chunk_index": chunk_index,
                "model": model,
                "candidate_count": candidate_count,
                "qa_detail_mode": qa_detail_mode,
                "question_type_plan": question_type_plan,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response_type": response_type,
                "response_dump": response_dump,
                "raw_response": raw,
                "parse_error": parse_error,
                "items_raw_count": len(raw_items),
                "items_validated_count": len(normalized),
                "dropped_validation_reasons": dropped_reasons,
            }
        )
    return normalized[: max(1, int(candidate_count))]


def call_evidence_answer_llm(
    *,
    client: Any,
    model: str,
    candidate: Dict[str, Any],
    generation_unit: Dict[str, Any],
    qa_detail_mode: str,
    prompt_language: str,
    request_timeout: int,
    item_normalizer_with_reason: Callable[..., Tuple[Optional[Dict[str, Any]], str]],
    source_override_handler: Callable[..., None],
    fixed_knowledge_category: Optional[str] = None,
    fixed_knowledge_category_confidence: Optional[float] = None,
    fixed_knowledge_category_reason: str = "",
    chunk_index: Optional[int] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    answer_audit: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    source_chunk = generation_unit.get("source_chunk") or {}
    source_chunk_text = str(source_chunk.get("text") or "").strip()
    source_unit_text = str(generation_unit.get("source_unit_text") or "").strip()
    unit_text = str(generation_unit.get("qa_generation_unit_text") or "").strip()
    prompt_unit_text = _ensure_evidence_paths_for_prompt(unit_text, generation_unit)
    evidence_ref_map = generation_unit.get("llm_evidence_ref_map")
    if not isinstance(evidence_ref_map, dict):
        evidence_ref_map = {}
    evidence_text_by_ref = generation_unit.get("llm_evidence_text_by_ref")
    if not isinstance(evidence_text_by_ref, dict):
        evidence_text_by_ref = {}
    llm_summary_hops = generation_unit.get("llm_summary_hops")
    llm_summary_hops = llm_summary_hops if isinstance(llm_summary_hops, list) else []
    valid_hop_refs = {
        str(hop.get("hop_ref") or "").strip()
        for hop in llm_summary_hops
        if isinstance(hop, dict) and str(hop.get("hop_ref") or "").strip()
    }
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        prompt_unit_text or source_chunk_text,
    )
    system_prompt = build_evidence_answer_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        qa_detail_mode=qa_detail_mode,
        question_type=str(candidate.get("question_type") or "简答题"),
    )
    candidate_question = str(candidate.get("question") or "").strip()
    question_type = str(candidate.get("question_type") or "简答题").strip() or "简答题"
    source_unit_payload = generation_unit.get("source_unit")
    if not isinstance(source_unit_payload, dict):
        source_unit_payload = {}
    user_content = (
        f"题目：{candidate_question}\n"
        f"题目形式：{question_type}\n\n"
        f"可读证据：\n{prompt_unit_text}"
    )
    if llm_summary_hops:
        hop_lines: List[str] = []
        for hop in llm_summary_hops:
            if not isinstance(hop, dict):
                continue
            hop_lines.append(
                f"{hop.get('hop_ref')}：{hop.get('sub_question')}\n"
                f"绑定证据：{'、'.join(str(value) for value in hop.get('evidence_refs') or [])}"
            )
        user_content += (
            "\n\n必须覆盖的原子子问题：\n"
            + "\n".join(hop_lines)
            + "\n请在 evidence_usage.hop_refs 中标明每条证据支撑的 HOP。"
        )

    response_type: Optional[str] = None
    response_dump: Any = None
    raw = ""
    parse_error: Optional[str] = None
    dropped_reason = ""
    initial_raw = ""
    answer_attempt_count = 0
    answer_retry_reason = ""
    answer_retry_error = ""

    def _write_answer_audit(final_reason: str) -> None:
        if not isinstance(answer_audit, dict):
            return
        answer_audit.update(
            {
                "answer_attempt_count": answer_attempt_count,
                "answer_retry_reason": answer_retry_reason or None,
                "answer_retry_error": answer_retry_error or None,
                "final_reason": final_reason,
            }
        )

    def _normalize_answer(raw_items: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
        if not raw_items:
            return None, "missing_items"
        raw_item = raw_items[0]
        model_item = dict(raw_item)
        model_item["question"] = candidate_question
        model_item["question_type"] = question_type
        normalized, reason = item_normalizer_with_reason(
            model_item,
            language_code=language_code,
            expected_question_type=question_type,
            fixed_knowledge_category=fixed_knowledge_category,
            fixed_knowledge_category_confidence=fixed_knowledge_category_confidence,
            fixed_knowledge_category_reason=fixed_knowledge_category_reason,
        )
        if not normalized:
            return None, reason
        raw_usage = raw_item.get("evidence_usage")
        if isinstance(raw_usage, list):
            normalized["evidence_usage"] = _restore_evidence_usage_ids(
                raw_usage,
                evidence_ref_map,
                valid_hop_refs=valid_hop_refs or None,
            )[:12]
        contract_ok, contract_reason = _evidence_usage_covers_contract(
            normalized.get("evidence_usage") or [],
            source_unit=source_unit_payload,
        )
        return (normalized, "ok") if contract_ok else (None, contract_reason)

    try:
        answer_attempt_count = 1
        raw = client.create_chat_completion_text(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=float(request_timeout),
        ).strip()
        response_type = "str"
        response_dump = safe_response_dump(raw)
        raw_items = _parse_json_items(raw)
    except Exception as exc:
        parse_error = str(exc)
        raw_items = []
        if debug_writer:
            debug_writer(
                {
                    "event": "evidence_answer_llm_call",
                    "chunk_index": chunk_index,
                    "candidate": candidate,
                    "generation_unit": {
                        "qa_generation_unit_id": generation_unit.get("qa_generation_unit_id"),
                        "evidence_chunk_ids": generation_unit.get("evidence_chunk_ids"),
                    },
                    "knowledge_category": fixed_knowledge_category,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "response_type": response_type,
                    "response_dump": response_dump,
                    "raw_response": raw,
                    "initial_raw_response": initial_raw,
                    "answer_attempt_count": answer_attempt_count,
                    "answer_retry_reason": answer_retry_reason or None,
                    "answer_retry_error": answer_retry_error or None,
                    "parse_error": parse_error,
                }
            )
        _write_answer_audit("answer_generation_error")
        raise

    initial_raw = raw
    normalized_item, dropped_reason = _normalize_answer(raw_items)
    retry_instruction = _answer_retry_instruction(
        dropped_reason,
        evidence_ref_map,
        llm_summary_hops=llm_summary_hops,
    )
    if normalized_item is None and retry_instruction:
        answer_attempt_count = 2
        answer_retry_reason = dropped_reason
        retry_content = f"{user_content}\n\n补充要求：{retry_instruction}"
        try:
            raw = client.create_chat_completion_text(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=float(request_timeout),
            ).strip()
            response_type = "str"
            response_dump = safe_response_dump(raw)
            raw_items = _parse_json_items(raw)
            normalized_item, dropped_reason = _normalize_answer(raw_items)
            user_content = retry_content
        except Exception as exc:
            answer_retry_error = str(exc)
    if normalized_item:
        source_override_handler(
            normalized_item,
            chunk_text=source_unit_text or source_chunk_text,
            language_code=language_code,
        )
        normalized_item["question"] = candidate_question
        evidence_usage = normalized_item.get("evidence_usage")
        primary_ids: List[str] = []
        if isinstance(evidence_usage, list):
            for entry in evidence_usage:
                if not isinstance(entry, dict) or entry.get("role") not in {"primary_source", "primary_visual"}:
                    continue
                chunk_id = str(entry.get("chunk_id") or "").strip()
                if chunk_id and chunk_id not in primary_ids:
                    primary_ids.append(chunk_id)
        source_chunks = generation_unit.get("source_chunks") or []
        chunks_by_id = {
            str(chunk.get("chunk_id") or "").strip(): chunk
            for chunk in source_chunks
            if isinstance(chunk, dict) and str(chunk.get("chunk_id") or "").strip()
        }
        primary_chunks = [chunks_by_id[chunk_id] for chunk_id in primary_ids if chunk_id in chunks_by_id]
        if normalized_item and not primary_chunks:
            primary_chunks = [source_chunk] if isinstance(source_chunk, dict) else []
            primary_ids = [
                str(source_chunk.get("chunk_id") or "").strip()
            ] if source_chunk else []
        primary_source = primary_chunks[0] if primary_chunks else source_chunk
        if normalized_item:
            normalized_item["source_chunk_id"] = primary_source.get("chunk_id")
            normalized_item["source_chunk_index"] = primary_source.get("chunk_index")
            normalized_item["source_chunk_title_path"] = primary_source.get("title_path")
            normalized_item["source_chunk_ids"] = primary_ids
            normalized_item["source_chunk_indexes"] = [
                chunk.get("chunk_index") for chunk in primary_chunks if chunk.get("chunk_index") is not None
            ]
            normalized_item["source_chunk_title_paths"] = [
                chunk.get("title_path") for chunk in primary_chunks if str(chunk.get("title_path") or "").strip()
            ]
            normalized_item["evidence_chunk_ids"] = generation_unit.get("evidence_chunk_ids") or []
            normalized_item["qa_generation_unit_id"] = generation_unit.get("qa_generation_unit_id")
            normalized_item["qa_generation_unit_text"] = unit_text
            normalized_item["qa_evaluation_evidence_text"] = _build_qa_evaluation_evidence_text(
                normalized_item.get("evidence_usage"),
                evidence_text_by_ref,
                fallback=unit_text,
            )
            normalized_item["evidence_mode"] = str(
                source_unit_payload.get("evidence_mode") or "text"
            )
            normalized_item["required_image_refs"] = list(
                source_unit_payload.get("required_image_ids") or []
            )
            normalized_item["qa_generation_subject_label"] = str(
                source_unit_payload.get("subject_label") or ""
            )
            normalized_item["qa_generation_required_material_ids"] = list(
                source_unit_payload.get("required_material_ids") or []
            )
            normalized_item["qa_generation_optional_material_ids"] = list(
                source_unit_payload.get("optional_material_ids") or []
            )
            normalized_item["qa_generation_summary_hops"] = [
                dict(hop)
                for hop in source_unit_payload.get("summary_hops") or []
                if isinstance(hop, dict)
            ]
            normalized_item["retrieval_trace"] = generation_unit.get("retrieval_trace") or {}
            normalized_item["source"] = primary_source.get("chunk_id") or normalized_item.get("source")
            normalized_item["text_for_embedding"] = (
                f"{candidate_question} [SEP] {normalized_item.get('answer') or ''}"
            )

    if debug_writer:
        debug_writer(
            {
                "event": "evidence_answer_llm_call",
                "chunk_index": chunk_index,
                "candidate": candidate,
                "generation_unit": {
                    "qa_generation_unit_id": generation_unit.get("qa_generation_unit_id"),
                    "evidence_chunk_ids": generation_unit.get("evidence_chunk_ids"),
                    "retrieval_trace": generation_unit.get("retrieval_trace"),
                    "summary_hops": llm_summary_hops,
                },
                "knowledge_category": fixed_knowledge_category,
                "model": model,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response_type": response_type,
                "response_dump": response_dump,
                "raw_response": raw,
                "initial_raw_response": initial_raw,
                "answer_attempt_count": answer_attempt_count,
                "answer_retry_reason": answer_retry_reason or None,
                "answer_retry_error": answer_retry_error or None,
                "parse_error": parse_error,
                "items_raw_count": len(raw_items),
                "items_validated_count": 1 if normalized_item else 0,
                "dropped_reason": "" if normalized_item else dropped_reason,
            }
        )
    final_reason = "ok" if normalized_item else dropped_reason
    _write_answer_audit(final_reason)
    return normalized_item, final_reason



__all__ = [
    "apply_question_type_plan",
    "build_question_type_plan",
    "call_candidate_question_llm",
    "call_evidence_answer_llm",
    "call_question_editor_llm",
    "call_scenario_planner_llm",
    "normalize_question_type_mode",
    "normalize_question_type_weights",
    "normalize_question_types",
]
