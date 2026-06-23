"""Judge whether an image description fits its original chunk context."""

from __future__ import annotations

import json
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.llm import LLMClientConfig

from .models import ImageAnchorContext, PlacementDecision


def normalize_fit_score(value: float) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.65
    return max(0.0, min(1.0, score))


def parse_placement_response(image_id: str, raw_response: str, min_score: float) -> PlacementDecision:
    raw = str(raw_response or "").strip()
    payload = None
    try:
        payload = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return PlacementDecision(
            image_id=image_id,
            accepted=False,
            score=0.0,
            reason="placement judge returned non-json response",
            raw_response=raw,
            error="invalid_json",
        )
    accepted = bool(payload.get("accepted"))
    score = normalize_fit_score(payload.get("score", 0.0))
    reason = str(payload.get("reason") or "").strip()
    return PlacementDecision(
        image_id=image_id,
        accepted=accepted and score >= min_score,
        score=score,
        reason=reason or ("accepted" if accepted else "rejected"),
        raw_response=raw,
    )


class ImagePlacementJudge:
    def __init__(
        self,
        *,
        enabled: bool = True,
        min_score: float = 0.65,
        llm_config: Optional["LLMClientConfig"] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_score = normalize_fit_score(min_score)
        self.llm_config = llm_config

    def judge(self, *, anchor: ImageAnchorContext, description: str) -> PlacementDecision:
        image_id = anchor.image_id
        clean_description = str(description or "").strip()
        if not clean_description:
            return PlacementDecision(image_id=image_id, accepted=False, score=0.0, reason="empty description")
        if not self.enabled:
            return PlacementDecision(image_id=image_id, accepted=True, score=1.0, reason="fit check disabled")
        if self.llm_config is None:
            return PlacementDecision(
                image_id=image_id,
                accepted=False,
                score=0.0,
                reason="fit check enabled but no llm config was supplied",
                error="missing_llm_config",
            )

        from app.services.llm import get_llm_client_pool

        client = get_llm_client_pool().get_client(self.llm_config)
        prompt = (
            "请判断图片解析结果是否适合回填到原图片所在 chunk 的位置。"
            "只输出 JSON，格式为 {\"accepted\": true/false, \"score\": 0到1, \"reason\": \"原因\"}。\n\n"
            f"chunk 标题路径：{anchor.chunk.title_path}\n"
            f"chunk 摘要：{anchor.chunk.summary}\n\n"
            f"chunk 正文：{anchor.chunk.text[:3000]}\n\n"
            f"图片解析结果：{clean_description}"
        )
        try:
            raw = client.create_chat_completion_text(
                model=self.llm_config.model_name,
                temperature=0.0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": "你是严谨的文档上下文一致性判断器。"},
                    {"role": "user", "content": prompt},
                ],
            )
            return parse_placement_response(image_id, raw, self.min_score)
        except Exception as exc:
            return PlacementDecision(
                image_id=image_id,
                accepted=False,
                score=0.0,
                reason=f"placement judge failed: {exc}",
                error=str(exc),
            )
