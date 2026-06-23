"""Chunk summary generation for image context enrichment."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.llm import LLMClientConfig

from .models import ChunkContext


def normalize_summary_mode(value: str) -> str:
    mode = str(value or "lightweight").strip().lower()
    return "llm" if mode == "llm" else "lightweight"


def _excerpt(text: str, limit: int = 900) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    half = max(100, limit // 2)
    return value[:half] + "\n...\n" + value[-half:]


class ChunkSummaryService:
    def __init__(
        self,
        *,
        mode: str = "lightweight",
        llm_config: Optional["LLMClientConfig"] = None,
        max_workers: int = 4,
    ) -> None:
        self.mode = normalize_summary_mode(mode)
        self.llm_config = llm_config
        self.max_workers = max(1, int(max_workers or 1))

    def summarize(self, chunks: Iterable[ChunkContext]) -> List[ChunkContext]:
        chunk_list = list(chunks)
        if self.mode != "llm" or self.llm_config is None:
            return [self._with_lightweight_summary(chunk) for chunk in chunk_list]

        summaries: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(chunk_list)))) as executor:
            future_map = {executor.submit(self._summarize_with_llm, chunk): chunk for chunk in chunk_list}
            for future in as_completed(future_map):
                chunk = future_map[future]
                try:
                    summaries[chunk.chunk_index] = future.result()
                except Exception:
                    summaries[chunk.chunk_index] = self._build_lightweight_summary(chunk)

        return [
            ChunkContext(
                chunk_index=chunk.chunk_index,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                title_path=chunk.title_path,
                path_summary=chunk.path_summary,
                summary=summaries.get(chunk.chunk_index) or self._build_lightweight_summary(chunk),
            )
            for chunk in chunk_list
        ]

    def _with_lightweight_summary(self, chunk: ChunkContext) -> ChunkContext:
        return ChunkContext(
            chunk_index=chunk.chunk_index,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            title_path=chunk.title_path,
            path_summary=chunk.path_summary,
            summary=self._build_lightweight_summary(chunk),
        )

    def _build_lightweight_summary(self, chunk: ChunkContext) -> str:
        parts = []
        if chunk.title_path:
            parts.append(f"标题路径：{chunk.title_path}")
        if chunk.path_summary:
            parts.append(f"路径摘要：{chunk.path_summary}")
        parts.append(f"内容摘录：{_excerpt(chunk.text)}")
        return "\n".join(part for part in parts if part.strip()).strip()

    def _summarize_with_llm(self, chunk: ChunkContext) -> str:
        from app.services.llm import get_llm_client_pool

        client = get_llm_client_pool().get_client(self.llm_config)
        user_payload = {
            "chunk_id": chunk.chunk_id,
            "title_path": chunk.title_path,
            "path_summary": chunk.path_summary,
            "text": _excerpt(chunk.text, limit=2600),
        }
        raw = client.create_chat_completion_text(
            model=self.llm_config.model_name,
            temperature=0.1,
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "你是文档 chunk 摘要助手。请输出一段简洁中文摘要，用于帮助图片解析判断上下文。",
                },
                {
                    "role": "user",
                    "content": "请概括以下 chunk 的主题、关键实体、约束、结论和与图片可能相关的信息：\n"
                    + json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        )
        return raw.strip() or self._build_lightweight_summary(chunk)
