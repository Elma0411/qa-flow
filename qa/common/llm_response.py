# 文件作用：安全提取和序列化大模型响应内容。
# 关联说明：被 LLM 调用链路复用，统一处理 OpenAI/兼容接口响应。

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def extract_first_choice_content(resp: Any) -> str:
    """
    Best-effort extract the first choice message content from an OpenAI-compatible response.

    Supports:
    - OpenAI Python v1 objects (resp.choices[0].message.content)
    - dict-like responses (resp["choices"][0]["message"]["content"] / ["text"])
    - plain string responses (already the content)
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp

    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                text = c0.get("text")
                if isinstance(text, str):
                    return text
        # Some gateways return {"content": "..."} directly.
        if isinstance(resp.get("content"), str):
            return resp.get("content")  # type: ignore[return-value]
        return ""

    # OpenAI v1: resp.choices[0].message.content
    try:
        choices = getattr(resp, "choices", None)
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            # message may be dict or object
            msg = getattr(c0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
                # message dict
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
            # completion text fallback
            text = getattr(c0, "text", None)
            if isinstance(text, str):
                return text
            if isinstance(c0, dict) and isinstance(c0.get("text"), str):
                return c0["text"]
    except Exception:
        return ""
    return ""


def safe_response_dump(resp: Any, *, max_chars: int = 20000) -> Any:
    """
    Convert response to JSON-serializable payload for debugging.
    Large payloads are truncated to keep logs manageable.
    """
    if resp is None:
        return None
    if isinstance(resp, (int, float, bool)):
        return resp
    if isinstance(resp, str):
        s = resp
        return s if len(s) <= max_chars else (s[:max_chars] + " ...<truncated>")
    if isinstance(resp, dict):
        # Try to JSON-dump and truncate.
        try:
            s = json.dumps(resp, ensure_ascii=False)
            if len(s) <= max_chars:
                return resp
            return {"_truncated": True, "preview": s[:max_chars] + " ...<truncated>"}
        except Exception:
            return {"_repr": _truncate(repr(resp), max_chars)}
    if isinstance(resp, list):
        try:
            s = json.dumps(resp, ensure_ascii=False)
            if len(s) <= max_chars:
                return resp
            return {"_truncated": True, "preview": s[:max_chars] + " ...<truncated>"}
        except Exception:
            return {"_repr": _truncate(repr(resp), max_chars)}

    # OpenAI v1 responses are pydantic models; prefer model_dump when available.
    if hasattr(resp, "model_dump"):
        try:
            data = resp.model_dump()
            return safe_response_dump(data, max_chars=max_chars)
        except Exception:
            pass

    return {"_type": type(resp).__name__, "_repr": _truncate(repr(resp), max_chars)}


def _truncate(text: str, max_chars: int) -> str:
    s = str(text or "")
    return s if len(s) <= max_chars else (s[:max_chars] + " ...<truncated>")


__all__ = ["extract_first_choice_content", "safe_response_dump"]

