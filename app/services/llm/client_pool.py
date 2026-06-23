"""Shared LLM/VLM client pool.

The pool follows the dw VLM client contract:
`create_chat_completion_text`, `public_signature`, `recreate`, and `close`.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .vlm_client import VLMClientConfig, create_vlm_client


class LLMClientProtocol(Protocol):
    def create_chat_completion_text(
        self,
        *,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        ...

    def public_signature(self) -> Dict[str, Any]:
        ...

    def recreate(self, reason: str) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class LLMClientConfig:
    api_base: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_type: Optional[str] = None
    model_version: Optional[str] = None
    timeout_seconds: float = 180.0

    def to_vlm_config(self) -> VLMClientConfig:
        return VLMClientConfig.from_values(
            api_base=self.api_base,
            model_name=self.model_name,
            api_key=self.api_key,
            api_type=self.api_type,
            model_version=self.model_version,
            timeout_seconds=self.timeout_seconds,
        )


def _secret_fingerprint(value: Optional[str]) -> str:
    if value is None:
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _pool_key(config: VLMClientConfig) -> Tuple[Any, ...]:
    return (
        config.api_type,
        config.base_url,
        config.model_name,
        config.model_version,
        _secret_fingerprint(config.api_key),
        config.timeout_seconds,
        config.stream,
        config.max_concurrent_requests,
        config.min_interval_seconds,
        config.top_p,
        config.presence_penalty,
    )


class LLMClientPool:
    def __init__(self, max_size: int = 8) -> None:
        self.max_size = max(1, int(max_size))
        self._lock = threading.RLock()
        self._clients: "OrderedDict[Tuple[Any, ...], LLMClientProtocol]" = OrderedDict()

    def get_client(self, config: LLMClientConfig) -> LLMClientProtocol:
        vlm_config = config.to_vlm_config()
        key = _pool_key(vlm_config)
        with self._lock:
            existing = self._clients.pop(key, None)
            if existing is not None:
                self._clients[key] = existing
                return existing

            client = create_vlm_client(vlm_config)
            self._clients[key] = client
            while len(self._clients) > self.max_size:
                _old_key, old_client = self._clients.popitem(last=False)
                try:
                    old_client.close()
                except Exception:
                    pass
            return client

    def close_all(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass


_DEFAULT_POOL = LLMClientPool()


def get_llm_client_pool() -> LLMClientPool:
    return _DEFAULT_POOL

