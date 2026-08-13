"""Process-wide BGE cross-encoder reranker service."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from app.core.runtime_paths import MODELS_DIR


DEFAULT_RERANKER_MODEL_NAME = "bge-reranker-v2-m3"


class RerankerService:
    def __init__(self, model_path: Optional[str] = None) -> None:
        configured = str(
            model_path
            or os.getenv("QA_RERANKER_MODEL_PATH")
            or (Path(MODELS_DIR) / DEFAULT_RERANKER_MODEL_NAME)
        ).strip()
        self.model_path = Path(configured).expanduser().resolve()
        self.device = str(os.getenv("QA_RERANKER_DEVICE") or "auto").strip().lower()
        self.batch_size = max(1, int(os.getenv("QA_RERANKER_BATCH_SIZE") or 8))
        self.max_length = max(64, int(os.getenv("QA_RERANKER_MAX_LENGTH") or 512))
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._resolved_device = "cpu"

    def _validate_model_files(self) -> None:
        required = ("config.json", "tokenizer_config.json", "model.safetensors")
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise RuntimeError(
                "BGE reranker model is incomplete at "
                f"{self.model_path}; missing: {', '.join(missing)}"
            )

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._validate_model_files()
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception as exc:
                raise RuntimeError(
                    "BGE reranker requires torch and transformers in the QA runtime"
                ) from exc
            resolved_device = self.device
            if resolved_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    f"QA_RERANKER_DEVICE={resolved_device} requested CUDA, but CUDA is unavailable"
                )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                )
                model.eval()
                model.to(resolved_device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load BGE reranker from {self.model_path}: {exc}"
                ) from exc
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._resolved_device = resolved_device

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        clean_query = str(query or "").strip()
        clean_passages = [str(passage or "").strip() for passage in passages]
        if not clean_query:
            raise ValueError("reranker query cannot be empty")
        if any(not passage for passage in clean_passages):
            raise ValueError("reranker passages cannot be empty")
        if not clean_passages:
            return []
        self.load()
        scores: List[float] = []
        assert self._torch is not None and self._tokenizer is not None and self._model is not None
        with self._inference_lock, self._torch.inference_mode():
            for start in range(0, len(clean_passages), self.batch_size):
                batch = clean_passages[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [[clean_query, passage] for passage in batch],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                encoded = {
                    key: value.to(self._resolved_device)
                    for key, value in encoded.items()
                }
                logits = self._model(**encoded, return_dict=True).logits.view(-1).float()
                scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores

    def rank(self, query: str, pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        identifiers = [str(identifier) for identifier, _passage in pairs]
        scores = self.score(query, [passage for _identifier, passage in pairs])
        ranked = list(zip(identifiers, scores))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def close(self) -> None:
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            if self._torch is not None and self._resolved_device.startswith("cuda"):
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass
            self._torch = None


_RERANKER_SERVICE = RerankerService()


def get_reranker_service() -> RerankerService:
    return _RERANKER_SERVICE


__all__ = [
    "DEFAULT_RERANKER_MODEL_NAME",
    "RerankerService",
    "get_reranker_service",
]
