import base64
import json
import math
import os
import re
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError
except ImportError:  # pragma: no cover - optional until API requests are made
    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

try:
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover - optional local inference dependency
    AutoProcessor = None

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional local inference dependency
    LLM = None
    SamplingParams = None

from app.services.document_processing.ocr_processor.ocr_models import ImageInfo, OCRResult

from .classification_client import (
    ClassificationResult,
    classify_images_batch,
    default_classification_result,
)
from .image_models import AnalysisResult, ImageDescription
from .prompt_registry import (
    REWRITE_SYSTEM_PROMPT,
    TABLE_REWRITE_PROMPT_TEMPLATE,
    get_prompt_config,
)
from app.services.llm.vlm_client import (
    VLMClientConfig,
    VLMConnectionError,
    VLMHTTPStatusError,
    create_vlm_client,
    normalize_vlm_api_type,
)

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - Pillow may arrive transitively in deployment
    Image = None
    ImageOps = None


warnings.filterwarnings("ignore")


class _PrintLogger:
    @staticmethod
    def debug(message: str, *args, exc_info: bool = False):
        text = message % args if args else message
        print(f"[DEBUG] {text}", flush=True)
        if exc_info:
            traceback.print_exc()

    @staticmethod
    def info(message: str, *args):
        text = message % args if args else message
        print(f"[INFO] {text}", flush=True)

    @staticmethod
    def warning(message: str, *args):
        text = message % args if args else message
        print(f"[WARNING] {text}", flush=True)

    @staticmethod
    def error(message: str, *args):
        text = message % args if args else message
        print(f"[ERROR] {text}", flush=True)

    @staticmethod
    def exception(message: str, *args):
        text = message % args if args else message
        print(f"[ERROR] {text}", flush=True)
        traceback.print_exc()


logger = _PrintLogger()

EMPTY_CONTEXT_PLACEHOLDER = "（无可用上下文）"
TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*[:\-+| ]{3,}\s*$")
ASCII_BOX_PATTERN = re.compile(r"^\s*\+[-+]+\+\s*$")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid integer for %s=%r, using default %s", name, raw, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r, using default %s", name, raw, default)
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_confidence_threshold(value: float) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError("classification_confidence_threshold must be a number between 0.0 and 1.0")

    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("classification_confidence_threshold must be between 0.0 and 1.0")
    return threshold


@dataclass(frozen=True)
class APIRequestConfig:
    timeout_seconds: float = _get_env_float("VLM_API_TIMEOUT_SECONDS", 180.0)
    max_retries: int = _get_env_int("VLM_API_MAX_RETRIES", 4)
    retry_base_delay_seconds: float = _get_env_float("VLM_API_RETRY_BASE_DELAY_SECONDS", 3.0)
    server_error_cooldown_seconds: float = _get_env_float(
        "VLM_API_SERVER_ERROR_COOLDOWN_SECONDS",
        15.0,
    )
    inter_request_delay_seconds: float = _get_env_float(
        "VLM_API_INTER_REQUEST_DELAY_SECONDS",
        0.0,
    )
    max_tokens: int = _get_env_int("VLM_API_MAX_TOKENS", 1536)
    temperature: float = _get_env_float("VLM_API_TEMPERATURE", 0.2)
    max_context_chars: int = _get_env_int("VLM_API_MAX_CONTEXT_CHARS", 1200)
    max_image_side: int = _get_env_int("VLM_API_MAX_IMAGE_SIDE", 1600)
    jpeg_quality: int = _get_env_int("VLM_API_JPEG_QUALITY", 90)
    recreate_client_on_error: bool = _get_env_bool("VLM_API_RECREATE_CLIENT_ON_ERROR", True)


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _truncate_context_tail(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return "..." + value[-max_chars:]


def _truncate_context_head(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _normalize_context_text(text: str) -> str:
    value = (text or "").strip()
    return value or EMPTY_CONTEXT_PLACEHOLDER


def _preview_text(text: str, limit: int = 160) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _safe_file_size(file_path: str) -> Optional[int]:
    try:
        return Path(file_path).stat().st_size
    except OSError:
        return None


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as file_obj:
        data = base64.b64encode(file_obj.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()[1:]
    mime = ext if ext in ("png", "bmp") else "jpeg"
    return f"data:image/{mime};base64,{data}"


def optimize_image_to_base64(
    image_path: str,
    *,
    max_image_side: int,
    jpeg_quality: int,
) -> str:
    if Image is None or ImageOps is None or max_image_side <= 0:
        return image_to_base64(image_path)

    try:
        with Image.open(image_path) as img:
            working = ImageOps.exif_transpose(img)
            width, height = working.size
            longest_edge = max(width, height)
            if longest_edge > max_image_side:
                scale = max_image_side / float(longest_edge)
                new_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                working = working.resize(new_size, resampling)

            if working.mode not in ("RGB", "L"):
                working = working.convert("RGB")

            output = BytesIO()
            save_kwargs = {"format": "JPEG", "quality": _clamp_int(jpeg_quality, 50, 100)}
            if working.mode == "L":
                save_kwargs["format"] = "PNG"
            else:
                working = working.convert("RGB")
            working.save(output, **save_kwargs)
            data = base64.b64encode(output.getvalue()).decode("utf-8")
            mime = "png" if save_kwargs["format"] == "PNG" else "jpeg"
            return f"data:image/{mime};base64,{data}"
    except Exception as exc:
        logger.warning("Failed to optimize image %s, falling back to raw bytes: %s", image_path, exc)
        return image_to_base64(image_path)


def is_invalid_response(response: str) -> bool:
    if not (response and response.strip()):
        return True

    words = response.strip().split()
    if len(words) < 3:
        return False

    word_count: Dict[str, int] = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1

    return max(word_count.values()) / len(words) > 0.7


def contains_structured_table_output(response: str) -> bool:
    if not response:
        return False

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return False

    pipe_like_lines = sum(
        1 for line in lines if line.count("|") >= 2 or TABLE_LINE_PATTERN.match(line)
    )
    separator_like_lines = sum(
        1 for line in lines if TABLE_SEPARATOR_PATTERN.match(line) or ASCII_BOX_PATTERN.match(line)
    )
    tab_like_lines = sum(1 for line in lines if line.count("\t") >= 2)

    return pipe_like_lines >= 1 or separator_like_lines >= 1 or tab_like_lines >= 1


def render_prompt_text(prompt_template: str, a_context: str, b_context: str) -> str:
    return prompt_template.format(
        a_context=_normalize_context_text(a_context),
        b_context=_normalize_context_text(b_context),
    )


def build_table_rewrite_prompt(a_context: str, b_context: str, raw_text: str) -> str:
    return TABLE_REWRITE_PROMPT_TEMPLATE.format(
        a_context=_normalize_context_text(a_context),
        b_context=_normalize_context_text(b_context),
        raw_text=(raw_text or "").strip(),
    )


def prepare_api_prompt(
    a_context: str,
    b_context: str,
    image_urls: List[str],
    prompt_template: str,
) -> List[Dict]:
    content = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    content.append(
        {
            "type": "text",
            "text": prompt_template.format(a_context=a_context, b_context=b_context),
        }
    )

    return [
        {
            "role": "system",
            "content": "你是专业的多模态文档解析助手。请只输出可直接插入文档的正文内容。",
        },
        {"role": "user", "content": content},
    ]


def prepare_vllm_inputs(messages, processor):
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {},
        "mm_processor_kwargs": {},
    }


def build_api_messages(
    system_prompt: str,
    prompt_text: str,
    image_urls: Optional[List[str]] = None,
) -> List[Dict]:
    content = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in (image_urls or [])
    ]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def build_vllm_messages(
    system_prompt: str,
    prompt_text: str,
    image_paths: Optional[List[str]] = None,
) -> List[Dict]:
    content = [
        {"type": "image", "image": str(image_path)}
        for image_path in (image_paths or [])
    ]
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _normalize_image_info(img_info: Any) -> Dict[str, Any]:
    if isinstance(img_info, ImageInfo):
        return img_info.to_dict()
    if isinstance(img_info, dict):
        return img_info
    raise TypeError(f"Unsupported image info type: {type(img_info)}")


def _resolve_image_path(file_path: str, ocr_output_dir: str) -> str:
    path = Path(file_path)
    if path.is_absolute():
        return str(path)

    if path.exists():
        return str(path)

    return str(Path(ocr_output_dir) / path)


def _extract_message_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
            else:
                text_value = getattr(item, "text", None)
            if text_value:
                text_parts.append(str(text_value).strip())
        return "\n".join(part for part in text_parts if part).strip()
    return str(content or "").strip()


class BatchVLMDocParser:
    def __init__(
        self,
        a_contexts: List[str],
        b_contexts: List[str],
        images_list: List[List[str]],
        image_ids: List[str],
        pdf_name: str,
        output_dir: str,
        model_path: Optional[str] = None,
        num_gpus: int = 2,
        use_api: bool = True,
        api_base: str = None,
        model_name: str = None,
        api_key: str = None,
        vlm_api_type: str = "openai",
        model_version: str = None,
        vlm_client: Optional[Any] = None,
        enable_classification: bool = False,
        classifier_api_base: str = None,
        classifier_timeout: int = 30,
        classification_confidence_threshold: float = 0.0,
    ):
        assert len({len(a_contexts), len(b_contexts), len(images_list), len(image_ids)}) == 1

        self.a_contexts = a_contexts
        self.b_contexts = b_contexts
        self.images_list = images_list
        self.image_ids = image_ids
        self.pdf_name = pdf_name
        self.output_dir = output_dir
        self.use_api = use_api
        self.request_config = APIRequestConfig()
        self.api_client_config = None
        self.vlm_api_type = "local_vllm"
        self.api_base = ""
        self.model_name = str(model_name or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model_version = model_version or ""
        if self.use_api:
            self.vlm_api_type = normalize_vlm_api_type(vlm_api_type)
            if vlm_client is not None and getattr(vlm_client, "config", None) is not None:
                self.api_client_config = vlm_client.config
            else:
                self.api_client_config = VLMClientConfig.from_values(
                    api_base=api_base,
                    model_name=model_name,
                    api_key=api_key,
                    api_type=self.vlm_api_type,
                    model_version=model_version,
                    timeout_seconds=self.request_config.timeout_seconds,
                )
            self.vlm_api_type = self.api_client_config.api_type
            self.api_base = self.api_client_config.base_url
            self.model_name = self.api_client_config.model_name
            self.api_key = self.api_client_config.api_key
            self.model_version = self.api_client_config.model_version
        self.enable_classification = enable_classification
        self.classifier_api_base = classifier_api_base
        self.classifier_timeout = classifier_timeout
        self.classification_confidence_threshold = _normalize_confidence_threshold(
            classification_confidence_threshold
        )

        self.current_index = 0
        self.raw_results: Dict[int, str] = {}
        self.image_types: Dict[int, str] = {}
        self.prompt_keys: Dict[int, str] = {}
        self.error_info: Dict[int, str] = {}
        self.classification_results: Dict[int, ClassificationResult] = {}
        self.encoded_image_cache: Dict[str, str] = {}
        self._owns_api_client = False

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        if self.use_api:
            if vlm_client is None:
                self.client = self._create_api_client()
                self._owns_api_client = True
            else:
                self.client = vlm_client
            logger.info(
                "Using VLM API type=%s base=%s model=%s timeout=%ss retries=%s max_tokens=%s max_image_side=%s shared_client=%s",
                self.vlm_api_type,
                self.api_base,
                self.model_name,
                self.request_config.timeout_seconds,
                self.request_config.max_retries,
                self.request_config.max_tokens,
                self.request_config.max_image_side,
                not self._owns_api_client,
            )
        else:
            if AutoProcessor is None or LLM is None or SamplingParams is None:
                raise ImportError(
                    "Local VLLM mode requires optional dependencies: transformers and vllm"
                )
            resolved_model_path = str(model_path or "").strip()
            if not resolved_model_path:
                raise ValueError("Local VLM model path is not configured; pass model_path when use_api=False")
            logger.info("Using local vLLM model path=%s", resolved_model_path)
            self.processor = AutoProcessor.from_pretrained(
                resolved_model_path,
                trust_remote_code=True,
            )
            self.llm = LLM(
                model=resolved_model_path,
                tensor_parallel_size=num_gpus,
                max_model_len=8192,
                mm_encoder_tp_mode="data",
                seed=0,
            )
            self.sampling_params = SamplingParams(
                temperature=self.request_config.temperature,
                max_tokens=self.request_config.max_tokens,
                top_p=0.9,
            )

    def _create_api_client(self) -> Any:
        return create_vlm_client(self.api_client_config)

    def _close_api_client(self) -> None:
        if not self._owns_api_client:
            return
        client = getattr(self, "client", None)
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close VLM API client", exc_info=True)

    def _recreate_api_client(self, reason: str) -> None:
        if not self.use_api:
            return
        if self._owns_api_client:
            self._close_api_client()
            self.client = self._create_api_client()
            logger.warning("Recreated VLM API client: %s", reason)
            return

        recreate = getattr(self.client, "recreate", None)
        if callable(recreate):
            recreate(reason)

    def _get_encoded_image_url(self, image_path: str) -> str:
        cached = self.encoded_image_cache.get(image_path)
        if cached is not None:
            return cached

        encoded = optimize_image_to_base64(
            image_path,
            max_image_side=self.request_config.max_image_side,
            jpeg_quality=self.request_config.jpeg_quality,
        )
        self.encoded_image_cache[image_path] = encoded
        logger.info(
            "Prepared image payload path=%s original_bytes=%s encoded_chars=%s",
            image_path,
            _safe_file_size(image_path),
            len(encoded),
        )
        return encoded

    def _classify_images(self):
        default_results = {
            idx: default_classification_result(image_id)
            for idx, image_id in enumerate(self.image_ids)
        }

        if not self.enable_classification:
            logger.info("Image classification disabled, using unified prompt for all %s images", len(self.image_ids))
            self.classification_results = default_results
            return

        logger.info(
            "Starting image classification for %s images via %s threshold=%.3f",
            len(self.image_ids),
            self.classifier_api_base or "default classifier endpoint",
            self.classification_confidence_threshold,
        )

        valid_indices: List[int] = []
        valid_ids: List[str] = []
        valid_paths: List[str] = []

        for idx, image_id in enumerate(self.image_ids):
            images = self.images_list[idx]
            if not images:
                default_results[idx] = default_classification_result(image_id, "missing image path")
                continue

            valid_indices.append(idx)
            valid_ids.append(image_id)
            valid_paths.append(images[0])

        service_results = classify_images_batch(
            image_ids=valid_ids,
            image_paths=valid_paths,
            api_base=self.classifier_api_base,
            timeout=self.classifier_timeout,
        )
        for idx, result in zip(valid_indices, service_results):
            default_results[idx] = result
            if result.status != "success":
                logger.warning(
                    "Image classification failed for %s, fallback to unified prompt: %s",
                    result.image_id,
                    result.error_message,
                )
            else:
                logger.info(
                    "Image classification result for %s: %s (%s, %.3f)",
                    result.image_id,
                    result.display_name,
                    result.category_key,
                    result.confidence,
                )

        self.classification_results = default_results

    def _resolve_prompt(self, index: int):
        classification = self.classification_results.get(
            index,
            default_classification_result(self.image_ids[index]),
        )
        category_key = None
        if classification.status == "success":
            confidence = float(classification.confidence or 0.0)
            if not self.enable_classification or confidence >= self.classification_confidence_threshold:
                category_key = classification.category_key
            else:
                logger.info(
                    "Image classification confidence below threshold for %s: %.3f < %.3f, using unified prompt",
                    classification.image_id,
                    confidence,
                    self.classification_confidence_threshold,
                )
        prompt_config = get_prompt_config(category_key, enable_classification=self.enable_classification)
        self.image_types[index] = prompt_config.display_name
        self.prompt_keys[index] = prompt_config.category_key
        return classification, prompt_config

    def _build_retry_delay(self, attempt_index: int, is_server_error: bool) -> float:
        delay = self.request_config.retry_base_delay_seconds * (attempt_index + 1)
        if is_server_error:
            delay += self.request_config.server_error_cooldown_seconds
        return max(0.0, delay)

    def _get_prompt_contexts(self, index: int) -> Tuple[str, str]:
        return (
            _truncate_context_tail(
                self.a_contexts[index],
                self.request_config.max_context_chars,
            ),
            _truncate_context_head(
                self.b_contexts[index],
                self.request_config.max_context_chars,
            ),
        )

    def _rewrite_structured_output_api(self, image_index: int, raw_text: str) -> str:
        a_context, b_context = self._get_prompt_contexts(image_index)
        logger.info(
            "Starting structured-output rewrite for %s raw_chars=%s",
            self.image_ids[image_index],
            len(raw_text or ""),
        )
        messages = build_api_messages(
            REWRITE_SYSTEM_PROMPT,
            build_table_rewrite_prompt(a_context, b_context, raw_text),
        )
        rewritten_text, error_message = self._call_api_with_retry(image_index, messages)
        if rewritten_text and not is_invalid_response(rewritten_text):
            logger.info(
                "Structured-output rewrite succeeded for %s rewritten_chars=%s preview=%r",
                self.image_ids[image_index],
                len(rewritten_text),
                _preview_text(rewritten_text),
            )
            return rewritten_text.strip()

        logger.warning(
            "Structured-output rewrite failed for %s: %s",
            self.image_ids[image_index],
            error_message or "invalid rewrite response",
        )
        return raw_text

    def _rewrite_structured_output_vllm(self, image_index: int, raw_text: str) -> str:
        a_context, b_context = self._get_prompt_contexts(image_index)
        logger.info(
            "Starting local structured-output rewrite for %s raw_chars=%s",
            self.image_ids[image_index],
            len(raw_text or ""),
        )
        messages = build_vllm_messages(
            REWRITE_SYSTEM_PROMPT,
            build_table_rewrite_prompt(a_context, b_context, raw_text),
        )
        try:
            rewrite_start = time.perf_counter()
            outputs = self.llm.generate(
                [prepare_vllm_inputs(messages, self.processor)],
                self.sampling_params,
            )
            rewritten_text = outputs[0].outputs[0].text.strip()
            if rewritten_text and not is_invalid_response(rewritten_text):
                logger.info(
                    "Local structured-output rewrite succeeded for %s in %.2fs rewritten_chars=%s preview=%r",
                    self.image_ids[image_index],
                    time.perf_counter() - rewrite_start,
                    len(rewritten_text),
                    _preview_text(rewritten_text),
                )
                return rewritten_text
        except Exception as exc:
            logger.warning(
                "Local rewrite failed for %s: %s",
                self.image_ids[image_index],
                exc,
            )

        return raw_text

    def _normalize_generated_text(self, image_index: int, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""

        if contains_structured_table_output(text):
            logger.info(
                "Detected structured table output for %s, applying rewrite pass",
                self.image_ids[image_index],
            )
            text = (
                self._rewrite_structured_output_api(image_index, text)
                if self.use_api
                else self._rewrite_structured_output_vllm(image_index, text)
            ).strip()

        return text

    def _call_api_with_retry(self, image_index: int, messages: List[Dict]) -> Tuple[Optional[str], str]:
        image_id = self.image_ids[image_index]
        last_error = "unknown api error"

        for attempt in range(self.request_config.max_retries):
            if attempt > 0 and self.request_config.recreate_client_on_error:
                self._recreate_api_client(f"retry attempt {attempt + 1} for {image_id}")

            try:
                attempt_start = time.perf_counter()
                logger.info(
                    "Calling VLM API for %s attempt=%s/%s",
                    image_id,
                    attempt + 1,
                    self.request_config.max_retries,
                )
                response_text = self.client.create_chat_completion_text(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.request_config.temperature,
                    max_tokens=self.request_config.max_tokens,
                )
                elapsed = time.perf_counter() - attempt_start
                logger.info(
                    "VLM API succeeded for %s attempt=%s/%s in %.2fs response_chars=%s preview=%r",
                    image_id,
                    attempt + 1,
                    self.request_config.max_retries,
                    elapsed,
                    len(response_text),
                    _preview_text(response_text),
                )
                return response_text, ""
            except (APIStatusError, VLMHTTPStatusError) as exc:
                status_code = getattr(exc, "status_code", None)
                is_server_error = status_code is not None and int(status_code) >= 500
                last_error = f"http {status_code}: {exc}"
                logger.warning(
                    "VLM API status error for %s (attempt %s/%s): %s",
                    image_id,
                    attempt + 1,
                    self.request_config.max_retries,
                    exc,
                )
                if attempt < self.request_config.max_retries - 1:
                    delay = self._build_retry_delay(attempt, is_server_error)
                    logger.info("Retrying %s after %.2fs due to API status error", image_id, delay)
                    time.sleep(delay)
            except (APIConnectionError, APITimeoutError, VLMConnectionError) as exc:
                last_error = f"connection error: {exc}"
                logger.warning(
                    "VLM API connection error for %s (attempt %s/%s): %s",
                    image_id,
                    attempt + 1,
                    self.request_config.max_retries,
                    exc,
                )
                if attempt < self.request_config.max_retries - 1:
                    delay = self._build_retry_delay(attempt, True)
                    logger.info("Retrying %s after %.2fs due to connection error", image_id, delay)
                    time.sleep(delay)
            except Exception as exc:
                last_error = f"unexpected error: {exc}"
                logger.warning(
                    "Unexpected VLM API error for %s (attempt %s/%s): %s",
                    image_id,
                    attempt + 1,
                    self.request_config.max_retries,
                    exc,
                )
                if attempt < self.request_config.max_retries - 1:
                    delay = self._build_retry_delay(attempt, False)
                    logger.info("Retrying %s after %.2fs due to unexpected error", image_id, delay)
                    time.sleep(delay)

        return None, last_error

    def _batch_inference_api(self, start: int, end: int):
        for i in range(start, end):
            self.error_info[i] = ""
            try:
                _, prompt_config = self._resolve_prompt(i)
                a_context, b_context = self._get_prompt_contexts(i)
                image_paths = self.images_list[i]
                logger.info(
                    "Starting API image analysis image_id=%s prompt=%s image_type=%s images=%s context_before_chars=%s context_after_chars=%s first_image=%s",
                    self.image_ids[i],
                    prompt_config.category_key,
                    prompt_config.display_name,
                    len(image_paths),
                    len(a_context),
                    len(b_context),
                    image_paths[0] if image_paths else "",
                )
                image_start = time.perf_counter()
                messages = build_api_messages(
                    prompt_config.system_prompt,
                    render_prompt_text(prompt_config.prompt_template, a_context, b_context),
                    image_urls=[
                        self._get_encoded_image_url(image_path)
                        for image_path in image_paths
                    ],
                )
                raw_text, error_message = self._call_api_with_retry(i, messages)
                if not raw_text:
                    self.error_info[i] = error_message or "api call failed"
                    self.raw_results[i] = ""
                    logger.error(
                        "API image analysis failed image_id=%s prompt=%s error=%s elapsed=%.2fs",
                        self.image_ids[i],
                        prompt_config.category_key,
                        self.error_info[i],
                        time.perf_counter() - image_start,
                    )
                    continue

                raw_text = self._normalize_generated_text(i, raw_text)
                if is_invalid_response(raw_text):
                    self.error_info[i] = "model returned invalid content (empty or repetitive)"
                    self.raw_results[i] = ""
                    logger.error(
                        "API image analysis produced invalid output image_id=%s prompt=%s elapsed=%.2fs",
                        self.image_ids[i],
                        prompt_config.category_key,
                        time.perf_counter() - image_start,
                    )
                else:
                    self.raw_results[i] = raw_text
                    logger.info(
                        "Completed API image analysis image_id=%s prompt=%s elapsed=%.2fs output_chars=%s preview=%r",
                        self.image_ids[i],
                        prompt_config.category_key,
                        time.perf_counter() - image_start,
                        len(raw_text),
                        _preview_text(raw_text),
                    )

                if self.request_config.inter_request_delay_seconds > 0:
                    logger.info(
                        "Sleeping %.2fs before next image request",
                        self.request_config.inter_request_delay_seconds,
                    )
                    time.sleep(self.request_config.inter_request_delay_seconds)
            except Exception as exc:
                self.error_info[i] = f"processing failed: {exc}"
                self.raw_results[i] = ""
                logger.exception("Image analysis failed for %s", self.image_ids[i])

    def _batch_inference_vllm(self, start: int, end: int):
        messages_list = []
        indices = []

        for i in range(start, end):
            self.error_info[i] = ""
            _, prompt_config = self._resolve_prompt(i)
            a_context, b_context = self._get_prompt_contexts(i)
            logger.info(
                "Queueing local VLLM image analysis image_id=%s prompt=%s image_type=%s images=%s context_before_chars=%s context_after_chars=%s first_image=%s",
                self.image_ids[i],
                prompt_config.category_key,
                prompt_config.display_name,
                len(self.images_list[i]),
                len(a_context),
                len(b_context),
                self.images_list[i][0] if self.images_list[i] else "",
            )
            messages_list.append(
                build_vllm_messages(
                    prompt_config.system_prompt,
                    render_prompt_text(prompt_config.prompt_template, a_context, b_context),
                    image_paths=self.images_list[i],
                )
            )
            indices.append(i)

        inputs = [prepare_vllm_inputs(msg, self.processor) for msg in messages_list]

        try:
            batch_start = time.perf_counter()
            outputs = self.llm.generate(inputs, self.sampling_params)
            logger.info(
                "Local VLLM batch completed images=%s elapsed=%.2fs",
                len(indices),
                time.perf_counter() - batch_start,
            )
            for idx, output in zip(indices, outputs):
                raw_text = self._normalize_generated_text(idx, output.outputs[0].text.strip())
                if is_invalid_response(raw_text):
                    self.error_info[idx] = "model returned invalid content (empty or repetitive)"
                    self.raw_results[idx] = ""
                    logger.error(
                        "Local VLLM produced invalid output image_id=%s",
                        self.image_ids[idx],
                    )
                else:
                    self.raw_results[idx] = raw_text
                    logger.info(
                        "Completed local VLLM image analysis image_id=%s output_chars=%s preview=%r",
                        self.image_ids[idx],
                        len(raw_text),
                        _preview_text(raw_text),
                    )
        except Exception as exc:
            for idx in indices:
                self.error_info[idx] = f"local vllm inference failed: {exc}"
                self.raw_results[idx] = ""
                logger.error("Local vLLM inference failed for %s: %s", self.image_ids[idx], exc)

    def process_all_automatically(self, batch_size: int = 1) -> AnalysisResult:
        start_time = time.perf_counter()
        logger.info(
            "Starting VLM analysis pdf=%s total_images=%s mode=%s classification=%s threshold=%.3f output_dir=%s",
            self.pdf_name,
            len(self.image_ids),
            "api" if self.use_api else "local_vllm",
            self.enable_classification,
            self.classification_confidence_threshold,
            self.output_dir,
        )
        self._classify_images()

        while self.current_index < len(self.a_contexts):
            start = self.current_index
            end = min(start + batch_size, len(self.a_contexts))
            logger.info("Analyzing images %s-%s", start, end - 1)

            if self.use_api:
                self._batch_inference_api(start, end)
            else:
                self._batch_inference_vllm(start, end)

            self.current_index = end

        processing_time = time.perf_counter() - start_time
        result = self._pack_results(processing_time=processing_time)
        self._persist_result_files(result)
        self._close_api_client()
        logger.info(
            "Finished VLM analysis pdf=%s analyzed=%s/%s elapsed=%.2fs output_dir=%s",
            self.pdf_name,
            result.analyzed_images,
            result.total_images,
            processing_time,
            self.output_dir,
        )
        return result

    def _pack_results(self, processing_time: float = 0.0) -> AnalysisResult:
        descriptions: List[ImageDescription] = []
        analyzed = 0

        for idx, img_id in enumerate(self.image_ids):
            text = self.raw_results.get(idx, "")
            error_msg = self.error_info.get(idx, "")
            if text and not error_msg:
                analyzed += 1
                descriptions.append(
                    ImageDescription(
                        image_id=img_id,
                        description=text,
                        image_type=self.image_types.get(idx, ""),
                        prompt_key=self.prompt_keys.get(idx, ""),
                        status="success",
                    )
                )
            else:
                descriptions.append(
                    ImageDescription(
                        image_id=img_id,
                        description="",
                        image_type=self.image_types.get(idx, ""),
                        prompt_key=self.prompt_keys.get(idx, ""),
                        status="error",
                        error_message=error_msg or "no valid content generated",
                    )
                )

        return AnalysisResult(
            pdf_name=self.pdf_name,
            total_images=len(self.image_ids),
            analyzed_images=analyzed,
            descriptions=descriptions,
            processing_time=processing_time,
            output_dir=Path(self.output_dir),
        )

    def _persist_result_files(self, result: AnalysisResult) -> None:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result.save_descriptions(str(output_dir / "image_descriptions.json"))

        summary = {
            "pdf_name": result.pdf_name,
            "total_images": result.total_images,
            "analyzed_images": result.analyzed_images,
            "failed_images": result.total_images - result.analyzed_images,
            "processing_time": result.processing_time,
            "use_api": self.use_api,
            "vlm_api_type": self.vlm_api_type if self.use_api else "",
            "api_base": self.api_base if self.use_api else "",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "classification_enabled": self.enable_classification,
            "classification_confidence_threshold": self.classification_confidence_threshold,
            "request_config": asdict(self.request_config),
            "results": [
                {
                    "image_id": desc.image_id,
                    "status": desc.status,
                    "image_type": desc.image_type,
                    "prompt_key": desc.prompt_key,
                    "classification_category_key": (
                        self.classification_results.get(idx).category_key
                        if self.classification_results.get(idx)
                        else ""
                    ),
                    "classification_confidence": (
                        self.classification_results.get(idx).confidence
                        if self.classification_results.get(idx)
                        else None
                    ),
                    "classification_status": (
                        self.classification_results.get(idx).status
                        if self.classification_results.get(idx)
                        else ""
                    ),
                    "classification_error_message": (
                        self.classification_results.get(idx).error_message
                        if self.classification_results.get(idx)
                        else ""
                    ),
                    "error_message": desc.error_message,
                }
                for idx, desc in enumerate(result.descriptions)
            ],
        }
        with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)
        logger.info(
            "Persisted image analysis artifacts descriptions=%s summary=%s",
            output_dir / "image_descriptions.json",
            output_dir / "analysis_summary.json",
        )


def analyze_images_simple(
    ocr_result: OCRResult,
    output_dir: str,
    use_api: bool = True,
    api_base: str = None,
    model_name: str = None,
    api_key: str = None,
    vlm_api_type: str = "openai",
    model_version: str = None,
    vlm_client: Optional[Any] = None,
    enable_classification: bool = False,
    classifier_api_base: str = None,
    classifier_timeout: int = 30,
    classification_confidence_threshold: float = 0.0,
) -> AnalysisResult:
    normalized_infos = [_normalize_image_info(info) for info in ocr_result.images_info]

    return BatchVLMDocParser(
        a_contexts=[info.get("context_before", "") for info in normalized_infos],
        b_contexts=[info.get("context_after", "") for info in normalized_infos],
        images_list=[
            [_resolve_image_path(str(info.get("file_path", "")), str(ocr_result.output_dir))]
            for info in normalized_infos
        ],
        image_ids=[info.get("image_id", "") for info in normalized_infos],
        pdf_name=ocr_result.pdf_name,
        output_dir=output_dir,
        use_api=use_api,
        api_base=api_base,
        model_name=model_name,
        api_key=api_key,
        vlm_api_type=vlm_api_type,
        model_version=model_version,
        vlm_client=vlm_client,
        enable_classification=enable_classification,
        classifier_api_base=classifier_api_base,
        classifier_timeout=classifier_timeout,
        classification_confidence_threshold=classification_confidence_threshold,
    ).process_all_automatically(batch_size=1)
