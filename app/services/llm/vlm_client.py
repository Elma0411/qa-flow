import logging
import os
import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional until API client is created
    OpenAI = None


logger = logging.getLogger(__name__)

DEFAULT_VLM_API_TYPE = "openai"


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _resolve_required_config_value(
    value: Optional[str],
    *,
    env_name: str,
    param_name: str,
    label: str,
) -> str:
    resolved = _first_non_empty(value, os.getenv(env_name))
    if resolved is None:
        raise ValueError(
            f"{label} is not configured; pass {param_name} or set {env_name}"
        )
    return resolved


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float for %s=%r, using default %s", name, raw, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid integer for %s=%r, using default %s", name, raw, default)
        return default


def normalize_openai_base_url(api_base: Optional[str]) -> str:
    raw = (api_base or "").strip()
    if not raw:
        return ""

    normalized = raw.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def normalize_vlm_api_type(api_type: Optional[str]) -> str:
    raw = (api_type or os.getenv("VLM_API_TYPE", DEFAULT_VLM_API_TYPE)).strip().lower()
    normalized = raw.replace("-", "_")
    if normalized in {"", "openai", "openai_compatible"}:
        return "openai"
    if normalized in {"lmp", "lmpcloud", "lmp_cloud", "lmp_cloud_original", "lmp_http"}:
        return "lmp_cloud"
    raise ValueError(f"Unsupported vlm_api_type: {api_type!r}")


def normalize_lmp_cloud_endpoint(api_base: Optional[str]) -> str:
    raw = (api_base or "").strip()
    if not raw:
        return ""

    normalized = raw.rstrip("/")
    if normalized.endswith("/V2"):
        normalized = normalized[: -len("/V2")]

    if normalized.endswith("/api/vlm/chat/completions"):
        return normalized
    if normalized.endswith("/api/vlm"):
        return normalized + "/chat/completions"
    if normalized.endswith("/lmp-cloud-ias-server"):
        return normalized + "/api/vlm/chat/completions"

    return normalized + "/lmp-cloud-ias-server/api/vlm/chat/completions"


def normalize_vlm_endpoint(api_base: Optional[str], api_type: Optional[str]) -> str:
    normalized_type = normalize_vlm_api_type(api_type)
    if normalized_type == "lmp_cloud":
        return normalize_lmp_cloud_endpoint(api_base)
    return normalize_openai_base_url(api_base)


@dataclass(frozen=True)
class VLMClientConfig:
    base_url: str
    model_name: str
    api_key: str
    api_type: str = DEFAULT_VLM_API_TYPE
    model_version: str = ""
    timeout_seconds: float = 180.0
    stream: bool = True
    max_concurrent_requests: int = 1
    min_interval_seconds: float = 0.0
    top_p: float = 0.8
    presence_penalty: float = 1.0

    @classmethod
    def from_values(
        cls,
        *,
        api_base: Optional[str],
        model_name: Optional[str],
        api_key: Optional[str],
        api_type: Optional[str] = None,
        model_version: Optional[str] = None,
        timeout_seconds: float,
    ) -> "VLMClientConfig":
        normalized_type = normalize_vlm_api_type(api_type)
        resolved_api_base = _resolve_required_config_value(
            api_base,
            env_name="VLM_API_BASE",
            param_name="api_base",
            label="VLM API base",
        )
        resolved_model_name = _resolve_required_config_value(
            model_name,
            env_name="VLM_MODEL_NAME",
            param_name="model_name",
            label="VLM model name",
        )
        resolved_api_key = _resolve_required_config_value(
            api_key,
            env_name="VLM_API_KEY",
            param_name="api_key",
            label="VLM API key",
        )
        return cls(
            base_url=normalize_vlm_endpoint(resolved_api_base, normalized_type),
            model_name=resolved_model_name,
            api_key=resolved_api_key,
            api_type=normalized_type,
            model_version=_first_non_empty(model_version, os.getenv("VLM_MODEL_VERSION")) or "",
            timeout_seconds=float(timeout_seconds),
            stream=_get_env_bool("VLM_API_STREAM", True),
            max_concurrent_requests=max(1, _get_env_int("VLM_API_MAX_CONCURRENT_REQUESTS", 1)),
            min_interval_seconds=max(0.0, _get_env_float("VLM_API_MIN_INTERVAL_SECONDS", 0.0)),
            top_p=_get_env_float("VLM_API_TOP_P", 0.8),
            presence_penalty=_get_env_float("VLM_API_PRESENCE_PENALTY", 1.0),
        )


class VLMHTTPStatusError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"http {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class VLMConnectionError(RuntimeError):
    pass


class OpenAICompatibleVLMClient:
    """
    Process-scoped OpenAI-compatible VLM client with keep-alive reuse,
    optional streaming, and per-client request throttling.
    """

    def __init__(self, config: VLMClientConfig):
        self.config = config
        self._client_lock = threading.RLock()
        self._lifecycle_condition = threading.Condition()
        self._active_calls = 0
        self._recreating = False
        self._request_gate = threading.BoundedSemaphore(config.max_concurrent_requests)
        self._throttle_lock = threading.Lock()
        self._last_request_at = 0.0
        self._client = self._create_client()

    def _create_client(self) -> OpenAI:
        if OpenAI is None:
            raise ImportError("OpenAI-compatible VLM client requires the openai package")
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )

    def public_signature(self) -> Dict[str, Any]:
        return {
            "api_type": self.config.api_type,
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "timeout_seconds": self.config.timeout_seconds,
            "stream": self.config.stream,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "min_interval_seconds": self.config.min_interval_seconds,
        }

    def close(self) -> None:
        self._wait_for_active_calls()
        with self._client_lock:
            client = self._client
            self._client = None

        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("Failed to close VLM API client", exc_info=True)

    def recreate(self, reason: str) -> None:
        with self._lifecycle_condition:
            self._recreating = True
            while self._active_calls > 0:
                self._lifecycle_condition.wait()

        try:
            with self._client_lock:
                old_client = self._client
                self._client = self._create_client()

            close = getattr(old_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to close stale VLM API client", exc_info=True)
            logger.warning("Recreated VLM API client: %s", reason)
        finally:
            with self._lifecycle_condition:
                self._recreating = False
                self._lifecycle_condition.notify_all()

    def create_chat_completion_text(
        self,
        *,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        with self._request_gate:
            self._begin_call()
            try:
                self._throttle_if_needed()
                with self._client_lock:
                    client = self._client
                if client is None:
                    raise RuntimeError("VLM API client has been closed")

                request_model = model or self.config.model_name
                if self.config.stream:
                    return self._stream_chat_completion_text(
                        client=client,
                        model=request_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                response = client.chat.completions.create(
                    model=request_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                return extract_message_text(response)
            finally:
                self._end_call()

    def _begin_call(self) -> None:
        with self._lifecycle_condition:
            while self._recreating:
                self._lifecycle_condition.wait()
            self._active_calls += 1

    def _end_call(self) -> None:
        with self._lifecycle_condition:
            self._active_calls = max(0, self._active_calls - 1)
            if self._active_calls == 0:
                self._lifecycle_condition.notify_all()

    def _wait_for_active_calls(self) -> None:
        with self._lifecycle_condition:
            while self._active_calls > 0:
                self._lifecycle_condition.wait()

    def _throttle_if_needed(self) -> None:
        if self.config.min_interval_seconds <= 0:
            return

        with self._throttle_lock:
            now = time.perf_counter()
            wait_seconds = self.config.min_interval_seconds - (now - self._last_request_at)
            if wait_seconds > 0:
                logger.info("Throttling VLM request for %.2fs", wait_seconds)
                time.sleep(wait_seconds)
            self._last_request_at = time.perf_counter()

    def _stream_chat_completion_text(
        self,
        *,
        client: OpenAI,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        chunks: List[str] = []
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for event in stream:
            piece = extract_stream_delta_text(event)
            if piece:
                chunks.append(piece)
        return "".join(chunks)


class LmpCloudVLMClient:
    """
    Client for the LMP Cloud original HTTP multimodal chat endpoint.

    It keeps the same public method as OpenAICompatibleVLMClient so the
    analyzer can switch clients without changing its retry/integration flow.
    """

    def __init__(self, config: VLMClientConfig):
        self.config = config
        self._lifecycle_condition = threading.Condition()
        self._active_calls = 0
        self._recreating = False
        self._request_gate = threading.BoundedSemaphore(config.max_concurrent_requests)
        self._throttle_lock = threading.Lock()
        self._last_request_at = 0.0

    def public_signature(self) -> Dict[str, Any]:
        return {
            "api_type": self.config.api_type,
            "base_url": self.config.base_url,
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "timeout_seconds": self.config.timeout_seconds,
            "stream": self.config.stream,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "min_interval_seconds": self.config.min_interval_seconds,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
        }

    def close(self) -> None:
        self._wait_for_active_calls()

    def recreate(self, reason: str) -> None:
        self._wait_for_active_calls()
        logger.warning("LMP Cloud VLM client recreate requested; no persistent HTTP client to recreate: %s", reason)

    def create_chat_completion_text(
        self,
        *,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        with self._request_gate:
            self._begin_call()
            try:
                self._throttle_if_needed()
                payload = {
                    "model": model or self.config.model_name,
                    "messages": normalize_lmp_cloud_messages(messages),
                    "stream": bool(self.config.stream),
                    "temperature": temperature,
                    "top_p": self.config.top_p,
                    "presence_penalty": self.config.presence_penalty,
                }
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if self.config.model_version:
                    payload["modelVersion"] = self.config.model_version
                return self._post_chat_completion(payload)
            finally:
                self._end_call()

    def _begin_call(self) -> None:
        with self._lifecycle_condition:
            while self._recreating:
                self._lifecycle_condition.wait()
            self._active_calls += 1

    def _end_call(self) -> None:
        with self._lifecycle_condition:
            self._active_calls = max(0, self._active_calls - 1)
            if self._active_calls == 0:
                self._lifecycle_condition.notify_all()

    def _wait_for_active_calls(self) -> None:
        with self._lifecycle_condition:
            while self._active_calls > 0:
                self._lifecycle_condition.wait()

    def _throttle_if_needed(self) -> None:
        if self.config.min_interval_seconds <= 0:
            return

        with self._throttle_lock:
            now = time.perf_counter()
            wait_seconds = self.config.min_interval_seconds - (now - self._last_request_at)
            if wait_seconds > 0:
                logger.info("Throttling LMP Cloud VLM request for %.2fs", wait_seconds)
                time.sleep(wait_seconds)
            self._last_request_at = time.perf_counter()

    def _post_chat_completion(self, payload: Dict[str, Any]) -> str:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": self.config.api_key,
        }
        req = urlrequest.Request(
            self.config.base_url,
            data=request_body,
            headers=headers,
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=self.config.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                if self.config.stream or "text/event-stream" in content_type.lower():
                    return self._read_event_stream(response)

                body = response.read().decode("utf-8", errors="replace")
                if not body.strip():
                    return ""
                return extract_lmp_cloud_message_text(json.loads(body))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VLMHTTPStatusError(exc.code, detail) from exc
        except (urlerror.URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise VLMConnectionError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid LMP Cloud VLM response JSON: {exc}") from exc

    def _read_event_stream(self, response: Any) -> str:
        chunks: List[str] = []
        latest_snapshot = ""
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("event:"):
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if not line or line == "[DONE]":
                if line == "[DONE]":
                    break
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping non-JSON LMP Cloud VLM stream line: %r", line)
                continue

            piece = extract_lmp_cloud_stream_delta_text(event)
            if piece:
                if lmp_cloud_stream_event_is_snapshot(event):
                    latest_snapshot = piece
                else:
                    chunks.append(piece)
        return latest_snapshot or "".join(chunks)


def create_vlm_client(config: VLMClientConfig) -> Any:
    if normalize_vlm_api_type(config.api_type) == "lmp_cloud":
        return LmpCloudVLMClient(config)
    return OpenAICompatibleVLMClient(config)


def extract_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    return _content_to_text(content)


def extract_stream_delta_text(event: Any) -> str:
    choices = getattr(event, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", "") if delta is not None else ""
    return _content_to_text(content)


def normalize_lmp_cloud_messages(messages: List[Dict]) -> List[Dict]:
    normalized_messages: List[Dict] = []
    image_count = 0
    skipped_images = 0
    pending_system_texts: List[str] = []

    for message in messages:
        role = str(message.get("role", "user") or "user")
        content = message.get("content", "")

        if role == "system":
            system_text = _message_like_to_text(content).strip()
            if system_text:
                pending_system_texts.append(system_text)
            continue

        if not isinstance(content, list):
            text_items = [{"type": "text", "text": str(content)}]
            image_items: List[Dict[str, Any]] = []
            other_items: List[Dict[str, Any]] = []
            if pending_system_texts and role == "user":
                system_text = "\n\n".join(pending_system_texts)
                text_items[0]["text"] = system_text + "\n\n" + text_items[0]["text"]
                pending_system_texts = []
            normalized_messages.append(
                {
                    "role": role,
                    "content": text_items + image_items + other_items,
                }
            )
            continue

        text_items: List[Dict[str, Any]] = []
        image_items: List[Dict[str, Any]] = []
        other_items: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                text_items.append({"type": "text", "text": str(item)})
                continue

            item_type = item.get("type")
            if item_type == "image_url":
                image_url = item.get("image_url") or {}
                if isinstance(image_url, dict):
                    image_value = image_url.get("url", "")
                else:
                    image_value = str(image_url)
                if image_value and image_count == 0:
                    image_items.append({"type": "image_base64", "image": image_value})
                    image_count += 1
                elif image_value:
                    skipped_images += 1
                continue
            if item_type == "image_base64":
                image_value = item.get("image", "")
                if image_value and image_count == 0:
                    image_items.append({"type": "image_base64", "image": image_value})
                    image_count += 1
                elif image_value:
                    skipped_images += 1
                continue
            if item_type == "text":
                text_items.append({"type": "text", "text": str(item.get("text", ""))})
                continue
            other_items.append(item)

        if pending_system_texts and role == "user":
            system_text = "\n\n".join(pending_system_texts)
            if text_items:
                text_items[0]["text"] = system_text + "\n\n" + text_items[0]["text"]
            else:
                text_items.insert(0, {"type": "text", "text": system_text})
            pending_system_texts = []

        normalized_messages.append(
            {
                "role": role,
                "content": text_items + image_items + other_items,
            }
        )

    if pending_system_texts:
        normalized_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n\n".join(pending_system_texts),
                    }
                ],
            }
        )

    if skipped_images:
        logger.warning("LMP Cloud endpoint only supports one image; ignored %s extra image(s)", skipped_images)
    return normalized_messages


def extract_lmp_cloud_message_text(response_body: Any) -> str:
    if isinstance(response_body, dict) and "data" in response_body and "choices" not in response_body:
        return extract_lmp_cloud_message_text(response_body.get("data"))

    choices = response_body.get("choices") if isinstance(response_body, dict) else None
    if not choices:
        return ""

    choice = choices[0]
    if not isinstance(choice, dict):
        return _content_to_text(choice)

    for key in ("message", "delta", "content", "text"):
        text = _message_like_to_text(choice.get(key))
        if text:
            return text
    return ""


def extract_lmp_cloud_stream_delta_text(event: Any) -> str:
    if isinstance(event, dict) and "data" in event and "choices" not in event:
        return extract_lmp_cloud_stream_delta_text(event.get("data"))
    if isinstance(event, str):
        try:
            return extract_lmp_cloud_stream_delta_text(json.loads(event))
        except json.JSONDecodeError:
            return event
    return extract_lmp_cloud_message_text(event)


def lmp_cloud_stream_event_is_snapshot(event: Any) -> bool:
    if isinstance(event, dict) and "data" in event and "choices" not in event:
        return lmp_cloud_stream_event_is_snapshot(event.get("data"))
    if not isinstance(event, dict):
        return False
    choices = event.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    return "message" in choice and "delta" not in choice


def _message_like_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "content" in value:
            return _content_to_text(value.get("content"))
        if "text" in value:
            return _content_to_text(value.get("text"))
        return ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _message_like_to_text(item)
            if text:
                parts.append(text)
        return "".join(parts)
    return _content_to_text(value)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)
