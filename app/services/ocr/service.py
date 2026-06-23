# 文件作用：调用 OCR 服务并将上传文件解析为可处理文本。
# 关联说明：依赖 config.py，供 pipeline_batch_routes 在切块前解析上传文件。

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import UploadFile

from . import config as ocr_config_service
from app.services.storage import read_uploaded_file_content

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".srt",
}

OCR_EXTENSIONS = {
    ".pdf",
    ".ofd",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".docx",
    ".doc",
}

OCR_CONTENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def resolve_ocr_batch_url() -> str:
    try:
        profile = ocr_config_service.get_active_profile()
        if str(profile.get("provider") or "").strip().lower() == "batch_ocr":
            url = str(profile.get("post_url") or "").strip()
            if url:
                return url
    except Exception:
        pass
    return os.environ.get("OCR_BATCH_URL", "").strip()


def resolve_ocr_timeout_seconds(
    requested_timeout: Optional[int],
    *,
    default_timeout: Optional[int] = None,
) -> int:
    if requested_timeout is None:
        env_timeout = str(os.environ.get("OCR_TIMEOUT_SECONDS") or "").strip()
        if env_timeout:
            requested_timeout = env_timeout  # type: ignore[assignment]
        elif default_timeout is not None:
            requested_timeout = default_timeout
        else:
            requested_timeout = 600
    try:
        timeout_value = int(requested_timeout)
    except (TypeError, ValueError):
        timeout_value = 600
    return max(1, min(timeout_value, 3600))


def _get_extension(filename: Optional[str]) -> str:
    if not filename:
        return ""
    _, ext = os.path.splitext(filename)
    return ext.lower()


def _guess_content_format_from_filename(filename: Optional[str]) -> str:
    ext = _get_extension(filename)
    return "markdown" if ext in MARKDOWN_EXTENSIONS else "text"


def _basename_any_sep(filename: Optional[str]) -> str:
    if not filename:
        return "file"
    name = filename.replace("\\", "/")
    return os.path.basename(name) or "file"


def _peek_upload_file(upload_file: UploadFile, limit: int = 8192) -> bytes:
    try:
        return upload_file.file.read(limit)
    finally:
        upload_file.file.seek(0)


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control_chars = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return (control_chars / max(1, len(sample))) > 0.12


def file_requires_ocr(upload_file: UploadFile) -> bool:
    ext = _get_extension(upload_file.filename)
    content_type = (upload_file.content_type or "").lower()

    if ext in TEXT_EXTENSIONS or content_type.startswith("text/"):
        return False
    if ext in OCR_EXTENSIONS or content_type.startswith("image/") or content_type in OCR_CONTENT_TYPES:
        return True
    sample = _peek_upload_file(upload_file)
    return _looks_binary(sample)


def should_auto_ocr(upload_file: UploadFile, ocr_enabled: bool) -> bool:
    return bool(ocr_enabled) and file_requires_ocr(upload_file)


def _normalize_text_from_ocr_entry(entry: Dict[str, Any]) -> str:
    normalized = _normalize_ocr_entry_content(entry)
    content = normalized.get("content")
    if isinstance(content, str) and content.strip():
        return content
    pages = entry.get("pages") or []
    if isinstance(pages, list):
        page_texts: List[str] = []
        for page in pages:
            if isinstance(page, dict):
                t = page.get("text")
                if isinstance(t, str) and t.strip():
                    page_texts.append(t)
        return "\n\n".join(page_texts)
    return ""


def _normalize_markdown_from_ocr_entry(entry: Dict[str, Any]) -> Optional[str]:
    normalized = _normalize_ocr_entry_content(entry)
    value = normalized.get("markdown_content")
    return value if isinstance(value, str) and value.strip() else None


def _normalize_plain_text_from_ocr_entry(entry: Dict[str, Any]) -> Optional[str]:
    normalized = _normalize_ocr_entry_content(entry)
    value = normalized.get("plain_text")
    return value if isinstance(value, str) and value.strip() else None


def _decode_bytes_to_text(payload: bytes) -> str:
    if not payload:
        return ""
    for encoding in ("utf-8", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _extract_filename_from_content_disposition(content_disposition: str) -> Optional[str]:
    text = str(content_disposition or "").strip()
    if not text:
        return None
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', text, flags=re.IGNORECASE)
    if not match:
        return None
    return os.path.basename(match.group(1).strip()) or None


def _guess_content_format_from_response(response: httpx.Response) -> str:
    content_type = str(response.headers.get("content-type") or "").strip().lower()
    if "markdown" in content_type:
        return "markdown"
    filename = _extract_filename_from_content_disposition(
        str(response.headers.get("content-disposition") or "")
    )
    if _guess_content_format_from_filename(filename) == "markdown":
        return "markdown"
    return "text"


def _normalize_requested_output_format(raw_value: Optional[str]) -> Optional[str]:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"markdown", "md", "ocr_markdown", "ocr-md", "ocr_md"}:
        return "markdown"
    if normalized == "text":
        return "text"
    return None


def _build_file_content_record(
    *,
    filename: Optional[str],
    content: Optional[str],
    status: str,
    ocr_seconds: Optional[float],
    error: Optional[str] = None,
    content_format: Optional[str] = None,
    markdown_content: Optional[str] = None,
    plain_text: Optional[str] = None,
    ocr_pages: Optional[List[Dict[str, Any]]] = None,
    ocr_raw_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_content = content if isinstance(content, str) else None
    normalized_markdown = markdown_content if isinstance(markdown_content, str) and markdown_content.strip() else None
    normalized_plain = plain_text if isinstance(plain_text, str) and plain_text.strip() else None
    normalized_format = str(content_format or "").strip().lower()
    if normalized_format not in {"markdown", "text"}:
        normalized_format = "markdown" if normalized_markdown else "text"
    if normalized_format == "markdown" and normalized_markdown is None and isinstance(normalized_content, str):
        normalized_markdown = normalized_content
    if normalized_format == "text" and normalized_plain is None and isinstance(normalized_content, str):
        normalized_plain = normalized_content

    return {
        "filename": filename,
        "content": normalized_content,
        "size": len(normalized_content) if isinstance(normalized_content, str) else 0,
        "status": status,
        "error": error,
        "ocr_seconds": ocr_seconds,
        "content_format": normalized_format,
        "markdown_content": normalized_markdown,
        "plain_text": normalized_plain,
        "ocr_pages": ocr_pages if isinstance(ocr_pages, list) else [],
        "ocr_raw_entry": dict(ocr_raw_entry) if isinstance(ocr_raw_entry, dict) else None,
    }


def _normalize_ocr_entry_content(entry: Dict[str, Any]) -> Dict[str, Any]:
    content_format = str(entry.get("content_format") or "").strip().lower()
    if content_format not in {"markdown", "text"}:
        content_format = ""

    markdown_value = entry.get("markdown_content")
    plain_text_value = entry.get("plain_text")
    text_value = entry.get("text")

    markdown_content = markdown_value.strip() if isinstance(markdown_value, str) and markdown_value.strip() else None
    plain_text = plain_text_value.strip() if isinstance(plain_text_value, str) and plain_text_value.strip() else None
    text = text_value.strip() if isinstance(text_value, str) and text_value.strip() else None

    if not markdown_content and content_format == "markdown" and text:
        markdown_content = text
    if not plain_text and content_format == "text" and text:
        plain_text = text

    if not content_format:
        if markdown_content:
            content_format = "markdown"
        else:
            content_format = "text"

    preferred_content = markdown_content or text or plain_text or ""
    if content_format == "text":
        preferred_content = plain_text or text or markdown_content or ""

    return {
        "content": preferred_content,
        "content_format": content_format,
        "markdown_content": markdown_content,
        "plain_text": plain_text or (text if content_format == "text" else None),
    }


def _extract_content_from_http_response(
    response: httpx.Response,
    response_mode: str,
    *,
    requested_output_format: Optional[str] = None,
) -> Dict[str, Any]:
    mode = str(response_mode or "").strip().lower() or "text"
    if mode == "file":
        text = _decode_bytes_to_text(response.content or b"")
        content_format = _guess_content_format_from_response(response)
        requested_hint = _normalize_requested_output_format(requested_output_format)
        if content_format == "text" and requested_hint == "markdown":
            content_format = "markdown"
        return _normalize_ocr_entry_content(
            {
                "text": text,
                "content_format": content_format,
                "markdown_content": text if content_format == "markdown" else None,
                "plain_text": text if content_format == "text" else None,
            }
        )
    if mode == "structured_json":
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, str):
            return _normalize_ocr_entry_content({"text": payload, "content_format": "text"})
        if isinstance(payload, dict):
            normalized = _normalize_ocr_entry_content(payload)
            if normalized["content"]:
                return normalized
        fallback_text = response.text or _decode_bytes_to_text(response.content or b"")
        return _normalize_ocr_entry_content({"text": fallback_text, "content_format": "text"})

    # default: text
    text_value = response.text or ""
    if text_value and "\ufffd" not in text_value:
        return _normalize_ocr_entry_content(
            {
                "text": text_value,
                "content_format": _guess_content_format_from_response(response),
            }
        )
    return _normalize_ocr_entry_content(
        {
            "text": _decode_bytes_to_text(response.content or b""),
            "content_format": _guess_content_format_from_response(response),
        }
    )


async def _call_batch_ocr_service(
    *,
    ocr_url: str,
    upload_files: List[UploadFile],
    batch_field: str,
    timeout_seconds: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    multipart_files: List[Tuple[str, Tuple[str, Any, str]]] = []
    for index, upload_file in enumerate(upload_files):
        original_name = _basename_any_sep(upload_file.filename)
        internal_name = f"__{index:04d}__{original_name}"
        upload_file.file.seek(0)
        multipart_files.append(
            (
                batch_field,
                (
                    internal_name,
                    upload_file.file,
                    upload_file.content_type or "application/octet-stream",
                ),
            )
        )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            ocr_url,
            files=multipart_files,
            timeout=httpx.Timeout(timeout_seconds),
        )

    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, str):
        if len(upload_files) != 1:
            raise RuntimeError(
                "OCR service returned an unstructured string for multiple files; "
                "please upgrade OCR service to return structured JSON per file"
            )
        return (
            [
                {
                    "filename": multipart_files[0][1][0],
                    "status": "success",
                    "pages": [{"page": 1, "text": payload}],
                    "text": payload,
                }
            ],
            [],
        )

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected OCR response type: {type(payload).__name__}")

    results = payload.get("results") or []
    errors = payload.get("errors") or []
    if not isinstance(results, list) or not isinstance(errors, list):
        raise RuntimeError("OCR response schema invalid: results/errors must be lists")
    return results, errors


async def _call_process_api_service(
    *,
    ocr_url: str,
    upload_files: List[UploadFile],
    file_field: str,
    extra_form_fields: Dict[str, str],
    response_mode: str,
    timeout_seconds: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    requested_output_format = _normalize_requested_output_format(
        extra_form_fields.get("output_format")
    )
    async with httpx.AsyncClient() as client:
        for index, upload_file in enumerate(upload_files):
            original_name = _basename_any_sep(upload_file.filename)
            internal_name = f"__{index:04d}__{original_name}"
            upload_file.file.seek(0)
            try:
                response = await client.post(
                    ocr_url,
                    data=extra_form_fields or None,
                    files={
                        file_field: (
                            internal_name,
                            upload_file.file,
                            upload_file.content_type or "application/octet-stream",
                        )
                    },
                    timeout=httpx.Timeout(timeout_seconds),
                )
                response.raise_for_status()
                content_payload = _extract_content_from_http_response(
                    response,
                    response_mode,
                    requested_output_format=requested_output_format,
                )
                results.append(
                    {
                        "filename": internal_name,
                        "status": "success",
                        "pages": [{"page": 1, "text": str(content_payload.get("content") or "")}],
                        "text": str(content_payload.get("content") or ""),
                        "content_format": str(content_payload.get("content_format") or "text"),
                        "markdown_content": content_payload.get("markdown_content"),
                        "plain_text": content_payload.get("plain_text"),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "filename": internal_name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
    return results, errors


async def batch_ocr_upload_files(
    upload_files: List[UploadFile],
    timeout_seconds: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Call the external OCR batch service with multiple files.

    Returns (results, errors) where each item is a dict containing at least:
    - filename: the multipart filename used when calling OCR
    - status: success|error
    - text: extracted text (for success)
    - pages: optional page list (for success)
    - error: error message (for error)
    """
    profile = ocr_config_service.get_active_profile()
    provider = str(profile.get("provider") or "batch_ocr").strip().lower() or "batch_ocr"
    ocr_url = str(profile.get("post_url") or "").strip() or resolve_ocr_batch_url()
    request_cfg = profile.get("request") if isinstance(profile.get("request"), dict) else {}
    response_cfg = profile.get("response") if isinstance(profile.get("response"), dict) else {}

    batch_field = str(request_cfg.get("batch_field") or "files").strip() or "files"
    file_field = str(request_cfg.get("file_field") or "file").strip() or "file"
    extra_fields = request_cfg.get("extra_form_fields")
    extra_fields = extra_fields if isinstance(extra_fields, dict) else {}
    extra_fields_str: Dict[str, str] = {str(k): str(v) for k, v in extra_fields.items()}
    response_mode = str(response_cfg.get("mode") or "").strip().lower() or (
        "structured_json" if provider == "batch_ocr" else "text"
    )

    if not ocr_url:
        raise RuntimeError(
            "OCR URL is not set; configure it via /ui (OCR 配置管理) or set OCR_BATCH_URL"
        )

    if provider == "process_api":
        return await _call_process_api_service(
            ocr_url=ocr_url,
            upload_files=upload_files,
            file_field=file_field,
            extra_form_fields=extra_fields_str,
            response_mode=response_mode,
            timeout_seconds=timeout_seconds,
        )

    return await _call_batch_ocr_service(
        ocr_url=ocr_url,
        upload_files=upload_files,
        batch_field=batch_field,
        timeout_seconds=timeout_seconds,
    )


async def resolve_uploaded_files_with_auto_ocr(
    upload_files: List[UploadFile],
    *,
    ocr_enabled: bool = True,
    ocr_timeout_seconds: Optional[int] = None,
    ocr_fail_fast: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert a list of uploaded files into text inputs for the pipeline.

    - Text files are decoded locally.
    - Non-text files are sent to the OCR batch service.

    Returns (file_contents, ocr_summary).
    """
    file_contents: List[Dict[str, Any]] = []
    ocr_summary: List[Dict[str, Any]] = []

    ocr_candidates: List[Tuple[int, UploadFile]] = []

    for index, upload_file in enumerate(upload_files):
        requires_ocr = file_requires_ocr(upload_file)
        if requires_ocr and not ocr_enabled:
            error_message = "OCR is disabled (ocr_enabled=false) but file requires OCR"
            file_contents.append(
                _build_file_content_record(
                    filename=upload_file.filename,
                    content=None,
                    status="error",
                    error=error_message,
                    ocr_seconds=0.0,
                    content_format=_guess_content_format_from_filename(upload_file.filename),
                )
            )
            ocr_summary.append(
                {
                    "filename": upload_file.filename,
                    "status": "error",
                    "error": error_message,
                    "ocr_seconds": 0.0,
                }
            )
            continue

        if requires_ocr:
            ocr_candidates.append((index, upload_file))
            file_contents.append(
                _build_file_content_record(
                    filename=upload_file.filename,
                    content=None,
                    status="pending_ocr",
                    ocr_seconds=None,
                    content_format="text",
                )
            )
            continue

        try:
            text_content = read_uploaded_file_content(upload_file)
            content_format = _guess_content_format_from_filename(upload_file.filename)
            file_contents.append(
                _build_file_content_record(
                    filename=upload_file.filename,
                    content=text_content,
                    status="success",
                    ocr_seconds=0.0,
                    content_format=content_format,
                    markdown_content=text_content if content_format == "markdown" else None,
                    plain_text=text_content if content_format == "text" else None,
                )
            )
        except Exception as exc:
            file_contents.append(
                _build_file_content_record(
                    filename=upload_file.filename,
                    content=None,
                    status="error",
                    error=str(exc),
                    ocr_seconds=0.0,
                    content_format=_guess_content_format_from_filename(upload_file.filename),
                )
            )

    if not ocr_candidates:
        return file_contents, ocr_summary

    ordered_ocr_files = [f for _, f in ocr_candidates]
    profile_timeout = None
    try:
        profile_timeout = int(ocr_config_service.get_active_profile().get("timeout_seconds") or 0) or None
    except Exception:
        profile_timeout = None
    timeout_value = resolve_ocr_timeout_seconds(ocr_timeout_seconds, default_timeout=profile_timeout)
    ocr_started = time.perf_counter()
    try:
        results, errors = await batch_ocr_upload_files(ordered_ocr_files, timeout_value)
    except Exception as exc:
        ocr_elapsed = time.perf_counter() - ocr_started
        ocr_share = ocr_elapsed / max(1, len(ocr_candidates))
        error_message = f"OCR request failed: {str(exc)}"
        for index, upload_file in ocr_candidates:
            file_contents[index] = _build_file_content_record(
                filename=upload_file.filename,
                content=None,
                status="error",
                error=error_message,
                ocr_seconds=ocr_share,
                content_format="text",
            )
            ocr_summary.append(
                {
                    "filename": upload_file.filename,
                    "status": "error",
                    "error": error_message,
                    "ocr_seconds": ocr_share,
                }
            )
        if ocr_fail_fast:
            raise RuntimeError(error_message)
        return file_contents, ocr_summary

    ocr_elapsed = time.perf_counter() - ocr_started
    ocr_share = ocr_elapsed / max(1, len(ocr_candidates))

    result_by_filename: Dict[str, Dict[str, Any]] = {}
    for entry in results:
        if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
            result_by_filename[entry["filename"]] = entry

    error_by_filename: Dict[str, Dict[str, Any]] = {}
    for entry in errors:
        if isinstance(entry, dict) and isinstance(entry.get("filename"), str):
            error_by_filename[entry["filename"]] = entry

    for relative_index, (original_index, upload_file) in enumerate(ocr_candidates):
        internal_name = f"__{relative_index:04d}__{_basename_any_sep(upload_file.filename)}"
        entry = result_by_filename.get(internal_name)
        error_entry = error_by_filename.get(internal_name)

        if isinstance(entry, dict) and entry.get("status") == "success":
            normalized_text = _normalize_text_from_ocr_entry(entry)
            markdown_content = _normalize_markdown_from_ocr_entry(entry)
            plain_text = _normalize_plain_text_from_ocr_entry(entry)
            pages = entry.get("pages") if isinstance(entry.get("pages"), list) else []
            content_format = str(entry.get("content_format") or "").strip().lower() or (
                "markdown" if markdown_content else "text"
            )
            file_contents[original_index] = {
                **_build_file_content_record(
                    filename=upload_file.filename,
                    content=normalized_text,
                    status="success",
                    ocr_seconds=ocr_share,
                    content_format=content_format,
                    markdown_content=markdown_content,
                    plain_text=plain_text,
                    ocr_pages=pages,
                    ocr_raw_entry=entry,
                )
            }
            ocr_summary.append(
                {
                    "filename": upload_file.filename,
                    "status": "success",
                    "page_count": len(pages) if isinstance(pages, list) else 0,
                    "content_format": content_format,
                    "has_markdown": bool(markdown_content),
                    "ocr_seconds": ocr_share,
                }
            )
            continue

        error_message = None
        if isinstance(error_entry, dict):
            error_message = error_entry.get("error") or error_entry.get("detail") or "OCR failed"
        if not error_message:
            error_message = "OCR result missing for file"

        file_contents[original_index] = _build_file_content_record(
            filename=upload_file.filename,
            content=None,
            status="error",
            error=str(error_message),
            ocr_seconds=ocr_share,
            content_format="text",
            ocr_raw_entry=entry if isinstance(entry, dict) else error_entry if isinstance(error_entry, dict) else None,
        )
        ocr_summary.append(
            {
                "filename": upload_file.filename,
                "status": "error",
                "error": str(error_message),
                "ocr_seconds": ocr_share,
            }
        )

    if ocr_fail_fast and any(item.get("status") == "error" for item in file_contents):
        raise RuntimeError("At least one file failed during OCR (ocr_fail_fast=true)")

    return file_contents, ocr_summary


__all__ = [
    "resolve_ocr_batch_url",
    "resolve_ocr_timeout_seconds",
    "should_auto_ocr",
    "batch_ocr_upload_files",
    "resolve_uploaded_files_with_auto_ocr",
]
