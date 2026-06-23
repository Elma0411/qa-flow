# 文件作用：作为 OCR 配置与自动解析服务的公共 facade。
# 关联说明：聚合 config 和 service，分别处理 OCR 配置与文件解析。

"""Public facade for OCR configuration and file resolution services."""

from .store import (
    OCRConfigStore,
    activate_profile,
    delete_profile,
    get_active_profile,
    get_profile,
    list_profiles,
    upsert_profile,
)
from .service import (
    batch_ocr_upload_files,
    file_requires_ocr,
    resolve_ocr_batch_url,
    resolve_ocr_timeout_seconds,
    resolve_uploaded_files_with_auto_ocr,
    should_auto_ocr,
)

__all__ = [
    "OCRConfigStore",
    "activate_profile",
    "batch_ocr_upload_files",
    "delete_profile",
    "file_requires_ocr",
    "get_active_profile",
    "get_profile",
    "list_profiles",
    "resolve_ocr_batch_url",
    "resolve_ocr_timeout_seconds",
    "resolve_uploaded_files_with_auto_ocr",
    "should_auto_ocr",
    "upsert_profile",
]
