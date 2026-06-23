# 文件作用：定义 Easy Dataset 切块流程的异常类型。
# 关联说明：被 easy_dataset.py 及其内部拆分模块共用，避免异常类形成循环依赖。

from __future__ import annotations

from typing import Any, Dict, Optional

class EasyDatasetChunkingError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = str(code or "chunking_failed").strip() or "chunking_failed"
        self.message = str(message or self.code).strip() or self.code
        self.details = dict(details or {})

__all__ = ["EasyDatasetChunkingError"]
