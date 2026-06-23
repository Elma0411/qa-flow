# 文件作用：提供服务端本地时间与时区相关工具函数。
# 关联说明：为 pipeline_state、routers 和作业服务提供统一时间格式。

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


_PIPELINE_TIMESTAMP_FIELDS = frozenset(
    {"updated_at", "started_at", "finished_at", "completed_at", "created_at"}
)


def _server_local_tz():
    return datetime.now().astimezone().tzinfo or timezone.utc


def now_server_local_iso(*, timespec: str = "microseconds") -> str:
    return datetime.now().astimezone().isoformat(timespec=timespec)


def parse_datetime_to_local(
    value: Any,
    *,
    naive_assumption: str = "local",
) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=_server_local_tz())
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except Exception:
            try:
                dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
            except Exception:
                return None

    if dt.tzinfo is None:
        tzinfo = timezone.utc if naive_assumption == "utc" else _server_local_tz()
        dt = dt.replace(tzinfo=tzinfo)
    return dt.astimezone(_server_local_tz())


def to_local_epoch_seconds(
    value: Any,
    *,
    naive_assumption: str = "local",
) -> int | None:
    dt = parse_datetime_to_local(value, naive_assumption=naive_assumption)
    if dt is None:
        return None
    return int(dt.timestamp())


def normalize_pipeline_timestamps(payload: Any) -> Any:
    if isinstance(payload, list):
        return [normalize_pipeline_timestamps(item) for item in payload]
    if not isinstance(payload, dict):
        return payload

    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in _PIPELINE_TIMESTAMP_FIELDS and isinstance(value, str):
            local_dt = parse_datetime_to_local(value, naive_assumption="utc")
            normalized[key] = (
                local_dt.isoformat(timespec="microseconds")
                if local_dt is not None
                else value
            )
            continue
        normalized[key] = normalize_pipeline_timestamps(value)
    return normalized


__all__ = [
    "normalize_pipeline_timestamps",
    "now_server_local_iso",
    "parse_datetime_to_local",
    "to_local_epoch_seconds",
]
