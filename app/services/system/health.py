# 文件作用：检查运行环境、依赖、目录和外部服务连通性。
# 关联说明：读取 core 配置并检查 services 依赖，是系统路由的服务实现。

from __future__ import annotations

import os
import platform
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from typing import Any

from app.core.config import CONFIG
from app.core.runtime_paths import DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR


def test_api_connection(client: Any, model: str) -> Tuple[bool, str]:
    try:
        result = client.create_chat_completion_text(
            model=model,
            messages=[{"role": "user", "content": "请回复'OK'"}],
            temperature=0.0,
            max_tokens=5,
            timeout=15,
        ).strip()
        return True, result
    except Exception as exc:
        return False, str(exc)


def run_environment_check() -> Dict[str, Any]:
    started = time.perf_counter()
    checks: List[Dict[str, Any]] = []

    _add_check(
        checks,
        "api_process",
        "API 进程",
        "api",
        "ok",
        "FastAPI 进程可响应环境检测请求",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    _check_runtime_paths(checks)
    _check_python_dependencies(checks)
    _check_models(checks)
    _check_sqlite(checks)
    _check_milvus(checks)
    _check_llm(checks)
    _check_ocr(checks)
    _check_cuda(checks)

    summary = {"ok": 0, "warning": 0, "error": 0}
    for item in checks:
        status = str(item.get("status") or "error")
        if status not in summary:
            status = "error"
        summary[status] += 1

    if summary["error"]:
        overall_status = "error"
    elif summary["warning"]:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "summary": summary,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "checked_at": int(time.time()),
        "checks": checks,
    }


def _add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    name: str,
    category: str,
    status: str,
    message: str,
    details: Dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "id": check_id,
            "name": name,
            "category": category,
            "status": status if status in {"ok", "warning", "error"} else "error",
            "message": str(message or ""),
            "details": details or {},
        }
    )


def _safe_error(exc: BaseException) -> str:
    return str(exc).strip()[:600] or exc.__class__.__name__


def _mask_secret(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return "***"
    return f"{raw[:4]}***{raw[-4:]}"


def _check_runtime_paths(checks: List[Dict[str, Any]]) -> None:
    path_items = {
        "runtime_root": CONFIG.get("runtime_root"),
        "outputs_dir": CONFIG.get("outputs_dir"),
        "uploads_dir": CONFIG.get("uploads_dir"),
        "models_dir": CONFIG.get("models_dir"),
    }
    details: Dict[str, Any] = {}
    errors: List[str] = []
    for key, raw_path in path_items.items():
        path = Path(str(raw_path or "")).resolve()
        item: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "writable": False}
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".envcheck_", dir=str(path), delete=True) as tmp:
                tmp.write(b"ok")
                tmp.flush()
            item["exists"] = True
            item["writable"] = True
        except Exception as exc:
            errors.append(f"{key}: {_safe_error(exc)}")
            item["error"] = _safe_error(exc)
        details[key] = item

    _add_check(
        checks,
        "runtime_paths",
        "运行目录读写",
        "storage",
        "error" if errors else "ok",
        "运行目录可写" if not errors else "部分运行目录不可写",
        details,
    )


def _check_python_dependencies(checks: List[Dict[str, Any]]) -> None:
    modules = [
        "fastapi",
        "openai",
        "pymilvus",
        "sentence_transformers",
        "torch",
        "transformers",
    ]
    details: Dict[str, Any] = {}
    missing: List[str] = []
    for module_name in modules:
        try:
            module = __import__(module_name)
            details[module_name] = {
                "available": True,
                "version": str(getattr(module, "__version__", "")),
            }
        except Exception as exc:
            missing.append(module_name)
            details[module_name] = {"available": False, "error": _safe_error(exc)}

    _add_check(
        checks,
        "python_dependencies",
        "Python 关键依赖",
        "dependency",
        "error" if missing else "ok",
        "关键依赖可导入" if not missing else f"缺少或无法导入依赖：{', '.join(missing)}",
        details,
    )


def _check_models(checks: List[Dict[str, Any]]) -> None:
    unsup = CONFIG.get("unsupervised") if isinstance(CONFIG.get("unsupervised"), dict) else {}
    milvus_cfg = CONFIG.get("milvus") if isinstance(CONFIG.get("milvus"), dict) else {}
    model_items = {
        "embedding_model": milvus_cfg.get("embedding_model"),
        "coverage_embed_model": unsup.get("coverage_embed_model_path"),
        "nli_model": unsup.get("nli_model_path"),
        "qa_model": unsup.get("qa_model_path"),
        "knowledge_tagger_model": DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR,
    }
    if bool(unsup.get("enable_fluency_ppl")):
        model_items["fluency_model"] = unsup.get("fluency_model_path")

    details: Dict[str, Any] = {}
    missing: List[str] = []
    for key, raw_path in model_items.items():
        path = Path(str(raw_path or "")).resolve()
        exists = path.exists()
        details[key] = {"path": str(path), "exists": exists}
        if not exists:
            missing.append(key)

    _add_check(
        checks,
        "model_files",
        "模型文件路径",
        "model",
        "error" if missing else "ok",
        "必要模型路径存在" if not missing else f"缺少模型路径：{', '.join(missing)}",
        details,
    )


def _check_sqlite(checks: List[Dict[str, Any]]) -> None:
    db_path = Path(str(os.environ.get("ADMIN_META_DB_PATH") or "")).resolve()
    if not str(os.environ.get("ADMIN_META_DB_PATH") or "").strip():
        db_path = Path(str(CONFIG.get("outputs_dir") or "")).resolve() / "admin_meta.sqlite3"
    try:
        from app.services.admin import init_db

        init_db()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS env_check_ping (id INTEGER PRIMARY KEY, ts INTEGER)")
            conn.execute("INSERT INTO env_check_ping(ts) VALUES (?)", (int(time.time()),))
            conn.execute("DELETE FROM env_check_ping WHERE id NOT IN (SELECT id FROM env_check_ping ORDER BY id DESC LIMIT 5)")
            conn.commit()
        finally:
            conn.close()
        _add_check(
            checks,
            "sqlite_admin_meta",
            "本地 SQLite 元数据",
            "database",
            "ok",
            "SQLite 元数据文件可读写",
            {"path": str(db_path)},
        )
    except Exception as exc:
        _add_check(
            checks,
            "sqlite_admin_meta",
            "本地 SQLite 元数据",
            "database",
            "error",
            f"SQLite 读写失败：{_safe_error(exc)}",
            {"path": str(db_path)},
        )


def _check_milvus(checks: List[Dict[str, Any]]) -> None:
    try:
        from app.services import milvus as milvus_service

        if not milvus_service.MILVUS_AVAILABLE:
            _add_check(
                checks,
                "milvus",
                "Milvus 向量库",
                "database",
                "error",
                "pymilvus 未安装，向量库不可用",
            )
            return

        if not milvus_service.milvus_client:
            ok, message = milvus_service.init_milvus()
            if not ok:
                _add_check(
                    checks,
                    "milvus",
                    "Milvus 向量库",
                    "database",
                    "error",
                    f"Milvus 初始化失败：{message}",
                    _milvus_config_details(),
                )
                return

        client = milvus_service.milvus_client
        field_names = []
        entity_count = None
        try:
            client.load()
        except Exception:
            pass
        try:
            field_names = [f.name for f in client.schema.fields]
        except Exception:
            field_names = []
        try:
            entity_count = int(client.num_entities)
        except Exception:
            entity_count = None

        details = _milvus_config_details()
        details.update({"fields": field_names, "entity_count": entity_count})
        _add_check(
            checks,
            "milvus",
            "Milvus 向量库",
            "database",
            "ok",
            "Milvus 主集合可连接",
            details,
        )
    except Exception as exc:
        _add_check(
            checks,
            "milvus",
            "Milvus 向量库",
            "database",
            "error",
            f"Milvus 检测失败：{_safe_error(exc)}",
            _milvus_config_details(),
        )


def _milvus_config_details() -> Dict[str, Any]:
    milvus_cfg = CONFIG.get("milvus") if isinstance(CONFIG.get("milvus"), dict) else {}
    return {
        "host": milvus_cfg.get("host"),
        "port": milvus_cfg.get("port"),
        "collection_name": milvus_cfg.get("collection_name"),
        "vector_dim": milvus_cfg.get("vector_dim"),
    }


def _check_llm(checks: List[Dict[str, Any]]) -> None:
    details = {
        "base_url": CONFIG.get("base_url"),
        "model": CONFIG.get("model"),
        "api_key": _mask_secret(CONFIG.get("api_key")),
    }
    if not str(CONFIG.get("api_key") or "").strip():
        _add_check(
            checks,
            "llm_endpoint",
            "LLM 接口",
            "endpoint",
            "error",
            "LLM_API_KEY 为空，无法调用大模型",
            details,
        )
        return
    try:
        from app.core.clients import get_default_openai_client

        success, result = test_api_connection(get_default_openai_client(), str(CONFIG.get("model") or ""))
        _add_check(
            checks,
            "llm_endpoint",
            "LLM 接口",
            "endpoint",
            "ok" if success else "error",
            "LLM 接口可调用" if success else f"LLM 接口调用失败：{result}",
            {**details, "response": str(result)[:300]},
        )
    except Exception as exc:
        _add_check(
            checks,
            "llm_endpoint",
            "LLM 接口",
            "endpoint",
            "error",
            f"LLM 接口检测失败：{_safe_error(exc)}",
            details,
        )


def _check_ocr(checks: List[Dict[str, Any]]) -> None:
    try:
        from app.services.ocr import get_active_profile

        profile = get_active_profile()
    except Exception as exc:
        _add_check(
            checks,
            "ocr_endpoint",
            "OCR 接口",
            "endpoint",
            "error",
            f"OCR 配置读取失败：{_safe_error(exc)}",
        )
        return

    post_url = str(profile.get("post_url") or "").strip()
    details = {
        "name": profile.get("name"),
        "provider": profile.get("provider"),
        "post_url": post_url,
        "timeout_seconds": profile.get("timeout_seconds"),
        "response_mode": (profile.get("response") or {}).get("mode") if isinstance(profile.get("response"), dict) else "",
    }
    if not post_url:
        _add_check(checks, "ocr_endpoint", "OCR 接口", "endpoint", "error", "OCR post_url 为空", details)
        return
    reachable, message, probe_details = _probe_http_endpoint(post_url, timeout=4)
    _add_check(
        checks,
        "ocr_endpoint",
        "OCR 接口",
        "endpoint",
        "ok" if reachable else "error",
        "OCR 地址可连通" if reachable else f"OCR 地址不可连通：{message}",
        {**details, **probe_details},
    )


def _probe_http_endpoint(url: str, *, timeout: int) -> Tuple[bool, str, Dict[str, Any]]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, "URL 格式不是 http/https", {"probe": "invalid_url"}
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.getcode())
            return True, f"HTTP {status_code}", {"probe_status": status_code}
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        if status_code < 500:
            return True, f"HTTP {status_code}", {"probe_status": status_code}
        return False, f"HTTP {status_code}", {"probe_status": status_code}
    except Exception as exc:
        return False, _safe_error(exc), {"probe_error": _safe_error(exc)}


def _check_cuda(checks: List[Dict[str, Any]]) -> None:
    require_cuda = str(os.environ.get("REQUIRE_CUDA") or "").strip().lower() in {"1", "true", "yes", "y"}
    try:
        import torch
    except Exception as exc:
        _add_check(
            checks,
            "cuda",
            "CUDA / GPU",
            "runtime",
            "error" if require_cuda else "warning",
            f"torch 不可导入，无法检测 CUDA：{_safe_error(exc)}",
            {"required": require_cuda},
        )
        return

    details: Dict[str, Any] = {
        "required": require_cuda,
        "torch_version": str(getattr(torch, "__version__", "")),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        _add_check(
            checks,
            "cuda",
            "CUDA / GPU",
            "runtime",
            "error" if require_cuda else "warning",
            "CUDA 不可用；CPU 镜像可继续运行，GPU 任务不可用",
            details,
        )
        return

    try:
        count = int(torch.cuda.device_count())
        devices = []
        for idx in range(count):
            device_info = {"index": idx, "name": torch.cuda.get_device_name(idx)}
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                device_info["memory_free_mb"] = int(free_bytes / 1024 / 1024)
                device_info["memory_total_mb"] = int(total_bytes / 1024 / 1024)
            except Exception:
                pass
            devices.append(device_info)
        if count > 0:
            with torch.cuda.device(0):
                tensor = torch.tensor([1.0], device="cuda")
                details["cuda_probe_value"] = float((tensor + 1).detach().cpu().item())
                del tensor
                torch.cuda.empty_cache()
        details["device_count"] = count
        details["devices"] = devices
        _add_check(
            checks,
            "cuda",
            "CUDA / GPU",
            "runtime",
            "ok",
            f"CUDA 可调用，检测到 {count} 张 GPU",
            details,
        )
    except Exception as exc:
        _add_check(
            checks,
            "cuda",
            "CUDA / GPU",
            "runtime",
            "error" if require_cuda else "warning",
            f"CUDA 探测失败：{_safe_error(exc)}",
            details,
        )


__all__ = ["run_environment_check", "test_api_connection"]
