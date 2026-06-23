# 文件作用：封装 OCR 配置档案的存储、校验与激活状态管理。
# 关联说明：作为 ocr 包内的状态型能力实现，被 facade 和解析服务共同调用。

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.core.config import CONFIG


class OCRConfigStore:
    def __init__(self, config_file: Optional[str] = None) -> None:
        self.config_file = config_file or os.path.join(CONFIG["outputs_dir"], "ocr_configs.json")
        self.allowed_providers = {"batch_ocr", "process_api"}
        self.allowed_response_modes = {"structured_json", "text", "file"}

    def _default_post_url(self) -> str:
        return (
            str(os.environ.get("OCR_BATCH_URL") or "").strip()
            or "http://batch_ocr:8080/batch_ocr"
        )

    def _normalize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        name = str(profile.get("name") or "").strip()
        provider = str(profile.get("provider") or "batch_ocr").strip().lower()
        post_url = str(profile.get("post_url") or "").strip()
        timeout_seconds = profile.get("timeout_seconds")

        request_obj = profile.get("request") if isinstance(profile.get("request"), dict) else {}
        response_obj = profile.get("response") if isinstance(profile.get("response"), dict) else {}

        batch_field = str(request_obj.get("batch_field") or "files").strip() or "files"
        file_field = str(request_obj.get("file_field") or "file").strip() or "file"

        extra_form_fields = request_obj.get("extra_form_fields")
        if not isinstance(extra_form_fields, dict):
            extra_form_fields = {}
        extra_form_fields_str: Dict[str, str] = {}
        for k, v in extra_form_fields.items():
            if k is None:
                continue
            key = str(k).strip()
            if not key:
                continue
            extra_form_fields_str[key] = str(v) if v is not None else ""

        response_mode = str(response_obj.get("mode") or "").strip().lower()
        if not response_mode:
            response_mode = "structured_json" if provider == "batch_ocr" else "text"

        normalized_timeout = 600
        try:
            if timeout_seconds is not None:
                normalized_timeout = int(timeout_seconds)
        except (TypeError, ValueError):
            normalized_timeout = 600
        normalized_timeout = max(1, min(normalized_timeout, 3600))

        return {
            "name": name,
            "provider": provider,
            "post_url": post_url,
            "timeout_seconds": normalized_timeout,
            "request": {
                "batch_field": batch_field,
                "file_field": file_field,
                "extra_form_fields": extra_form_fields_str,
            },
            "response": {"mode": response_mode},
        }

    def _validate_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_profile(profile)
        if not normalized["name"]:
            raise ValueError("配置名称不能为空")
        if normalized["provider"] not in self.allowed_providers:
            raise ValueError(f"provider 仅支持: {', '.join(sorted(self.allowed_providers))}")
        if not normalized["post_url"]:
            raise ValueError("post_url 不能为空（需要填写完整 POST 地址，包含 path）")
        if normalized["response"]["mode"] not in self.allowed_response_modes:
            raise ValueError(
                f"response.mode 仅支持: {', '.join(sorted(self.allowed_response_modes))}"
            )
        return normalized

    def _ensure_store(self) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        if not os.path.exists(self.config_file):
            default_profile = {
                "name": "default",
                "provider": "batch_ocr",
                "post_url": self._default_post_url(),
                "timeout_seconds": 600,
                "request": {"batch_field": "files", "file_field": "file", "extra_form_fields": {}},
                "response": {"mode": "structured_json"},
            }
            store: Dict[str, Any] = {"active": "default", "profiles": {"default": default_profile}}
            self._save_store(store)
            return store

        with open(self.config_file, "r", encoding="utf-8") as f:
            try:
                store = json.load(f)
            except Exception:
                store = {}
        if not isinstance(store, dict):
            store = {}

        profiles_obj = store.get("profiles")
        if not isinstance(profiles_obj, dict):
            profiles_obj = {}
        store["profiles"] = profiles_obj

        active = store.get("active")
        if not isinstance(active, str) or not active.strip():
            active = "default"
        store["active"] = active

        if not profiles_obj:
            profiles_obj["default"] = {
                "name": "default",
                "provider": "batch_ocr",
                "post_url": self._default_post_url(),
                "timeout_seconds": 600,
                "request": {"batch_field": "files", "file_field": "file", "extra_form_fields": {}},
                "response": {"mode": "structured_json"},
            }
            store["active"] = "default"
            self._save_store(store)

        return store

    def _save_store(self, store: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def list_profiles(self) -> Dict[str, Any]:
        return self._ensure_store()

    def upsert_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._validate_profile(profile)
        store = self._ensure_store()
        profiles: Dict[str, Any] = store["profiles"]
        profiles[normalized["name"]] = normalized
        self._save_store(store)
        return store

    def delete_profile(self, name: str) -> Dict[str, Any]:
        store = self._ensure_store()
        if name == store.get("active"):
            raise ValueError("无法删除当前激活的配置，请先切换到其他配置")
        profiles: Dict[str, Any] = store["profiles"]
        if name in profiles:
            profiles.pop(name)
            self._save_store(store)
        return store

    def activate_profile(self, name: str) -> Dict[str, Any]:
        store = self._ensure_store()
        profiles: Dict[str, Any] = store["profiles"]
        if name not in profiles:
            raise ValueError(f"配置不存在: {name}")
        store["active"] = name
        self._save_store(store)
        return store

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        store = self._ensure_store()
        profiles = store.get("profiles")
        if not isinstance(profiles, dict):
            return None
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            return None
        return self._normalize_profile(profile)

    def get_active_profile(self) -> Dict[str, Any]:
        store = self._ensure_store()
        active = str(store.get("active") or "").strip() or "default"
        profile = self.get_profile(active)
        if profile:
            return profile
        fallback = self.get_profile("default")
        if fallback:
            return fallback
        return self._normalize_profile(
            {
                "name": "default",
                "provider": "batch_ocr",
                "post_url": self._default_post_url(),
                "timeout_seconds": 600,
                "request": {"batch_field": "files", "file_field": "file", "extra_form_fields": {}},
                "response": {"mode": "structured_json"},
            }
        )


_STORE = OCRConfigStore()


def list_profiles() -> Dict[str, Any]:
    return _STORE.list_profiles()


def upsert_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return _STORE.upsert_profile(profile)


def delete_profile(name: str) -> Dict[str, Any]:
    return _STORE.delete_profile(name)


def activate_profile(name: str) -> Dict[str, Any]:
    return _STORE.activate_profile(name)


def get_profile(name: str) -> Optional[Dict[str, Any]]:
    return _STORE.get_profile(name)


def get_active_profile() -> Dict[str, Any]:
    return _STORE.get_active_profile()

