# 文件作用：封装大模型配置档案的存储、校验与激活状态管理。
# 关联说明：作为 llm_config 包内的状态型能力实现，被 facade 和路由共同调用。

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.core import clients
from app.core.config import CONFIG


class LLMConfigStore:
    def __init__(self, config_file: Optional[str] = None) -> None:
        self.config_file = config_file or os.path.join(CONFIG["outputs_dir"], "llm_configs.json")

    def _ensure_store(self) -> Dict[str, object]:
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        if not os.path.exists(self.config_file):
            default_profile = {
                "name": "default",
                "api_key": CONFIG.get("api_key") or "",
                "base_url": CONFIG.get("base_url") or "",
                "model": CONFIG.get("model") or "",
            }
            store = {"active": "default", "profiles": {"default": default_profile}}
            self._save_store(store)
            return store
        with open(self.config_file, "r", encoding="utf-8") as f:
            try:
                store = json.load(f)
            except Exception:
                store = {}
        if "profiles" not in store or not isinstance(store["profiles"], dict):
            store["profiles"] = {}
        if "active" not in store:
            store["active"] = "default"
        if not store["profiles"]:
            store["profiles"]["default"] = {
                "name": "default",
                "api_key": CONFIG.get("api_key") or "",
                "base_url": CONFIG.get("base_url") or "",
                "model": CONFIG.get("model") or "",
            }
            store["active"] = "default"
            self._save_store(store)
        return store

    def _save_store(self, store: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)

    def list_profiles(self) -> Dict[str, object]:
        return self._ensure_store()

    def upsert_profile(self, name: str, api_key: str, base_url: str, model: str) -> Dict[str, object]:
        if not name:
            raise ValueError("配置名称不能为空")
        store = self._ensure_store()
        store["profiles"][name] = {
            "name": name,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
        }
        self._save_store(store)
        return store

    def delete_profile(self, name: str) -> Dict[str, object]:
        store = self._ensure_store()
        if name == store.get("active"):
            raise ValueError("无法删除当前激活的配置，请先切换到其他配置")
        if name in store["profiles"]:
            store["profiles"].pop(name)
            self._save_store(store)
        return store

    def activate_profile(self, name: str) -> Dict[str, object]:
        store = self._ensure_store()
        profile: Optional[Dict[str, str]] = store["profiles"].get(name)  # type: ignore
        if not profile:
            raise ValueError(f"配置不存在: {name}")
        CONFIG["api_key"] = profile.get("api_key") or ""
        CONFIG["base_url"] = profile.get("base_url") or ""
        CONFIG["model"] = profile.get("model") or ""

        unsup = CONFIG.get("unsupervised")
        if isinstance(unsup, dict) and not bool(unsup.get("hypothesis_llm_locked", False)):
            unsup["hypothesis_api_key"] = CONFIG.get("api_key") or ""
            unsup["hypothesis_base_url"] = CONFIG.get("base_url") or ""
            unsup["hypothesis_model"] = CONFIG.get("model") or ""
            CONFIG["unsupervised"] = unsup

        store["active"] = name
        self._save_store(store)
        clients.clear_default_client_cache()
        return store


LLM_CONFIG_STORE = LLMConfigStore()


def list_profiles() -> Dict[str, object]:
    return LLM_CONFIG_STORE.list_profiles()


def upsert_profile(name: str, api_key: str, base_url: str, model: str) -> Dict[str, object]:
    return LLM_CONFIG_STORE.upsert_profile(name, api_key, base_url, model)


def delete_profile(name: str) -> Dict[str, object]:
    return LLM_CONFIG_STORE.delete_profile(name)


def activate_profile(name: str) -> Dict[str, object]:
    return LLM_CONFIG_STORE.activate_profile(name)
