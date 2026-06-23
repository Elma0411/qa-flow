from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List

import uvicorn


ROOT_DIR = Path(__file__).resolve().parent.parent


def build_reload_paths() -> List[str]:
    reload_dirs = [ROOT_DIR / "app", ROOT_DIR / "qa", ROOT_DIR / "scripts"]
    return [str(p) for p in reload_dirs if p.exists()]


def build_reload_excludes() -> List[str]:
    return [
        "volumes/*",
        "volumes/**",
        "milvus_data/*",
        "milvus_data/**",
        "runtime_assets/*",
        "runtime_assets/**",
        "outputs/*",
        "outputs/**",
        "qa/outputs/*",
        "qa/outputs/**",
        "qa/uploads/*",
        "qa/uploads/**",
        "static/*",
        "static/**",
        ".git/*",
        ".git/**",
    ]


def run_api(*, host: str = "0.0.0.0", port: int = 12000, reload: bool = True) -> None:
    run_kwargs: Dict[str, object] = {"host": host, "port": port, "reload": reload}
    params = inspect.signature(uvicorn.run).parameters
    reload_dirs = build_reload_paths()
    reload_excludes = build_reload_excludes()
    if "reload_dirs" in params:
        run_kwargs["reload_dirs"] = reload_dirs
        if "reload_excludes" in params:
            run_kwargs["reload_excludes"] = reload_excludes
    elif "reload_excludes" in params:
        run_kwargs["reload_excludes"] = reload_excludes
    else:
        print("⚠️ 当前 uvicorn 不支持 reload_dirs/reload_excludes，已自动关闭 reload 以避免扫描数据目录崩溃")
        run_kwargs["reload"] = False
    if "app_dir" in params:
        run_kwargs["app_dir"] = str(ROOT_DIR)
    uvicorn.run("app.main:app", **run_kwargs)

