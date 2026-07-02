#!/usr/bin/env python3
"""Diagnose lmp_cloud LLM/VLM connectivity using the current qa-flow client rules."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.llm.vlm_client import (  # noqa: E402
    VLMClientConfig,
    create_vlm_client,
    normalize_lmp_cloud_endpoint,
    normalize_lmp_cloud_messages,
    normalize_vlm_api_type,
)


def mask_secret(value: Optional[str]) -> str:
    text = str(value or "")
    if not text:
        return "<empty>"
    if len(text) <= 10:
        return f"{text[:2]}***{text[-2:]}"
    return f"{text[:6]}***{text[-4:]}"


def default_config_candidates() -> List[Path]:
    candidates: List[Path] = []
    outputs_dir = os.environ.get("APP_OUTPUTS_DIR")
    if outputs_dir:
        candidates.append(Path(outputs_dir) / "llm_configs.json")
    candidates.append(REPO_ROOT / "runtime_assets" / "outputs" / "llm_configs.json")
    candidates.append(Path("/app/runtime_assets/outputs/llm_configs.json"))
    return candidates


def find_default_config_file() -> Optional[Path]:
    for path in default_config_candidates():
        if path.exists():
            return path
    return None


def load_profile(path: Path, profile_name: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        store = json.load(f)
    profiles = store.get("profiles") or {}
    active = str(store.get("active") or "default")
    name = profile_name or active
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(key) for key in profiles.keys())) or "<none>"
        raise SystemExit(f"Profile not found: {name}. Available profiles: {available}")
    return name, profile


def resolve_config(args: argparse.Namespace) -> Tuple[str, Dict[str, Any], Optional[Path]]:
    config_path: Optional[Path] = Path(args.config_file).expanduser() if args.config_file else find_default_config_file()
    profile_name = args.profile
    profile: Dict[str, Any] = {}
    loaded_name = "<args>"

    if config_path and config_path.exists():
        loaded_name, profile = load_profile(config_path, profile_name)
    elif not any([args.api_key, args.base_url, args.model]):
        tried = "\n  ".join(str(path) for path in default_config_candidates())
        raise SystemExit(
            "No llm_configs.json found and no direct arguments were provided.\n"
            "Tried:\n  "
            + tried
            + "\nUse --api-key --base-url --model --api-type lmp_cloud instead."
        )

    merged = dict(profile)
    for key in ("api_key", "base_url", "model", "api_type", "model_version"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    merged.setdefault("api_type", "openai")
    merged.setdefault("model_version", "")
    return loaded_name, merged, config_path


def build_messages(prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a connectivity test endpoint. Reply briefly."},
        {"role": "user", "content": prompt},
    ]


def build_lmp_payload(config: Dict[str, Any], *, stream: bool, prompt: str, max_tokens: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": str(config.get("model") or ""),
        "messages": normalize_lmp_cloud_messages(build_messages(prompt)),
        "stream": stream,
        "temperature": 0.0,
        "top_p": float(os.environ.get("VLM_API_TOP_P", "0.8") or 0.8),
        "presence_penalty": float(os.environ.get("VLM_API_PRESENCE_PENALTY", "1.0") or 1.0),
        "max_tokens": max_tokens,
    }
    model_version = str(config.get("model_version") or "").strip()
    if model_version:
        payload["modelVersion"] = model_version
    return payload


def candidate_endpoints(raw_base_url: str) -> List[Tuple[str, str]]:
    raw = str(raw_base_url or "").strip().rstrip("/")
    if not raw:
        return []

    bases = [raw]
    if raw.endswith("/V2"):
        bases.append(raw[: -len("/V2")])
    if raw.endswith("/api/vlm/chat/completions"):
        bases.append(raw[: -len("/api/vlm/chat/completions")])
    if raw.endswith("/api/vlm"):
        bases.append(raw[: -len("/api/vlm")])
    if raw.endswith("/lmp-cloud-ias-server"):
        bases.append(raw[: -len("/lmp-cloud-ias-server")])

    candidates: List[Tuple[str, str]] = [("current-code-normalized", normalize_lmp_cloud_endpoint(raw))]
    for base in bases:
        base = base.rstrip("/")
        if not base:
            continue
        candidates.extend(
            [
                ("raw-as-post-url", base),
                ("base+/api/vlm/chat/completions", base + "/api/vlm/chat/completions"),
                (
                    "base+/lmp-cloud-ias-server/api/vlm/chat/completions",
                    base + "/lmp-cloud-ias-server/api/vlm/chat/completions",
                ),
                ("base+/V2/api/vlm/chat/completions", base + "/V2/api/vlm/chat/completions"),
            ]
        )

    seen = set()
    unique: List[Tuple[str, str]] = []
    for label, url in candidates:
        if url not in seen:
            seen.add(url)
            unique.append((label, url))
    return unique


def socket_probe(url: str, timeout: float) -> Tuple[bool, str]:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False, "URL has no host"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return False, f"DNS failed: {exc}"
    addresses = sorted({info[4][0] for info in infos})
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"TCP ok: {host}:{port}; resolved={','.join(addresses[:5])}"
    except OSError as exc:
        return False, f"TCP failed: {host}:{port}; resolved={','.join(addresses[:5])}; error={exc}"


def trim_body(body: str, limit: int) -> str:
    body = body.replace("\r", "\\r")
    if len(body) <= limit:
        return body
    return body[:limit] + f"... <truncated {len(body) - limit} chars>"


def post_json(
    url: str,
    *,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
    body_limit: int,
) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": api_key,
        },
        method="POST",
    )
    started = time.time()
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "elapsed_ms": int((time.time() - started) * 1000),
                "content_type": response.headers.get("Content-Type", ""),
                "body": trim_body(body, body_limit),
            }
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_ms": int((time.time() - started) * 1000),
            "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
            "body": trim_body(body, body_limit),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "elapsed_ms": int((time.time() - started) * 1000),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def print_json_block(title: str, value: Any) -> None:
    print(f"\n## {title}")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_current_client_probe(
    config: Dict[str, Any],
    *,
    timeout: float,
    stream: bool,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    old_stream = os.environ.get("VLM_API_STREAM")
    os.environ["VLM_API_STREAM"] = "true" if stream else "false"
    try:
        vlm_config = VLMClientConfig.from_values(
            api_base=str(config.get("base_url") or ""),
            model_name=str(config.get("model") or ""),
            api_key=str(config.get("api_key") or ""),
            api_type=str(config.get("api_type") or "lmp_cloud"),
            model_version=str(config.get("model_version") or ""),
            timeout_seconds=timeout,
        )
        client = create_vlm_client(vlm_config)
        started = time.time()
        text = client.create_chat_completion_text(
            messages=build_messages(prompt),
            model=str(config.get("model") or ""),
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=None,
        )
        return {
            "ok": True,
            "elapsed_ms": int((time.time() - started) * 1000),
            "client_signature": client.public_signature(),
            "text": text,
        }
    except Exception as exc:
        signature = None
        try:
            signature = VLMClientConfig.from_values(
                api_base=str(config.get("base_url") or ""),
                model_name=str(config.get("model") or ""),
                api_key=str(config.get("api_key") or ""),
                api_type=str(config.get("api_type") or "lmp_cloud"),
                model_version=str(config.get("model_version") or ""),
                timeout_seconds=timeout,
            )
        except Exception:
            pass
        public_signature = None
        if signature is not None:
            public_signature = {
                "api_type": signature.api_type,
                "base_url": signature.base_url,
                "model_name": signature.model_name,
                "model_version": signature.model_version,
                "timeout_seconds": signature.timeout_seconds,
                "stream": signature.stream,
            }
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "client_signature": public_signature,
        }
    finally:
        if old_stream is None:
            os.environ.pop("VLM_API_STREAM", None)
        else:
            os.environ["VLM_API_STREAM"] = old_stream


def summarize_endpoint_results(results: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for item in results:
        status = item.get("result", {}).get("status")
        label = item.get("label")
        url = item.get("url")
        if status == 404:
            lines.append(f"- 404 at {label}: {url}")
        elif status in {401, 403}:
            lines.append(f"- auth failure ({status}) at {label}: URL reached, check api_key/header format")
        elif status is not None and 400 <= int(status) < 500:
            lines.append(f"- client error ({status}) at {label}: URL reached, check model/modelVersion/body")
        elif status is not None and 200 <= int(status) < 300:
            lines.append(f"- success ({status}) at {label}: {url}")
    if not lines:
        lines.append("- no HTTP status received; check DNS/TCP/proxy/TLS from this server")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose qa-flow lmp_cloud LLM config and endpoint routing."
    )
    parser.add_argument("--config-file", help="Path to llm_configs.json. Defaults to APP_OUTPUTS_DIR or runtime_assets.")
    parser.add_argument("--profile", help="Profile name in llm_configs.json. Defaults to active profile.")
    parser.add_argument("--api-key", dest="api_key", help="Override API key.")
    parser.add_argument("--base-url", dest="base_url", help="Override base_url.")
    parser.add_argument("--model", help="Override model.")
    parser.add_argument("--api-type", dest="api_type", default=None, help="Override api_type, usually lmp_cloud.")
    parser.add_argument("--model-version", dest="model_version", default=None, help="Override modelVersion.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="请只回复 OK")
    parser.add_argument("--body-limit", type=int, default=1200)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use stream=true/false. Defaults to VLM_API_STREAM or current code default true.",
    )
    parser.add_argument(
        "--probe-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="POST to common lmp_cloud endpoint candidates.",
    )
    args = parser.parse_args()

    profile_name, config, config_path = resolve_config(args)
    stream = args.stream
    if stream is None:
        stream = str(os.environ.get("VLM_API_STREAM", "true")).strip().lower() in {"1", "true", "yes", "on"}

    api_key = str(config.get("api_key") or "")
    base_url = str(config.get("base_url") or "")
    model = str(config.get("model") or "")
    api_type = normalize_vlm_api_type(str(config.get("api_type") or "lmp_cloud"))
    config["api_type"] = api_type

    print("# qa-flow lmp_cloud LLM diagnosis")
    print(f"repo_root: {REPO_ROOT}")
    print(f"config_file: {config_path or '<direct args>'}")
    print(f"profile: {profile_name}")
    print(f"api_type: {api_type}")
    print(f"api_key: {mask_secret(api_key)}")
    print(f"raw_base_url: {base_url}")
    print(f"current_code_normalized_url: {normalize_lmp_cloud_endpoint(base_url) if api_type == 'lmp_cloud' else base_url}")
    print(f"model: {model}")
    print(f"model_version: {str(config.get('model_version') or '<empty>')}")
    print(f"stream: {stream}")

    missing = [name for name, value in [("api_key", api_key), ("base_url", base_url), ("model", model)] if not value]
    if missing:
        print_json_block("Config Error", {"missing": missing})
        return 2
    if api_type != "lmp_cloud":
        print_json_block("Config Warning", {"message": "api_type is not lmp_cloud; override with --api-type lmp_cloud if needed."})

    normalized_url = normalize_lmp_cloud_endpoint(base_url)
    tcp_ok, tcp_message = socket_probe(normalized_url, min(args.timeout, 10.0))
    print_json_block("DNS/TCP Probe For Current URL", {"ok": tcp_ok, "detail": tcp_message})

    current_probe = run_current_client_probe(
        config,
        timeout=args.timeout,
        stream=stream,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    print_json_block("Current qa-flow Client Probe", current_probe)

    if not args.probe_candidates:
        return 0 if current_probe.get("ok") else 1

    payload = build_lmp_payload(config, stream=stream, prompt=args.prompt, max_tokens=args.max_tokens)
    payload_preview = dict(payload)
    payload_preview["messages"] = payload_preview["messages"][:1] + [{"role": "user", "content": "<omitted>"}]
    print_json_block("Request Shape", payload_preview)

    endpoint_results: List[Dict[str, Any]] = []
    for label, url in candidate_endpoints(base_url):
        result = post_json(
            url,
            api_key=api_key,
            payload=payload,
            timeout=args.timeout,
            body_limit=args.body_limit,
        )
        endpoint_results.append({"label": label, "url": url, "result": result})
    print_json_block("Endpoint Candidate POST Results", endpoint_results)

    print("\n## Reading The Result")
    for line in summarize_endpoint_results(endpoint_results):
        print(line)
    print("- current qa-flow code uses current_code_normalized_url and Authorization: <api_key> without Bearer.")
    print("- if only a non-current candidate succeeds, update the saved base_url or the normalization rule.")
    return 0 if current_probe.get("ok") or any(item["result"].get("ok") for item in endpoint_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
