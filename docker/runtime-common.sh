#!/usr/bin/env bash

set -Eeuo pipefail

RUNTIME_PIDS=()

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
        *) return 1 ;;
    esac
}

set_runtime_defaults() {
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
    export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
    export APP_RUNTIME_ROOT="${APP_RUNTIME_ROOT:-/app/runtime_assets}"
    export APP_MODELS_DIR="${APP_MODELS_DIR:-/app/runtime_assets/models}"
    export APP_OUTPUTS_DIR="${APP_OUTPUTS_DIR:-/app/runtime_assets/outputs}"
    export APP_UPLOADS_DIR="${APP_UPLOADS_DIR:-/app/runtime_assets/uploads}"
    export MODEL_BASE_DIR="${MODEL_BASE_DIR:-/app/runtime_assets/models/ocr}"
    export CLASSIFIER_MODEL_DIR="${CLASSIFIER_MODEL_DIR:-/app/runtime_assets/models/image_classifier}"
    export CLASSIFIER_CLASS_CONFIG_FILE="${CLASSIFIER_CLASS_CONFIG_FILE:-}"
    export CLASSIFIER_API_BASE="${CLASSIFIER_API_BASE:-http://127.0.0.1:10488}"
    export OCR_USE_GPU="${OCR_USE_GPU:-true}"
    export OCR_REPLACE_IMAGES="${OCR_REPLACE_IMAGES:-true}"
    export VLM_API_BASE="${VLM_API_BASE:-}"
    export VLM_MODEL_NAME="${VLM_MODEL_NAME:-}"
    export VLM_API_KEY="${VLM_API_KEY:-}"
    export VLM_API_TYPE="${VLM_API_TYPE:-openai}"
    export VLM_MODEL_VERSION="${VLM_MODEL_VERSION:-}"
    export SOFFICE_BINARY="${SOFFICE_BINARY:-/usr/bin/soffice}"
    export QA_FLOW_API_RELOAD="${QA_FLOW_API_RELOAD:-true}"
    export MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
    export MILVUS_PORT="${MILVUS_PORT:-19530}"
    export MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
    export MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
    export TORCH_HOME="${TORCH_HOME:-/app/runtime_assets/cache/torch}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/app/runtime_assets/cache/transformers}"
    export HF_HOME="${HF_HOME:-/app/runtime_assets/cache/huggingface}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/milvus/lib:/usr/lib/libreoffice/program:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}"
}

prepare_runtime_dirs() {
    mkdir -p \
        "${APP_RUNTIME_ROOT}" \
        "${APP_MODELS_DIR}" \
        "${APP_OUTPUTS_DIR}" \
        "${APP_UPLOADS_DIR}" \
        "${TORCH_HOME}" \
        "${TRANSFORMERS_CACHE}" \
        "${HF_HOME}" \
        /data/etcd \
        /data/minio \
        /data/milvus \
        /var/log/apiuse
}

start_background() {
    local name="$1"
    shift
    echo "[apiuse] starting ${name}..."
    "$@" &
    RUNTIME_PIDS+=("$!")
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout_seconds="${3:-120}"
    local elapsed=0

    until curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; do
        if (( elapsed >= timeout_seconds )); then
            echo "[apiuse] ${name} did not become ready within ${timeout_seconds}s: ${url}" >&2
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "[apiuse] ${name} is ready: ${url}"
}

seed_default_ocr_profile() {
    if ! is_truthy "${SEED_LOCAL_OCR_PROFILE:-true}"; then
        return 0
    fi

    python - <<'PY'
import json
import os
from pathlib import Path

runtime_root = Path(os.environ.get("APP_RUNTIME_ROOT") or "/app/runtime_assets")
outputs_dir = Path(os.environ.get("APP_OUTPUTS_DIR") or runtime_root / "outputs")
config_path = outputs_dir / "ocr_configs.json"

if config_path.exists():
    print(f"[apiuse] OCR config exists, keeping it: {config_path}")
    raise SystemExit(0)

outputs_dir.mkdir(parents=True, exist_ok=True)
profile = {
    "name": "qa-flow-local-ocr",
    "provider": "process_api",
    "post_url": "http://127.0.0.1:11169/process",
    "timeout_seconds": 600,
    "request": {
        "batch_field": "files",
        "file_field": "file",
        "extra_form_fields": {
            "output_format": "text",
            "docx_strategy": "pdf",
        },
    },
    "response": {"mode": "file"},
}
store = {"active": profile["name"], "profiles": {profile["name"]: profile}}
config_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[qa-flow] seeded local OCR profile: {config_path}")
PY
}

start_infra_services() {
    prepare_runtime_dirs

    start_background "etcd" \
        /usr/local/bin/etcd \
            --data-dir=/data/etcd \
            --listen-client-urls=http://0.0.0.0:2379 \
            --advertise-client-urls=http://127.0.0.1:2379 \
            --listen-peer-urls=http://0.0.0.0:2380 \
            --initial-advertise-peer-urls=http://127.0.0.1:2380 \
            --initial-cluster=default=http://127.0.0.1:2380 \
            --name=default

    wait_for_http "etcd" "http://127.0.0.1:2379/health" 60

    start_background "MinIO" \
        env MINIO_ROOT_USER="${MINIO_ROOT_USER}" MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}" \
        /usr/local/bin/minio server /data/minio --console-address ":9001"

    wait_for_http "MinIO" "http://127.0.0.1:9000/minio/health/live" 60

    start_background "Milvus" \
        /bin/bash -lc 'cd /milvus && export LD_LIBRARY_PATH="/milvus/lib:/usr/lib/libreoffice/program:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}" && exec /milvus/bin/milvus run standalone'

    wait_for_http "Milvus" "http://127.0.0.1:9091/healthz" 180
}

start_optional_classifier() {
    if ! is_truthy "${START_CLASSIFIER_SERVICE:-true}"; then
        return 0
    fi

    start_background "image classifier API" \
        python -u -m uvicorn app.services.image_understanding.classifier_service.main:app \
            --host 0.0.0.0 \
            --port 10488
    wait_for_http "image classifier API" "http://127.0.0.1:10488/health" 120
}

stop_runtime_children() {
    local status="${1:-$?}"
    trap - EXIT INT TERM
    if ((${#RUNTIME_PIDS[@]} > 0)); then
        echo "[apiuse] stopping child processes..."
        kill "${RUNTIME_PIDS[@]}" >/dev/null 2>&1 || true
        wait "${RUNTIME_PIDS[@]}" >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
