#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime-common.sh"

set_runtime_defaults
trap 'stop_runtime_children $?' EXIT INT TERM

echo "[qa-flow] starting runtime..."

start_infra_services
seed_default_ocr_profile
start_optional_classifier

start_background "QA Flow OCR API" \
    python -u /app/scripts/start_ocr_api.py

start_background "QA Flow API" \
    python -u -c 'from scripts._launch_api import run_api; run_api(reload=False)'

wait_for_http "QA Flow OCR API" "http://127.0.0.1:11169/health" 180
wait_for_http "QA Flow API" "http://127.0.0.1:12000/health" 180

echo "[qa-flow] services are running: QA Flow API :12000, QA Flow OCR API :11169"
wait -n "${RUNTIME_PIDS[@]}"
