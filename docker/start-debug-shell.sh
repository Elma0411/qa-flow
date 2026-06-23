#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime-common.sh"

set_runtime_defaults
trap 'stop_runtime_children $?' EXIT INT TERM

echo "[apiuse] starting debug shell..."

if is_truthy "${DEBUG_START_INFRA:-true}"; then
    start_infra_services
    seed_default_ocr_profile
else
    prepare_runtime_dirs
fi

cat <<'EOF'
[apiuse] main APIs are not started in this debug container.
[apiuse] useful manual commands:
  python -u /app/scripts/start_ocr_api.py
  python -u -c 'from scripts._launch_api import run_api; run_api(reload=False)'
  check-merged-runtime
EOF

/bin/bash -l
stop_runtime_children $?
