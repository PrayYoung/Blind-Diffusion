#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${VAST_CONFIG_FILE:-$ROOT_DIR/infra/vast_config.env}"

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

if command -v vast-run >/dev/null 2>&1; then
  exec vast-run run --project-root "$ROOT_DIR" "$@"
fi

echo "vast-run is not installed. Install it first or expose it on PATH." >&2
exit 1
