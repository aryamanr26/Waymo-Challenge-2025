#!/usr/bin/env bash
# Run Waymo E2E tf.data loader smoke test (GCS list + one batch).
#
# Usage:
#   ./scripts/check_waymo_dataloader.sh
#   WAYMO_NUM_TEMPORAL_FRAMES=5 ./scripts/check_waymo_dataloader.sh
#
# Requires: GCP credentials for gs:// access, tensorflow, gcsfs, waymo-open-dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${SCRIPT_DIR}/check_waymo_dataloader.py" ]]; then
  echo "ERROR: missing ${SCRIPT_DIR}/check_waymo_dataloader.py" >&2
  exit 1
fi

exec python3 "${SCRIPT_DIR}/check_waymo_dataloader.py" "$@"
