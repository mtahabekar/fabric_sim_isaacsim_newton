#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_FILE="${1:-${SCRIPT_DIR}/../config/full_sim.yaml}"

set +u
source /opt/ros/jazzy/setup.bash
set -u

if [[ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]]; then
  echo "Workspace install/setup.bash was not found." >&2
  echo "Build first with: cd ${WORKSPACE_DIR} && colcon build --symlink-install" >&2
  exit 1
fi

set +u
source "${WORKSPACE_DIR}/install/setup.bash"
set -u

exec ros2 launch fabric_sim_bringup full_sim.launch.py config_file:="${CONFIG_FILE}"
