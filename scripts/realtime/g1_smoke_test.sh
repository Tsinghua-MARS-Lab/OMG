#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-holomotion_orin_deploy}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
HOLOMOTION_DEPLOY_DIR="${HOLOMOTION_DEPLOY_DIR:-/home/unitree/holomotion/deployment/unitree_g1_ros2_29dof}"
OMG_DIR="${OMG_DIR:-/home/unitree/OMG}"
HOLOMOTION_CONFIG="${HOLOMOTION_CONFIG:-${HOLOMOTION_DEPLOY_DIR}/src/config/g1_29dof_holomotion.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/holomotion_deploy/bin/python}"
LOWSTATE_TOPIC="${LOWSTATE_TOPIC:-/lowstate}"
CYCLONEDDS_INTERFACE="${CYCLONEDDS_INTERFACE:-eth0}"
RUN_POLICY_DRY_RUN="${RUN_POLICY_DRY_RUN:-1}"
DRY_RUN_CONFIG="${DRY_RUN_CONFIG:-/tmp/omg_policy_node_smoke.yaml}"
LATEST_OBS_URI="${LATEST_OBS_URI:-tcp://127.0.0.1:6001}"
LATEST_OBS_TOPIC="${LATEST_OBS_TOPIC:-obs65}"
LATEST_OBS_MODE="${LATEST_OBS_MODE:-connect}"
ZMQ_JITTER_DELAY_FRAMES="${ZMQ_JITTER_DELAY_FRAMES:-0}"
MAX_DATA_AGE="${MAX_DATA_AGE:-0.6}"
DRIVER_DURATION="${DRIVER_DURATION:-12}"
DRIVER_RATE="${DRIVER_RATE:-50}"
A_START="${A_START:-1.0}"
B_START="${B_START:-2.5}"
POLICY_READY_SLEEP="${POLICY_READY_SLEEP:-12}"

usage() {
  cat <<'EOF'
Run the Unitree G1 OMG realtime smoke test.

This script does not start HoloMotion main_node, does not enable motors, and does
not publish /lowcmd. It first checks host/container state, paths, ROS
environment, Unitree messages, lowstate visibility, OMG imports, config
files. It then starts only HoloMotion policy_node_29dof against OMG
obs65 and runs the dry-run ROS driver to validate motion-policy action output
without robot torque commands.

Environment overrides:
  CONTAINER_NAME         default: holomotion_orin_deploy
  DOCKER_BIN            default: docker, set to "sudo docker" if needed
  HOLOMOTION_DEPLOY_DIR default: /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof
  OMG_DIR        default: /home/unitree/OMG
  HOLOMOTION_CONFIG     default: $HOLOMOTION_DEPLOY_DIR/src/config/g1_29dof_holomotion.yaml
  LOWSTATE_TOPIC        default: /lowstate
  CYCLONEDDS_INTERFACE  default: eth0
  RUN_POLICY_DRY_RUN    default: 1; set to 0 for preflight-only checks
  LATEST_OBS_URI        default: tcp://127.0.0.1:6001
  LATEST_OBS_TOPIC      default: obs65
  DRIVER_DURATION       default: 12
  A_START / B_START     default: 1.0 / 2.5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

pass() {
  printf '[preflight:ok] %s\n' "$*"
}

fail() {
  printf '[preflight:fail] %s\n' "$*" >&2
  exit 1
}

run_host() {
  printf '[preflight:run] %s\n' "$*"
  "$@"
}

docker_exec() {
  ${DOCKER_BIN} exec "$CONTAINER_NAME" "$@"
}

docker_exec_bash() {
  ${DOCKER_BIN} exec "$CONTAINER_NAME" bash -lc "$1"
}

command -v ${DOCKER_BIN%% *} >/dev/null 2>&1 || fail "docker command not found: ${DOCKER_BIN}"
run_host ${DOCKER_BIN} inspect "$CONTAINER_NAME" >/tmp/omg_preflight_container.json \
  || fail "container not found: ${CONTAINER_NAME}"
if [[ "$(${DOCKER_BIN} inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]]; then
  fail "container is not running: ${CONTAINER_NAME}"
fi
pass "container is running: ${CONTAINER_NAME}"

IMAGE="$(${DOCKER_BIN} inspect -f '{{.Config.Image}}' "$CONTAINER_NAME")"
pass "container image: ${IMAGE}"

[[ -d "${OMG_DIR}" ]] || fail "OMG directory missing on host: ${OMG_DIR}"
if git -C "${OMG_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  pass "OMG commit: $(git -C "${OMG_DIR}" rev-parse --short HEAD)"
  if [[ -n "$(git -C "${OMG_DIR}" status --short)" ]]; then
    printf '[preflight:warn] OMG worktree has local changes\n' >&2
    git -C "${OMG_DIR}" status --short >&2
  fi
else
  fail "OMG directory is not a git checkout: ${OMG_DIR}"
fi

docker_exec_bash "test -d '${HOLOMOTION_DEPLOY_DIR}'" || fail "HoloMotion deploy dir missing in container: ${HOLOMOTION_DEPLOY_DIR}"
docker_exec_bash "test -f '${HOLOMOTION_CONFIG}'" || fail "HoloMotion config missing in container: ${HOLOMOTION_CONFIG}"
docker_exec_bash "test -x '${PYTHON_BIN}'" || fail "Python not executable in container: ${PYTHON_BIN}"
pass "container paths exist"

docker_exec_bash "
set -euo pipefail
source /opt/ros/humble/setup.bash
source /root/unitree_ros2/setup.sh
cd '${HOLOMOTION_DEPLOY_DIR}'
source install/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>${CYCLONEDDS_INTERFACE}</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
'${PYTHON_BIN}' - <<'PY'
import importlib
for name in ['rclpy', 'unitree_hg.msg', 'zmq', 'yaml', 'numpy']:
    importlib.import_module(name)
print('python_imports_ok')
PY
ros2 topic list >/tmp/omg_preflight_topics.txt
grep -qx '${LOWSTATE_TOPIC}' /tmp/omg_preflight_topics.txt
" || fail "container ROS/Python/lowstate check failed; verify CYCLONEDDS_INTERFACE=${CYCLONEDDS_INTERFACE} and LOWSTATE_TOPIC=${LOWSTATE_TOPIC}"
pass "ROS environment and lowstate topic are visible: ${LOWSTATE_TOPIC}"

docker_exec_bash "
set -euo pipefail
cd '${OMG_DIR}'
PYTHONPATH='${OMG_DIR}/src':\$PYTHONPATH '${PYTHON_BIN}' - <<'PY'
import importlib
for name in [
    'omg.cli.realtime.holomotion_real_bridge',
    'omg.cli.realtime.planner_server',
    'omg.realtime.protocol',
    'omg.realtime.transport',
]:
    importlib.import_module(name)
print('omg_realtime_imports_ok')
PY
" || fail "OMG realtime imports failed inside container"
pass "OMG realtime imports work"

pass "preflight complete; run policy dry-run next before enabling true robot motion"

if [[ "${RUN_POLICY_DRY_RUN}" != "1" ]]; then
  pass "policy dry-run skipped; set RUN_POLICY_DRY_RUN=1 to enable"
  exit 0
fi

ROS_ENV="
source /opt/ros/humble/setup.bash
source /root/unitree_ros2/setup.sh
cd '$HOLOMOTION_DEPLOY_DIR'
source install/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>${CYCLONEDDS_INTERFACE}</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
export LD_LIBRARY_PATH=/host_gpu:/cuda_base:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/lib/aarch64-linux-gnu/:/root/miniconda3/envs/holomotion_deploy/lib:/root/miniconda3/envs/holomotion_deploy/lib/stubs:\$LD_LIBRARY_PATH
"

printf '[dry-run:run] generating config: %s\n' "${DRY_RUN_CONFIG}"
${DOCKER_BIN} exec -i "$CONTAINER_NAME" env \
  HOLOMOTION_CONFIG="$HOLOMOTION_CONFIG" \
  DRY_RUN_CONFIG="$DRY_RUN_CONFIG" \
  LATEST_OBS_URI="$LATEST_OBS_URI" \
  LATEST_OBS_TOPIC="$LATEST_OBS_TOPIC" \
  LATEST_OBS_MODE="$LATEST_OBS_MODE" \
  ZMQ_JITTER_DELAY_FRAMES="$ZMQ_JITTER_DELAY_FRAMES" \
  MAX_DATA_AGE="$MAX_DATA_AGE" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import yaml

src = Path(os.environ["HOLOMOTION_CONFIG"])
dst = Path(os.environ["DRY_RUN_CONFIG"])
with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

vr = cfg.setdefault("vr", {})
vr.update(
    {
        "enable_teleop_reference": True,
        "latest_obs_zmq_uri": os.environ["LATEST_OBS_URI"],
        "latest_obs_zmq_topic": os.environ["LATEST_OBS_TOPIC"],
        "latest_obs_zmq_mode": os.environ["LATEST_OBS_MODE"],
        "latest_obs_zmq_conflate": True,
        "zmq_jitter_delay_frames": int(os.environ["ZMQ_JITTER_DELAY_FRAMES"]),
        "max_data_age": float(os.environ["MAX_DATA_AGE"]),
        "require_vr_data_for_motion": True,
        "timing_debug_enabled": True,
        "timing_debug_log_interval_sec": 1.0,
        "timing_debug_log_per_loop": False,
    }
)
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(dst)
PY

printf '[dry-run:run] starting HoloMotion policy_node_29dof\n'
docker_exec_bash "$ROS_ENV
'$PYTHON_BIN' install/humanoid_control/lib/humanoid_control/policy_node_29dof --ros-args -p config_path:='$DRY_RUN_CONFIG'
" &
POLICY_PID=$!

cleanup() {
  set +e
  ${DOCKER_BIN} exec "$CONTAINER_NAME" bash -lc "pkill -f policy_node_29dof || true" >/dev/null 2>&1
  wait "$POLICY_PID" >/dev/null 2>&1
}
trap cleanup EXIT

printf '[dry-run:run] waiting %ss for policy setup\n' "${POLICY_READY_SLEEP}"
sleep "$POLICY_READY_SLEEP"

printf '[dry-run:run] running OMG driver\n'
docker_exec_bash "$ROS_ENV
PYTHONPATH='$OMG_DIR/src':\$PYTHONPATH '$PYTHON_BIN' -m omg.cli.realtime.policy_node_smoke_driver \
  --duration '$DRIVER_DURATION' \
  --rate '$DRIVER_RATE' \
  --a-start '$A_START' \
  --b-start '$B_START'
"

pass "policy dry-run complete; no /lowcmd publisher was started"
