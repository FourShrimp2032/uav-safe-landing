#!/usr/bin/env bash
# Run one landing in Gazebo.
#
#   ./scripts/run_sim.sh c1_field          # with the Gazebo window (see the browser tab)
#   HEADLESS=1 ./scripts/run_sim.sh c2_hill
#
# Worlds: c1_field, c2_hill, c3_town, c4_dynamic
# Optional: SPAWN_DX / SPAWN_DY move the start point (metres).
#           START_DELAY waits this many seconds before take-off, so the walking
#           person in c4_dynamic is somewhere else in their loop.
set -eo pipefail
WORLD=${1:-c1_field}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PX4_DIR=${PX4_DIR:-/opt/PX4-Autopilot}
DX=${SPAWN_DX:-0}
DY=${SPAWN_DY:-0}

set +u
source /opt/ros/jazzy/setup.bash
[ -f /ws/install/setup.bash ] || "$REPO/scripts/build.sh"
source /ws/install/setup.bash

# Let PX4's Gazebo launcher find our drone model, textures and worlds.
GZ_DIR="$PX4_DIR/Tools/simulation/gz"
ln -sfn "$REPO/sim/models/x500_landing" "$GZ_DIR/models/x500_landing"
ln -sfn "$REPO/sim/models/sl_assets" "$GZ_DIR/models/sl_assets"
for w in "$REPO"/sim/worlds/*.sdf; do ln -sfn "$w" "$GZ_DIR/worlds/$(basename "$w")"; done
# Our airframe (x500 + no RC / no ground station needed to arm).
for d in "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes" \
         "$PX4_DIR/build/px4_sitl_default/etc/init.d-posix/airframes"; do
  if [ -d "$d" ]; then ln -sfn "$REPO/sim/px4/airframes/4950_gz_x500_landing" "$d/"; fi
done

SPAWN=$(python3 - "$REPO" "$WORLD" "$DX" "$DY" <<'PY'
import sys, yaml
repo, world, dx, dy = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
c = yaml.safe_load(open(f"{repo}/ros2_ws/src/safe_landing/config/worlds.yaml"))[world]
print(f"{c['spawn_x'] + dx},{c['spawn_y'] + dy},0.3,0,0,0")
PY
)

cleanup() {
  pkill -f "ros2 launch safe_landing" 2>/dev/null || true
  pkill -x px4 2>/dev/null || true
  pkill -f "gz sim" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
cleanup

if [ "${HEADLESS:-0}" = "1" ]; then export HEADLESS=1; else unset HEADLESS; fi
mkdir -p /tmp/px4
echo "Starting PX4 + Gazebo ($WORLD, spawn $SPAWN)..."
(cd "$PX4_DIR" && PX4_SYS_AUTOSTART=4950 PX4_SIM_MODEL=gz_x500_landing PX4_GZ_WORLD="$WORLD" \
   PX4_GZ_MODEL_POSE="$SPAWN" ./build/px4_sitl_default/bin/px4 -d > /tmp/px4/px4.log 2>&1) &

for i in $(seq 1 180); do
  grep -q "Ready for takeoff" /tmp/px4/px4.log 2>/dev/null && break
  sleep 1
done
grep -q "Ready for takeoff" /tmp/px4/px4.log || echo "WARNING: PX4 didn't report 'Ready for takeoff' yet; continuing (see /tmp/px4/px4.log)"
if [ "${START_DELAY:-0}" != "0" ]; then
  echo "Waiting ${START_DELAY} s before take-off..."
  sleep "$START_DELAY"
fi
echo "PX4 ready. Starting MAVROS and the landing node..."
ros2 launch safe_landing sim.launch.py world:="$WORLD" spawn_dx:="$DX" spawn_dy:="$DY" \
  log_dir:=/root/landing_logs
