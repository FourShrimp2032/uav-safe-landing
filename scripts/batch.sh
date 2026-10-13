#!/usr/bin/env bash
# KPI-4: run N headless landings in one world, then summarize.
#
#   ./scripts/batch.sh c1_field 30
#
# Each run starts from a slightly different spot (up to 2 m away) and after
# a random wait (0-40 s), so the walking person in c4_dynamic arrives at a
# different moment of the landing each time.
set -o pipefail
WORLD=${1:-c1_field}
N=${2:-30}
TIMEOUT=${RUN_TIMEOUT:-600}   # wall-clock seconds per run
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOGS=/root/landing_logs/$WORLD
mkdir -p "$LOGS"

for i in $(seq 1 "$N"); do
  DX=$(python3 -c "import random; print(round(random.uniform(-2, 2), 2))")
  DY=$(python3 -c "import random; print(round(random.uniform(-2, 2), 2))")
  WAIT=$(python3 -c "import random; print(random.randint(0, 40))")
  before=$(ls "$LOGS"/*_result.json 2>/dev/null | wc -l)
  echo "=== run $i/$N (spawn offset $DX, $DY, start after $WAIT s) ==="
  # setsid: the run gets its own process group, so stopping it below reaches
  # every process it started (PX4, Gazebo, MAVROS, bridge, node, recorder).
  # env --default-signal: bash makes background jobs ignore Ctrl+C (SIGINT)
  # and children inherit that, so without it ROS nodes can't be shut down
  # politely and get left behind, piling up run after run.
  HEADLESS=1 SPAWN_DX=$DX SPAWN_DY=$DY START_DELAY=$WAIT setsid env --default-signal=INT,QUIT \
    "$REPO/scripts/run_sim.sh" "$WORLD" > "/tmp/run_${WORLD}_$i.log" 2>&1 &
  pid=$!
  for s in $(seq 1 "$TIMEOUT"); do
    now=$(ls "$LOGS"/*_result.json 2>/dev/null | wc -l)
    [ "$now" -gt "$before" ] && break
    sleep 1
  done
  [ "$now" -gt "$before" ] || echo "run $i timed out (log: /tmp/run_${WORLD}_$i.log)"
  # Stop the whole run: TERM first so the recording gets closed properly,
  # then KILL whatever is still there.
  kill -TERM -- -"$pid" 2>/dev/null
  for s in $(seq 1 15); do kill -0 -- -"$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  "$REPO/scripts/kill_sim.sh"
  sleep 3
done
python3 "$REPO/tools/summarize.py" "$WORLD"
