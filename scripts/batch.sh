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
  HEADLESS=1 SPAWN_DX=$DX SPAWN_DY=$DY START_DELAY=$WAIT "$REPO/scripts/run_sim.sh" "$WORLD" > "/tmp/run_${WORLD}_$i.log" 2>&1 &
  pid=$!
  for s in $(seq 1 "$TIMEOUT"); do
    now=$(ls "$LOGS"/*_result.json 2>/dev/null | wc -l)
    [ "$now" -gt "$before" ] && break
    sleep 1
  done
  [ "$now" -gt "$before" ] || echo "run $i timed out (log: /tmp/run_${WORLD}_$i.log)"
  # Close the recording first, gently, so the bag gets finished and indexed
  # (killed hard it ends up empty and debug_video.py can't read it).
  pkill -INT -f "ros2 bag record" 2>/dev/null
  for s in $(seq 1 15); do pgrep -f "ros2 bag record" > /dev/null || break; sleep 1; done
  # Stop this run. Killing only run_sim.sh isn't enough: bash runs its exit
  # trap after the ros2 launch in the foreground returns, which it never does.
  pkill -f "ros2 launch safe_landing" 2>/dev/null
  pkill -x px4 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 3
done
python3 "$REPO/tools/summarize.py" "$WORLD"
