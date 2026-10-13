#!/usr/bin/env bash
# Stop every simulator process: PX4, Gazebo and all ROS nodes of a run.
# Polite first (so the bag recorder can close its file), then forced.
PATTERN="ros2 launch safe_landing|ros2 bag record|gz sim|mavros_node|parameter_bridge|landing_node"
pkill -TERM -f "$PATTERN" 2>/dev/null
pkill -TERM -x px4 2>/dev/null
for _ in $(seq 1 10); do
  pgrep -f "$PATTERN" > /dev/null || pgrep -x px4 > /dev/null || exit 0
  sleep 1
done
pkill -KILL -f "$PATTERN" 2>/dev/null
pkill -KILL -x px4 2>/dev/null
exit 0
