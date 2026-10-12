#!/usr/bin/env bash
# Build the ROS 2 package into /ws (inside the container).
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source /opt/ros/jazzy/setup.bash
mkdir -p /ws/src
ln -sfn "$REPO/ros2_ws/src/safe_landing" /ws/src/safe_landing
cd /ws && colcon build --symlink-install
echo "Built. Run: source /ws/install/setup.bash"
