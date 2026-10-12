"""Everything on the ROS 2 side of the simulation.

Start PX4 + Gazebo first (scripts/run_sim.sh does both), then:

    ros2 launch safe_landing sim.launch.py world:=c1_field

Starts the Gazebo->ROS bridge, MAVROS (connected to PX4 SITL), the landing
node and a rosbag recording of everything except raw camera frames.
"""
import os
import time

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    share = get_package_share_directory("safe_landing")
    world = LaunchConfiguration("world").perform(context)
    log_root = os.path.expanduser(LaunchConfiguration("log_dir").perform(context))
    record = LaunchConfiguration("record").perform(context).lower() == "true"
    with open(os.path.join(share, "config", "worlds.yaml")) as f:
        worlds = yaml.safe_load(f)
    if world not in worlds:
        raise RuntimeError(f"unknown world {world}; known: {', '.join(worlds)}")
    cfg = worlds[world]
    # run_sim.sh may spawn the drone away from the default spot (batch runs);
    # PX4's local frame starts wherever it spawns.
    spawn_x = float(cfg["spawn_x"]) + float(LaunchConfiguration("spawn_dx").perform(context))
    spawn_y = float(cfg["spawn_y"]) + float(LaunchConfiguration("spawn_dy").perform(context))
    mission_x = float(cfg["spawn_x"]) + float(cfg["mission_x"]) - spawn_x
    mission_y = float(cfg["spawn_y"]) + float(cfg["mission_y"]) - spawn_y
    log_dir = os.path.join(log_root, world)
    os.makedirs(log_dir, exist_ok=True)

    actions = [
        Node(package="ros_gz_bridge", executable="parameter_bridge", name="gz_bridge",
             parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml")}],
             output="screen"),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(get_package_share_directory("mavros"), "launch", "px4.launch")),
            launch_arguments={"fcu_url": "udp://:14540@127.0.0.1:14557"}.items()),
        Node(package="safe_landing", executable="landing_node", name="safe_landing",
             parameters=[os.path.join(share, "config", "params.yaml"),
                         {"use_sim_time": True, "world": world, "log_dir": log_dir,
                          "mission_x": mission_x, "mission_y": mission_y,
                          "spawn_x": spawn_x, "spawn_y": spawn_y}],
             output="screen"),
    ]
    if record:
        # NFR-04: everything the node sees and decides, at full rate.
        bag = os.path.join(log_dir, "bag_" + time.strftime("%Y%m%d_%H%M%S"))
        actions.append(ExecuteProcess(cmd=[
            "ros2", "bag", "record", "-o", bag,
            "/mavros/state", "/mavros/extended_state", "/mavros/local_position/pose",
            "/mavros/setpoint_position/local", "/landing/status",
            "/landing/range_down/scan", "/landing/range_tilt/scan",
            "/landing/debug/look", "/landing/debug/feel_points"], output="log"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="c1_field"),
        DeclareLaunchArgument("log_dir", default_value="~/landing_logs"),
        DeclareLaunchArgument("record", default_value="true"),
        DeclareLaunchArgument("spawn_dx", default_value="0.0"),
        DeclareLaunchArgument("spawn_dy", default_value="0.0"),
        OpaqueFunction(function=setup),
    ])
