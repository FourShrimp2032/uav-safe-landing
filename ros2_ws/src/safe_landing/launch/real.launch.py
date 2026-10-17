"""Everything on the ROS 2 side of the real drone (runs on the Raspberry Pi).

    ros2 launch safe_landing real.launch.py

Starts MAVROS on the serial link to the flight controller, the Pi camera
(camera_ros), camera_prep (undistort -> /landing/camera/image), the
rangefinder driver, the landing node and a rosbag recording. The landing
node waits until the pilot switches to OFFBOARD; switching out of OFFBOARD
takes control back at any moment.

Bench test without flying: the same launch with the propellers off. The
node stays in WAIT_OFFBOARD, and every topic can be checked (see
docs/real_drone.md).
"""
import os
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    share = get_package_share_directory("safe_landing")
    arg = lambda name: LaunchConfiguration(name).perform(context)
    params = [os.path.join(share, "config", "params.yaml"),
              os.path.join(share, "config", "params_real.yaml")]
    log_dir = os.path.join(os.path.expanduser(arg("log_dir")), "real")
    os.makedirs(log_dir, exist_ok=True)

    actions = [
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(get_package_share_directory("mavros"), "launch", "px4.launch")),
            launch_arguments={"fcu_url": arg("fcu_url")}.items()),
        Node(package="camera_ros", executable="camera_node", name="camera",
             parameters=[{"width": int(arg("raw_width")), "height": int(arg("raw_height")),
                          "format": "RGB888"}],
             output="screen"),
        Node(package="safe_landing", executable="camera_prep", name="camera_prep",
             parameters=params, remappings=[("image_raw", "/camera/image_raw")], output="screen"),
        Node(package="safe_landing", executable="range_driver", name="range_driver",
             parameters=params, output="screen"),
        Node(package="safe_landing", executable="landing_node", name="safe_landing",
             parameters=params + [{"world": "real", "log_dir": log_dir}],
             output="screen"),
    ]
    if arg("record").lower() == "true":
        topics = ["/mavros/state", "/mavros/extended_state", "/mavros/local_position/pose",
                  "/mavros/setpoint_position/local", "/landing/status",
                  "/landing/range_down/scan", "/landing/range_tilt/scan",
                  "/landing/debug/look", "/landing/debug/feel_points"]
        # Camera frames are big (~14 MB/s), but they are the training data
        # for plan B, so they can be switched on for data-collection flights.
        if arg("record_camera").lower() == "true":
            topics.append("/landing/camera/image")
        bag = os.path.join(log_dir, "bag_" + time.strftime("%Y%m%d_%H%M%S"))
        actions.append(ExecuteProcess(cmd=["ros2", "bag", "record", "-o", bag, *topics], output="log"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("fcu_url", default_value="/dev/ttyAMA0:921600"),
        DeclareLaunchArgument("raw_width", default_value="2304"),   # Camera Module 3 full-FOV binned mode
        DeclareLaunchArgument("raw_height", default_value="1296"),
        DeclareLaunchArgument("log_dir", default_value="~/landing_logs"),
        DeclareLaunchArgument("record", default_value="true"),
        DeclareLaunchArgument("record_camera", default_value="false"),
        OpaqueFunction(function=setup),
    ])
