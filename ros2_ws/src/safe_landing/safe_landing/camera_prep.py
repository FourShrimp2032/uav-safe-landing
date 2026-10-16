"""Real camera frames -> /landing/camera/image, the ideal pinhole image the
landing node expects (the Gazebo camera publishes that topic in the sim).

Subscribes to the raw frames from camera_ros, removes the lens distortion
with the calibration, turns the frame so its top points to the nose, and
publishes it at camera_width x camera_height with camera_hfov_deg (the same
parameters the landing node uses, so both agree by construction).
Without a calibration file it only resizes, and says so: fine for a bench
test, not for flying.
"""
from __future__ import annotations

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .camera_model import load_calibration, max_hfov_deg, rotate, scaled, undistort_maps


class CameraPrep(Node):
    def __init__(self):
        super().__init__("camera_prep")
        d = self.declare_parameter
        self.w = d("camera_width", 1280).value
        self.h = d("camera_height", 720).value
        self.hfov = d("camera_hfov_deg", 95.0).value
        self.rot = d("camera_rotate_deg", 0).value
        self.max_rate = d("camera_rate_hz", 5.0).value     # the node looks at 5 frames/s in the sim
        path = os.path.expanduser(d("calibration_file", "").value)
        self.cal = None
        if path:
            self.cal = load_calibration(path)
            widest = max_hfov_deg(self.cal)
            self.get_logger().info(f"calibration {path}: widest usable HFOV {widest:.1f} deg, "
                                   f"using {self.hfov:.1f} deg")
            if self.hfov > widest:
                self.get_logger().warn("camera_hfov_deg is wider than the lens covers: "
                                       "the image edges will be black (counted as bad ground)")
        else:
            self.get_logger().warn("no calibration_file: frames are only resized. Do not fly like this.")
        self.maps = None
        self.bridge = CvBridge()
        self.last_t = -1e9
        self.pub = self.create_publisher(Image, "/landing/camera/image", qos_profile_sensor_data)
        self.create_subscription(Image, "image_raw", self.on_image, qos_profile_sensor_data)

    def on_image(self, msg: Image) -> None:
        t = self.get_clock().now().nanoseconds * 1e-9
        if t - self.last_t < 1.0 / self.max_rate - 0.01:
            return
        self.last_t = t
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if self.cal is not None:
            if self.maps is None:
                h, w = img.shape[:2]
                self.maps = undistort_maps(scaled(self.cal, w, h), self.w, self.h, self.hfov)
            img = cv2.remap(img, *self.maps, cv2.INTER_LINEAR)
        elif img.shape[1] != self.w or img.shape[0] != self.h:
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        out = self.bridge.cv2_to_imgmsg(rotate(img, self.rot), "bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CameraPrep()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
