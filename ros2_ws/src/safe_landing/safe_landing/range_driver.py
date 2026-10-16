"""Real-drone rangefinders -> the two LaserScan topics the landing node reads
(the same ones the Gazebo bridge publishes in the simulation).

  tilted TF-Luna  : wired to the Pi's UART, read here directly
  downward TF-Luna: wired to the flight controller, so PX4 can land on it even
                    if the Pi stops; read back through MAVROS (sensor_msgs/Range).
                    down_source:=serial reads it from a second Pi UART instead.

Every frame the sensor sends is published, also when the reading is not
usable (weak return in sunlight, out of range): it goes out as +inf. The
landing node then knows the sensor is alive but sees nothing, which is
different from a sensor that stopped talking (no messages -> abort).
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Range

from .tfluna import MAX_RANGE_M, MIN_RANGE_M, FrameParser


class SerialTFLuna:
    def __init__(self, port: str, baud: int):
        import serial  # python3-serial; imported here so tests don't need it
        self.port = serial.Serial(port, baud, timeout=0)
        self.parser = FrameParser()

    def read(self):
        n = self.port.in_waiting
        return self.parser.feed(self.port.read(n)) if n else []


class RangeDriver(Node):
    def __init__(self):
        super().__init__("range_driver")
        d = self.declare_parameter
        self.rate = d("rate_hz", 50.0).value          # the simulated sensors run at 50 Hz
        self.tilt = SerialTFLuna(d("tilt_port", "/dev/ttyAMA3").value, d("baud", 115200).value)
        self.down_source = d("down_source", "mavros").value
        self.tilt_pub = self.create_publisher(LaserScan, "/landing/range_tilt/scan", qos_profile_sensor_data)
        self.down_pub = self.create_publisher(LaserScan, "/landing/range_down/scan", qos_profile_sensor_data)
        self.down = None
        if self.down_source == "serial":
            self.down = SerialTFLuna(d("down_port", "/dev/ttyAMA4").value, d("baud", 115200).value)
        elif self.down_source == "mavros":
            topic = d("down_topic", "/mavros/distance_sensor/rangefinder_pub").value
            self.create_subscription(Range, topic, self.on_down_range, qos_profile_sensor_data)
            self.get_logger().info(f"downward rangefinder from {topic}")
        else:
            raise ValueError(f"down_source must be mavros or serial, not {self.down_source}")
        self.last = {"tilt": None, "down": None}
        self.count = {"tilt": 0, "down": 0}
        self.create_timer(1.0 / 200.0, self.poll)     # TF-Luna sends 100 frames/s
        self.create_timer(1.0 / self.rate, self.publish)
        self.create_timer(5.0, self.report)

    def poll(self) -> None:
        for name, sensor in (("tilt", self.tilt), ("down", self.down)):
            if sensor is None:
                continue
            readings = sensor.read()
            if readings:
                r = readings[-1]
                self.last[name] = r.distance_m if r.valid else math.inf

    def on_down_range(self, msg: Range) -> None:
        r = msg.range
        ok = math.isfinite(r) and max(msg.min_range, MIN_RANGE_M) <= r <= min(msg.max_range, MAX_RANGE_M)
        self.down_pub.publish(self.scan(r if ok else math.inf, msg.header.frame_id or "range_down"))
        self.count["down"] += 1

    def publish(self) -> None:
        # Publish the newest reading once, then wait for a new one: a
        # sensor that stops sending must stop the messages too.
        for name, pub in (("tilt", self.tilt_pub), ("down", self.down_pub)):
            if name == "down" and self.down is None:
                continue
            r = self.last[name]
            if r is None:
                continue
            pub.publish(self.scan(r, f"range_{name}"))
            self.last[name] = None
            self.count[name] += 1

    def scan(self, r: float, frame: str) -> LaserScan:
        m = LaserScan()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = frame
        m.angle_min = m.angle_max = 0.0
        m.angle_increment = 0.0
        m.range_min, m.range_max = MIN_RANGE_M, MAX_RANGE_M
        m.ranges = [float(r)]
        return m

    def report(self) -> None:
        self.get_logger().info(f"rangefinders, messages in last 5 s: tilt {self.count['tilt']}, "
                               f"down {self.count['down']}")
        self.count = {"tilt": 0, "down": 0}


def main(args=None):
    rclpy.init(args=args)
    node = RangeDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
