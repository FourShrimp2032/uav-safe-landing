"""The safe-landing state machine ("look, then feel").

    TAKEOFF / WAIT_OFFBOARD -> GOTO_MISSION -> LOOK -> MOVE -> CONFIRM -> DESCEND -> LAND
                                                ^                            |
                                                +-------- GO_AROUND <--------+
                                                          (after max_attempts: RTL)

LOOK     camera picks open spots near the mission centre (look.py)
MOVE     fly over the best one at look altitude
CONFIRM  look again from straight above; skip the spot if it no longer looks clear
DESCEND  go down while turning; rangefinder points check slope and bumps (feel.py);
         the camera keeps watching for moving objects (motion.py)
LAND     PX4 AUTO.LAND does the touchdown
GO_AROUND  climb back, mark the spot as bad, look again

Talks to PX4 through MAVROS: position setpoints in OFFBOARD mode, and
set_mode for AUTO.LAND / AUTO.RTL. Every decision is logged to CSV at
10 Hz and a result JSON is written when the run ends.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import ExtendedState, State as FcuState
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String

from .feel import FeelParams, GroundFeel
from .geometry import (Camera, beam_direction, beam_hit, intrinsics_from_fov, quat_to_rot,
                       rot_yaw, yaw_to_quat)
from .look import LookParams, draw_debug, find_spots
from .motion import MotionDetector, MotionParams

LANDED_STATE_ON_GROUND = 1


class S(Enum):
    WAIT_FCU = "WAIT_FCU"
    TAKEOFF = "TAKEOFF"
    WAIT_OFFBOARD = "WAIT_OFFBOARD"
    GOTO_MISSION = "GOTO_MISSION"
    LOOK = "LOOK"
    MOVE = "MOVE"
    CONFIRM = "CONFIRM"
    DESCEND = "DESCEND"
    LAND = "LAND"
    GO_AROUND = "GO_AROUND"
    RTL = "RTL"
    DONE = "DONE"


OFFBOARD_STATES = {S.GOTO_MISSION, S.LOOK, S.MOVE, S.CONFIRM, S.DESCEND, S.GO_AROUND}


class SafeLandingNode(Node):
    def __init__(self):
        super().__init__("safe_landing")
        d = self.declare_parameter
        self.world = d("world", "unknown").value
        self.auto_takeoff = d("auto_takeoff", True).value
        self.use_mission = d("use_mission_point", True).value
        self.mission_xy = np.array([d("mission_x", 0.0).value, d("mission_y", 0.0).value])
        # Where the local frame's origin is in the Gazebo world (sim only),
        # so results can be checked against the world's ground truth.
        self.origin_world = np.array([d("spawn_x", 0.0).value, d("spawn_y", 0.0).value])
        self.look_alt = d("look_altitude_m", 15.0).value
        self.feel_start_agl = d("feel_start_agl_m", 5.0).value
        self.final_agl = d("final_agl_m", 1.5).value
        self.v_fast = d("descent_speed_fast", 1.0).value
        self.v_slow = d("descent_speed_slow", 0.4).value
        self.yaw_rate = math.radians(d("yaw_rate_deg_s", 45.0).value)
        self.max_attempts = d("max_attempts", 3).value
        self.max_looks = d("max_looks", 8).value
        self.final_hold_timeout = d("final_hold_timeout_s", 10.0).value
        self.moving_buffer = d("moving_buffer_m", 1.5).value
        self.slope_block_radius = d("slope_block_radius_m", 4.0).value
        self.log_dir = os.path.expanduser(d("log_dir", "~/landing_logs").value)
        # camera
        w = d("camera_width", 960).value
        h = d("camera_height", 540).value
        hfov = math.radians(d("camera_hfov_deg", 66.0).value)
        self.cam = Camera(*intrinsics_from_fov(w, h, hfov),
                          offset_body=d("camera_offset", [0.05, 0.0, -0.06]).value)
        self.camera_timeout = d("camera_timeout_s", 2.0).value
        # rangefinders
        self.down_offset = np.array(d("range_down_offset", [-0.03, 0.0, -0.06]).value)
        self.tilt_offset = np.array(d("range_tilt_offset", [0.0, 0.0, -0.06]).value)
        self.down_dir = beam_direction(0.0)
        self.tilt_dir = beam_direction(d("range_tilt_deg", 15.0).value)
        self.range_timeout = d("range_timeout_s", 0.5).value

        self.look_params = LookParams(
            required_radius_m=d("required_radius_m", 1.0).value,
            edge_threshold=d("look_edge_threshold", 6.0).value,
            bad_cell_fraction=d("look_bad_cell_fraction", 0.10).value)
        self.feel_params = FeelParams(
            check_radius_m=d("feel_check_radius_m", 1.3).value,
            max_slope_deg=d("max_slope_deg", 10.0).value,
            bump_threshold_m=d("bump_threshold_m", 0.12).value,
            min_points=d("feel_min_points", 80).value)
        self.R = self.look_params.required_radius_m

        self.bridge = CvBridge()
        self.motion = MotionDetector(MotionParams())

        # --- state ---
        self.fcu = FcuState()
        self.landed_state = 0
        self.R_wb = np.eye(3)
        self.t_wb = None
        self.home_z = None
        self.ground_z = 0.0
        self.down_hit = None
        self.last_down = -1e9
        self.last_tilt = -1e9
        self.img = None
        self.img_t = -1e9
        self.img_pose = None
        self.moving_world: list[tuple[float, float, float]] = []   # x, y, radius (m)
        self.moving_t = -1e9
        self.sp = None             # [x, y, z, yaw]
        self.state = S.WAIT_FCU
        self.state_t = self.now()
        self.offboard_seen = False
        self.last_req = -1e9
        self.queue: list[np.ndarray] = []
        self.target = None
        self.bad_spots: list[tuple[float, float, float]] = []
        # (x, y, ux, uy): spot rejected for slope and the uphill direction there
        self.slope_hints: list[tuple[float, float, float, float]] = []
        self.feel: GroundFeel | None = None
        self.feel_result = None
        self.attempts = 0
        self.looks = 0
        self.arrive_t = None
        self.hold_t = None
        self.events: list[dict] = []
        self.look_times: list[float] = []
        self.degraded: list[str] = []
        self.result_written = False
        self.tick = 0
        self.start_wall = time.time()

        # --- ROS I/O ---
        self.create_subscription(FcuState, "/mavros/state", self.on_state, qos_profile_sensor_data)
        self.create_subscription(ExtendedState, "/mavros/extended_state", self.on_ext,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, "/landing/camera/image", self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/landing/range_down/scan", self.on_down,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/landing/range_tilt/scan", self.on_tilt,
                                 qos_profile_sensor_data)
        self.sp_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)
        self.status_pub = self.create_publisher(String, "/landing/status", 10)
        self.debug_pub = self.create_publisher(Image, "/landing/debug/look", 2)
        self.cloud_pub = self.create_publisher(PointCloud2, "/landing/debug/feel_points", 2)
        self.arm_cli = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")

        os.makedirs(self.log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.world}_{stamp}"
        self.csv_file = open(os.path.join(self.log_dir, f"{self.run_id}_log.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["t", "state", "x", "y", "z", "yaw", "agl", "target_x", "target_y",
                           "feel_points", "feel_sectors", "slope_deg", "bumps", "moving",
                           "camera_ok", "ranges_ok", "mode", "armed"])
        self.dt = 0.05
        self.create_timer(self.dt, self.step)
        self.get_logger().info(f"safe_landing ready, world={self.world}, logs in {self.log_dir}")

    # ------------------------------------------------------------------ utils
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def goto(self, new: S, why: str = "") -> None:
        self.get_logger().info(f"{self.state.value} -> {new.value} {why}")
        self.events.append({"t": round(self.now(), 2), "from": self.state.value, "to": new.value,
                            "why": why})
        self.state = new
        self.state_t = self.now()

    def in_state(self) -> float:
        return self.now() - self.state_t

    def yaw(self) -> float:
        return rot_yaw(self.R_wb)

    def agl(self) -> float:
        if self.t_wb is None:
            return float("nan")
        if self.down_hit is not None and self.now() - self.last_down < self.range_timeout:
            return float(self.t_wb[2] - self.down_hit[2])
        return float(self.t_wb[2] - self.ground_z)

    def camera_ok(self) -> bool:
        return self.img is not None and self.now() - self.img_t < self.camera_timeout

    def ranges_ok(self) -> bool:
        n = self.now()
        return n - self.last_down < self.range_timeout and n - self.last_tilt < self.range_timeout

    def hdist(self, xy) -> float:
        return float(np.hypot(self.t_wb[0] - xy[0], self.t_wb[1] - xy[1]))

    def request_mode(self, mode: str) -> None:
        if self.fcu.mode != mode and self.now() - self.last_req > 1.0 and self.mode_cli.service_is_ready():
            self.mode_cli.call_async(SetMode.Request(custom_mode=mode))
            self.last_req = self.now()

    def request_arm(self) -> None:
        if not self.fcu.armed and self.now() - self.last_req > 1.0 and self.arm_cli.service_is_ready():
            self.arm_cli.call_async(CommandBool.Request(value=True))
            self.last_req = self.now()

    # -------------------------------------------------------------- callbacks
    def on_state(self, msg: FcuState) -> None:
        self.fcu = msg

    def on_ext(self, msg: ExtendedState) -> None:
        self.landed_state = msg.landed_state

    def on_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        self.t_wb = np.array([p.x, p.y, p.z])
        self.R_wb = quat_to_rot(q.x, q.y, q.z, q.w)
        if self.home_z is None:
            self.home_z = p.z
            self.ground_z = p.z

    def _range(self, msg: LaserScan):
        if not msg.ranges:
            return None
        r = msg.ranges[0]
        if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
            return None
        return r

    def on_down(self, msg: LaserScan) -> None:
        self.last_down = self.now()
        r = self._range(msg)
        if r is None or self.t_wb is None:
            self.down_hit = None
            return
        self.down_hit = beam_hit(r, self.R_wb, self.t_wb, self.down_offset, self.down_dir)
        self.ground_z = 0.9 * self.ground_z + 0.1 * float(self.down_hit[2])
        if self.feel is not None:
            self.feel.add_point(*self.down_hit)

    def on_tilt(self, msg: LaserScan) -> None:
        self.last_tilt = self.now()
        r = self._range(msg)
        if r is None or self.t_wb is None or self.feel is None:
            return
        self.feel.add_point(*beam_hit(r, self.R_wb, self.t_wb, self.tilt_offset, self.tilt_dir))

    def on_image(self, msg: Image) -> None:
        if self.t_wb is None:
            return
        self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.img_t = self.now()
        self.img_pose = (self.R_wb.copy(), self.t_wb.copy())
        # Watch for moving objects whenever we're airborne and high enough
        # for the picture to mean something.
        agl = self.agl()
        if self.state in OFFBOARD_STATES and agl > self.final_agl:
            gsd = self.cam.gsd(agl)
            blobs = self.motion.update(self.img, gsd)
            world = []
            for b in blobs:
                xy = self.cam.pixel_to_ground(b.u, b.v, *self.img_pose, self.ground_z)
                if xy is not None:
                    world.append((xy[0], xy[1], b.radius_px * gsd))
            self.moving_world = world
            if world:
                self.moving_t = self.now()
            if self.state == S.DESCEND:
                self.publish_debug(None, blobs)
        else:
            self.motion.reset()

    # ------------------------------------------------------------ main loop
    def step(self) -> None:
        self.tick += 1
        if self.t_wb is not None and self.sp is None:
            self.sp = [*self.t_wb, self.yaw()]
        if self.offboard_seen and self.state in OFFBOARD_STATES and self.fcu.mode != "OFFBOARD":
            self.goto(S.DONE, f"pilot took over (mode {self.fcu.mode})")
            self.finish("pilot_override")
        handler = getattr(self, f"st_{self.state.value.lower()}")
        handler()
        if self.fcu.mode == "OFFBOARD":
            self.offboard_seen = True
        self.publish_setpoint()
        if self.tick % 2 == 0:
            self.log_row()

    def publish_setpoint(self) -> None:
        if self.sp is None or self.state in (S.LAND, S.RTL, S.DONE):
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, self.sp[:3])
        qx, qy, qz, qw = yaw_to_quat(self.sp[3])
        msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        self.sp_pub.publish(msg)

    # ------------------------------------------------------------ states
    def st_wait_fcu(self) -> None:
        if self.fcu.connected and self.t_wb is not None and self.in_state() > 2.0:
            self.goto(S.TAKEOFF if self.auto_takeoff else S.WAIT_OFFBOARD)

    def st_takeoff(self) -> None:
        # PX4 only accepts OFFBOARD after it has been receiving setpoints.
        self.sp = [self.sp[0], self.sp[1], self.home_z + self.look_alt, self.sp[3]]
        if self.in_state() < 2.0:
            return
        if self.fcu.mode != "OFFBOARD":
            self.request_mode("OFFBOARD")
        elif not self.fcu.armed:
            self.request_arm()
        elif abs(self.t_wb[2] - self.sp[2]) < 0.7:
            self.goto(S.GOTO_MISSION, f"at {self.look_alt:.0f} m")

    def st_wait_offboard(self) -> None:
        # Real drone: the pilot flies there and flips to OFFBOARD. Until then
        # keep the setpoint on the current position so the switch is smooth.
        self.sp = [*self.t_wb, self.yaw()]
        if self.fcu.mode == "OFFBOARD" and self.fcu.armed:
            self.home_z = self.t_wb[2] - self.look_alt  # assume we're at look altitude
            self.ground_z = self.home_z
            if not self.use_mission:
                self.mission_xy = self.t_wb[:2].copy()
            self.goto(S.GOTO_MISSION, "pilot handed over")

    def st_goto_mission(self) -> None:
        if not self.use_mission:
            self.mission_xy = self.t_wb[:2].copy()
        self.sp = [self.mission_xy[0], self.mission_xy[1], self.ground_z + self.look_alt, self.sp[3]]
        if self.hdist(self.mission_xy) < 0.5 and abs(self.t_wb[2] - self.sp[2]) < 0.7:
            self.goto(S.LOOK, "over mission point")

    def st_look(self) -> None:
        if self.in_state() < 1.0:
            return  # let the drone settle
        if not self.camera_ok():
            if "camera" not in self.degraded:
                self.degraded.append("camera")
            self.queue = [self.t_wb[:2].copy()]
            self.next_target("DEGRADED: no camera, checking the spot below with rangefinders only")
            return
        if self.img_t < self.state_t + 1.0:
            return  # wait for a frame taken after settling
        self.looks += 1
        if self.looks > self.max_looks:
            self.goto(S.RTL, "too many looks without a landing")
            return
        t0 = time.perf_counter()
        result, _ = self.run_look()
        self.look_times.append(time.perf_counter() - t0)
        R_wb, t_wb = self.img_pose
        self.queue = []
        for c in result.candidates:
            xy = self.cam.pixel_to_ground(c.u, c.v, R_wb, t_wb, self.ground_z)
            if xy is not None:
                self.queue.append(np.array(xy))
        self.publish_debug(result)
        if not self.queue:
            self.goto(S.RTL, "FR-10: no safe zone in view")
            return
        self.next_target(f"{len(self.queue)} candidates, picked in {self.look_times[-1] * 1000:.0f} ms")

    def run_look(self):
        R_wb, t_wb = self.img_pose
        gsd = self.cam.gsd(max(t_wb[2] - self.ground_z, 1.0))
        blocked = []
        for (x, y, r) in self.bad_spots + self.recent_moving(self.moving_buffer):
            uv = self.cam.ground_to_pixel(x, y, self.ground_z, R_wb, t_wb)
            if uv is not None:
                blocked.append((uv[0], uv[1], r / gsd))
        sides = []
        for (x, y, ux, uy) in self.slope_hints:
            a = self.cam.ground_to_pixel(x, y, self.ground_z, R_wb, t_wb)
            b = self.cam.ground_to_pixel(x + ux, y + uy, self.ground_z, R_wb, t_wb)
            if a is not None and b is not None:
                sides.append((a[0], a[1], b[0] - a[0], b[1] - a[1]))
        center = self.cam.ground_to_pixel(self.mission_xy[0], self.mission_xy[1], self.ground_z,
                                          R_wb, t_wb)
        return find_spots(self.img, gsd, center, blocked, self.look_params, sides), gsd

    def recent_moving(self, extra: float):
        if self.now() - self.moving_t > 2.0:
            return []
        return [(x, y, r + extra) for (x, y, r) in self.moving_world]

    def next_target(self, why: str) -> None:
        self.target = self.queue.pop(0)
        self.arrive_t = None
        self.goto(S.MOVE, why)

    def st_move(self) -> None:
        self.sp = [self.target[0], self.target[1], self.ground_z + self.look_alt, self.sp[3]]
        if self.hdist(self.target) < 0.4:
            self.arrive_t = self.arrive_t or self.now()
            if self.now() - self.arrive_t > 1.0:
                self.goto(S.CONFIRM, f"over target ({self.target[0]:.1f}, {self.target[1]:.1f})")
        else:
            self.arrive_t = None

    def st_confirm(self) -> None:
        if not self.camera_ok():
            self.start_descent("DEGRADED: no camera, skipping confirm")
            return
        if self.img_t < self.state_t + 0.5:
            return
        result, gsd = self.run_look()
        R_wb, t_wb = self.img_pose
        uv = self.cam.ground_to_pixel(self.target[0], self.target[1], self.ground_z, R_wb, t_wb)
        clearance = result.clearance_at(*uv) if uv is not None else 0.0
        self.publish_debug(result, target_uv=uv)
        if clearance >= 0.8 * self.R:
            self.start_descent(f"confirmed, clearance {clearance:.2f} m")
            return
        self.bad_spots.append((self.target[0], self.target[1], self.R))
        if self.queue:
            self.next_target(f"spot no longer clear ({clearance:.2f} m), next candidate")
        else:
            self.goto(S.LOOK, f"spot no longer clear ({clearance:.2f} m), looking again")

    def start_descent(self, why: str) -> None:
        self.feel = GroundFeel(self.target, self.feel_params)
        self.feel_result = None
        self.hold_t = None
        self.goto(S.DESCEND, why)

    def st_descend(self) -> None:
        agl = self.agl()
        x, y, z, yaw = self.sp
        # FR-11: lost rangefinders while relying on them -> abort
        if agl < self.feel_start_agl + 1.0 and not self.ranges_ok():
            if "rangefinder" not in self.degraded:
                self.degraded.append("rangefinder")
            self.goto(S.RTL, "FR-11: rangefinder stopped answering")
            return
        # FR-05 / FR-10: something moving near the landing spot
        for (mx, my, mr) in self.recent_moving(0.0):
            if math.hypot(mx - self.target[0], my - self.target[1]) < self.R + self.moving_buffer + mr:
                self.go_around(f"moving object at ({mx:.1f}, {my:.1f})")
                return
        if self.tick % 2 == 0:
            self.feel_result = self.feel.evaluate()
            if self.tick % 10 == 0:
                self.publish_cloud()
        fr = self.feel_result
        if fr is not None and fr.unsafe:
            self.go_around(fr.reason)
            return
        if agl > self.final_agl:
            v = self.v_fast if agl > self.feel_start_agl + 0.5 else self.v_slow
            z = max(z - v * self.dt, self.t_wb[2] - 1.0)
        elif fr is not None and fr.safe:
            self.goto(S.LAND, f"ground OK: slope {fr.slope_deg:.1f} deg, {fr.n_points} points")
            return
        else:
            self.hold_t = self.hold_t or self.now()
            if self.now() - self.hold_t > self.final_hold_timeout:
                self.go_around("ground check never completed")
                return
        if agl < self.feel_start_agl + 0.5:
            yaw = yaw + self.yaw_rate * self.dt
        self.sp = [self.target[0], self.target[1], z, yaw]

    def go_around(self, why: str) -> None:
        radius = self.slope_block_radius if why.startswith("slope") else self.R + 0.5
        self.bad_spots.append((self.target[0], self.target[1], radius))
        fr = self.feel_result
        if why.startswith("slope") and fr is not None and math.hypot(*fr.uphill) > 1e-3:
            # Flat ground is more likely downhill (the foot of a hill, a valley
            # floor), so stop looking on the uphill side of this spot.
            n = math.hypot(*fr.uphill)
            self.slope_hints.append((self.target[0], self.target[1],
                                     fr.uphill[0] / n, fr.uphill[1] / n))
        self.attempts += 1
        self.feel = None
        self.goto(S.GO_AROUND, f"FR-10: {why}")

    def st_go_around(self) -> None:
        self.sp = [self.t_wb[0], self.t_wb[1], self.ground_z + self.look_alt, self.sp[3]]
        if abs(self.t_wb[2] - self.sp[2]) < 0.7:
            if self.attempts >= self.max_attempts:
                self.goto(S.RTL, f"{self.attempts} failed attempts")
            else:
                self.goto(S.LOOK, f"attempt {self.attempts + 1}")

    def st_land(self) -> None:
        self.request_mode("AUTO.LAND")
        if self.landed_state == LANDED_STATE_ON_GROUND and self.in_state() > 2.0:
            self.goto(S.DONE, "touchdown")
            self.finish("landed")

    def st_rtl(self) -> None:
        self.request_mode("AUTO.RTL")
        if self.landed_state == LANDED_STATE_ON_GROUND and self.in_state() > 5.0:
            self.goto(S.DONE, "back home")
            self.finish("rtl")

    def st_done(self) -> None:
        pass

    # ------------------------------------------------------------ output
    def finish(self, outcome: str) -> None:
        if self.result_written:
            return
        self.result_written = True
        fr = self.feel_result
        landed_xy = self.t_wb[:2].tolist() if self.t_wb is not None else None
        res = {
            "world": self.world,
            "outcome": outcome,
            "target_xy": self.target.tolist() if self.target is not None else None,
            "landed_xy": landed_xy,
            "target_world_xy": (None if self.target is None
                                else (self.target + self.origin_world).tolist()),
            "landed_world_xy": (None if landed_xy is None
                                else (np.array(landed_xy) + self.origin_world).tolist()),
            "landing_error_m": (self.hdist(self.target) if self.target is not None
                                and outcome == "landed" else None),
            "attempts": self.attempts,
            "looks": self.looks,
            "look_compute_s": self.look_times,
            "degraded": self.degraded,
            "feel": None if fr is None else {
                "points": fr.n_points, "sectors": fr.sectors_covered,
                "slope_deg": round(fr.slope_deg, 2), "bumps": fr.bump_points,
                "max_residual_m": round(fr.max_residual_m, 3)},
            "bad_spots": self.bad_spots,
            "slope_hints": self.slope_hints,
            "events": self.events,
            "sim_time_s": round(self.now(), 1),
            "wall_time_s": round(time.time() - self.start_wall, 1),
        }
        path = os.path.join(self.log_dir, f"{self.run_id}_result.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        self.csv_file.flush()
        self.get_logger().info(f"RESULT {outcome} -> {path}")

    def log_row(self) -> None:
        if self.t_wb is None:
            return
        fr = self.feel_result if self.state == S.DESCEND else None
        tgt = self.target if self.target is not None else (float("nan"), float("nan"))
        row = [f"{self.now():.2f}", self.state.value, *(f"{v:.3f}" for v in self.t_wb),
               f"{self.yaw():.3f}", f"{self.agl():.2f}", f"{tgt[0]:.2f}", f"{tgt[1]:.2f}",
               fr.n_points if fr else "", fr.sectors_covered if fr else "",
               f"{fr.slope_deg:.2f}" if fr else "", fr.bump_points if fr else "",
               len(self.moving_world), int(self.camera_ok()), int(self.ranges_ok()),
               self.fcu.mode, int(self.fcu.armed)]
        self.csv.writerow(row)
        status = {"state": self.state.value, "agl": round(self.agl(), 2),
                  "target": None if self.target is None else [round(v, 2) for v in self.target],
                  "attempts": self.attempts, "feel": fr.reason if fr else None,
                  "moving": len(self.moving_world)}
        self.status_pub.publish(String(data=json.dumps(status)))

    def publish_debug(self, result, blobs=(), target_uv=None) -> None:
        if self.debug_pub.get_subscription_count() == 0 or self.img is None:
            return
        if result is None:
            img = self.img.copy()
            for b in blobs:
                cv2.circle(img, (int(b.u), int(b.v)), int(max(b.radius_px, 8)), (255, 0, 255), 2)
        else:
            img = draw_debug(self.img, result, target_uv,
                             [(b.u, b.v, b.radius_px) for b in blobs])
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))

    def publish_cloud(self) -> None:
        if self.feel is None or self.cloud_pub.get_subscription_count() == 0:
            return
        h = Header(frame_id="map", stamp=self.get_clock().now().to_msg())
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(h, self.feel.points.tolist()))

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SafeLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
