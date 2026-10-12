"""Frames and conversions shared by the camera and rangefinder code.

Conventions (the same ones MAVROS uses):
  * world frame: ENU, x east, y north, z up (MAVROS local_position).
  * body frame: FLU, x forward, y left, z up.
  * camera optical frame: x right in the image, y down in the image, z out
    of the lens.

The camera points straight down with the top of the image towards the
drone's nose (see sim/models/x500_landing/model.sdf). In body coordinates
that means: image-right = -y (right), image-down = -x (backwards),
lens axis = -z (down).
"""
from __future__ import annotations

import math

import numpy as np

# Columns are the optical x, y, z axes written in body coordinates.
R_BODY_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])


def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix (body -> world) from a unit quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a pure yaw rotation."""
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def rot_yaw(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def intrinsics_from_fov(width: int, height: int, hfov_rad: float) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy for an ideal pinhole camera (square pixels)."""
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    return fx, fx, width / 2.0, height / 2.0


class Camera:
    """Projects between image pixels and points on a horizontal ground plane."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 offset_body=(0.0, 0.0, 0.0)):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.offset_body = np.asarray(offset_body, float)

    def pixel_to_ground(self, u: float, v: float, R_wb: np.ndarray, t_wb: np.ndarray,
                        ground_z: float):
        """World (x, y) where pixel (u, v) hits the plane z = ground_z, or None."""
        ray_opt = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        ray_w = R_wb @ (R_BODY_OPTICAL @ ray_opt)
        if ray_w[2] > -1e-6:
            return None
        origin = t_wb + R_wb @ self.offset_body
        s = (ground_z - origin[2]) / ray_w[2]
        if s <= 0:
            return None
        p = origin + s * ray_w
        return float(p[0]), float(p[1])

    def ground_to_pixel(self, x: float, y: float, ground_z: float, R_wb: np.ndarray,
                        t_wb: np.ndarray):
        """Pixel (u, v) where world point (x, y, ground_z) appears, or None."""
        origin = t_wb + R_wb @ self.offset_body
        p_opt = R_BODY_OPTICAL.T @ (R_wb.T @ (np.array([x, y, ground_z]) - origin))
        if p_opt[2] <= 1e-6:
            return None
        return (float(self.fx * p_opt[0] / p_opt[2] + self.cx),
                float(self.fy * p_opt[1] / p_opt[2] + self.cy))

    def gsd(self, agl: float) -> float:
        """Metres per pixel at the image centre when looking straight down."""
        return agl / self.fx


def beam_direction(tilt_deg: float) -> np.ndarray:
    """Unit vector (body frame) of a rangefinder tilted forward from vertical."""
    t = math.radians(tilt_deg)
    return np.array([math.sin(t), 0.0, -math.cos(t)])


def beam_hit(range_m: float, R_wb: np.ndarray, t_wb: np.ndarray,
             offset_body: np.ndarray, dir_body: np.ndarray) -> np.ndarray:
    """World point measured by a rangefinder reading."""
    return t_wb + R_wb @ (np.asarray(offset_body, float) + range_m * np.asarray(dir_body, float))
