"""Turning the real (distorted) camera image into the ideal pinhole image the
landing node assumes.

The landing node treats the camera as a perfect pinhole with a given
horizontal field of view (camera_hfov_deg) and image size. The simulated
camera is exactly that. A real wide lens bends straight lines towards the
edges, so every frame is remapped with the calibration onto that same ideal
pinhole camera. The node's geometry then holds for the real camera too.

The calibration file is the one ROS camera_calibration writes (ost.yaml):
camera_matrix, distortion_model ("plumb_bob" or "equidistant"),
distortion_coefficients, image_width, image_height.

Kept free of ROS so it can be unit-tested anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from .geometry import intrinsics_from_fov


@dataclass
class Calibration:
    K: np.ndarray            # 3x3
    D: np.ndarray            # distortion coefficients
    model: str               # "plumb_bob" or "equidistant"
    width: int
    height: int


def load_calibration(path: str) -> Calibration:
    with open(path) as f:
        c = yaml.safe_load(f)
    K = np.array(c["camera_matrix"]["data"], float).reshape(3, 3)
    D = np.array(c["distortion_coefficients"]["data"], float).ravel()
    model = c.get("distortion_model", "plumb_bob")
    if model not in ("plumb_bob", "equidistant"):
        raise ValueError(f"unsupported distortion model {model}")
    return Calibration(K, D, model, int(c["image_width"]), int(c["image_height"]))


def target_matrix(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx, fy, cx, cy = intrinsics_from_fov(width, height, math.radians(hfov_deg))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)


def scaled(cal: Calibration, width: int, height: int) -> Calibration:
    """The same calibration for frames captured at another resolution
    (same sensor crop, just scaled)."""
    sx, sy = width / cal.width, height / cal.height
    K = cal.K.copy()
    K[0] *= sx
    K[1] *= sy
    return Calibration(K, cal.D, cal.model, width, height)


def undistort_maps(cal: Calibration, out_w: int, out_h: int, hfov_deg: float):
    """Remap tables from the raw frame to the ideal out_w x out_h pinhole
    with the given horizontal field of view. Pixels the lens never saw come
    out black; the landing node already treats dark pixels as bad ground."""
    P = target_matrix(out_w, out_h, hfov_deg)
    if cal.model == "equidistant":
        return cv2.fisheye.initUndistortRectifyMap(cal.K, cal.D[:4], np.eye(3), P,
                                                   (out_w, out_h), cv2.CV_16SC2)
    return cv2.initUndistortRectifyMap(cal.K, cal.D, np.eye(3), P, (out_w, out_h), cv2.CV_16SC2)


def max_hfov_deg(cal: Calibration) -> float:
    """Widest horizontal FOV the ideal image can have without black edges at
    mid-height (useful to pick camera_hfov_deg after calibrating)."""
    pts = np.array([[[0.0, cal.height / 2]], [[cal.width - 1.0, cal.height / 2]]])
    if cal.model == "equidistant":
        und = cv2.fisheye.undistortPoints(pts, cal.K, cal.D[:4])
    else:
        und = cv2.undistortPoints(pts, cal.K, cal.D)
    xl, xr = und[0, 0, 0], und[1, 0, 0]
    return math.degrees(math.atan(-xl) + math.atan(xr))


def rotate(img: np.ndarray, deg: int) -> np.ndarray:
    """Turn the frame so the top of the image points to the drone's nose.
    The long side of the sensor must run left-right across the drone, as in
    the simulation, so only 0 and 180 degrees are allowed."""
    deg %= 360
    if deg == 0:
        return img
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    raise ValueError("mount the camera with its long side across the drone (rotate 0 or 180)")
