import math

import numpy as np

from safe_landing.geometry import (Camera, beam_direction, beam_hit, intrinsics_from_fov,
                                   quat_to_rot, rot_yaw, yaw_to_quat)


def cam():
    return Camera(*intrinsics_from_fov(960, 540, math.radians(66)))


def test_image_centre_is_below_drone():
    c = cam()
    xy = c.pixel_to_ground(480, 270, np.eye(3), np.array([3.0, 4.0, 15.0]), 0.0)
    assert np.allclose(xy, (3.0, 4.0))


def test_image_top_is_drone_nose():
    c = cam()
    # yaw 0: nose points east (+x); top of the image must be east of the drone
    x, y = c.pixel_to_ground(480, 0, np.eye(3), np.array([0.0, 0.0, 10.0]), 0.0)
    assert x > 3 and abs(y) < 1e-6
    # image right = drone's right = south (-y) when facing east
    x, y = c.pixel_to_ground(959, 270, np.eye(3), np.array([0.0, 0.0, 10.0]), 0.0)
    assert y < -5 and abs(x) < 1e-6


def test_round_trip_with_attitude():
    c = cam()
    yaw = 0.7
    qx, qy, qz, qw = yaw_to_quat(yaw)
    R = quat_to_rot(qx, qy, qz, qw)
    # add a small roll/pitch
    roll = 0.05
    Rr = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R = R @ Rr
    t = np.array([1.0, 2.0, 12.0])
    for (u, v) in [(100, 50), (480, 270), (900, 500)]:
        xy = c.pixel_to_ground(u, v, R, t, 0.5)
        uv = c.ground_to_pixel(xy[0], xy[1], 0.5, R, t)
        assert np.allclose(uv, (u, v), atol=1e-6)
    assert abs(rot_yaw(quat_to_rot(qx, qy, qz, qw)) - yaw) < 1e-9


def test_gsd_meets_kpi1_at_15m():
    assert cam().gsd(15.0) <= 0.02 + 1e-3


def test_tilted_beam_hits_ahead():
    d = beam_direction(15.0)
    p = beam_hit(3.0 / math.cos(math.radians(15)), np.eye(3), np.array([0, 0, 3.0]),
                 np.zeros(3), d)
    assert abs(p[2]) < 1e-9
    assert abs(p[0] - 3.0 * math.tan(math.radians(15))) < 1e-9
