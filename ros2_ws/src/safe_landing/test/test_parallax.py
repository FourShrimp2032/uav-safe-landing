"""Descending next to a house: roofs and walls seem to move (parallax), but
must not abort the landing, while a walking person still must."""
import math

import cv2
import numpy as np
import pytest

import make_worlds as mw
from render3d import render
from safe_landing.geometry import Camera, intrinsics_from_fov, quat_to_rot, yaw_to_quat
from safe_landing.look import find_spots
from safe_landing.motion import MotionDetector

W, H = 1280, 720   # Pi Camera Module 3 Wide, as in params.yaml
CAM = Camera(*intrinsics_from_fov(W, H, math.radians(102)), offset_body=(0.05, 0, -0.06))
HALF = Camera(*intrinsics_from_fov(W // 2, H // 2, math.radians(102)), offset_body=(0.05, 0, -0.06))
BUFFER = 1.0 + 1.5   # landing circle + moving buffer (params.yaml)


@pytest.fixture(scope="module")
def tex():
    return mw.make_textures(np.random.default_rng(0))


def view(objects, x, y, z, yaw, tex):
    R, t = quat_to_rot(*yaw_to_quat(yaw)), np.array([x, y, z])
    img = render(objects, HALF, R, t, W // 2, H // 2, tex)   # half size: faster, same detail as the detector uses
    return cv2.resize(img, (W, H)), R, t


def descend(objects, x, y, tex, z_from, z_to, person=None):
    """Like the node: confirm map at 10 m, then descend and watch. Returns
    the moving objects that would have aborted the landing."""
    img, R, t = view(objects, x, y, 10.0, 0.3, tex)
    res = find_spots(img, CAM.gsd(10.0))
    static = np.array([CAM.pixel_to_ground((c + .5) * res.cell_px, (r + .5) * res.cell_px, R, t, 0.0)
                       for r, c in zip(*np.nonzero(res.cell_bad))]).reshape(-1, 2)
    det, aborts, z, yaw, k = MotionDetector(), [], z_from, 0.3, 0
    while z > z_to:
        objs = list(objects) + ([person(k)] if person else [])
        img, R, t = view(objs, x, y, z, yaw, tex)
        agl = z - 0.06
        for b in det.update(img, CAM.gsd(agl), CAM.ground_to_pixel(x, y, 0.0, R, t), agl):
            mx, my = CAM.pixel_to_ground(b.u, b.v, R, t, 0.0)
            if len(static) and (np.hypot(*(static - (mx, my)).T) < 0.71 * 0.25 + 0.5).any():
                continue
            if math.hypot(mx - x, my - y) < BUFFER + b.radius_px * CAM.gsd(agl):
                aborts.append((round(z, 1), round(mx, 1), round(my, 1)))
        z -= (1.0 if z > 5.5 else 0.4) / 5      # 5 Hz camera
        yaw += math.radians(45) / 5 if z < 5.5 else 0.0
        k += 1
    return aborts


def test_descent_next_to_house_is_not_an_abort(tex):
    # The spot from the failed town run, 1.9 m from a 5 m house.
    town = mw.WORLDS["c3_town"]["objects"]
    assert descend(town, 5.0, 1.1, tex, 10.0, 5.0) == []


def test_person_walking_in_still_aborts(tex):
    def person(k):   # 2 m/s towards the spot
        return {"kind": "person", "x": 6.0 + 4.0 - 0.4 * k, "y": 0.0}
    aborts = descend(mw.WORLDS["c4_dynamic"]["objects"], 6.0, 0.0, tex, 12.0, 10.0, person)
    assert aborts, "person not detected"


def test_person_walking_in_near_the_ground_aborts(tex):
    # The c4_dynamic run that went wrong: the person walked in from the side
    # while the drone was only a few metres up. The wide camera still sees
    # them in time to abort before the commit height.
    def person(k):   # 2 m/s towards the spot, starting 5 m away
        return {"kind": "person", "x": 6.0, "y": -5.0 + 0.4 * k}
    aborts = descend(mw.WORLDS["c4_dynamic"]["objects"], 6.0, 0.0, tex, 3.0, 0.7, person)
    assert aborts, "person not detected"
