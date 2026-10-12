"""Fly the descent spiral over the analytic test worlds and check FEEL."""
import math

import numpy as np

from make_worlds import WORLDS, terrain_height
from safe_landing.feel import FeelParams, GroundFeel, fit_plane
from safe_landing.geometry import beam_direction, beam_hit


def surface_height(world, x, y):
    h = terrain_height(world, x, y)
    for o in world["objects"]:
        if o["kind"] == "rock":
            s = o["size"]
            if abs(x - o["x"]) < s / 2 and abs(y - o["y"]) < 0.4 * s:
                h = max(h, terrain_height(world, o["x"], o["y"]) + 0.6 * s)
        if o["kind"] == "bush":
            d2 = (x - o["x"]) ** 2 + (y - o["y"]) ** 2
            if d2 < o["r"] ** 2:
                h = max(h, 0.6 * o["r"] + math.sqrt(o["r"] ** 2 - d2))
    return h


def raycast(world, origin, direction, max_range=12.0, step=0.01):
    for r in np.arange(0.3, max_range, step):
        p = origin + r * direction
        if p[2] <= surface_height(world, p[0], p[1]):
            return r
    return None


def fly_spiral(world, target, top=5.0, bottom=1.5, v=0.4, yaw_rate=math.radians(45), rate=50,
               noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    feel = GroundFeel(target, FeelParams())
    ground = terrain_height(world, *target)
    d_down, d_tilt = beam_direction(0), beam_direction(15)
    z, yaw, dt = ground + top, 0.0, 1.0 / rate
    while z > ground + bottom:
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        t = np.array([target[0], target[1], z])
        for d in (d_down, d_tilt):
            r = raycast(world, t, R @ d)
            if r is not None:
                feel.add_point(*beam_hit(r + rng.normal(0, noise), R, t, np.zeros(3), d))
        z -= v * dt
        yaw += yaw_rate * dt
        res = feel.evaluate()
        if res.unsafe:
            return res
    return feel.evaluate()


def test_flat_grass_is_safe():
    res = fly_spiral({"objects": [], "hill": None}, (0.0, 0.0))
    assert res.complete and res.safe, res.reason
    assert res.slope_deg < 2


def test_slope_is_rejected():
    world = WORLDS["c2_hill"]
    res = fly_spiral(world, tuple(world["mission_xy"]))  # about 20 deg there
    assert res.unsafe and res.reason.startswith("slope"), res.reason


def test_flat_ground_beside_the_hill_is_safe():
    res = fly_spiral(WORLDS["c2_hill"], (-2.0, 0.0))
    assert res.safe, res.reason


def test_rock_in_landing_circle_is_rejected():
    world = WORLDS["c1_field"]
    res = fly_spiral(world, (8.6, 4.2))  # among the rock cluster
    assert res.unsafe and res.reason.startswith("bump"), res.reason


def test_spiral_covers_the_circle():
    res = fly_spiral({"objects": [], "hill": None}, (0.0, 0.0))
    assert res.sectors_covered == 8 and res.n_points > 200


def test_plane_fit_ignores_one_outlier():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1, 1, (200, 2))
    z = 0.1 * xy[:, 0] + rng.normal(0, 0.01, 200)
    z[0] += 1.0
    coef, res = fit_plane(np.c_[xy, z], 0.12)
    assert abs(coef[0] - 0.1) < 0.01 and abs(res[0]) > 0.9


def test_slope_wins_over_early_bumps():
    # 20 deg slope where the first readings are noisy (as on Gazebo's
    # triangulated hill): the verdict must be "slope", so the drone learns
    # which way is uphill.
    rng = np.random.default_rng(3)
    feel = GroundFeel((0.0, 0.0), FeelParams())
    k = math.tan(math.radians(20))
    for i in range(300):
        a = i * 0.15
        r = 0.4 + 0.9 * (i % 50) / 50
        x, y = r * math.cos(a), r * math.sin(a)
        z = k * x + (0.2 if i < 5 else rng.normal(0, 0.01))
        feel.add_point(x, y, z)
        res = feel.evaluate()
        if res.unsafe:
            break
    assert res.unsafe and res.reason.startswith("slope"), res.reason
    assert res.uphill[0] > 0.3 and abs(res.uphill[1]) < 0.1
