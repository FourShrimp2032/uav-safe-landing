import numpy as np
import pytest

from make_worlds import WORLDS, distance_to_hazards, render_topdown, terrain_slope_deg
from safe_landing.look import LookParams, find_spots

GSD = 0.0202  # 66 deg camera, 960 px, 15 m


def to_world(c, center):
    return center[0] + (c.u - 480) * GSD, center[1] - (c.v - 270) * GSD


@pytest.mark.parametrize("name", list(WORLDS))
def test_candidates_avoid_visible_hazards(name):
    world = WORLDS[name]
    center = world["mission_xy"]
    img = render_topdown(world, center, gsd_m=GSD)
    res = find_spots(img, GSD)
    assert res.candidates, "no landing spot found"
    for c in res.candidates:
        x, y = to_world(c, center)
        assert distance_to_hazards(world, x, y) >= 0.9, (name, x, y)


def test_plain_grass_is_not_flagged():
    world = {"objects": [], "hill": None}
    img = render_topdown(world, (0, 0), gsd_m=GSD)
    res = find_spots(img, GSD)
    assert res.cell_bad.mean() < 0.02
    best = res.candidates[0]
    assert best.distance_m < 0.3      # FR-06: closest to the mission centre


def test_rocks_on_mission_point_push_spot_away():
    world = WORLDS["c1_field"]
    center = world["mission_xy"]
    res = find_spots(render_topdown(world, center, gsd_m=GSD), GSD)
    best = res.candidates[0]
    assert 1.0 < best.distance_m < 4.0


def test_blocked_spot_is_skipped():
    world = {"objects": [], "hill": None}
    img = render_topdown(world, (0, 0), gsd_m=GSD)
    res = find_spots(img, GSD, blocked_uv=[(480, 270, 2.0 / GSD)])
    best = res.candidates[0]
    assert best.distance_m > 2.0


def test_camera_cannot_see_the_hill():
    # Documented limit: grass on a 20 deg slope looks like flat grass, so
    # LOOK happily proposes it. FEEL must catch it (see test_feel.py).
    world = WORLDS["c2_hill"]
    center = world["mission_xy"]
    res = find_spots(render_topdown(world, center, gsd_m=GSD), GSD)
    x, y = to_world(res.candidates[0], center)
    assert terrain_slope_deg(world, x, y) > 10


def test_find_spots_is_fast_enough():
    import time
    world = WORLDS["c3_town"]
    img = render_topdown(world, world["mission_xy"], gsd_m=GSD)
    find_spots(img, GSD)
    t0 = time.perf_counter()
    for _ in range(5):
        find_spots(img, GSD, params=LookParams())
    assert (time.perf_counter() - t0) / 5 < 0.2   # NFR-01 budget on this machine


def test_small_soil_patch_as_bright_as_grass_is_flagged():
    # KPI-1: 20 x 20 cm at 10 m. Brown soil passes the colour rule and is
    # about as bright as the grass, so only the colour edges give it away.
    gsd = 0.0193
    img = render_topdown({"objects": [], "hill": None}, (0, 0), gsd_m=gsd)
    side = round(0.2 / gsd)
    spots = [(150, 100), (400, 300), (700, 220), (820, 420)]
    for u, v in spots:
        img[v:v + side, u:u + side] = (48, 84, 120)   # BGR
    res = find_spots(img, gsd)
    for u, v in spots:
        c = side // 2
        r0, c0 = (v + c) // res.cell_px, (u + c) // res.cell_px
        assert res.cell_bad[r0 - 1:r0 + 2, c0 - 1:c0 + 2].any(), (u, v)


def test_green_pond_edge_is_flagged():
    world = WORLDS["c5_shadows_water"]
    pond = next(o for o in world["objects"] if o.get("color") == "murky")
    img = render_topdown(world, (pond["x"], pond["y"]), gsd_m=GSD)
    res = find_spots(img, GSD)
    rad = int(pond["r"] / GSD)
    ring = []
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        u, v = 480 + rad * np.cos(a), 270 - rad * np.sin(a)
        r, c = int(v // res.cell_px), int(u // res.cell_px)
        ring.append(res.cell_bad[r - 1:r + 2, c - 1:c + 2].any())
    assert sum(ring) >= 12
