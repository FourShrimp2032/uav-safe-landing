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
