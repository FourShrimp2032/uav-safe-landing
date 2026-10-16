"""Look + feel together, the way the node chains them, without Gazebo."""
import math

from make_worlds import WORLDS, distance_to_hazards, render_topdown, terrain_slope_deg
from safe_landing.look import find_spots
from test_feel import fly_spiral

GSD = 0.0202
W, H = 960, 540


def run_attempts(world, max_attempts=3, use_slope_hints=True):
    """Return (landed_xy or None, attempts) for a drone hovering over the
    mission point, retrying like landing_node does."""
    cx, cy = world["mission_xy"]
    img = render_topdown(world, (cx, cy), gsd_m=GSD)
    to_px = lambda x, y: (W / 2 + (x - cx) / GSD, H / 2 - (y - cy) / GSD)  # noqa: E731
    bad, sides = [], []
    for attempt in range(max_attempts):
        res = find_spots(img, GSD, blocked_uv=bad, blocked_sides=sides)
        if not res.candidates:
            return None, attempt
        c = res.candidates[0]
        x, y = cx + (c.u - W / 2) * GSD, cy - (c.v - H / 2) * GSD
        feel = fly_spiral(world, (x, y))
        if feel.safe:
            return (x, y), attempt
        radius = 4.0 if feel.reason.startswith("slope") else 1.5
        bad.append((*to_px(x, y), radius / GSD))
        if feel.reason.startswith("slope") and use_slope_hints:
            n = math.hypot(*feel.uphill)
            a, b = to_px(x, y), to_px(x + feel.uphill[0] / n, y + feel.uphill[1] / n)
            sides.append((a[0], a[1], b[0] - a[0], b[1] - a[1]))
    return None, max_attempts


def test_hill_lands_on_flat_ground_after_slope_rejection():
    world = WORLDS["c2_hill"]
    landed, attempts = run_attempts(world)
    assert landed is not None, "never found flat ground"
    assert attempts >= 1                       # the first pick was on the slope
    assert terrain_slope_deg(world, *landed) < 10
    assert distance_to_hazards(world, *landed) >= 0.5


def test_field_lands_first_try():
    world = WORLDS["c1_field"]
    landed, attempts = run_attempts(world)
    assert landed is not None and attempts == 0
    assert distance_to_hazards(world, *landed) >= 0.5


def test_shadows_water_lands_on_grass_first_try():
    world = WORLDS["c5_shadows_water"]
    landed, attempts = run_attempts(world)
    assert landed is not None and attempts == 0
    assert distance_to_hazards(world, *landed) >= 0.5


def test_blue_water_is_rejected_by_colour():
    world = WORLDS["c5_shadows_water"]
    pond = next(o for o in world["objects"] if o["kind"] == "pond" and o["color"] == "blue")
    img = render_topdown(world, (pond["x"], pond["y"]), gsd_m=GSD)
    res = find_spots(img, GSD)
    rows, cols = res.cell_bad.shape
    centre = res.cell_bad[rows // 2 - 3:rows // 2 + 3, cols // 2 - 3:cols // 2 + 3]
    assert centre.all()
