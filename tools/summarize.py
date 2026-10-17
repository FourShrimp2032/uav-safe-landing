#!/usr/bin/env python3
"""Summarize landing results against the world's ground truth (KPI-2/3/4).

    python3 tools/summarize.py c1_field [logs_dir]

Reads <logs_dir>/<world>/*_result.json (default logs dir: /root/landing_logs
in the container, ./logs on your Mac).
"""
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_worlds import WORLDS, actor_xy, distance_to_hazards, terrain_slope_deg  # noqa: E402
from make_worlds import object_footprint  # noqa: E402

MAX_SLOPE = 10.0
MIN_CLEARANCE = 0.5
PERSON_CLEARANCE = 1.5   # landing circle (1 m) + half a person
PERSON_WINDOW = 4.0      # seconds before ground contact that count
CONTACT_AGL = 0.15       # height (m) that counts as ground contact
MAX_ERROR = 0.30


def judge(world, r, contact_t=None):
    """Success = landed on ground that is really safe. Every test world has
    safe ground near the mission point, so an RTL counts as a failure."""
    if r["outcome"] != "landed" or not r.get("landed_world_xy"):
        return False, r["outcome"]
    x, y = r["landed_world_xy"]
    slope = terrain_slope_deg(world, x, y)
    clear = distance_to_hazards(world, x, y)
    if slope >= MAX_SLOPE:
        return False, f"landed on {slope:.0f} deg slope"
    if clear < MIN_CLEARANCE:
        return False, f"landed at ({x:.1f}, {y:.1f}), {clear:.2f} m from {nearest_hazard(world, x, y)}"
    person = closest_person(world, r, x, y, contact_t)
    if person is not None and person < PERSON_CLEARANCE:
        return False, f"person {person:.1f} m away at touchdown"
    return True, "safe landing"


def nearest_hazard(world, x, y):
    """Name of the hazard that is closest to (x, y), e.g. "pond (murky) at (5.0, 1.0)"."""
    def gap(o):
        return distance_to_hazards({"objects": [o]}, x, y)
    o = min(world["objects"], key=gap)
    kind = o["kind"] + (f" ({o['color']})" if "color" in o else "")
    shape, _ = object_footprint(o)
    return f"{kind} at ({o['x']:.1f}, {o['y']:.1f})" + (" edge" if shape == "rect" else "")


def contact_time(result_path: Path):
    """When the drone first reached the ground, from the height log. The
    "touchdown" event comes 1-2 s later (PX4's land detector takes a while)."""
    log = result_path.with_name(result_path.name.replace("_result.json", "_log.csv"))
    if not log.exists():
        return None
    with open(log, newline="") as f:
        for row in csv.DictReader(f):
            if row["state"] == "LAND" and float(row["agl"]) <= CONTACT_AGL:
                return float(row["t"])
    return None


def closest_person(world, r, x, y, contact_t=None):
    """Closest the walking person came to the landing point in the last
    seconds before ground contact. Someone walking up to a drone that is
    already on the ground is not a landing failure."""
    if not world.get("actor"):
        return None
    t_td = contact_t
    if t_td is None:
        t_td = next((e["t"] for e in r.get("events", []) if e["why"] == "touchdown"), None)
    if t_td is None:
        return None
    ts = [t_td - PERSON_WINDOW + 0.1 * i for i in range(int(10 * PERSON_WINDOW) + 1)]
    return min(math.hypot(px - x, py - y) for px, py in (actor_xy(world, t) for t in ts))


def main():
    world_name = sys.argv[1] if len(sys.argv) > 1 else "c1_field"
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/root/landing_logs")
    if not root.exists():
        root = Path(__file__).resolve().parents[1] / "logs"
    files = sorted((root / world_name).glob("*_result.json"))
    if not files:
        print(f"no results in {root / world_name}")
        return
    world = WORLDS[world_name]
    ok, errors, looks, skipped = 0, [], [], 0
    print(f"{'run':<34} {'outcome':<16} {'tries':>5} {'err m':>6}  verdict")
    for f in files:
        r = json.loads(f.read_text())
        if r["outcome"] == "no_takeoff":
            # The simulator was too slow for PX4 to arm: not a landing.
            skipped += 1
            print(f"{f.stem[:34]:<34} {r['outcome']:<16} {'-':>5} {'-':>6}  simulator problem, not counted")
            continue
        good, why = judge(world, r, contact_time(f))
        ok += good
        if r.get("landing_error_m") is not None:
            errors.append(r["landing_error_m"])
        looks += r.get("look_compute_s", [])
        err = f"{r['landing_error_m']:.2f}" if r.get("landing_error_m") is not None else "-"
        print(f"{f.stem[:34]:<34} {r['outcome']:<16} {r['attempts']:>5} {err:>6}  {why}")
    n = len(files) - skipped
    print()
    if skipped:
        print(f"{skipped} run(s) never took off (simulator too slow) and are not counted")
    if n == 0:
        return
    print(f"KPI-4 success rate: {ok}/{n} = {100 * ok / n:.0f} %  (target >= 90 % over >= 30 runs)")
    if errors:
        within = sum(e <= MAX_ERROR for e in errors)
        print(f"KPI-3 landing error: mean {statistics.mean(errors):.2f} m, max {max(errors):.2f} m, "
              f"{within}/{len(errors)} within {MAX_ERROR} m")
    if looks:
        print(f"KPI-2 spot choice compute: mean {1000 * statistics.mean(looks):.0f} ms, "
              f"max {1000 * max(looks):.0f} ms  (target <= 2-3 s)")


if __name__ == "__main__":
    main()
