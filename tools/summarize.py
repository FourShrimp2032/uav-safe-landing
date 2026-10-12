#!/usr/bin/env python3
"""Summarize landing results against the world's ground truth (KPI-2/3/4).

    python3 tools/summarize.py c1_field [logs_dir]

Reads <logs_dir>/<world>/*_result.json (default logs dir: /root/landing_logs
in the container, ./logs on your Mac).
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_worlds import WORLDS, distance_to_hazards, terrain_slope_deg  # noqa: E402

MAX_SLOPE = 10.0
MIN_CLEARANCE = 0.5
MAX_ERROR = 0.30


def judge(world, r):
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
        return False, f"landed {clear:.2f} m from a hazard"
    return True, "safe landing"


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
    ok, errors, looks = 0, [], []
    print(f"{'run':<34} {'outcome':<16} {'tries':>5} {'err m':>6}  verdict")
    for f in files:
        r = json.loads(f.read_text())
        good, why = judge(world, r)
        ok += good
        if r.get("landing_error_m") is not None:
            errors.append(r["landing_error_m"])
        looks += r.get("look_compute_s", [])
        err = f"{r['landing_error_m']:.2f}" if r.get("landing_error_m") is not None else "-"
        print(f"{f.stem[:34]:<34} {r['outcome']:<16} {r['attempts']:>5} {err:>6}  {why}")
    n = len(files)
    print()
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
