"""FEEL: check the ground under the drone with rangefinder points.

While the drone descends and turns, every rangefinder reading becomes a 3D
ground point (geometry.beam_hit). The tilted beam draws a spiral around the
landing spot and the downward beam covers its centre.

Points within `check_radius_m` of the target are fitted with a plane
z = a*x + b*y + c:

  * slope = angle between the plane and horizontal  -> FR-04 tilt limit
  * residual = how far each point is above/below the plane
      -> rocks, holes, bushes, or something that walked under the drone

The fit is repeated without the worst points, so one big rock doesn't tilt
the plane and hide itself.

Bumps only count inside `bump_radius_m`, the landing circle the camera
promised to be clear. The ring beyond it helps the slope fit, but a bush or
rock out there is not under the drone and must not abort the landing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FeelParams:
    check_radius_m: float = 1.3
    bump_radius_m: float = 1.0       # = the landing circle radius (look.required_radius_m)
    min_points: int = 80
    sectors: int = 8                 # coverage is counted in angular sectors
    min_sectors: int = 7
    inner_radius_m: float = 0.3      # points closer than this don't count for coverage
    max_slope_deg: float = 10.0
    bump_threshold_m: float = 0.12
    min_bump_points: int = 3         # a single noisy reading is not a rock
    early_slope_points: int = 40     # judge slope early once this many points are in


@dataclass
class FeelResult:
    n_points: int
    sectors_covered: int
    slope_deg: float
    bump_points: int
    max_residual_m: float
    ground_z: float          # plane height at the target
    uphill: tuple            # (dz/dx, dz/dy) of the fitted plane: points uphill
    complete: bool           # enough points and coverage to decide
    safe: bool               # complete and nothing wrong
    unsafe: bool             # something is definitely wrong (abort now)
    reason: str


def fit_plane(pts: np.ndarray, threshold: float, iterations: int = 3):
    """Least-squares plane z = a x + b y + c with outlier rejection."""
    use = np.ones(len(pts), bool)
    coef = np.array([0.0, 0.0, float(np.median(pts[:, 2]))])
    for _ in range(iterations):
        if use.sum() < 3:
            break
        A = np.c_[pts[use, 0], pts[use, 1], np.ones(use.sum())]
        coef, *_ = np.linalg.lstsq(A, pts[use, 2], rcond=None)
        res = pts[:, 2] - (coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2])
        new_use = np.abs(res) <= threshold
        if new_use.sum() < 3 or np.array_equal(new_use, use):
            break
        use = new_use
    res = pts[:, 2] - (coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2])
    return coef, res


class GroundFeel:
    def __init__(self, target_xy, params: FeelParams | None = None):
        self.p = params or FeelParams()
        self.tx, self.ty = float(target_xy[0]), float(target_xy[1])
        self._pts: list[tuple[float, float, float]] = []

    def add_point(self, x: float, y: float, z: float) -> bool:
        """Keep the point if it is near the target. Returns True if kept."""
        if math.hypot(x - self.tx, y - self.ty) > self.p.check_radius_m:
            return False
        self._pts.append((x, y, z))
        return True

    @property
    def points(self) -> np.ndarray:
        return np.array(self._pts, float).reshape(-1, 3)

    def evaluate(self) -> FeelResult:
        p = self.p
        pts = self.points
        n = len(pts)
        if n < 3:
            return FeelResult(n, 0, 0.0, 0, 0.0, float("nan"), (0.0, 0.0), False, False, False,
                              "collecting")

        dx, dy = pts[:, 0] - self.tx, pts[:, 1] - self.ty
        ring = np.hypot(dx, dy) >= p.inner_radius_m
        ang = (np.arctan2(dy[ring], dx[ring]) + math.pi) / (2 * math.pi) * p.sectors
        sectors = len(np.unique(np.clip(ang.astype(int), 0, p.sectors - 1)))

        coef, res = fit_plane(pts, p.bump_threshold_m)
        slope = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
        inside = np.hypot(dx, dy) <= p.bump_radius_m
        bumps = int((inside & (np.abs(res) > p.bump_threshold_m)).sum())
        ground_z = float(coef[0] * self.tx + coef[1] * self.ty + coef[2])
        complete = n >= p.min_points and sectors >= p.min_sectors

        # No verdict from a handful of points: on a curved hillside the first
        # few look like bumps. Slope is judged before bumps, because a slope
        # verdict also tells the drone which way to search next.
        ready = complete or (n >= p.early_slope_points and sectors >= 4)
        reason = "ok"
        unsafe = False
        if ready and slope > p.max_slope_deg:
            unsafe, reason = True, f"slope {slope:.1f} deg > {p.max_slope_deg:.0f}"
        elif ready and bumps >= p.min_bump_points:
            unsafe, reason = True, f"bump: {bumps} points off the ground by >{p.bump_threshold_m:.2f} m"
        elif not complete:
            reason = f"collecting ({n} pts, {sectors}/{p.sectors} sectors)"
        max_res = float(np.abs(res[inside]).max()) if inside.any() else 0.0
        return FeelResult(n, sectors, slope, bumps, max_res, ground_z,
                          (float(coef[0]), float(coef[1])), complete, complete and not unsafe, unsafe, reason)
