"""LOOK: find open, uniform ground in a downward camera image.

The image is cut into square cells (0.25 m on the ground by default). A cell
is bad when too many of its pixels are either

  * not ground-coloured: grey (rock, asphalt, concrete), blue (water),
    very dark (holes, deep shadow) or a hue that isn't grass or dirt, or
  * "busy": strong edges after blurring away fine grass texture, which is
    what bushes, tree canopies, rocks, roofs and objects look like.

Bad cells are then grown by a distance transform: every good cell gets its
distance to the nearest bad cell (its clearance). Spots with enough
clearance are ranked by clearance and by distance to the mission centre
(FR-06/07), and the best few are returned.

There is no 3D here at all: slopes and hidden rocks are left to the
rangefinder check (feel.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LookParams:
    cell_m: float = 0.25
    bad_cell_fraction: float = 0.10     # cell is bad above this share of bad pixels
    required_radius_m: float = 1.0      # FR-06: R ~ 1.5 x drone size
    distance_weight: float = 0.5        # metres of clearance worth 1 m closer to the centre
    clearance_credit_m: float = 3.0     # extra room counts up to this much clearance
    max_candidates: int = 3
    min_separation_m: float = 3.0       # between returned candidates
    # Colour rules, OpenCV HSV (H 0-179, S and V 0-255).
    min_saturation: int = 45
    min_value: int = 40
    green_hue: tuple = (30, 90)
    brown_hue: tuple = (9, 30)
    # Texture rule: blur away grass detail (sigma in metres), then count
    # pixels with a gradient stronger than this (grey levels per pixel).
    texture_sigma_m: float = 0.04
    edge_threshold: float = 6.0
    # Colour edges: same blur, gradient of the Lab a/b (colour) channels.
    # Catches objects as bright as the grass but a different colour.
    colour_edge_threshold: float = 2.0


@dataclass
class Candidate:
    u: float            # pixel column of the spot centre
    v: float            # pixel row
    clearance_m: float  # distance to the nearest bad cell
    distance_m: float   # distance to the mission centre
    score: float


@dataclass
class LookResult:
    cell_px: int
    cell_bad: np.ndarray        # rows x cols bool
    clearance_m: np.ndarray     # rows x cols float
    candidates: list = field(default_factory=list)

    def clearance_at(self, u: float, v: float) -> float:
        r, c = int(v // self.cell_px), int(u // self.cell_px)
        if 0 <= r < self.clearance_m.shape[0] and 0 <= c < self.clearance_m.shape[1]:
            return float(self.clearance_m[r, c])
        return 0.0


def bad_pixel_mask(bgr: np.ndarray, gsd_m: float, p: LookParams) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= p.green_hue[0]) & (h <= p.green_hue[1])
    brown = (h >= p.brown_hue[0]) & (h <= p.brown_hue[1])
    colour_ok = (green | brown) & (s >= p.min_saturation) & (v >= p.min_value)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma_px = max(0.8, p.texture_sigma_m / gsd_m)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma_px)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    busy = np.hypot(gx, gy) > p.edge_threshold

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    g2 = np.zeros_like(gray)
    for ch in (1, 2):
        b = cv2.GaussianBlur(lab[..., ch], (0, 0), sigma_px)
        g2 += (cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8.0) ** 2
        g2 += (cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8.0) ** 2
    busy |= np.sqrt(g2) > p.colour_edge_threshold
    return ~colour_ok | busy


def find_spots(bgr: np.ndarray, gsd_m: float, center_uv=None, blocked_uv=(),
               params: LookParams | None = None, blocked_sides=()) -> LookResult:
    """Rank landing spots in one image.

    gsd_m      ground size of one pixel (metres), e.g. Camera.gsd(agl)
    center_uv  pixel of the mission centre (default: image centre)
    blocked_uv list of (u, v, radius_px) circles to treat as bad, e.g. spots
               that already failed the rangefinder check or moving objects
    blocked_sides  list of (u, v, du, dv): everything on the (du, dv) side of
               the line through (u, v) is bad. Used to stop searching uphill
               after a slope was found.
    """
    p = params or LookParams()
    h, w = bgr.shape[:2]
    cell_px = max(4, int(round(p.cell_m / gsd_m)))
    rows, cols = h // cell_px, w // cell_px
    if rows < 3 or cols < 3:
        return LookResult(cell_px, np.ones((max(rows, 1), max(cols, 1)), bool),
                          np.zeros((max(rows, 1), max(cols, 1))), [])
    img = bgr[:rows * cell_px, :cols * cell_px]
    bad_px = bad_pixel_mask(img, gsd_m, p)
    frac = bad_px.reshape(rows, cell_px, cols, cell_px).mean(axis=(1, 3))
    cell_bad = frac > p.bad_cell_fraction

    cc = (np.arange(cols) + 0.5) * cell_px
    rr = (np.arange(rows) + 0.5) * cell_px
    U, V = np.meshgrid(cc, rr)
    for (bu, bv, brad) in blocked_uv:
        cell_bad |= (U - bu) ** 2 + (V - bv) ** 2 <= brad ** 2
    for (su, sv, du, dv) in blocked_sides:
        cell_bad |= (U - su) * du + (V - sv) * dv > 0

    # Clearance: distance (in cells) to the nearest bad cell. The image
    # border counts as bad, because we can't see what's beyond it.
    good = np.pad((~cell_bad).astype(np.uint8), 1, constant_values=0)
    dist = cv2.distanceTransform(good, cv2.DIST_L2, 5)[1:-1, 1:-1]
    clearance = np.maximum(dist - 0.5, 0.0) * p.cell_m

    cu, cv_ = center_uv if center_uv is not None else (w / 2.0, h / 2.0)
    dist_center = np.hypot(U - cu, V - cv_) * gsd_m
    eligible = clearance >= p.required_radius_m
    score = np.minimum(clearance, p.clearance_credit_m) - p.distance_weight * dist_center
    score = np.where(eligible, score, -np.inf)

    picks = []
    for idx in np.argsort(score, axis=None)[::-1]:
        r, c = divmod(int(idx), cols)
        if not np.isfinite(score[r, c]) or len(picks) >= p.max_candidates:
            break
        if all(np.hypot(U[r, c] - q.u, V[r, c] - q.v) * gsd_m >= p.min_separation_m for q in picks):
            picks.append(Candidate(float(U[r, c]), float(V[r, c]), float(clearance[r, c]),
                                   float(dist_center[r, c]), float(score[r, c])))
    return LookResult(cell_px, cell_bad, clearance, picks)


def draw_debug(bgr: np.ndarray, result: LookResult, target_uv=None, moving=()) -> np.ndarray:
    """Overlay bad cells (red), candidates (yellow), target (cyan), moving objects (magenta)."""
    out = bgr.copy()
    cp = result.cell_px
    rows, cols = result.cell_bad.shape
    tint = np.zeros_like(out)
    mask = np.kron(result.cell_bad.astype(np.uint8), np.ones((cp, cp), np.uint8)).astype(bool)
    tint[:rows * cp, :cols * cp][mask] = (0, 0, 255)
    out = cv2.addWeighted(out, 1.0, tint, 0.45, 0)
    for i, c in enumerate(result.candidates):
        cv2.circle(out, (int(c.u), int(c.v)), 10, (0, 255, 255), 2)
        cv2.putText(out, str(i + 1), (int(c.u) + 12, int(c.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
    if target_uv is not None:
        cv2.drawMarker(out, (int(target_uv[0]), int(target_uv[1])), (255, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
    for (u, v, r) in moving:
        cv2.circle(out, (int(u), int(v)), int(max(r, 8)), (255, 0, 255), 2)
    return out
