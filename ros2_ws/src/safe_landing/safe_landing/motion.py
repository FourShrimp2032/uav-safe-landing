"""Moving-object detection from a downward camera (FR-05).

The drone itself moves a little between frames, so frames can't be
subtracted directly. Instead, corners are tracked from the previous frame to
the current one (optical flow), and the previous frame is warped onto the
current one with the homography they give (the ground is roughly a plane, and a
homography maps one view of a plane exactly onto another). Whatever still
differs after the warp didn't move with the ground: a person, an animal, a
car.

A blob only counts once it has been seen in at least `persist_frames` of the
last few frames, which filters out single-frame noise and lighting flicker.

Tall things (roofs, walls, tree tops) are not on the ground plane, so while
the drone descends they don't line up after the warp either: they grow
outwards from the point straight below the camera faster than the ground does
(parallax). When the caller passes that point (`epipole_uv`), blobs whose own
motion points straight away from it are taken as parallax and dropped. A
person walking towards or across the landing spot still counts; one walking
straight away from it is ignored, which is safe.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .look import LookParams, bad_pixel_mask


@dataclass
class MotionParams:
    work_width: int = 480           # frames are shrunk to this width first
    diff_threshold: int = 28        # grey-level change that counts as "changed"
    min_area_m2: float = 0.08       # smaller blobs are ignored (a person from above is ~0.15 m^2)
    persist_frames: int = 2         # seen in this many of the last `history` frames
    history: int = 3
    persist_radius_m: float = 1.0   # blob centres this close count as the same object
    min_matches: int = 30          # tracked corners needed to trust the alignment
    min_second_plane: int = 15     # corners needed to fit the ground behind a big roof
    parallax_max_angle_deg: float = 35.0  # blob motion this close to "outwards" = parallax
    parallax_min_flow_px: float = 0.5     # slower blobs have no reliable direction


@dataclass
class Blob:
    u: float          # full-resolution pixel coordinates
    v: float
    radius_px: float
    area_m2: float


class MotionDetector:
    def __init__(self, params: MotionParams | None = None):
        self.p = params or MotionParams()
        self.prev = None
        self.prev_agl = None
        self.prev_ground = None
        self.look_params = LookParams()
        self.recent = deque(maxlen=self.p.history)
        self.reliable = False

    def reset(self) -> None:
        self.prev = None
        self.prev_agl = None
        self.prev_ground = None
        self.recent.clear()

    def update(self, bgr: np.ndarray, gsd_m: float, epipole_uv=None, agl=None) -> list[Blob]:
        """Feed one frame; returns confirmed moving blobs in this frame.

        epipole_uv  pixel straight below the camera, given only while the
                    drone is descending; enables the parallax filter.
        agl         height above the ground (m). With it, the frame-to-frame
                    alignment locks onto the ground even when a big roof
                    fills much of the picture.
        """
        h, w = bgr.shape[:2]
        scale = self.p.work_width / float(w)
        small_bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gsd_small = gsd_m / scale
        blobs: list[Blob] = []
        self.reliable = False
        if self.prev is not None:
            growth = None
            if agl is not None and self.prev_agl is not None and agl > 0.5:
                growth = (self.prev_agl / agl) ** 2
            H = self._track(self.prev, small, growth, self.prev_ground)
            if H is not None:
                self.reliable = True
                epi = None if epipole_uv is None else (epipole_uv[0] * scale, epipole_uv[1] * scale)
                blobs = self._diff_blobs(small, H, gsd_small, scale, epi)

        self.prev = small
        self.prev_agl = agl
        # Corners are tracked only on open ground (grass, dirt), so roofs and
        # tree tops don't pull the alignment off the ground plane.
        ground = (~bad_pixel_mask(small_bgr, gsd_small, self.look_params)).astype(np.uint8) * 255
        self.prev_ground = ground if np.count_nonzero(ground) > 0.2 * ground.size else None
        self.recent.append(blobs)
        return self._persistent(blobs, gsd_m)

    def _track(self, prev, cur, ground_growth=None, mask=None):
        """Homography prev -> cur from corners tracked with optical flow.
        (Corner tracking copes with low-contrast grass better than feature
        descriptors do.)

        While the drone changes height, a big roof in view can win the fit.
        So with ground_growth (how much the ground should magnify between the
        two frames, from the change in height) up to three planes are fitted
        in turn, and the one whose magnification matches the ground is kept.
        """
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.005, minDistance=8, mask=mask)
        if pts is None or len(pts) < self.p.min_matches:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < self.p.min_matches:
            return None
        a, b = pts[ok], nxt[ok]
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
        if H is None or inl is None or inl.sum() < self.p.min_matches:
            return None
        if ground_growth is None:
            return H
        best, best_err = H, abs(np.log(_growth(H, cur.shape) / ground_growth))
        rest = inl.ravel() == 0
        for _ in range(2):
            if rest.sum() < self.p.min_second_plane:
                break
            idx = np.flatnonzero(rest)
            H2, inl2 = cv2.findHomography(a[idx], b[idx], cv2.RANSAC, 2.0)
            if H2 is None or inl2 is None or inl2.sum() < self.p.min_second_plane:
                break
            err = abs(np.log(_growth(H2, cur.shape) / ground_growth))
            if err < best_err:
                best, best_err = H2, err
            rest[idx[inl2.ravel() == 1]] = False
        return best

    def _diff_blobs(self, cur, H, gsd_small, scale, epipole=None) -> list[Blob]:
        hh, ww = cur.shape
        warped = cv2.warpPerspective(self.prev, H, (ww, hh))
        valid = cv2.warpPerspective(np.full_like(self.prev, 255), H, (ww, hh))
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        diff = cv2.absdiff(cur, warped)
        diff[valid == 0] = 0
        mask = (diff > self.p.diff_threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        flow = None
        out = []
        for i in range(1, n):
            area_m2 = stats[i, cv2.CC_STAT_AREA] * gsd_small ** 2
            if area_m2 >= self.p.min_area_m2:
                if epipole is not None:
                    if flow is None:
                        flow = cv2.calcOpticalFlowFarneback(warped, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    if self._is_parallax(flow[labels == i], cents[i], epipole):
                        continue
                r = 0.5 * max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                out.append(Blob(cents[i][0] / scale, cents[i][1] / scale, r / scale, area_m2))
        return out

    def _is_parallax(self, flow_px: np.ndarray, centre, epipole) -> bool:
        """True if the blob's motion points straight away from the epipole."""
        f = np.median(flow_px, axis=0)
        mag = float(np.hypot(*f))
        out = np.array([centre[0] - epipole[0], centre[1] - epipole[1]])
        out_n = float(np.hypot(*out))
        if mag < self.p.parallax_min_flow_px or out_n < 1.0:
            return False
        cos = float(f @ out) / (mag * out_n)
        return cos >= np.cos(np.radians(self.p.parallax_max_angle_deg))

    def _persistent(self, blobs, gsd_m) -> list[Blob]:
        need = self.p.persist_frames
        if need <= 1:
            return blobs
        rad_px = self.p.persist_radius_m / gsd_m
        confirmed = []
        for b in blobs:
            seen = sum(any(np.hypot(b.u - o.u, b.v - o.v) <= rad_px + o.radius_px for o in frame)
                       for frame in self.recent)
            if seen >= need:
                confirmed.append(b)
        return confirmed


def _growth(H: np.ndarray, shape) -> float:
    """How much H magnifies a small patch at the image centre."""
    h, w = shape[:2]
    c = np.array([[[w / 2, h / 2], [w / 2 + 50, h / 2], [w / 2, h / 2 + 50]]], np.float32)
    p = cv2.perspectiveTransform(c, H)[0]
    (ax, ay), (bx, by) = p[1] - p[0], p[2] - p[0]
    return abs(float(ax * by - ay * bx)) / 2500.0
