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
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionParams:
    work_width: int = 480           # frames are shrunk to this width first
    diff_threshold: int = 28        # grey-level change that counts as "changed"
    min_area_m2: float = 0.08       # smaller blobs are ignored (a person from above is ~0.15 m^2)
    persist_frames: int = 2         # seen in this many of the last `history` frames
    history: int = 3
    persist_radius_m: float = 1.0   # blob centres this close count as the same object
    min_matches: int = 30          # tracked corners needed to trust the alignment


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
        self.recent = deque(maxlen=self.p.history)
        self.reliable = False

    def reset(self) -> None:
        self.prev = None
        self.recent.clear()

    def update(self, bgr: np.ndarray, gsd_m: float) -> list[Blob]:
        """Feed one frame; returns confirmed moving blobs in this frame."""
        h, w = bgr.shape[:2]
        scale = self.p.work_width / float(w)
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        gsd_small = gsd_m / scale
        blobs: list[Blob] = []
        self.reliable = False
        if self.prev is not None:
            H = self._track(self.prev, small)
            if H is not None:
                self.reliable = True
                blobs = self._diff_blobs(small, H, gsd_small, scale)

        self.prev = small
        self.recent.append(blobs)
        return self._persistent(blobs, gsd_m)

    def _track(self, prev, cur):
        """Homography prev -> cur from corners tracked with optical flow.
        (Corner tracking copes with low-contrast grass better than feature
        descriptors do.)"""
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.005, minDistance=8)
        if pts is None or len(pts) < self.p.min_matches:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < self.p.min_matches:
            return None
        H, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
        if H is None or inl is None or inl.sum() < self.p.min_matches:
            return None
        return H

    def _diff_blobs(self, cur, H, gsd_small, scale) -> list[Blob]:
        hh, ww = cur.shape
        warped = cv2.warpPerspective(self.prev, H, (ww, hh))
        valid = cv2.warpPerspective(np.full_like(self.prev, 255), H, (ww, hh))
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
        diff = cv2.absdiff(cur, warped)
        diff[valid == 0] = 0
        mask = (diff > self.p.diff_threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):
            area_m2 = stats[i, cv2.CC_STAT_AREA] * gsd_small ** 2
            if area_m2 >= self.p.min_area_m2:
                r = 0.5 * max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
                out.append(Blob(cents[i][0] / scale, cents[i][1] / scale, r / scale, area_m2))
        return out

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
