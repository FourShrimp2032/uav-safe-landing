import cv2

from make_worlds import render_topdown
from safe_landing.motion import MotionDetector

GSD = 0.0202


def frames(person_path, drift_px=(0, 0), rot_deg=0.0):
    base = render_topdown({"objects": [], "hill": None}, (0, 0), size_px=(700, 1100), gsd_m=GSD)
    out = []
    for i, pos in enumerate(person_path):
        img = base.copy()
        if pos is not None:
            u, v = pos
            cv2.ellipse(img, (u, v), (14, 22), 0, 0, 360, (40, 40, 160), -1)  # person ~0.6 x 0.9 m
        M = cv2.getRotationMatrix2D((550, 350), rot_deg * i, 1.0)
        M[:, 2] += (drift_px[0] * i, drift_px[1] * i)
        img = cv2.warpAffine(img, M, (1100, 700))
        out.append(img[80:620, 70:1030])  # 960 x 540 crop
    return out


def test_walking_person_is_detected_while_drone_drifts():
    path = [(300 + 20 * i, 350) for i in range(6)]   # ~0.4 m per frame = 2 m/s at 5 Hz
    det = MotionDetector()
    found = [det.update(f, GSD) for f in frames(path, drift_px=(3, -2), rot_deg=1.0)]
    assert det.reliable
    assert any(found[2:]), "person not detected"


def test_static_scene_has_no_motion():
    det = MotionDetector()
    found = [det.update(f, GSD) for f in frames([None] * 6, drift_px=(4, 3), rot_deg=1.5)]
    assert det.reliable
    assert not any(found)
