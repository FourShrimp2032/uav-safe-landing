"""Real-drone drivers that can be checked without the hardware: the TF-Luna
serial protocol and the camera undistortion onto the ideal pinhole."""
import math

import cv2
import numpy as np
import pytest
import yaml

from safe_landing.camera_model import (Calibration, load_calibration, max_hfov_deg, rotate,
                                       scaled, target_matrix, undistort_maps)
from safe_landing.geometry import Camera, intrinsics_from_fov
from safe_landing.tfluna import FrameParser, make_frame, parse_frame


# ------------------------------------------------------------------ TF-Luna
def test_frame_round_trip():
    r = parse_frame(make_frame(437, amp=900, temp_raw=2248))
    assert r.distance_m == pytest.approx(4.37)
    assert r.amp == 900 and r.temp_c == pytest.approx(25.0)
    assert r.valid


def test_bad_checksum_rejected():
    f = bytearray(make_frame(437))
    f[8] ^= 0xFF
    assert parse_frame(bytes(f)) is None


def test_weak_or_overexposed_return_is_not_valid():
    assert not parse_frame(make_frame(437, amp=40)).valid
    assert not parse_frame(make_frame(437, amp=65535)).valid
    assert not parse_frame(make_frame(5, amp=900)).valid       # below 0.2 m


def test_parser_handles_garbage_and_split_frames():
    stream = b"\x00\x59\x13" + make_frame(120) + b"\x59\x59\x01" + make_frame(250) + make_frame(300)
    p = FrameParser()
    out = []
    for i in range(0, len(stream), 4):          # arrives in small, misaligned chunks
        out += p.feed(stream[i:i + 4])
    assert [round(r.distance_m, 2) for r in out] == [1.2, 2.5, 3.0]


def test_parser_finds_frame_after_fake_header():
    # 0x59 0x59 inside the data must not lose the real frame that follows
    f = make_frame(0x5959)
    p = FrameParser()
    assert [r.distance_m for r in p.feed(f + make_frame(100))] == [pytest.approx(228.73), 1.0]


# ------------------------------------------------------------------ camera
def fake_calibration(w=1152, h=648):
    """A wide lens with noticeable barrel distortion."""
    f = 0.55 * w
    K = np.array([[f, 0, w / 2 + 6], [0, f, h / 2 - 4], [0, 0, 1]], float)
    D = np.array([-0.28, 0.08, 0.0005, -0.0003, -0.01])
    return Calibration(K, D, "plumb_bob", w, h)


def distort(points_norm, cal):
    """Ideal normalised image coords -> raw pixels through the lens model."""
    obj = np.c_[points_norm, np.ones(len(points_norm))].astype(np.float64)
    px, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), cal.K, cal.D)
    return px.reshape(-1, 2)


def test_undistorted_image_matches_the_node_camera():
    """A point on the ground must land on the pixel the landing node expects."""
    cal = fake_calibration()
    W, H, hfov = 640, 360, 90.0
    mx, my = undistort_maps(cal, W, H, hfov)
    node_cam = Camera(*intrinsics_from_fov(W, H, math.radians(hfov)))
    # pixel (u, v) of the output image should sample the raw pixel where the
    # same viewing ray hits the distorted sensor
    us, vs = np.meshgrid(np.linspace(20, W - 20, 9), np.linspace(20, H - 20, 5))
    norm = np.c_[(us.ravel() - node_cam.cx) / node_cam.fx, (vs.ravel() - node_cam.cy) / node_cam.fy]
    want = distort(norm, cal)
    maps = cv2.convertMaps(mx, my, cv2.CV_32FC1)
    got = np.c_[maps[0][vs.ravel().astype(int), us.ravel().astype(int)],
                maps[1][vs.ravel().astype(int), us.ravel().astype(int)]]
    assert np.abs(got - want).max() < 1.5


def test_straight_line_comes_out_straight():
    cal = fake_calibration()
    raw = np.zeros((cal.height, cal.width), np.uint8)
    # a straight world line, as the bent lens records it
    xs = np.linspace(-0.8, 0.8, 400)
    pts = distort(np.c_[xs, np.full_like(xs, 0.35)], cal)
    cv2.polylines(raw, [pts.astype(np.int32)], False, 255, 3)
    out = cv2.remap(raw, *undistort_maps(cal, 640, 360, 90.0), cv2.INTER_LINEAR)
    ys, xs_ = np.nonzero(out > 128)
    fit = np.polyfit(xs_, ys, 1)
    assert np.abs(np.polyval(fit, xs_) - ys).max() < 3.0


def test_scaled_and_max_hfov(tmp_path):
    cal = fake_calibration()
    half = scaled(cal, cal.width // 2, cal.height // 2)
    assert half.K[0, 0] == pytest.approx(cal.K[0, 0] / 2)
    assert 90 < max_hfov_deg(cal) < 150
    f = tmp_path / "ost.yaml"
    f.write_text(yaml.safe_dump({
        "image_width": cal.width, "image_height": cal.height,
        "camera_matrix": {"rows": 3, "cols": 3, "data": cal.K.ravel().tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": cal.D.tolist()}}))
    loaded = load_calibration(str(f))
    assert np.allclose(loaded.K, cal.K) and np.allclose(loaded.D, cal.D)


def test_target_matrix_matches_node_intrinsics():
    K = target_matrix(1280, 720, 95.0)
    fx, fy, cx, cy = intrinsics_from_fov(1280, 720, math.radians(95.0))
    assert np.allclose(K, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def test_rotation_only_keeps_long_side_across():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert rotate(img, 180)[0, 0] == 11
    with pytest.raises(ValueError):
        rotate(img, 90)
