"""Perspective renderer for tests: what the downward camera sees of a world,
including the sides of houses and the parallax of tall things, which the
flat top-down render (make_worlds.render_topdown) can't show."""
import numpy as np

import make_worlds as mw
from safe_landing.geometry import R_BODY_OPTICAL


def solids(objects):
    out = []
    for o in objects:
        k = o["kind"]
        if k == "house":
            w, d, h = o["w"], o["d"], o["h"]
            out.append(("box", (o["x"] - w / 2, o["y"] - d / 2, 0, o["x"] + w / 2, o["y"] + d / 2, h), "wall"))
            out.append(("box", (o["x"] - w / 2 - .2, o["y"] - d / 2 - .2, h,
                                o["x"] + w / 2 + .2, o["y"] + d / 2 + .2, h + .2), f"roof_{o.get('roof', 'red')}"))
        elif k == "person":
            out.append(("box", (o["x"] - .25, o["y"] - .2, 0, o["x"] + .25, o["y"] + .2, 1.7), "wall"))
        elif k == "tree":
            out.append(("sph", (o["x"], o["y"], 3 + .7 * o["r"], o["r"]), "foliage"))
        elif k == "bush":
            out.append(("sph", (o["x"], o["y"], .6 * o["r"], o["r"]), "foliage"))
    return out


def render(objects, cam, R_wb, t_wb, width, height, tex):
    """BGR image of `objects` on flat grass from camera pose (R_wb, t_wb)."""
    js, is_ = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
    rays = np.stack([(js - cam.cx) / cam.fx, (is_ - cam.cy) / cam.fy, np.ones_like(js)], -1)
    d = rays @ (R_wb @ R_BODY_OPTICAL).T
    o = t_wb + R_wb @ cam.offset_body
    best = np.where(d[..., 2] < 0, -o[2] / d[..., 2], np.inf)
    names = ["grass"]
    hit_id = np.zeros(best.shape, np.int16)
    for kind, p, name in solids(objects):
        if name not in names:
            names.append(name)
        if kind == "box":
            lo, hi = np.array(p[:3]), np.array(p[3:])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - o) / d, (hi - o) / d
            tn = np.nanmax(np.minimum(t1, t2), -1)
            tf = np.nanmin(np.maximum(t1, t2), -1)
            hit = (tn <= tf) & (tn > 0) & (tn < best)
        else:
            oc = o - np.array(p[:3])
            b, a = d @ oc, (d * d).sum(-1)
            disc = b * b - a * (oc @ oc - p[3] ** 2)
            tn = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
            hit = (tn > 0) & (tn < best)
        best = np.where(hit, tn, best)
        hit_id[hit] = names.index(name)
    P = o + best[..., None] * d
    img = np.zeros((height, width, 3))
    for i, name in enumerate(names):
        m = hit_id == i
        if not m.any():
            continue
        t = tex[name]
        if t.shape[:2] == (1, 1):
            img[m] = t[0, 0]
            continue
        n, s = t.shape[0], mw.GRASS_TILE_M if name == "grass" else 2.0
        a, b = (P[m, 0] + P[m, 1], P[m, 2]) if name == "wall" else (P[m, 0], P[m, 1])
        img[m] = t[((-b / s) * n).astype(int) % n, ((a / s) * n).astype(int) % n]
    return np.clip(img[..., ::-1], 0, 255).astype(np.uint8)
