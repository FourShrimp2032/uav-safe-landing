#!/usr/bin/env python3
"""Generate the Gazebo test worlds, their textures and ground-truth files.

Everything under sim/worlds, sim/ground_truth and sim/models/sl_assets is
produced by this script, so edit the WORLDS table below rather than the SDF
files. Run it from the repository root:

    python3 tools/make_worlds.py

The same WORLDS table is used by tools/summarize.py (to check whether a
landing spot was really safe) and by the unit tests (to render a top-down
picture of a world without running Gazebo).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "sim" / "models" / "sl_assets"
TEXTURES = ASSETS / "materials" / "textures"
WORLDS_DIR = ROOT / "sim" / "worlds"
GT_DIR = ROOT / "sim" / "ground_truth"
PKG_WORLDS_YAML = ROOT / "ros2_ws" / "src" / "safe_landing" / "config" / "worlds.yaml"

TERRAIN_SIZE_M = 128.0
HEIGHTMAP_PX = 257          # heightmaps must be 2^n + 1 pixels
GRASS_TILE_M = 10.24        # 1024 px grass texture -> 1 cm per pixel

# ---------------------------------------------------------------------------
# The four test scenarios (C1-C4 in the spec).
#
# Object kinds and their parameters:
#   rock  x y size            box about size x 0.8*size x 0.6*size
#   bush  x y r               textured sphere, sunk 40 % into the ground
#   tree  x y r               trunk + canopy sphere of radius r at about 4 m
#   house x y w d h [roof]    walls + flat roof (roof: "red" or "grey")
#   road  x y w d             thin asphalt slab
# ---------------------------------------------------------------------------
WORLDS = {
    "c1_field": {
        "title": "C1 ideal: flat field with bushes and rocks",
        "mission_xy": [8.0, 4.0],
        "spawn_xy": [0.0, 0.0],
        "shadows": False,
        "hill": None,
        "objects": [
            # rock cluster right on the mission point, so the drone must move
            {"kind": "rock", "x": 8.0, "y": 4.0, "size": 0.30},
            {"kind": "rock", "x": 8.6, "y": 3.5, "size": 0.25},
            {"kind": "rock", "x": 7.4, "y": 4.6, "size": 0.22},
            {"kind": "rock", "x": 8.3, "y": 5.0, "size": 0.20},
            {"kind": "bush", "x": 5.0, "y": 6.5, "r": 0.8},
            {"kind": "bush", "x": 11.5, "y": 2.0, "r": 0.7},
            {"kind": "bush", "x": 10.5, "y": 7.5, "r": 0.9},
            {"kind": "bush", "x": 4.0, "y": 1.0, "r": 0.6},
            {"kind": "bush", "x": 13.0, "y": 5.5, "r": 0.6},
            {"kind": "rock", "x": 6.0, "y": 2.5, "size": 0.25},
            {"kind": "rock", "x": 10.0, "y": 4.8, "size": 0.20},
            {"kind": "rock", "x": 2.5, "y": 6.0, "size": 0.30},
        ],
    },
    "c2_hill": {
        "title": "C2 hard: grassy hill, the camera can't see the slope",
        "mission_xy": [7.0, 0.0],
        "spawn_xy": [0.0, 0.0],
        "shadows": False,
        # cos^2 bump: height H at the centre, zero at radius R.
        # Steepest slope = H*pi/(2R) -> about 25 deg for H=3, R=10.
        "hill": {"x": 14.0, "y": 0.0, "height": 3.0, "radius": 10.0},
        "objects": [
            {"kind": "bush", "x": 1.5, "y": 5.0, "r": 0.7},
            {"kind": "rock", "x": 2.0, "y": -3.0, "size": 0.25},
            {"kind": "rock", "x": -1.0, "y": 4.0, "size": 0.25},
        ],
    },
    "c3_town": {
        "title": "C3 urban: houses, trees and a road; land in the courtyard",
        "mission_xy": [5.0, 6.0],     # on the roof of house A
        "spawn_xy": [0.0, -1.0],
        "shadows": True,
        "hill": None,
        "objects": [
            {"kind": "road", "x": 0.0, "y": -6.0, "w": 60.0, "d": 5.0},
            {"kind": "house", "x": 5.0, "y": 6.0, "w": 8.0, "d": 6.0, "h": 5.0, "roof": "grey"},
            {"kind": "house", "x": 19.0, "y": 6.0, "w": 8.0, "d": 6.0, "h": 6.0, "roof": "red"},
            {"kind": "house", "x": 12.0, "y": 17.0, "w": 10.0, "d": 6.0, "h": 5.0, "roof": "red"},
            {"kind": "tree", "x": 12.0, "y": 11.0, "r": 1.8},
            {"kind": "tree", "x": -2.0, "y": 9.0, "r": 2.0},
            {"kind": "tree", "x": 25.0, "y": 11.0, "r": 1.6},
            {"kind": "bush", "x": 14.0, "y": 2.0, "r": 0.6},
        ],
    },
    "c4_dynamic": {
        "title": "C4 dynamic: a person walks into the landing zone",
        "mission_xy": [6.0, 0.0],
        "spawn_xy": [0.0, 0.0],
        "shadows": False,
        "hill": None,
        "objects": [
            {"kind": "bush", "x": 3.0, "y": 4.0, "r": 0.7},
            {"kind": "bush", "x": 10.0, "y": -3.0, "r": 0.8},
            {"kind": "rock", "x": 8.0, "y": 3.0, "size": 0.25},
        ],
        # Loops every 36 s: waits away from the zone, walks onto the mission
        # point (about 2 m/s, the spec's limit), stands, then walks away.
        "actor": [
            # time, x, y, yaw
            [0.0, 6.0, -12.0, 1.5708],
            [12.0, 6.0, -12.0, 1.5708],
            [18.0, 6.0, 0.0, 1.5708],
            [22.0, 6.0, 0.0, 0.0],
            [26.0, 14.0, 0.0, -1.5708],
            [32.0, 14.0, -12.0, 3.1416],
            [36.0, 6.0, -12.0, 1.5708],
        ],
    },
}


# ---------------------------------------------------------------------------
# Ground truth helpers (also used by summarize.py and the tests)
# ---------------------------------------------------------------------------
def terrain_height(world: dict, x: float, y: float) -> float:
    hill = world.get("hill")
    if not hill:
        return 0.0
    r = math.hypot(x - hill["x"], y - hill["y"])
    if r >= hill["radius"]:
        return 0.0
    return hill["height"] * math.cos(math.pi * r / (2 * hill["radius"])) ** 2


def terrain_slope_deg(world: dict, x: float, y: float) -> float:
    hill = world.get("hill")
    if not hill:
        return 0.0
    r = math.hypot(x - hill["x"], y - hill["y"])
    if r >= hill["radius"]:
        return 0.0
    k = math.pi / (2 * hill["radius"])
    return math.degrees(math.atan(hill["height"] * k * abs(math.sin(2 * k * r))))


def object_footprint(obj: dict) -> tuple[str, tuple]:
    """Horizontal footprint used for the safety check: ("circle", (x, y, r))
    or ("rect", (xmin, ymin, xmax, ymax))."""
    k = obj["kind"]
    if k == "rock":
        return "circle", (obj["x"], obj["y"], obj["size"] * 0.6)
    if k == "bush":
        return "circle", (obj["x"], obj["y"], obj["r"])
    if k == "tree":
        return "circle", (obj["x"], obj["y"], obj["r"])
    if k in ("house", "road"):
        hw, hd = obj["w"] / 2, obj["d"] / 2
        return "rect", (obj["x"] - hw, obj["y"] - hd, obj["x"] + hw, obj["y"] + hd)
    raise ValueError(k)


def actor_xy(world: dict, t: float):
    """Where the walking person is at simulation time t (None if no person).
    Straight lines between waypoints; Gazebo smooths the corners a little."""
    wps = world.get("actor")
    if not wps:
        return None
    t = t % wps[-1][0]
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(wps, wps[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return wps[-1][1], wps[-1][2]


def distance_to_hazards(world: dict, x: float, y: float) -> float:
    """Distance from (x, y) to the nearest hazard edge (negative = inside)."""
    best = math.inf
    for obj in world["objects"]:
        shape, p = object_footprint(obj)
        if shape == "circle":
            d = math.hypot(x - p[0], y - p[1]) - p[2]
        else:
            dx = max(p[0] - x, 0.0, x - p[2])
            dy = max(p[1] - y, 0.0, y - p[3])
            d = math.hypot(dx, dy)
            if dx == 0 and dy == 0:
                d = -min(x - p[0], p[2] - x, y - p[1], p[3] - y)
        best = min(best, d)
    return best


# ---------------------------------------------------------------------------
# Procedural textures. Noise is made in the frequency domain, so every
# texture tiles seamlessly.
# ---------------------------------------------------------------------------
def periodic_noise(size: int, wavelength_px: float, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal((size, size))
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    fr = np.hypot(fx, fy)
    f0 = 1.0 / wavelength_px
    band = np.exp(-((fr - f0) ** 2) / (2 * (0.6 * f0) ** 2))
    band[0, 0] = 0
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * band))
    return out / (out.std() + 1e-9)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(path)


def make_textures(rng: np.random.Generator) -> dict[str, np.ndarray]:
    tex = {}
    n = 1024  # 1 cm per pixel over a 10.24 m tile
    base = np.array([72, 118, 42], float)
    shade = (1 + 0.10 * periodic_noise(n, 300, rng)
               + 0.06 * periodic_noise(n, 40, rng)
               + 0.05 * periodic_noise(n, 3, rng))
    hue = 0.06 * periodic_noise(n, 500, rng)
    grass = base[None, None, :] * shade[..., None]
    grass[..., 0] *= 1 + hue
    grass[..., 2] *= 1 - hue
    tex["grass"] = grass

    m = 256
    leaf = periodic_noise(m, 8, rng) + 0.5 * periodic_noise(m, 3, rng)
    foliage = np.array([38, 80, 28], float)[None, None, :] * (1 + 0.45 * leaf[..., None])
    foliage[leaf < -1.0] *= 0.35  # dark gaps between leaves
    tex["foliage"] = foliage

    speck = periodic_noise(m, 4, rng)
    tex["rock"] = np.array([128, 126, 120], float)[None, None, :] * (1 + 0.22 * speck[..., None])

    tex["asphalt"] = np.array([62, 62, 64], float)[None, None, :] * (
        1 + 0.12 * periodic_noise(m, 2, rng)[..., None])

    tex["bark"] = np.array([90, 62, 40], float)[None, None, :] * (
        1 + 0.25 * periodic_noise(m, 6, rng)[..., None])

    tex["wall"] = np.array([200, 186, 160], float)[None, None, :] * (
        1 + 0.04 * periodic_noise(m, 10, rng)[..., None])

    yy, xx = np.mgrid[0:m, 0:m]
    tiles = ((yy % 16) < 2) | ((xx + 8 * ((yy // 16) % 2)) % 16 < 1)
    for name, col in (("roof_red", [160, 58, 42]), ("roof_grey", [105, 105, 110])):
        t = np.array(col, float)[None, None, :] * (1 + 0.08 * periodic_noise(m, 5, rng)[..., None])
        t[tiles] *= 0.55
        tex[name] = t
    return tex


def write_assets(tex: dict[str, np.ndarray]) -> None:
    TEXTURES.mkdir(parents=True, exist_ok=True)
    for name, arr in tex.items():
        save_rgb(TEXTURES / f"{name}.png", arr)
    flat_normal = np.zeros((64, 64, 3))
    flat_normal[...] = [128, 128, 255]
    save_rgb(TEXTURES / "flat_normal.png", flat_normal)

    flat = np.zeros((HEIGHTMAP_PX, HEIGHTMAP_PX), np.uint8)
    Image.fromarray(flat, "L").save(ASSETS / "materials" / "textures" / "heightmap_flat.png")


def write_hill_heightmap(world: dict, name: str, z_size: float) -> None:
    half = TERRAIN_SIZE_M / 2
    coords = np.linspace(-half, half, HEIGHTMAP_PX)
    img = np.zeros((HEIGHTMAP_PX, HEIGHTMAP_PX))
    for row, yv in enumerate(coords[::-1]):        # image row 0 is +y (north)
        for col, xv in enumerate(coords):
            img[row, col] = terrain_height(world, xv, yv)
    Image.fromarray(np.round(img / z_size * 255).astype(np.uint8), "L").save(
        TEXTURES / f"heightmap_{name}.png")


# ---------------------------------------------------------------------------
# SDF writing
# ---------------------------------------------------------------------------
def tex_material(name: str) -> str:
    return f"""<material>
            <diffuse>1 1 1 1</diffuse>
            <specular>0.05 0.05 0.05 1</specular>
            <pbr><metal>
              <albedo_map>model://sl_assets/materials/textures/{name}.png</albedo_map>
              <roughness>0.9</roughness><metalness>0</metalness>
            </metal></pbr>
          </material>"""


def static_model(name: str, pose: str, geometry: str, texture: str, collide: bool = True) -> str:
    col = f"<collision name='c'><geometry>{geometry}</geometry></collision>" if collide else ""
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{pose}</pose>
      <link name="link">
        {col}
        <visual name="v">
          <geometry>{geometry}</geometry>
          {tex_material(texture)}
        </visual>
      </link>
    </model>"""


def objects_sdf(world: dict, rng: np.random.Generator) -> str:
    out = []
    for i, o in enumerate(world["objects"]):
        k = o["kind"]
        z0 = terrain_height(world, o["x"], o["y"])
        if k == "rock":
            s = o["size"]
            yaw = rng.uniform(0, math.pi)
            out.append(static_model(f"rock_{i}", f"{o['x']} {o['y']} {z0 + 0.3 * s} 0 0 {yaw:.3f}",
                                    f"<box><size>{s} {0.8 * s} {0.6 * s}</size></box>", "rock"))
        elif k == "bush":
            r = o["r"]
            out.append(static_model(f"bush_{i}", f"{o['x']} {o['y']} {z0 + 0.6 * r} 0 0 0",
                                    f"<sphere><radius>{r}</radius></sphere>", "foliage"))
        elif k == "tree":
            r = o["r"]
            trunk_h = 3.0
            out.append(static_model(f"trunk_{i}", f"{o['x']} {o['y']} {z0 + trunk_h / 2} 0 0 0",
                                    f"<cylinder><radius>0.15</radius><length>{trunk_h}</length></cylinder>",
                                    "bark"))
            out.append(static_model(f"canopy_{i}", f"{o['x']} {o['y']} {z0 + trunk_h + r * 0.7} 0 0 0",
                                    f"<sphere><radius>{r}</radius></sphere>", "foliage"))
        elif k == "house":
            w, d, h = o["w"], o["d"], o["h"]
            out.append(static_model(f"house_{i}", f"{o['x']} {o['y']} {z0 + h / 2} 0 0 0",
                                    f"<box><size>{w} {d} {h}</size></box>", "wall"))
            out.append(static_model(f"roof_{i}", f"{o['x']} {o['y']} {z0 + h + 0.1} 0 0 0",
                                    f"<box><size>{w + 0.4} {d + 0.4} 0.2</size></box>",
                                    f"roof_{o.get('roof', 'red')}"))
        elif k == "road":
            out.append(static_model(f"road_{i}", f"{o['x']} {o['y']} {z0 + 0.01} 0 0 0",
                                    f"<box><size>{o['w']} {o['d']} 0.02</size></box>", "asphalt",
                                    collide=False))
        else:
            raise ValueError(k)
    return "\n".join(out)


def actor_sdf(world: dict) -> str:
    wps = world.get("actor")
    if not wps:
        return ""
    url = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"
    points = "\n".join(
        f"          <waypoint><time>{t}</time><pose>{x} {y} 1.0 0 0 {yaw}</pose></waypoint>"
        for t, x, y, yaw in wps)
    return f"""
    <!-- The walking person. The mesh is downloaded from Gazebo Fuel the
         first time this world is opened, so the first run needs internet. -->
    <actor name="walker">
      <skin><filename>{url}</filename><scale>1.0</scale></skin>
      <animation name="walk">
        <filename>{url}</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walk" tension="0.6">
{points}
        </trajectory>
      </script>
    </actor>"""


def terrain_sdf(world: dict, name: str) -> str:
    hill = world.get("hill")
    if hill:
        z_size = hill["height"] * 1.25
        write_hill_heightmap(world, name, z_size)
        hm = f"heightmap_{name}.png"
        collision = f"""<collision name="terrain_collision">
          <geometry><heightmap>
            <uri>model://sl_assets/materials/textures/{hm}</uri>
            <size>{TERRAIN_SIZE_M} {TERRAIN_SIZE_M} {z_size}</size>
            <pos>0 0 0</pos>
          </heightmap></geometry>
        </collision>"""
    else:
        z_size = 1.0
        hm = "heightmap_flat.png"
        collision = """<collision name="ground_collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        </collision>"""
    return f"""
    <!-- Ground. A heightmap is used even for flat worlds because it is the
         one ground type whose grass texture repeats every few metres;
         without visible texture the camera can't tell grass from asphalt. -->
    <model name="terrain">
      <static>true</static>
      <link name="link">
        {collision}
        <visual name="terrain_visual">
          <geometry><heightmap>
            <use_terrain_paging>false</use_terrain_paging>
            <texture>
              <diffuse>model://sl_assets/materials/textures/grass.png</diffuse>
              <normal>model://sl_assets/materials/textures/flat_normal.png</normal>
              <size>{GRASS_TILE_M}</size>
            </texture>
            <uri>model://sl_assets/materials/textures/{hm}</uri>
            <size>{TERRAIN_SIZE_M} {TERRAIN_SIZE_M} {z_size}</size>
            <pos>0 0 0</pos>
          </heightmap></geometry>
        </visual>
      </link>
    </model>"""


def world_sdf(name: str, world: dict, rng: np.random.Generator) -> str:
    shadows = "true" if world["shadows"] else "false"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- {world['title']}
     Generated by tools/make_worlds.py. Edit that file, not this one. -->
<sdf version="1.9">
  <world name="{name}">
    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.7 0.8 0.95 1</background>
      <shadows>{shadows}</shadows>
    </scene>

    <light name="sun" type="directional">
      <pose>0 0 50 0 0 0</pose>
      <cast_shadows>{shadows}</cast_shadows>
      <intensity>1</intensity>
      <direction>-0.4 0.3 -0.87</direction>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>50.4501</latitude_deg>
      <longitude_deg>30.5234</longitude_deg>
      <elevation>150</elevation>
    </spherical_coordinates>
{terrain_sdf(world, name)}
{objects_sdf(world, rng)}
{actor_sdf(world)}
  </world>
</sdf>
"""


def ground_truth(name: str, world: dict) -> dict:
    return {
        "world": name,
        "title": world["title"],
        "mission_xy": world["mission_xy"],
        "spawn_xy": world["spawn_xy"],
        "hill": world["hill"],
        "objects": world["objects"],
        "actor_waypoints": world.get("actor"),
        "safety_rule": "landing is safe if slope < 10 deg and the nearest hazard edge is >= 0.5 m away",
    }


# ---------------------------------------------------------------------------
# Top-down orthographic picture of a world, for tests without Gazebo.
# ---------------------------------------------------------------------------
def render_topdown(world: dict, center_xy, size_px=(540, 960), gsd_m=0.02,
                   textures: dict | None = None, seed: int = 0) -> np.ndarray:
    """Return a BGR image (top of image = +y / north, right = +x / east),
    as seen by a downward camera with yaw 0 looking at center_xy."""
    rng = np.random.default_rng(seed)
    tex = textures or make_textures(rng)
    h, w = size_px
    cx, cy = center_xy
    js, is_ = np.meshgrid(np.arange(w), np.arange(h))
    X = cx + (js - w / 2 + 0.5) * gsd_m
    Y = cy - (is_ - h / 2 + 0.5) * gsd_m

    def sample(name, scale_m):
        t = tex[name]
        n = t.shape[0]
        u = ((X / scale_m) * n).astype(int) % n
        v = ((-Y / scale_m) * n).astype(int) % n
        return t[v, u]

    img = sample("grass", GRASS_TILE_M).copy()
    if world.get("hill"):
        # light shading so the hill isn't invisible, like a real renderer
        hgt = np.vectorize(lambda a, b: terrain_height(world, a, b))(X[::8, ::8], Y[::8, ::8])
        gy, gx = np.gradient(hgt)
        shade = 1 + 0.6 * (gx - gy) / 8
        shade = np.kron(shade, np.ones((8, 8)))[:h, :w]
        img *= shade[..., None]
    for o in world["objects"]:
        shape, p = object_footprint(o)
        name = {"rock": "rock", "bush": "foliage", "tree": "foliage",
                "road": "asphalt"}.get(o["kind"], f"roof_{o.get('roof', 'red')}")
        if shape == "circle":
            mask = (X - p[0]) ** 2 + (Y - p[1]) ** 2 <= p[2] ** 2
        else:
            mask = (X >= p[0]) & (X <= p[2]) & (Y >= p[1]) & (Y <= p[3])
        if mask.any():
            img[mask] = sample(name, 2.0)[mask]
    return np.clip(img[..., ::-1], 0, 255).astype(np.uint8)  # RGB -> BGR


def main() -> None:
    rng = np.random.default_rng(42)
    tex = make_textures(rng)
    write_assets(tex)
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for name, world in WORLDS.items():
        (WORLDS_DIR / f"{name}.sdf").write_text(world_sdf(name, world, rng))
        (GT_DIR / f"{name}.yaml").write_text(yaml.safe_dump(ground_truth(name, world), sort_keys=False))
        print(f"wrote {name}")
    # Per-world settings for the launch files. PX4's local frame starts at
    # the spawn point, so the mission point is given relative to it.
    launch_cfg = {name: {"spawn_x": w["spawn_xy"][0], "spawn_y": w["spawn_xy"][1],
                         "mission_x": w["mission_xy"][0] - w["spawn_xy"][0],
                         "mission_y": w["mission_xy"][1] - w["spawn_xy"][1]}
                  for name, w in WORLDS.items()}
    PKG_WORLDS_YAML.write_text("# Generated by tools/make_worlds.py\n" + yaml.safe_dump(launch_cfg))


if __name__ == "__main__":
    main()
