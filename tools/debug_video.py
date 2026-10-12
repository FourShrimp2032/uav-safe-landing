#!/usr/bin/env python3
"""Turn a run's rosbag into a video of what the landing camera saw and decided.

    python3 tools/debug_video.py c3_town            # newest run of that world
    python3 tools/debug_video.py /root/landing_logs/c3_town/bag_20261012_130000

Writes next to the bag (so it shows up in ./logs on your Mac):
  <bag>_look.mp4   every /landing/debug/look frame, with state and height on top
  <bag>_look/      one PNG per LOOK / CONFIRM decision, plus the last frame
                   before every go-around (NN_abort.png, shows why it aborted)

In the picture: red squares = ground the camera rejected, yellow circles =
candidate spots (1 = best), cyan cross = chosen target, magenta circles =
moving objects. Run it inside the container (it needs ROS 2 to read the bag).
"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

LOG_ROOT = "/root/landing_logs"
DECISION_STATES = {"LOOK", "CONFIRM"}


def find_bag(arg: str) -> str:
    if os.path.isdir(arg):
        return arg
    # bag_<time>_look/ is this tool's own picture folder, not a recording.
    bags = sorted(b for b in glob.glob(os.path.join(LOG_ROOT, arg, "bag_*"))
                  if os.path.isdir(b) and not b.endswith("_look"))
    if not bags:
        sys.exit(f"no bags in {os.path.join(LOG_ROOT, arg)}")
    return bags[-1]


def repair(bag: str) -> None:
    """A recording that was killed (Ctrl+C on a batch, a crash) has no
    metadata.yaml, so ROS can't open it. `ros2 bag reindex` rebuilds it."""
    if os.path.exists(os.path.join(bag, "metadata.yaml")):
        return
    files = glob.glob(os.path.join(bag, "*.mcap")) + glob.glob(os.path.join(bag, "*.db3"))
    if not files:
        sys.exit(f"{bag} has no recording in it")
    storage = "mcap" if files[0].endswith(".mcap") else "sqlite3"
    print(f"{bag} wasn't closed properly, rebuilding its index...")
    subprocess.run(["ros2", "bag", "reindex", bag, "-s", storage], check=False)
    if not os.path.exists(os.path.join(bag, "metadata.yaml")):
        sys.exit("couldn't repair it; try an older run: python3 tools/debug_video.py "
                 + os.path.join(os.path.dirname(bag), "bag_<time>"))


def to_bgr(msg) -> np.ndarray:
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return img if msg.encoding == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bag = find_bag(sys.argv[1]).rstrip("/")
    repair(bag)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if "/landing/debug/look" not in types:
        sys.exit("this bag has no /landing/debug/look frames")
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/landing/debug/look", "/landing/status"]))

    png_dir = bag + "_look"
    os.makedirs(png_dir, exist_ok=True)
    video, status, t0, n, n_png = None, {}, None, 0, 0
    last_img, last_state = None, None
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/landing/status":
            status = json.loads(msg.data)
            continue
        img = to_bgr(msg).copy()
        t0 = t0 if t0 is not None else t_ns
        state = status.get("state", "?")
        label = f"t={(t_ns - t0) / 1e9:5.1f}s  {state}  height {status.get('agl', '?')} m"
        if status.get("feel"):
            label += f"  ground: {status['feel']}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if video is None:
            video = cv2.VideoWriter(bag + "_look.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5,
                                    (img.shape[1], img.shape[0]))
        video.write(img)
        n += 1
        if state in DECISION_STATES:
            n_png += 1
            cv2.imwrite(os.path.join(png_dir, f"{n_png:02d}_{state.lower()}.png"), img)
        if state == "GO_AROUND" and last_state != "GO_AROUND" and last_img is not None:
            n_png += 1
            cv2.imwrite(os.path.join(png_dir, f"{n_png:02d}_abort.png"), last_img)
        last_img, last_state = img, state
    if video is None:
        sys.exit("no debug frames found")
    video.release()
    print(f"{n} frames -> {bag}_look.mp4")
    print(f"{n_png} decision pictures -> {png_dir}/")


if __name__ == "__main__":
    main()
