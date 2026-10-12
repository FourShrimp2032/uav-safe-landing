# UAV safe landing: "look, then feel"

A quadcopter lands safely on unprepared ground using one downward camera and two cheap laser rangefinders. It needs no LiDAR scanner and no stereo camera. Everything runs first in Gazebo (PX4 SITL + ROS 2 + MAVROS) and then moves to a real drone.

## How it works

1. **Look (10 m up):** the camera image is cut into 25 cm squares. A square is bad if it isn't grass or dirt coloured, or if its texture is busy (bushes, rocks, roofs, objects). The drone picks the open spot with at least 1 m clearance that is closest to the mission point. (`look.py`)
2. **Move** over that spot and **confirm** it from straight above.
3. **Descend and feel:** one rangefinder points straight down and one is tilted 15° forward. From 5 m down, the drone turns at 45°/s while descending, so the tilted laser dot draws a spiral over the landing circle. A plane is fitted to all the points:
   - tilt > 10° → slope, abort
   - points more than 12 cm off the plane → rock, hole or bush, abort

   (`feel.py`)
4. Meanwhile the camera watches for **moving objects** (people, animals, cars) by aligning consecutive frames on the ground and looking for what still changed. Anything moving near the landing circle triggers an abort, until the drone is 0.7 m above the ground. The camera is a wide-angle one (102°) so that it still sees a few metres around the spot near the ground. Roofs and tree tops also seem to move while the drone descends (they are closer, so they grow faster in the picture). That is told apart because they move straight outwards from the point below the drone, and because the camera already saw them as obstacles before the descent. (`motion.py`)
5. **Land** with PX4's AUTO.LAND if the ground check passed. Otherwise **go around**: climb back to 10 m, mark the spot as bad, and look again. After 3 failed attempts, return home (RTL).

The camera only decides *where to look*. The lasers decide *whether it's safe*, so a camera mistake (a shadow, a slope it can't see) costs a retry, not a crash.

## Hardware this simulates

| Part | Price | Mass |
|---|---|---|
| Raspberry Pi 5 (4 GB) + cooler | ~$65 | ~50 g |
| Pi Camera Module 3 Wide (102°), pointing down | ~$35 | ~4 g |
| 2× Benewake TF-Luna (down + 15° tilt) | ~$45 | ~10 g |
| 5 V BEC, cables, mount | ~$30 | ~70 g |
| **Total** | **~$175** | **~135 g, ~9 W** |

## Run it on a Mac

Gazebo and ROS 2 don't run natively on macOS, so everything runs in Docker and you watch Gazebo in your browser.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/). In Settings → Resources, give it at least **4 CPUs and 8 GB of memory**.
2. Clone this repo and build the image. The first build downloads several GB and compiles PX4, which takes 30–60 minutes.
   ```bash
   git clone https://github.com/FourShrimp2032/uav-safe-landing.git
   cd uav-safe-landing
   docker compose up -d --build
   ```
3. Open **http://localhost:6080/vnc.html** and click Connect. This is the container's screen, and Gazebo will appear here.
4. Open a shell inside the container and start a landing:
   ```bash
   docker compose exec sim bash
   ./scripts/run_sim.sh c1_field
   ```
   The first run builds the ROS package. The drone takes off, flies to the mission point at 10 m, picks a spot, and lands. The terminal prints every decision (`LOOK -> MOVE ...`).

Gazebo renders on the CPU inside Docker, so it runs slower than real time. That's expected, and the landing logic runs on simulation time, so the results are the same.

**Worlds:** `c1_field` (bushes and rocks), `c2_hill` (a slope the camera can't see), `c3_town` (houses, trees and a road), `c4_dynamic` (a person walks into the landing zone). `c4_dynamic` downloads the person model from Gazebo Fuel the first time, so it needs internet.

### Watching what the algorithm sees
In a second shell (`docker compose exec sim bash`):
```bash
ros2 run rqt_image_view rqt_image_view /landing/debug/look   # bad squares red, candidates yellow
ros2 topic echo /landing/status                              # state, height, ground check
rviz2                                                        # add PointCloud2 /landing/debug/feel_points (frame: map)
```

### Replaying what the camera saw
Every run records the debug picture in its rosbag. To turn the newest run of a world into a video:
```bash
python3 tools/debug_video.py c3_town
```
This writes `logs/c3_town/bag_<time>_look.mp4` and a folder with one picture per LOOK / CONFIRM decision, which you can open on your Mac. The colours are the same as in the live view.

### Results and KPIs
Each run writes to `logs/<world>/` on your Mac:
- `*_log.csv`: state, position, height, ground-check numbers, at 10 Hz (NFR-04)
- `*_result.json`: outcome, landing error, attempts, every state change and why
- `bag_*`: rosbag of all topics

For KPI-4 (≥ 30 runs per scenario), run a headless batch and summarize it:
```bash
./scripts/batch.sh c1_field 30
python3 tools/summarize.py c1_field
```
The summary checks every landing spot against the world's ground truth (slope < 10°, ≥ 0.5 m from any hazard) and reports KPI-2, KPI-3 and KPI-4.

## Repository layout

```
ros2_ws/src/safe_landing/
  safe_landing/look.py          camera: find open ground            (pure Python, unit-tested)
  safe_landing/feel.py          rangefinders: slope + bump check     (pure Python, unit-tested)
  safe_landing/motion.py        camera: moving objects               (pure Python, unit-tested)
  safe_landing/geometry.py      pixel <-> ground, beam -> 3D point   (pure Python, unit-tested)
  safe_landing/landing_node.py  ROS 2 node: state machine + MAVROS
  config/params.yaml            every tunable number
  launch/sim.launch.py          bridge + MAVROS + node + rosbag
sim/models/x500_landing/        PX4 x500 + camera + 2 rangefinders
sim/px4/airframes/              PX4 airframe: x500 that arms without RC or QGroundControl
sim/worlds/                     C1-C4 (generated, don't edit by hand)
tools/make_worlds.py            generates worlds, textures, ground truth
tools/summarize.py              KPI report
scripts/run_sim.sh, batch.sh    run one landing / a batch
docker/                         the simulation container
```

To change a world, edit the `WORLDS` table in `tools/make_worlds.py` and run `python3 tools/make_worlds.py`.

## Tests without Gazebo

The algorithms are tested on top-down renders of the same worlds, and on a simulated descent spiral over the analytic terrain. These tests run on any computer with Python, including your Mac:
```bash
pip install numpy opencv-python-headless pillow pyyaml pytest
python3 -m pytest ros2_ws/src/safe_landing/test
```
GitHub Actions runs them on every push.

## Status and known limits

- The algorithms pass their unit tests. **The full Gazebo run has not been tried end to end yet.** The first run may need small fixes: topic names, PX4 start-up timing, or thresholds tuned to how Gazebo actually renders the textures. Start with `c1_field`.
- **Shadows:** the drone's own shadow moves when the drone moves, which can look like a moving object. Shadows are turned off in C1, C2 and C4 and on in C3 (NFR-05). Expect to tune `motion.py` there.
- **Sun glare:** a reflection of the sun (on water, a wet roof, a car) slides across the picture as the drone moves, so the moving-object check can take it for something moving. The simulated sun has no specular light for this reason; test over shiny surfaces outdoors.
- **Thin obstacles** such as power lines and thin branches aren't seen reliably by either sensor.
- **Sparse coverage:** a small rock between two loops of the spiral can be missed. Descending slower or turning faster makes the spiral denser.
- **Real sunlight** shortens the TF-Luna's range on dark grass. The check starts at 5 m, so test outdoors early.

## Moving to the real drone

- Wire the downward TF-Luna to the flight controller, so PX4 uses it for landing even if the Pi crashes. The Pi reads it back through MAVROS (`/mavros/distance_sensor/...`). Wire the tilted one to the Pi (UART or I2C). A small driver node then publishes both as `sensor_msgs/LaserScan` on `/landing/range_down/scan` and `/landing/range_tilt/scan`.
- Use `camera_ros` (libcamera) for the Pi camera. Calibrate it, and put the real FOV and resolution in `params.yaml` *and* in `model.sdf`, so the simulation sees what the real camera sees. The wide lens bends straight lines near the edges, so undistort each frame with the calibration (`cv2.undistort`) before the node uses it; the simulated camera has no distortion.
- Set `auto_takeoff: false`. The pilot flies to the area and switches to OFFBOARD to hand over, and switching out of OFFBOARD at any time takes control back immediately.
- MAVROS `fcu_url` becomes the serial port, for example `/dev/ttyAMA0:921600`.
- Order: bench test, then hover and scan without landing (record bags), then land on a prepared pad, then rough ground. Always with a pilot ready to take over.
