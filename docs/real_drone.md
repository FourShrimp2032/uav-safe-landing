# Moving to the real drone

The simulation and the real drone run the same landing node. On the real
drone three small pieces replace Gazebo:

| Simulation | Real drone |
|---|---|
| Gazebo camera → `/landing/camera/image` | `camera_ros` → `camera_prep` (undistort, 5 Hz) → `/landing/camera/image` |
| Gazebo rangefinders → `/landing/range_*/scan` | `range_driver`: tilted TF-Luna on a Pi UART, downward TF-Luna through PX4 and MAVROS |
| MAVROS over UDP to PX4 SITL | MAVROS over the serial link to the flight controller |

Start everything with `ros2 launch safe_landing real.launch.py`.
Settings that differ from the simulation are in
`ros2_ws/src/safe_landing/config/params_real.yaml`.

Items marked **check** are written from the datasheets and the PX4/MAVROS
documentation, but haven't been tried on hardware yet. Confirm each one on
the bench before the first flight.

## Parts

Raspberry Pi 5 (4 GB) with the Active Cooler, a 64 GB A2 microSD card,
Camera Module 3 Wide with a Pi 5 camera cable (22-pin to 15-pin), 2× Benewake
TF-Luna (plus a spare), a 5 V / 5 A BEC, JST-GH cables and an M2.5 standoff
kit. The camera and the 15° rangefinder bracket are 3D-printed.

## Wiring

| From | To | Notes |
|---|---|---|
| Flight controller TELEM2: TX, RX, GND | Pi GPIO 15 (RX), GPIO 14 (TX), GND | UART0, `/dev/ttyAMA0`. Don't connect the 5 V wire |
| Downward TF-Luna: 5 V, GND, TXD, RXD | Flight controller TELEM3 (or GPS2): 5 V, GND, RX, TX | Pin 5 (config) left open = UART mode |
| Tilted TF-Luna: 5 V, GND, TXD, RXD | Pi 5 V, GND, GPIO 9 (RX), GPIO 8 (TX) | UART3, `/dev/ttyAMA3`. TF-Luna UART is 3.3 V, safe for the Pi |
| Camera | Pi CAM/DISP 0 | 22-pin end on the Pi |
| BEC 5 V / 5 A | Pi 5 V and GND pins (or USB-C) | From the flight battery |

TX always goes to RX. The downward TF-Luna points straight down. The tilted
one points 15° forward from vertical (towards the nose). The camera points
straight down, with the top of the image towards the nose and the long side
of the sensor across the drone. If it ends up mounted the other way round,
set `camera_rotate_deg: 180`.

Measure where the camera and both rangefinders sit relative to the drone's
centre. Put the numbers in `params_real.yaml` (metres, x forward, y left,
z up).

## Raspberry Pi

1. Install Ubuntu 24.04 Server 64-bit, then ROS 2 Jazzy (`ros-jazzy-ros-base`),
   `ros-jazzy-mavros`, `ros-jazzy-mavros-extras`, `ros-jazzy-cv-bridge`,
   `ros-jazzy-camera-calibration`, `python3-serial`, `python3-opencv`.
   Then install the MAVROS geographic datasets
   (`ros2 run mavros install_geographiclib_datasets.sh`).
2. **Camera (the hardest step).** `camera_ros` needs the Raspberry Pi
   fork of libcamera to see the Camera Module 3. Ubuntu's own libcamera
   doesn't support it. Build Raspberry Pi's libcamera and `camera_ros`
   from source in the workspace. **Check:** `cam --list` shows the
   imx708 sensor. A fallback that avoids this: Raspberry Pi OS with ROS 2
   Jazzy in Docker and the camera device passed through.
3. UARTs. Add these lines to `/boot/firmware/config.txt`:
   `dtparam=uart0=on` and `dtoverlay=uart3-pi5`. Remove the serial console
   from `cmdline.txt`. **Check:** after a reboot, `/dev/ttyAMA0` and
   `/dev/ttyAMA3` exist.
4. Build this workspace (`colcon build`) and source it.

## PX4 (set in QGroundControl)

| Parameter | Value | Why |
|---|---|---|
| `MAV_1_CONFIG` | TELEM 2 | MAVLink to the Pi |
| `MAV_1_MODE` | Onboard | |
| `SER_TEL2_BAUD` | 921600 | Must match `fcu_url` in the launch |
| `SENS_TFMINI_CFG` | TELEM 3 (where the downward TF-Luna is) | TF-Luna uses the same serial frames as the TFmini. **Check:** `DISTANCE_SENSOR` shows up in the MAVLink Inspector |
| `EKF2_RNG_CTRL` | Conditional | PX4 uses the rangefinder near the ground |
| `EKF2_RNG_POS_X/Y/Z` | measured offset | |
| Offboard loss action (`COM_OBL_RC_ACT`) | Land | If the Pi stops sending setpoints, PX4 lands by itself |

PX4 sends `DISTANCE_SENSOR` to the Pi at a low rate by default. To get
about 50 Hz, add a stream line for the TELEM2 port to
`etc/extras.txt` on the SD card, for example
`mavlink stream -d /dev/ttyS3 -s DISTANCE_SENSOR -r 50`. **Check** which
`/dev/ttyS*` is TELEM2 on your flight controller. The landing node works
with fewer messages too, but then it gets fewer ground points from the
downward beam.

**Check:** the MAVROS topic for the downward sensor. With MAVROS's default
`px4_config.yaml` it is `/mavros/distance_sensor/rangefinder_pub`. Confirm
with `ros2 topic list`, and set `down_topic` in `params_real.yaml` if it
differs.

## Calibrating the camera

Print a chessboard (for example 9 × 7 squares of 25 mm, which gives 8 × 6
inner corners). Then run:

```bash
ros2 run camera_ros camera_node --ros-args -p width:=2304 -p height:=1296
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 \
    --ros-args -r image:=/camera/image_raw
```

Move the board over the whole picture, especially the corners, then press
Save. Copy `ost.yaml` from the archive in `/tmp` to
`~/landing_calib/ost.yaml`. Calibrate at the same resolution the launch
uses (`raw_width`, `raw_height`). If the corners stay bent with the
default model, recalibrate with the fisheye model. `camera_prep` reads both.

On start, `camera_prep` logs the widest horizontal FOV the straightened
image can have. Set `camera_hfov_deg` in `params_real.yaml` a little below
that number.

## Bench test (propellers off)

1. `ros2 launch safe_landing real.launch.py`. The landing node should log
   `WAIT_FCU -> WAIT_OFFBOARD`.
2. `ros2 topic hz /landing/range_tilt/scan` should show about 50 Hz. Move
   a hand in front of the sensor, and `ros2 topic echo` should show the
   distance change. Check `/landing/range_down/scan` the same way.
   `range_driver` logs message counts every 5 s.
3. `ros2 topic hz /landing/camera/image` should show about 5 Hz. In
   `rqt_image_view`, the top of the image must be the nose side, and
   straight edges (a door frame) must look straight up to the corners.
4. Point both sensors at the sky or far away. They must keep publishing
   (as `inf`). If the messages stop instead, the landing node treats it as
   a sensor failure.

## Range outdoors

The simulated rangefinders reach 12 m. The TF-Luna datasheet gives 8 m on
a white target, 2.5 m on a dark (10 %) one, and less in strong sunlight.
Grass reflects the TF-Luna's 850 nm light fairly well, but measure the
real range over your field in sunlight before flying. The ground check
starts at 5 m (`feel_start_agl_m`). If the tilted beam doesn't reach the
ground reliably from there, lower `feel_start_agl_m` to what it does reach.

## First flights

Always fly with a pilot ready to take over. Switching out of OFFBOARD
returns control at once.

1. Hover at 10 m and switch to OFFBOARD over open ground. Record with
   `record_camera:=true`. Watch the look and descent, and take over
   before the touchdown.
2. Land on a prepared, flat pad.
3. Land on rough ground: grass, then bushes and rocks nearby.

The camera recordings from these flights are also the training data for
plan B (YOLO).
