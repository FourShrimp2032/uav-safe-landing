# Simulation environment: Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic +
# PX4 SITL + MAVROS, with a browser desktop (noVNC) so it runs on a Mac.
#
# Built by docker compose (see README). The first build downloads several
# GB and compiles PX4, so expect 30-60 minutes.
FROM ubuntu:24.04

ARG PX4_VERSION=v1.16.0
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Etc/UTC \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PX4_DIR=/opt/PX4-Autopilot

# Base tools + virtual desktop reachable from the browser
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release software-properties-common sudo \
        git wget nano tmux python3-pip python3-venv locales tzdata \
        xvfb x11vnc fluxbox novnc websockify xterm \
        mesa-utils libgl1-mesa-dri libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Jazzy, ros_gz (Gazebo Harmonic bridge) and MAVROS
RUN add-apt-repository -y universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
         > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-mavros ros-jazzy-mavros-extras \
        ros-jazzy-cv-bridge ros-jazzy-sensor-msgs-py \
        python3-colcon-common-extensions python3-rosdep \
        python3-opencv python3-yaml python3-pytest python3-numpy python3-pil \
    && rm -rf /var/lib/apt/lists/*

# MAVROS needs these geoid files to start
RUN /opt/ros/jazzy/lib/mavros/install_geographiclib_datasets.sh

# PX4 SITL. Its setup script also installs Gazebo Harmonic.
RUN git clone --branch ${PX4_VERSION} --depth 1 --recursive --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git ${PX4_DIR} \
    && bash ${PX4_DIR}/Tools/setup/ubuntu.sh --no-nuttx \
    && rm -rf /var/lib/apt/lists/*
RUN cd ${PX4_DIR} && make px4_sitl

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /root/.bashrc \
    && echo "cd /workspace/uav-safe-landing" >> /root/.bashrc

ENV DISPLAY=:1 LIBGL_ALWAYS_SOFTWARE=1
EXPOSE 6080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
