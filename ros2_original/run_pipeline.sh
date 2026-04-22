#!/bin/bash
# Launches all ROS2 nodes in separate terminals.
# Usage:  bash run_pipeline.sh [--video /path/to/video.mp4]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
SETUP="source /opt/ros/${ROS_DISTRO}/setup.bash && source ${ROS_WS}/install/setup.bash"

echo "================================================"
echo " Starting Perception Pipeline"
echo " ROS2: ${ROS_DISTRO} | Workspace: ${ROS_WS}"
echo "================================================"

# Build
echo "[1/5] Building workspace..."
cd "$ROS_WS" && source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --packages-select controller --symlink-install
if [ $? -ne 0 ]; then
    echo "[ERROR] colcon build failed."
    exit 1
fi
echo "[1/5] Build complete."

# Terminal 1 - Camera input (TCP or video file)
if [ "$1" = "--video" ] && [ -n "$2" ]; then
    echo "[2/5] Launching video publisher ($2)..."
    gnome-terminal --title="Video Publisher" -- bash -c \
        "${SETUP} && ros2 run controller video_publisher --ros-args -p video_path:=$2 -p loop:=true; exec bash"
else
    echo "[2/5] Launching TCP receiver node..."
    gnome-terminal --title="TCP Receiver" -- bash -c \
        "${SETUP} && ros2 run controller tcp_receiver_node; exec bash"
fi
sleep 2

# Terminal 2 - Lane Detection
echo "[3/5] Launching lane detection node..."
gnome-terminal --title="Lane Detection" -- bash -c \
    "${SETUP} && ros2 run controller lane_detection_node; exec bash"
sleep 2

# Terminal 3 - Object Detection
echo "[4/5] Launching object detection node..."
gnome-terminal --title="Object Detection" -- bash -c \
    "${SETUP} && ros2 run controller object_detection_node; exec bash"
sleep 2

# Terminal 4 - Visualization
echo "[5/5] Launching visualization node..."
gnome-terminal --title="Visualization" -- bash -c \
    "${SETUP} && ros2 run controller visualization_node; exec bash"

echo ""
echo "================================================"
echo " All nodes launched."
echo " Ctrl+C in each terminal to stop."
echo "================================================"
