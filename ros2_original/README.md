# Original ROS2 Pipeline

Initial ROS2 implementation of the perception pipeline, built by Omar. This was the first working version used during early development.

We ran into issues getting this version to run reliably for the capstone fair demo (cv2.imshow display problems in WSL2, lack of web-based viewer for the demo booth, and missing TensorRT support for real-time performance), so a newer implementation was built in `ros2_pipeline/` to address those gaps. This version is kept for reference.

## Architecture

```
Simulink (TCP) or Video File
        |
        v
tcp_receiver_node  /  video_publisher
        |
        | /camera/image_raw
        |
   +----+----+
   |         |
   v         v
lane_detection_node    object_detection_node
   |                   |
   | /lane_output      | /object_output
   |                   |
   +----+----+---------+
        |
        v
visualization_node  (cv2.imshow)
```

## Differences from ros2_pipeline/

| Feature | ros2_pipeline (newer) | ros2_original (this) |
|---------|----------------------|----------------------|
| ROS2 version | Jazzy | Humble |
| Image conversion | Raw bytes | cv_bridge |
| Visualization | MJPEG web viewer (:8080) | cv2.imshow (local window) |
| Seg preprocessing | Letterbox + FP16 | Direct resize to 640x384 |
| TensorRT/ONNX support | Yes (auto-fallback) | No (PyTorch only) |
| Lane splitting | Yes (post-processing) | No |
| Output format | JSON on /detections topic | Annotated Image on /object_output |

## Nodes

| Node | Subscribes | Publishes | Description |
|------|-----------|-----------|-------------|
| `tcp_receiver_node` | TCP socket | `/camera/image_raw` | Receives Simulink frames |
| `video_publisher` | Video file | `/camera/image_raw` | Publishes video for testing |
| `lane_detection_node` | `/camera/image_raw` | `/lane_output` | TwinLiteNetPlus segmentation |
| `object_detection_node` | `/camera/image_raw` | `/object_output` | YOLO26n + MobileNetV3 signs |
| `visualization_node` | `/lane_output`, `/object_output` | cv2.imshow | Fuses both outputs |

## Usage

### Build

```bash
source /opt/ros/humble/setup.bash   # or jazzy
cd ~/ros2_ws
ln -s /path/to/capstone-master/ros2_original src/controller
colcon build --packages-select controller --symlink-install
source install/setup.bash
```

### Run all nodes

```bash
# Terminal 1: TCP receiver (or video publisher for testing)
ros2 run controller tcp_receiver_node

# Terminal 2: Lane detection
ros2 run controller lane_detection_node

# Terminal 3: Object detection
ros2 run controller object_detection_node

# Terminal 4: Visualization
ros2 run controller visualization_node
```

### Test with a video file

```bash
ros2 run controller video_publisher --ros-args -p video_path:=/path/to/video.mp4 -p loop:=true
```

## Requirements

- ROS2 (Humble or Jazzy)
- `cv_bridge` (`sudo apt install ros-${ROS_DISTRO}-cv-bridge`)
- PyTorch, OpenCV, Ultralytics
- Model weights in `segmentation/weights/` and `detection/weights/` (shared with parent repo)
