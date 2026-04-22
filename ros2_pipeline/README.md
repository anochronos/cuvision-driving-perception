# Driving Pipeline - ROS2 Jazzy

A real-time autonomous driving perception pipeline built on ROS2 Jazzy. Receives camera frames via TCP (from MATLAB/Simulink or video files), runs drivable area and lane line segmentation with TwinLiteNetPlus, detects and classifies traffic signs with a two-stage YOLO + MobileNetV3 pipeline, and fuses everything into a live MJPEG web viewer.

## Architecture

```
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│  Simulink /  │  TCP  │  tcp_camera_node │  ROS  │  tlnp_node   │
│  Video File  │──────>│  (frame recv)    │──────>│  (seg model) │
└──────────────┘       └────────┬─────────┘       └──────┬───────┘
                                │                        │
                                │ /camera/image_raw      │ /tlnp/da_mask
                                │                        │ /tlnp/ll_mask
                                v                        v
                       ┌────────────────┐        ┌──────────────┐
                       │ detection_node │        │   viz_node   │
                       │ (sign detect)  │───────>│  (fusion +   │
                       └────────────────┘        │  web viewer) │
                         /detections             └──────┬───────┘
                                                        │
                                                   http://localhost:8080
```

### ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Raw BGR camera frames (1280x720) |
| `/tlnp/da_mask` | `sensor_msgs/Image` | Drivable area binary mask (mono8) |
| `/tlnp/ll_mask` | `sensor_msgs/Image` | Lane line binary mask (mono8) |
| `/detections` | `std_msgs/String` | JSON-encoded sign detections with bounding boxes, class names, and confidence scores |

## Nodes

### 1. tcp_camera_node

Receives raw video frames over TCP (or UDP) from an external source such as MATLAB/Simulink. Frames are expected in the Simulink default format: 720x1280x3, RGB, column-major (Fortran) order.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `0.0.0.0` | Listen address |
| `port` | `12345` | Listen port |
| `use_udp` | `false` | Use UDP instead of TCP |
| `image_h` | `720` | Expected frame height |
| `image_w` | `1280` | Expected frame width |
| `publish_rate` | `30.0` | Max publish rate (Hz) |

### 2. tlnp_node

Runs TwinLiteNetPlus inference for drivable area segmentation and lane line detection. Supports three inference backends with automatic fallback: TensorRT > ONNX Runtime > PyTorch. Includes frame skipping to maintain high effective output rates when GPU inference is the bottleneck.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_path` | `trained-models/large.pth` | Path to PyTorch weights |
| `config` | `large` | Model size: `nano`, `small`, `medium`, `large` |
| `img_size` | `640` | Inference input resolution |
| `use_half` | `true` | Use FP16 inference |
| `skip_frames` | `2` | Process every Nth frame (reuse masks between) |
| `engine_path` | `""` | TensorRT engine path (auto-detected from weight_path if empty) |
| `onnx_path` | `""` | ONNX model path (auto-detected from weight_path if empty) |

**Backend selection:**

The node checks for files at startup in this order:
1. `<weight_base>.engine` - TensorRT (GPU-specific, fastest)
2. `<weight_base>.onnx` - ONNX Runtime with CUDA provider
3. `<weight_base>.pth` - PyTorch (universal fallback)

### 3. detection_node

Two-stage traffic sign detection pipeline. Stage 1 uses a YOLO26n single-class detector to find sign locations in the frame. Stage 2 crops each detection and classifies it with a MobileNetV3-Small classifier. Includes IOU-based temporal smoothing to stabilize detections across frames.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detector_path` | (see config) | Path to YOLO detector weights (.pt) |
| `classifier_path` | (see config) | Path to MobileNetV3 classifier checkpoint (.pt) |
| `det_conf` | `0.15` | Detection confidence threshold |
| `det_iou` | `0.45` | Detection NMS IOU threshold |
| `cls_conf` | `0.5` | Classification confidence threshold |
| `det_imgsz` | `960` | Detection inference image size |
| `temporal_smoothing` | `true` | Enable cross-frame label smoothing |
| `smooth_frames` | `5` | Frames required for stable label |

### 4. viz_node

Fuses the raw camera image with segmentation masks and detection bounding boxes into a composite visualization. Serves the result as an MJPEG stream over HTTP.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `web_port` | `8080` | HTTP server port for MJPEG stream |
| `jpeg_quality` | `80` | JPEG compression quality (0-100) |
| `save_video` | `false` | Save output to MP4 file |
| `video_path` | `output_ros2.mp4` | Output video path (when save_video is true) |

**Visualization layers:**
- Green overlay: drivable area (from `/tlnp/da_mask`)
- Blue pixels: lane lines (from `/tlnp/ll_mask`)
- Colored bounding boxes: traffic sign detections with class labels and confidence
- FPS counter (top-left)

## Prerequisites

### System

- Ubuntu 24.04 (tested on WSL2)
- ROS2 Jazzy (desktop install)
- NVIDIA GPU with CUDA support
- Python 3.12 (system Python, required by ROS2 Jazzy)

### Model Weights

You need the following model weight files (not included in this repo):

| Model | File | Source |
|-------|------|--------|
| TwinLiteNetPlus-Large | `trained-models/large.pth` | [TwinLiteNetPlus](https://github.com/chiraqL/TwinLiteNetPlus) |
| YOLO26n Sign Detector | `weights/epoch90.pt` | Trained on MTSD dataset |
| MobileNetV3 Classifier | `classifier_model/best_classifier.pt` | Trained on MTSD crops |

## Installation

### 1. Install ROS2 Jazzy

```bash
sudo apt install ros-jazzy-desktop
```

### 2. Create Python virtual environment

ROS2 Jazzy requires system Python 3.12. Use a venv with `--system-site-packages` to inherit ROS2's `rclpy` bindings while allowing pip installs:

```bash
python3 -m venv ~/ros2_venv --system-site-packages
```

### 3. Install Python dependencies

```bash
~/ros2_venv/bin/pip install -r requirements.txt
```

### 4. Build the ROS2 workspace

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build --packages-select driving_pipeline --symlink-install
```

### 5. (Optional) Export TensorRT engine for faster inference

```bash
# Install optional deps
~/ros2_venv/bin/pip install onnx onnxruntime-gpu

# Export ONNX (portable) and TensorRT engine (GPU-specific)
~/ros2_venv/bin/python scripts/export_tensorrt.py \
    --weight /path/to/trained-models/large.pth \
    --config large
```

This creates `large.onnx` (commit to git) and `large.engine` (gitignore - GPU-specific). The `tlnp_node` auto-detects and uses the fastest available backend.

## Usage

### Launch the full pipeline

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export PYTHONPATH=~/ros2_venv/lib/python3.12/site-packages:$PYTHONPATH

ros2 launch driving_pipeline pipeline_launch.py
```

### Send a test video over TCP

```bash
~/ros2_venv/bin/python scripts/tcp_video_sender.py /path/to/video.mp4 --fps 15 --loop
```

### Connect from MATLAB/Simulink

Configure a TCP Client Send block to connect to `<WSL_IP>:12345`. Frames should be 720x1280x3 uint8, RGB, column-major (Fortran) order - this is the default format from Simulink's Video Display / To Workspace blocks.

### View output

Open in any browser:
```
http://localhost:8080
```

### Monitor ROS2 topics

```bash
# Check topic rates
ros2 topic hz /camera/image_raw
ros2 topic hz /tlnp/da_mask
ros2 topic hz /detections

# View detection JSON
ros2 topic echo /detections
```

## Configuration

All parameters are in `config/pipeline.yaml`. Edit and rebuild (or use `--symlink-install` to pick up changes automatically).

To override individual parameters at launch:

```bash
ros2 launch driving_pipeline pipeline_launch.py \
    tlnp_node:=__params:='{"skip_frames": 3, "img_size": 480}'
```

## Performance

Measured on NVIDIA RTX 4080 SUPER, Ubuntu 24.04 WSL2, 1280x720 input:

| Node | Backend | Rate | Notes |
|------|---------|------|-------|
| tcp_camera_node | TCP | 15 FPS | Limited by sender rate |
| tlnp_node | PyTorch FP16 | ~2.5 FPS | With skip_frames=2: effective 5 FPS output |
| tlnp_node | TensorRT FP16 | ~30+ FPS | After running export_tensorrt.py |
| detection_node | YOLO + MobileNetV3 | ~15 FPS | Temporal smoothing enabled |
| viz_node | CPU compositing | 30 FPS | MJPEG encoding + HTTP serving |

### Performance tuning

| Setting | Effect | Trade-off |
|---------|--------|-----------|
| `skip_frames: 3` | 3x fewer inferences | Masks update less frequently |
| `img_size: 480` | ~2x faster inference | Slight accuracy reduction |
| TensorRT export | 3-5x faster inference | GPU-specific engine, build time |
| `det_imgsz: 640` | Faster detection | May miss small/distant signs |
| `jpeg_quality: 60` | Less bandwidth | Lower visual quality in browser |

## Project Structure

```
driving_pipeline/
├── config/
│   └── pipeline.yaml           # All node parameters
├── driving_pipeline/
│   ├── __init__.py
│   ├── tcp_camera_node.py      # TCP/UDP frame receiver
│   ├── tlnp_node.py            # TwinLiteNetPlus segmentation (TRT/ONNX/PyTorch)
│   ├── detection_node.py       # Two-stage sign detection
│   └── viz_node.py             # Fusion + MJPEG web viewer
├── launch/
│   └── pipeline_launch.py      # Launch all 4 nodes
├── scripts/
│   ├── export_tensorrt.py      # ONNX/TensorRT export + benchmark
│   └── tcp_video_sender.py     # Send video files over TCP for testing
├── resource/
│   └── driving_pipeline
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
└── README.md
```

## Detection Output Format

The `/detections` topic publishes JSON strings with this schema:

```json
{
  "stamp": 1775246143.413,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": "speed_limit_50",
      "det_conf": 0.87,
      "cls_conf": 0.93
    }
  ]
}
```

## Troubleshooting

**"Address already in use" on port 12345 or 8080:**
```bash
fuser -k 12345/tcp
fuser -k 8080/tcp
```

**"No module named rclpy":**
Make sure ROS2 is sourced before activating the venv:
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
```

**TLNP node crashes with "No module named tqdm":**
```bash
~/ros2_venv/bin/pip install tqdm
```

**TensorRT engine fails to load:**
The `.engine` file is GPU-specific. Rebuild it on the target machine:
```bash
~/ros2_venv/bin/python scripts/export_tensorrt.py
```

**Detection node can't find weights:**
Update the paths in `config/pipeline.yaml` to match your local weight file locations.
