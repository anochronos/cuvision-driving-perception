# CARLA Simulator Pipeline

Real-time perception pipeline running on a live CARLA 0.9.15 feed. Connects to a CARLA server, attaches a camera to an ego vehicle, and runs TwinLiteNetPlus segmentation + two-stage traffic sign detection with results streamed to a browser.

## Architecture

```
CARLA Server (Windows) ──TCP──> carla_pipeline.py (WSL2)
                                  ├── TwinLiteNetPlus  (segmentation/../)
                                  ├── YOLO26n + MobileNetV3  (detection/../)
                                  └── MJPEG Web Viewer  (localhost:8080)
```

Models and weights are shared with the parent `segmentation/` and `detection/` directories, so no files are duplicated.

## Prerequisites

- **Windows**: CARLA 0.9.15 (`CarlaUE4.exe`)
- **WSL2**: conda env with CARLA client library, PyTorch, OpenCV
- Model weights in `segmentation/weights/` and `detection/weights/`

```bash
conda create -n carla python=3.10 -y
conda activate carla
pip install carla==0.9.15 torch torchvision opencv-python ultralytics
```

## Quick Start

### 1. Start CARLA on Windows

```
cd C:\path\to\CARLA_0.9.15\WindowsNoEditor
.\CarlaUE4.exe -windowed -quality-level=Epic
```

### 2. Run the pipeline from WSL

```bash
cd capstone-master/carla
conda run -n carla python carla_pipeline.py --carla-host <WINDOWS_IP>
```

Find the Windows IP with: `ip route show default | awk '{print $3}'`

### 3. Open the viewer

Navigate to http://localhost:8080 in a browser.

## Driving Modes

| Mode | Flag | Description |
|------|------|-------------|
| Autopilot | *(default)* | CARLA AI drives with NPC traffic |
| Manual | `--manual` | WASD controls from the browser |
| Attach | `--attach` | Hooks into an existing `manual_control.py` vehicle |

### Manual controls (browser)

| Key | Action |
|-----|--------|
| W / S | Throttle / Brake |
| A / D | Steer left / right |
| SPACE | Handbrake |
| T | Reset position (random spawn) |
| P | Toggle autopilot on/off |

## Weather & Map

```bash
# Weather presets: clear, cloudy, rain, sunset, night, snow, fog
python carla_pipeline.py --carla-host <IP> --weather snow

# Maps: Town01-Town07, Town10HD (default: Town03)
python carla_pipeline.py --carla-host <IP> --town Town04

# Combine
python carla_pipeline.py --carla-host <IP> --weather fog --town Town01 --manual
```

## Options

### Feature toggles

```bash
--no-lane-split       # Disable lane divider post-processing
--no-detection        # Disable traffic sign detection
```

### Model configuration

```bash
--seg-config large    # Segmentation model size: nano, small, medium, large
--seg-weight <path>   # Custom segmentation weights
--det-weight <path>   # Custom YOLO detector weights
--cls-weight <path>   # Custom MobileNetV3 classifier weights
--det-conf 0.20       # Detector confidence threshold
--cls-conf 0.50       # Classifier confidence threshold
```

### Camera & output

```bash
--res-x 1280          # Camera width (default: 1280)
--res-y 720           # Camera height (default: 720)
--web-port 8080       # Web viewer port
--save-video out.mp4  # Record output to video
```

### Lane splitting

```bash
--num-lanes 2         # Number of lanes to detect
--smooth-window 51    # Smoothing window size
--cutoff-bottom 0.08  # Bottom crop ratio (hood/dashboard)
--lane-thickness 4    # Divider line thickness
--lane-color 0 255 255  # BGR color (default: cyan)
```

## Example Commands

```bash
# Snow, manual driving, record video
python carla_pipeline.py --carla-host 172.29.240.1 \
  --manual --weather snow --save-video snow_drive.mp4

# Night, segmentation only, 1080p
python carla_pipeline.py --carla-host 172.29.240.1 \
  --weather night --no-detection --res-x 1920 --res-y 1080

# Attach to controller, fog, 3 lanes
python carla_pipeline.py --carla-host 172.29.240.1 \
  --attach --weather fog --num-lanes 3
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Connection timeout | Verify CARLA is running: `nc -z <IP> 2000` |
| WSL IP changed | `ip route show default \| awk '{print $3}'` |
| Spawn collision | Script auto-cleans on startup; restart CARLA if stuck |
| NPCs don't move | Restart CARLA server to reset traffic manager |
| Controls unresponsive | Click browser window to give it keyboard focus |
| RPC timeout after restart | Try `wsl --shutdown`, then relaunch |
