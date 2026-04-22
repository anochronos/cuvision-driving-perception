# CUVision - Autonomous Driving Perception Pipeline

A complete real-time autonomous driving perception system combining road segmentation, lane detection, and traffic sign recognition. Built for deployment via ROS2 Jazzy with MATLAB/Simulink integration.

## System Overview

```
Camera (Simulink/TCP or CARLA) ──> Road Segmentation (TwinLiteNetPlus)
                               ├──> Lane Line Detection (TwinLiteNetPlus)
                               └──> Traffic Sign Detection (YOLO26n + MobileNetV3)
                                        │
                                        v
                               ROS2 Fusion / MJPEG Web Viewer
```

### Components

| Component | Model | Params | Speed | Purpose |
|-----------|-------|--------|-------|---------|
| **Segmentation** | TwinLiteNetPlus-Large | 1.9M | ~150 FPS (TRT) | Drivable area + lane lines |
| **Sign Detector** | YOLO26n (single-class) | 2.6M | ~60 FPS | Locates traffic signs |
| **Sign Classifier** | MobileNetV3-Small | 2.5M | ~200 FPS | Identifies 252 sign types |
| **ROS2 Pipeline** | 4-node architecture | - | 15+ FPS end-to-end | Real-time fusion + web viewer |
| **CARLA Pipeline** | All of the above | - | 15+ FPS live | Simulator integration + browser controls |

## Directory Structure

```
capstone-master/
├── segmentation/               # TwinLiteNetPlus road segmentation
│   ├── model/                  # Network architecture
│   ├── hyperparameters/        # Training configs (base, snow finetune)
│   ├── weights/                # Trained model weights
│   ├── src/                    # Post-processing modules
│   │   ├── lane_splitter.py    # DA mask → lane divider post-processing
│   │   └── road_info.py        # OSM road metadata for lane-aware splitting
│   ├── train.py                # Training script
│   ├── BDD100K.py              # Dataset loader (with weather filtering)
│   ├── utils.py                # Validation, metrics, utilities
│   ├── demo.py                 # Image/video inference + lane splitting
│   ├── demoDataset.py          # Inference preprocessing
│   └── README.md
│
├── detection/                  # Two-stage traffic sign detection
│   ├── weights/                # YOLO detector weights
│   ├── classifier_model/       # MobileNetV3 classifier + class names
│   ├── train_single_class_detector.py
│   ├── train_classifier.py
│   ├── two_stage_pipeline.py   # Combined inference pipeline
│   └── README.md
│
├── ros2_pipeline/              # ROS2 Jazzy deployment package
│   ├── driving_pipeline/       # ROS2 node source code
│   ├── config/                 # Node parameters (YAML)
│   ├── launch/                 # Launch files
│   ├── scripts/                # Export tools, test utilities
│   └── README.md
│
├── carla/                      # CARLA 0.9.15 simulator integration
│   ├── carla_pipeline.py       # Main script (connect, drive, infer, view)
│   └── README.md
│
├── ros2_original/              # Original ROS2 implementation (by Omar)
│   ├── controller/             # ROS2 node source code
│   │   ├── tcp_receiver_node.py
│   │   ├── lane_detection_node.py
│   │   ├── object_detection_node.py
│   │   ├── visualization_node.py
│   │   └── video_publisher.py
│   └── README.md
│
├── scripts/                    # Benchmarking and evaluation
│   ├── benchmark_speed.py      # TLNP vs YOLOPX speed comparison
│   ├── fair_accuracy_benchmark.py  # Fair accuracy evaluation
│   └── eval_snow_subset.py     # Snow weather subset evaluation
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Install dependencies

```bash
# Create conda environment (recommended for training)
conda create -n driving python=3.10 -y
conda activate driving

pip install -r requirements.txt
```

### 2. Train models (or use provided weights)

See component READMEs for details:
- [Segmentation](segmentation/README.md)
- [Detection](detection/README.md)
- [CARLA Integration](carla/README.md)
- [ROS2 Deployment](ros2_pipeline/README.md)
- [Original ROS2 Pipeline](ros2_original/README.md)

### 3. Run inference

```bash
# Segmentation on video
cd segmentation
python demo.py --source /path/to/video.mp4 --weight weights/large.pth --config large

# Two-stage sign detection on video
cd detection
python two_stage_pipeline.py
```

### 4. Run with CARLA simulator

See [CARLA Pipeline README](carla/README.md) for full setup.

```bash
cd carla
conda run -n carla python carla_pipeline.py --carla-host <WINDOWS_IP>
# Open http://localhost:8080
```

### 5. Deploy with ROS2

See [ROS2 Pipeline README](ros2_pipeline/README.md) for full setup.

```bash
# Quick launch (after ROS2 setup)
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch driving_pipeline pipeline_launch.py
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 (6GB) | RTX 4080+ |
| VRAM | 4 GB | 8+ GB |
| RAM | 8 GB | 16+ GB |
| Storage | 10 GB (models + code) | 50+ GB (with BDD100K dataset) |

## Dataset

All models are trained on [BDD100K](https://www.bdd100k.com/):
- 100K driving videos, 10 cities
- Drivable area and lane line annotations
- Weather/scene labels (used for snow subset filtering)

Traffic sign detection uses [MTSD](https://www.mapillary.com/dataset/trafficsign) (Mapillary Traffic Sign Dataset):
- 250+ North American sign classes
- Converted to single-class for Stage 1 detector
- PNG templates for Stage 2 classifier training

## Benchmark Results

### Speed (RTX 4080 SUPER, FP16, 384x640)

| Model | Params | Mean (ms) | FPS |
|-------|--------|-----------|-----|
| TwinLiteNetPlus-Large | 1.9M | 3.37 | 297 |
| YOLOPX | 33.1M | 7.21 | 139 |

TwinLiteNetPlus is **2.14x faster** with **17x fewer parameters**.

### Accuracy (BDD100K val, 10K images)

| Metric | TLNP-Large | YOLOPX |
|--------|------------|--------|
| DA mIOU | 0.9291 | 0.9316 |
| DA Accuracy | 0.9752 | 0.9762 |
| LL IOU | 0.3420 | 0.2716 |
| LL Accuracy | 0.8223 | 0.7968 |

TLNP matches YOLOPX on drivable area and **beats it on lane line detection** (+26% IOU).

## Acknowledgments

- [TwinLiteNetPlus](https://github.com/chiraqL/TwinLiteNetPlus) - base segmentation model architecture and training code by chiraqL
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8/YOLO26n framework used for the sign detector
- [BDD100K](https://www.bdd100k.com/) - driving dataset for segmentation training and evaluation (Yu et al., 2020)
- [MTSD](https://www.mapillary.com/dataset/trafficsign) - Mapillary Traffic Sign Dataset for detector and classifier training (Ertler et al., 2020)
- [YOLOPX](https://github.com/CAIC-AD/YOLOPv2) - used as accuracy and speed baseline for benchmarking
- [CARLA](https://carla.org/) - open-source autonomous driving simulator (Dosovitskiy et al., 2017)
- [ROS2 Jazzy](https://docs.ros.org/en/jazzy/) - middleware for the deployment pipeline
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/) - deep learning framework
