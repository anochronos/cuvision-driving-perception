# TwinLiteNetPlus - Road Segmentation

Lightweight multi-task model for simultaneous drivable area segmentation and lane line detection. Based on [TwinLiteNetPlus](https://github.com/chiraqL/TwinLiteNetPlus) with added snow augmentation and weather-filtered training.

## Model Architecture

TwinLiteNetPlus uses a shared encoder with two parallel decoder heads:
- **Drivable Area (DA)**: 2-class segmentation (drivable vs background)
- **Lane Lines (LL)**: 2-class segmentation (lane vs background)

| Config | Parameters | FPS (RTX 4080) | DA mIOU | LL IOU |
|--------|-----------|-----------------|---------|--------|
| nano | 37K | 500+ | 0.886 | 0.268 |
| small | 200K | 450+ | 0.910 | 0.302 |
| medium | 800K | 380+ | 0.921 | 0.325 |
| **large** | **1.9M** | **297** | **0.929** | **0.342** |

## Files

| File | Description |
|------|-------------|
| `model/model.py` | Network architecture (encoder + dual decoder heads) |
| `model/config.py` | Channel configs for nano/small/medium/large variants |
| `train.py` | Training script with EMA, poly LR scheduler, checkpoint saving |
| `BDD100K.py` | Dataset class with snow augmentation and weather filtering |
| `utils.py` | Validation function, SegmentationMetric, loss functions |
| `demo.py` | Image and video inference with lane splitting visualization |
| `demoDataset.py` | Inference preprocessing (letterbox, transforms) |
| `src/lane_splitter.py` | Lane splitting post-processing (DA mask to lane dividers) |
| `src/road_info.py` | OpenStreetMap road metadata queries for lane-aware splitting |

## Training

### Prerequisites

1. Download BDD100K dataset:
   - Images: `bdd100k/images/100k/{train,val}/`
   - Drivable area masks: `bdd100k/images/bdd_seg_gt/{train,val}/`
   - Lane line masks: `bdd100k/images/bdd_lane_gt/{train,val}/`
   - Labels (for weather filtering): `bdd100k/labels/100k/{train,val}/`

2. Update the `bdd_root` path in the hyperparameter YAML to point to your BDD100K location.

### Train from scratch

```bash
python train.py \
    --hyp hyperparameters/twinlitev2_hyper.yaml \
    --config large \
    --num_epochs 100 \
    --savedir trained-models \
    --onGPU
```

### Fine-tune on snow data

Two fine-tuning configs are provided:

**Full dataset with snow augmentation** (`finetune_snow.yaml`):
- Trains on all BDD100K images with 30% probability of snow augmentation
- Start from pretrained `large.pth`

```bash
python train.py \
    --hyp hyperparameters/finetune_snow.yaml \
    --config large \
    --num_epochs 30 \
    --pretrained weights/large.pth \
    --savedir finetune_snow_large \
    --onGPU
```

**Snow-only subset** (`finetune_snow_only.yaml`):
- Trains only on images labeled as snowy weather (via `weather_filter` param)
- 60% snow augmentation probability, higher intensity
- Faster iteration, targeted improvement

```bash
python train.py \
    --hyp hyperparameters/finetune_snow_only.yaml \
    --config large \
    --num_epochs 30 \
    --pretrained weights/large.pth \
    --savedir finetune_snow_only \
    --onGPU
```

### Hyperparameter files

| File | Description |
|------|-------------|
| `twinlitev2_hyper.yaml` | Base training config (LR=0.001, no augmentation) |
| `finetune_snow.yaml` | Snow fine-tune (LR=0.0001, prob_snow=0.3, intensity=0.6) |
| `finetune_snow_only.yaml` | Snow-only subset (LR=0.0001, prob_snow=0.6, weather_filter=snowy) |

### Snow augmentation

The `BDD100K.py` dataset class includes a `road_snow_augment()` function that:
- Adds white/gray noise patches to simulate snow on the road surface
- Applies Gaussian blur for realistic coverage
- Controlled by `prob_snow` (probability per image) and `snow_intensity` (severity)

### Weather filtering

To train on specific weather conditions only, add to the hyperparameter YAML:

```yaml
weather_filter: "snowy"        # Single weather
# weather_filter: ["snowy", "rainy"]  # Multiple weathers
```

This reads BDD100K's JSON label files and filters images by the `attributes.weather` field.

## Inference

### Single image or video

```bash
python demo.py \
    --source /path/to/input.mp4 \
    --weight weights/large.pth \
    --config large \
    --save-dir output/
```

### Lane splitting (post-processing)

The demo includes a geometric post-processing pipeline that derives lane divider lines from the drivable area mask - no model retraining required. The pipeline in `src/lane_splitter.py` works as follows:

```
DA Mask → Morphological Cleanup → Bottom Cutoff → DA Accumulator (EMA)
       → Road Boundary Extraction → Turn Clamping
       → Divider Computation → Spatial Smoothing → Temporal EMA
       → Polyline Generation
```

| Component | Function | Purpose |
|-----------|----------|---------|
| `clean_da_mask()` | Morphological close + open (7px kernel) | Remove mask noise |
| `DAMaskAccumulator` | Rolling EMA (alpha=0.25, threshold=0.4) | Stabilize boundaries across frames |
| `extract_road_boundaries()` | Per-row left/right pixel scan | Find road edges |
| `estimate_lane_width_per_row()` | Perspective scaling from vanishing point | Estimate single-lane width |
| `clamp_boundaries_for_turns()` | Width-based boundary clamping | Prevent turn distortion |
| `compute_lane_dividers()` | Equal-width splitting with width gate | Place N-1 dividers |
| `smooth_divider()` | Moving average (51–101px window) | Reduce row-to-row jitter |
| `TemporalSmoother` | Per-pixel EMA (alpha=0.3) across frames | Reduce frame-to-frame flicker |

**Usage with lane splitting:**
```bash
python demo.py \
    --source /path/to/input.mp4 \
    --weight weights/large.pth \
    --config large \
    --num-lanes 2 \
    --smooth-window 51 \
    --save-dir output/
```

**With OSM road info** (uses real road width from OpenStreetMap):
```bash
python demo.py \
    --source /path/to/input.mp4 \
    --weight weights/large.pth \
    --config large \
    --num-lanes 2 \
    --gps-lat 43.65 --gps-lon -79.38 \
    --save-dir output/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-lanes` | 2 | Number of ego-direction lanes |
| `--smooth-window` | 51 | Spatial smoothing kernel size |
| `--cutoff-bottom` | 0.08 | Fraction of image bottom to ignore (hood) |
| `--lane-ratio` | 1.4 | Min road/lane width ratio to allow splitting |
| `--lane-thickness` | 4 | Divider line thickness in pixels |
| `--lane-color` | 0 255 255 | BGR color for lane dividers (cyan) |
| `--gps-lat` | None | GPS latitude for OSM road query |
| `--gps-lon` | None | GPS longitude for OSM road query |

### Evaluation

```bash
# Evaluate on full BDD100K val set (uses native val() from utils.py)
python -c "
from train import *
# loads model, runs val(), prints DA/LL metrics
"
```

## Weights

| File | Description | Size |
|------|-------------|------|
| `weights/large.pth` | Pretrained TwinLiteNetPlus-Large | 7.6 MB |

The weight file contains a `state_dict` only (no optimizer state). Load with:

```python
from model.model import TwinLiteNetPlus
from argparse import Namespace

args = Namespace(config='large', onGPU=True)
model = TwinLiteNetPlus(args).cuda().eval()
model.load_state_dict(torch.load('weights/large.pth', map_location='cuda'))
```
