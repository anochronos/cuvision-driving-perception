# Two-Stage Traffic Sign Detection

A two-stage pipeline for real-time traffic sign detection and classification:
- **Stage 1**: YOLO26n single-class detector (finds sign locations)
- **Stage 2**: MobileNetV3-Small classifier (identifies the sign type from 252 classes)

This design separates localization from classification, allowing each stage to be trained and optimized independently.

## Architecture

```
Input Frame
    │
    v
┌─────────────────────┐
│  Stage 1: YOLO26n   │   Single-class detection: "is there a sign here?"
│  (2.6M params)      │   Outputs: bounding boxes + confidence
└─────────┬───────────┘
          │ crop each detection
          v
┌─────────────────────┐
│  Stage 2: MobileNet │   Multi-class classification: "what sign is this?"
│  V3-Small (2.5M)    │   252 classes (250 signs + traffic_light + not_a_sign)
└─────────┬───────────┘
          │
          v
   Detections with class labels, bounding boxes, and confidence scores
```

### Why two stages?

1. **Single-class detection is more robust** - YOLO only needs to learn "sign vs not-sign", which generalizes much better than 250-class detection with limited per-class training data.
2. **Classifier trains on synthetic data** - MobileNetV3 trains on transparent PNG sign templates composited onto random backgrounds, so it doesn't need real-world labeled bounding boxes for every sign type.
3. **Each stage can be updated independently** - improve the detector without retraining the classifier, and vice versa.

## Files

| File | Description |
|------|-------------|
| `two_stage_pipeline.py` | Combined inference pipeline with temporal smoothing |
| `train_single_class_detector.py` | Train YOLO26n single-class sign detector |
| `train_classifier.py` | Train MobileNetV3-Small sign classifier |
| `weights/detector_yolo26n.pt` | Trained YOLO26n detector weights |
| `classifier_model/best_classifier.pt` | Trained MobileNetV3 classifier checkpoint |
| `classifier_model/class_names.json` | 252 class names (250 signs + traffic_light + not_a_sign) |

## Training

### Stage 1: YOLO26n Single-Class Detector

**Dataset**: MTSD (Mapillary Traffic Sign Dataset) converted to single-class YOLO format.

1. Prepare your MTSD data in YOLO format:
   ```
   data/output/
   ├── train/
   │   ├── images/    # .jpg files
   │   └── labels/    # .txt files (YOLO format)
   └── val/
       ├── images/
       └── labels/
   ```

2. Train:
   ```bash
   python train_single_class_detector.py
   ```

   This will:
   - Convert multi-class labels to single-class (all signs become class 0)
   - Create `data_single_class.yaml`
   - Train YOLOv8n for 100 epochs with sign-specific augmentation
   - Save best model to `data/single_class_detector/weights/best.pt`

**Training settings:**
| Setting | Value | Reason |
|---------|-------|--------|
| Model | YOLOv8n (pretrained) | Lightweight, fast |
| Epochs | 100 | Patience=30 for early stopping |
| Image size | 640 | Standard YOLO input |
| Batch size | 32 | Fits in 8GB VRAM |
| LR | 0.01 | Default YOLO LR |
| Augmentation | mosaic=0.5, scale=0.4, rotate=5 | Sign-specific; no horizontal flip |

### Stage 2: MobileNetV3-Small Classifier

**Dataset**: Transparent PNG sign templates composited onto road background crops.

1. Prepare sign template images:
   ```
   data/transparent_png_signs/    # One .png per sign class (e.g., Stop.png)
   data/road_backgrounds/         # Random road crops for compositing
   data/traffic_light_crops/      # (Optional) real traffic light crops
   ```

2. Train:
   ```bash
   python train_classifier.py
   ```

   This will:
   - Generate synthetic training data by compositing signs onto backgrounds
   - Add augmentation (scale, rotation, downscale-upscale, color jitter, blur)
   - Include negative samples (random background crops labeled "not_a_sign")
   - Train MobileNetV3-Small with ImageNet pretrained weights
   - Save best model to `classifier_model/best_classifier.pt`

**Training settings:**
| Setting | Value | Reason |
|---------|-------|--------|
| Model | MobileNetV3-Small (ImageNet pretrained) | Lightweight, accurate |
| Epochs | 30 | Synthetic data converges fast |
| Batch size | 64 | Small images (96x96) |
| LR | 0.001 | AdamW + cosine annealing |
| Image size | 96x96 | Matches typical crop sizes |
| Samples/class | 200 | Synthetic augmentation handles diversity |
| Negative multiplier | 3x | Reduces false positives |

**Synthetic data augmentation:**
- Random scale (0.5-0.95x)
- Random rotation (-15 to +15 degrees)
- Random background compositing (70% real road, 30% solid color)
- Downscale-upscale simulation (8-32px, mimics distant signs)
- Color jitter, Gaussian blur
- Random center offset

## Inference

### Process a video

```bash
python two_stage_pipeline.py
```

Edit the `main()` function to point to your video files and weight paths.

### Use as a library

```python
from two_stage_pipeline import TwoStagePipeline

pipeline = TwoStagePipeline(
    detector_path='weights/detector_yolo26n.pt',
    classifier_path='classifier_model/best_classifier.pt',
    det_conf=0.15,        # Low threshold - let classifier decide
    cls_conf=0.5,         # Classification confidence threshold
    det_imgsz=960,        # Larger = better for small/distant signs
    temporal_smoothing=True,
    smooth_frames=5,      # Frames needed for stable label
)

# Process single frame
detections = pipeline.process_frame(frame_bgr)
# Returns: [(x1, y1, x2, y2, class_name, det_conf, cls_conf), ...]

# Draw results
annotated = pipeline.draw_detections(frame_bgr, detections)
```

### Temporal smoothing

The pipeline tracks detections across frames using IOU matching:
- New detections are tracked by matching bounding boxes across consecutive frames
- Classification labels are smoothed over a sliding window (`smooth_frames`)
- The most frequent label in the window is used (majority voting)
- Stale tracks (not seen for 10 frames) are cleaned up automatically

This eliminates flickering labels and reduces single-frame false positives.

## Weights

| File | Size | Description |
|------|------|-------------|
| `weights/detector_yolo26n.pt` | 15 MB | YOLO26n single-class sign detector (epoch 90) |
| `classifier_model/best_classifier.pt` | 7 MB | MobileNetV3-Small 252-class classifier |
| `classifier_model/class_names.json` | 5 KB | Class name list (250 signs + traffic_light + not_a_sign) |

The classifier checkpoint contains the model state dict, class names, number of classes, and image size - everything needed to reconstruct the model at inference time.
