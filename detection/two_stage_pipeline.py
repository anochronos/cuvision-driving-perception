"""
Two-Stage Traffic Sign Detection Pipeline
Stage 1: YOLOv8n single-class detector (finds signs)
Stage 2: MobileNetV3-Small classifier (identifies sign type)
Designed to run alongside TwinLiteNet+ in real-time
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image


class SignClassifier:
    """Lightweight MobileNetV3-Small sign classifier"""

    def __init__(self, model_path, device='cuda'):
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        self.img_size = checkpoint['img_size']

        # Build model
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, self.num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, crop_bgr):
        """
        Classify a cropped sign image

        Args:
            crop_bgr: BGR image (numpy array) of cropped sign

        Returns:
            (class_name, confidence)
        """
        # Convert BGR to RGB PIL image
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # Transform and predict
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)

        return self.class_names[pred.item()], conf.item()

    def classify_batch(self, crops_bgr):
        """
        Classify multiple cropped sign images at once

        Args:
            crops_bgr: List of BGR images (numpy arrays)

        Returns:
            List of (class_name, confidence)
        """
        if not crops_bgr:
            return []

        tensors = []
        for crop in crops_bgr:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            tensors.append(self.transform(pil_img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)

        results = []
        for i in range(len(crops_bgr)):
            results.append((self.class_names[preds[i].item()], confs[i].item()))

        return results


class TwoStagePipeline:
    """
    Two-stage traffic sign detection pipeline

    Stage 1: YOLOv8n detects sign locations
    Stage 2: MobileNetV3 classifies each detected sign
    """

    def __init__(self, detector_path, classifier_path,
                 det_conf=0.25, det_iou=0.45, cls_conf=0.5,
                 det_imgsz=640, device='cuda',
                 temporal_smoothing=True, smooth_frames=3,
                 use_tiling=False, tile_overlap=0.25):
        """
        Args:
            detector_path: Path to single-class detector
            classifier_path: Path to MobileNetV3 classifier
            det_conf: Detection confidence threshold
            det_iou: Detection IOU threshold for NMS
            cls_conf: Classification confidence threshold
            det_imgsz: Detection inference image size
            device: 'cuda' or 'cpu'
            temporal_smoothing: Enable temporal smoothing to reduce flicker
            smooth_frames: Number of consecutive frames required for stable detection
            use_tiling: Split frame into tiles for better small object detection
            tile_overlap: Overlap ratio between tiles (0.0-0.5)
        """
        print("Loading two-stage pipeline...")

        # Stage 1: Detector
        self.detector = YOLO(detector_path)
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.det_imgsz = det_imgsz

        # Stage 2: Classifier
        self.classifier = SignClassifier(classifier_path, device)
        self.cls_conf = cls_conf

        # Tiling
        self.use_tiling = use_tiling
        self.tile_overlap = tile_overlap

        # Temporal smoothing
        self.temporal_smoothing = temporal_smoothing
        self.smooth_frames = smooth_frames
        self.track_history = {}  # track_id -> {last_box, labels, last_seen}
        self.next_track_id = 0
        self.frame_count = 0

        print(f"  Detector: {detector_path}")
        print(f"  Classifier: {classifier_path} ({self.classifier.num_classes} classes)")
        print(f"  Det conf: {det_conf}, Cls conf: {cls_conf}")
        print(f"  Tiling: {use_tiling} (overlap={tile_overlap})")
        print(f"  Temporal smoothing: {temporal_smoothing} ({smooth_frames} frames)")
        print("Pipeline ready!")

    def _detect_with_tiling(self, frame):
        """
        Run detection on overlapping tiles for better small object detection.
        Splits frame into a 2x2 grid with overlap, runs detector on each tile,
        then merges results back to full frame coordinates.
        """
        h, w = frame.shape[:2]
        overlap_px_x = int(w * self.tile_overlap)
        overlap_px_y = int(h * self.tile_overlap)

        tile_w = w // 2 + overlap_px_x
        tile_h = h // 2 + overlap_px_y

        # Define tile origins (x, y)
        tiles = [
            (0, 0),                              # top-left
            (w - tile_w, 0),                     # top-right
            (0, h - tile_h),                     # bottom-left
            (w - tile_w, h - tile_h),            # bottom-right
        ]

        all_boxes = []

        for (tx, ty) in tiles:
            tile = frame[ty:ty+tile_h, tx:tx+tile_w]

            results = self.detector(tile, conf=self.det_conf, iou=self.det_iou,
                                    imgsz=self.det_imgsz, verbose=False)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                det_conf = box.conf[0].item()

                # Map back to full frame coordinates
                x1 += tx
                y1 += ty
                x2 += tx
                y2 += ty

                all_boxes.append((x1, y1, x2, y2, det_conf))

        # Also run on the full frame to catch large signs
        results = self.detector(frame, conf=self.det_conf, iou=self.det_iou,
                                imgsz=self.det_imgsz, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = box.conf[0].item()
            all_boxes.append((x1, y1, x2, y2, det_conf))

        # NMS to remove duplicate detections from overlapping tiles
        if not all_boxes:
            return []

        all_boxes = self._nms_boxes(all_boxes, iou_threshold=0.5)
        return all_boxes

    def _nms_boxes(self, boxes, iou_threshold=0.5):
        """Simple NMS on raw boxes (x1,y1,x2,y2,conf)"""
        if not boxes:
            return []

        # Sort by confidence descending
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []

        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes
                     if self._iou(best[:4], b[:4]) < iou_threshold]

        return keep

    def _iou(self, box1, box2):
        """Calculate IOU between two boxes (x1,y1,x2,y2)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return intersection / (area1 + area2 - intersection)

    def _get_stable_label(self, box, class_name, cls_conf):
        """
        Track detections across frames and return stable label.
        Uses IOU matching to associate detections across frames.
        Returns the most frequent classification over recent frames.
        """
        if not self.temporal_smoothing:
            return class_name, cls_conf

        # Find matching track by IOU
        best_track_id = None
        best_iou = 0.3  # Minimum IOU to match

        for track_id, track in self.track_history.items():
            iou = self._iou(box, track['last_box'])
            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            # New track
            best_track_id = self.next_track_id
            self.next_track_id += 1
            self.track_history[best_track_id] = {
                'last_box': box,
                'labels': [],
                'last_seen': self.frame_count,
            }

        track = self.track_history[best_track_id]
        track['last_box'] = box
        track['last_seen'] = self.frame_count
        track['labels'].append((class_name, cls_conf))

        # Keep only recent frames
        track['labels'] = track['labels'][-self.smooth_frames * 2:]

        # Need minimum frames before showing label
        if len(track['labels']) < self.smooth_frames:
            return class_name, cls_conf  # Show raw until we have enough history

        # Get most common label in recent window
        recent = track['labels'][-self.smooth_frames:]
        label_counts = {}
        label_confs = {}
        for lbl, conf in recent:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            label_confs[lbl] = max(label_confs.get(lbl, 0), conf)

        # Return the most frequent label
        stable_label = max(label_counts, key=label_counts.get)
        return stable_label, label_confs[stable_label]

    def _cleanup_tracks(self):
        """Remove stale tracks not seen recently"""
        stale = [tid for tid, t in self.track_history.items()
                 if self.frame_count - t['last_seen'] > 10]
        for tid in stale:
            del self.track_history[tid]

    def process_frame(self, frame):
        """
        Process a single frame through the two-stage pipeline

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of detections: [(x1, y1, x2, y2, class_name, det_conf, cls_conf)]
        """
        self.frame_count += 1

        # Cleanup old tracks periodically
        if self.frame_count % 30 == 0:
            self._cleanup_tracks()

        # Stage 1: Detect signs
        if self.use_tiling:
            raw_boxes = self._detect_with_tiling(frame)
        else:
            results = self.detector(frame, conf=self.det_conf, iou=self.det_iou,
                                    imgsz=self.det_imgsz, verbose=False)
            raw_boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                det_conf = box.conf[0].item()
                raw_boxes.append((x1, y1, x2, y2, det_conf))

        if not raw_boxes:
            return []

        # Extract crops for all detections
        crops = []
        box_coords = []

        for (x1, y1, x2, y2, det_conf) in raw_boxes:
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Skip tiny crops
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
            box_coords.append((x1, y1, x2, y2, det_conf))

        if not crops:
            return []

        # Stage 2: Classify all crops in a batch
        classifications = self.classifier.classify_batch(crops)

        # Combine results, filtering out "not_a_sign" classifications
        detections = []
        for i, (class_name, cls_conf) in enumerate(classifications):
            if class_name in ('not_a_sign', 'traffic_light'):
                continue  # Reject non-sign detections
            if cls_conf >= self.cls_conf:
                x1, y1, x2, y2, det_conf = box_coords[i]
                box = (x1, y1, x2, y2)

                # Apply temporal smoothing
                stable_name, stable_conf = self._get_stable_label(
                    box, class_name, cls_conf
                )
                detections.append((x1, y1, x2, y2, stable_name, det_conf, stable_conf))

        return detections

    def draw_detections(self, frame, detections, line_width=2):
        """Draw detection boxes and labels on frame"""
        annotated = frame.copy()

        for (x1, y1, x2, y2, class_name, det_conf, cls_conf) in detections:
            # Color based on classification confidence
            if cls_conf > 0.8:
                color = (0, 255, 0)    # Green - high confidence
            elif cls_conf > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - lower confidence

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

            # Label
            label = f"{class_name} {cls_conf:.2f}"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           font_scale, thickness)

            # Background for text
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return annotated

    def process_video(self, video_path, output_path=None, save_video=True):
        """
        Process a video file through the pipeline

        Args:
            video_path: Path to input video
            output_path: Path to save output video (None = auto)
            save_video: Whether to save output video
        """
        print("\n" + "=" * 70)
        print("Two-Stage Pipeline - Video Processing")
        print("=" * 70)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            video_name = Path(video_path).stem
            output_path = f"output_{video_name}_twostage.mp4"

        print(f"  Input: {video_path}")
        print(f"  Resolution: {width}x{height} @ {fps}fps")
        print(f"  Frames: {total_frames}")
        print(f"  Output: {output_path}")

        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process frame
                detections = self.process_frame(frame)
                total_detections += len(detections)

                # Draw results
                annotated = self.draw_detections(frame, detections)

                if save_video:
                    out.write(annotated)

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_cur = frame_count / elapsed
                    progress = frame_count / total_frames * 100
                    print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"FPS: {fps_cur:.1f} | Dets: {len(detections)}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if out:
                out.release()

        elapsed = time.time() - start_time
        print(f"\nDone! {frame_count} frames, {total_detections} detections")
        print(f"Processing: {elapsed:.1f}s @ {frame_count/elapsed:.1f} FPS")
        if save_video:
            print(f"Saved to: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")


def main():
    detector_path = '/home/akshun-wsl/capstone-objectdet/data/mtsd_yolo26n_960/weights/best.pt'
    classifier_path = 'classifier_model/best_classifier.pt'

    pipeline = TwoStagePipeline(
        detector_path=detector_path,
        classifier_path=classifier_path,
        det_conf=0.15,
        det_iou=0.45,
        cls_conf=0.5,
        det_imgsz=960,
        device='cuda',
        temporal_smoothing=True,
        smooth_frames=5,
    )

    # Test on videos
    video_dir = '/home/akshun-wsl/capstone-objectdet/raw_video'
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    if not videos:
        print("No videos found!")
        return

    print(f"\nFound {len(videos)} videos:")
    for i, v in enumerate(videos):
        print(f"  {i+1}. {v}")

    for video in videos:
        video_path = os.path.join(video_dir, video)
        pipeline.process_video(video_path)


if __name__ == "__main__":
    main()
