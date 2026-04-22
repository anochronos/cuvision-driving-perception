#!/usr/bin/env python3
"""ROS2 node: two-stage traffic sign detection (YOLO26n + MobileNetV3) on /camera/image_raw."""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image as PILImage

# Add parent repo for shared detection weights
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


class SignClassifier:
    """MobileNetV3-Small sign classifier."""

    def __init__(self, model_path, device='cpu'):
        self.device = device

        checkpoint = torch.load(model_path, map_location=device)
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        self.img_size = checkpoint['img_size']

        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, self.num_classes
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def classify_batch(self, crops_bgr):
        if not crops_bgr:
            return []

        tensors = []
        for crop in crops_bgr:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(crop_rgb)
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


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        self.declare_parameter('detector_path',
            os.path.join(REPO_ROOT, 'detection', 'weights', 'detector_yolo26n.pt'))
        self.declare_parameter('classifier_path',
            os.path.join(REPO_ROOT, 'detection', 'classifier_model', 'best_classifier.pt'))
        self.declare_parameter('det_conf', 0.15)
        self.declare_parameter('det_iou', 0.45)
        self.declare_parameter('cls_conf', 0.5)
        self.declare_parameter('det_imgsz', 960)
        self.declare_parameter('process_every_n_frames', 2)

        detector_path = self.get_parameter('detector_path').value
        classifier_path = self.get_parameter('classifier_path').value
        self.det_conf = self.get_parameter('det_conf').value
        self.det_iou = self.get_parameter('det_iou').value
        self.cls_conf = self.get_parameter('cls_conf').value
        self.det_imgsz = self.get_parameter('det_imgsz').value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').value

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/object_output', 10)

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.get_logger().info('Loading two-stage detection pipeline...')
        self.detector = YOLO(detector_path)
        self.classifier = SignClassifier(classifier_path, self.device)
        self.get_logger().info(f'Detector: {detector_path}')
        self.get_logger().info(f'Classifier: {classifier_path} ({self.classifier.num_classes} classes)')

        self.frame_count = 0
        self.get_logger().info(f'Object Detection Node started on {self.device}')

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Stage 1: YOLO detect sign locations
        results = self.detector(frame, conf=self.det_conf, iou=self.det_iou,
                                imgsz=self.det_imgsz, verbose=False)

        crops = []
        box_coords = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = box.conf[0].item()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            crops.append(frame[y1:y2, x1:x2])
            box_coords.append((x1, y1, x2, y2, det_conf))

        annotated = frame.copy()

        if crops:
            # Stage 2: classify each crop
            classifications = self.classifier.classify_batch(crops)

            for i, (class_name, cls_conf) in enumerate(classifications):
                if class_name in ('not_a_sign', 'traffic_light'):
                    continue
                if cls_conf < self.cls_conf:
                    continue

                x1, y1, x2, y2, det_conf = box_coords[i]

                if cls_conf > 0.8:
                    color = (0, 255, 0)
                elif cls_conf > 0.6:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {cls_conf:.2f}"
                font_scale, thickness = 0.5, 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, thickness)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        output_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        output_msg.header = msg.header
        self.publisher_.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    object_node = ObjectDetectionNode()
    rclpy.spin(object_node)
    object_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
