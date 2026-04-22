"""ROS2 node: two-stage traffic sign detection (YOLO26n + MobileNetV3 classifier)."""

import sys
import json
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, '/home/akshun-wsl/capstone-objectdet/ObjectDetection')
from two_stage_pipeline import TwoStagePipeline


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        self.declare_parameter('detector_path',
            '/home/akshun-wsl/capstone-objectdet/ObjectDetection/data/mtsd_yolo26n_960/weights/best.pt')
        self.declare_parameter('classifier_path',
            '/home/akshun-wsl/capstone-objectdet/ObjectDetection/classifier_model/best_classifier.pt')
        self.declare_parameter('det_conf', 0.15)
        self.declare_parameter('det_iou', 0.45)
        self.declare_parameter('cls_conf', 0.5)
        self.declare_parameter('det_imgsz', 960)
        self.declare_parameter('temporal_smoothing', True)
        self.declare_parameter('smooth_frames', 5)

        self.pipeline = TwoStagePipeline(
            detector_path=self.get_parameter('detector_path').value,
            classifier_path=self.get_parameter('classifier_path').value,
            det_conf=self.get_parameter('det_conf').value,
            det_iou=self.get_parameter('det_iou').value,
            cls_conf=self.get_parameter('cls_conf').value,
            det_imgsz=self.get_parameter('det_imgsz').value,
            temporal_smoothing=self.get_parameter('temporal_smoothing').value,
            smooth_frames=self.get_parameter('smooth_frames').value,
        )

        self.get_logger().info('Two-stage detection pipeline loaded')

        self.sub = self.create_subscription(Image, '/camera/image_raw', self.on_image, 1)
        self.det_pub = self.create_publisher(String, '/detections', 10)

    def on_image(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        detections = self.pipeline.process_frame(frame)

        det_list = []
        for (x1, y1, x2, y2, class_name, det_conf, cls_conf) in detections:
            det_list.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'class': class_name,
                'det_conf': round(float(det_conf), 3),
                'cls_conf': round(float(cls_conf), 3),
            })

        det_msg = String()
        det_msg.data = json.dumps({
            'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'detections': det_list,
        })
        self.det_pub.publish(det_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
