#!/usr/bin/env python3
"""ROS2 node: fuses lane detection and object detection outputs into a combined display."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        self.bridge = CvBridge()

        self.lane_sub = self.create_subscription(
            Image, '/lane_output', self.lane_callback, 10)
        self.object_sub = self.create_subscription(
            Image, '/object_output', self.object_callback, 10)

        self.lane_img = None
        self.object_img = None
        self.lane_lock = Lock()
        self.object_lock = Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        self.get_logger().info('Visualization Node started')

    def lane_callback(self, msg):
        with self.lane_lock:
            self.lane_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.thread_pool.submit(self.create_overlay)

    def object_callback(self, msg):
        with self.object_lock:
            self.object_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.thread_pool.submit(self.create_overlay)

    def create_overlay(self):
        with self.lane_lock:
            lane = self.lane_img.copy() if self.lane_img is not None else None
        with self.object_lock:
            obj = self.object_img.copy() if self.object_img is not None else None

        if lane is None or obj is None:
            return

        if lane.shape != obj.shape:
            obj = cv2.resize(obj, (lane.shape[1], lane.shape[0]))

        # Start with lane overlay, add object detection boxes on top
        combined = lane.copy()
        diff = cv2.absdiff(obj, lane)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        combined[mask > 0] = obj[mask > 0]

        cv2.imshow('Combined Detection System', combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
