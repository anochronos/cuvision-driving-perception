#!/usr/bin/env python3
"""ROS2 node: runs TwinLiteNetPlus segmentation on /camera/image_raw and publishes overlayed result."""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from argparse import Namespace

# Add parent repo so we can import the shared segmentation model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)

from segmentation.model.model import TwinLiteNetPlus


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        self.declare_parameter('weight_path', os.path.join(REPO_ROOT, 'segmentation', 'weights', 'large.pth'))
        self.declare_parameter('config', 'large')
        self.declare_parameter('process_every_n_frames', 2)

        weight_path = self.get_parameter('weight_path').value
        config = self.get_parameter('config').value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').value

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/lane_output', 10)

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load TwinLiteNetPlus
        self.model = TwinLiteNetPlus(Namespace(config=config))
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.frame_count = 0
        self.get_logger().info(f'Lane Detection Node started ({config} on {self.device})')

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        img0 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess: resize to model input, normalize
        img = cv2.resize(img0, (640, 384))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            da_seg_out, ll_seg_out = self.model(img)

        # Post-process masks back to original resolution
        da_mask = torch.argmax(da_seg_out, dim=1).squeeze().cpu().numpy()
        ll_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy()

        da_mask = cv2.resize(da_mask.astype(np.uint8),
                             (img0.shape[1], img0.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        ll_mask = cv2.resize(ll_mask.astype(np.uint8),
                             (img0.shape[1], img0.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # Overlay: green drivable area, red lane lines
        img_result = img0.copy()
        img_result[da_mask == 1] = [0, 255, 0]
        img_result[ll_mask == 1] = [0, 0, 255]
        img_result = cv2.addWeighted(img0, 0.6, img_result, 0.4, 0)

        output_msg = self.bridge.cv2_to_imgmsg(img_result, encoding='bgr8')
        output_msg.header = msg.header
        self.publisher_.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)
    lane_node = LaneDetectionNode()
    rclpy.spin(lane_node)
    lane_node.destroy_node()

if __name__ == '__main__':
    main()
