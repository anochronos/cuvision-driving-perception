#!/usr/bin/env python3
"""ROS2 node: publishes video file frames to /camera/image_raw for testing without Simulink."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VideoPublisherNode(Node):
    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '')
        self.declare_parameter('loop', False)

        video_path = self.get_parameter('video_path').value
        self.loop = self.get_parameter('loop').value

        if not video_path:
            self.get_logger().error('No video_path parameter set')
            return

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
        self.frame_count = 0

        self.get_logger().info(f'Publishing video: {video_path} at {fps:.0f} FPS')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return
            else:
                self.get_logger().info('Video ended')
                self.cap.release()
                rclpy.shutdown()
                return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.publisher_.publish(msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Published {self.frame_count} frames')


def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisherNode()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()

if __name__ == '__main__':
    main()
