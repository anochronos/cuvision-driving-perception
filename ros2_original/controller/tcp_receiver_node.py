#!/usr/bin/env python3
"""ROS2 node: receives camera frames via TCP/UDP from Simulink and publishes to /camera/image_raw."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import socket
import threading
import numpy as np
import cv2


class TCPReceiverNode(Node):

    def __init__(self):
        super().__init__('tcp_camera_node')

        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 12345)
        self.declare_parameter('use_udp', False)
        self.declare_parameter('image_h', 720)
        self.declare_parameter('image_w', 1280)
        self.declare_parameter('publish_rate', 30.0)

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.use_udp = self.get_parameter('use_udp').value
        self.image_h = self.get_parameter('image_h').value
        self.image_w = self.get_parameter('image_w').value
        rate = self.get_parameter('publish_rate').value

        self.frame_bytes = self.image_h * self.image_w * 3
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0

        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / rate, self.publish_frame)

        self.recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.recv_thread.start()

        self.get_logger().info(
            f'TCP camera node listening on {self.host}:{self.port} '
            f'({"UDP" if self.use_udp else "TCP"}, {self.image_w}x{self.image_h})'
        )

    def _recv_exactly(self, sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(n - len(buf), 262144))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf.extend(chunk)
        return buf

    def _receive_loop(self):
        if self.use_udp:
            self._udp_loop()
        else:
            self._tcp_loop()

    def _tcp_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        self.get_logger().info('Waiting for Simulink TCP connection...')

        while rclpy.ok():
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self.get_logger().info(f'TCP connected from {addr}')

            while rclpy.ok():
                try:
                    data = self._recv_exactly(conn, self.frame_bytes)
                    pixels = np.frombuffer(data, dtype=np.uint8)
                    # Simulink sends column-major (Fortran order) RGB
                    frame = pixels.reshape((self.image_h, self.image_w, 3), order='F')
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_count += 1
                except ConnectionError:
                    self.get_logger().warn('TCP disconnected, waiting for reconnect...')
                    conn.close()
                    break
                except Exception as e:
                    self.get_logger().error(f'TCP error: {e}')

    def _udp_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        sock.bind((self.host, self.port))
        self.get_logger().info(f'UDP listening on {self.host}:{self.port}')

        frame_buf = np.zeros(self.frame_bytes, dtype=np.uint8)
        offset = 0

        while rclpy.ok():
            try:
                data = sock.recv(65535)
                pdata = np.frombuffer(data, dtype=np.uint8)
                end = offset + len(pdata)

                if end <= self.frame_bytes:
                    frame_buf[offset:end] = pdata
                    offset = end
                else:
                    remainder = self.frame_bytes - offset
                    frame_buf[offset:self.frame_bytes] = pdata[:remainder]
                    frame = frame_buf.reshape((self.image_h, self.image_w, 3), order='F')
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
                    leftover = pdata[remainder:]
                    frame_buf[:len(leftover)] = leftover
                    offset = len(leftover)
                    continue

                if offset >= self.frame_bytes:
                    frame = frame_buf.reshape((self.image_h, self.image_w, 3), order='F')
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
                    offset = 0
            except Exception as e:
                self.get_logger().error(f'UDP error: {e}')
                offset = 0

    def publish_frame(self):
        with self.frame_lock:
            frame = self.latest_frame
            if frame is None:
                return
            self.latest_frame = None

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TCPReceiverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
