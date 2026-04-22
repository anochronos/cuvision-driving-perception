"""ROS2 node: fuses camera + segmentation + detections, serves MJPEG web viewer."""

import json
import socket
import time
import threading
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')

        self.declare_parameter('web_port', 8080)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('save_video', False)
        self.declare_parameter('video_path', 'output_ros2.mp4')

        self.web_port = self.get_parameter('web_port').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        save_video = self.get_parameter('save_video').value
        video_path = self.get_parameter('video_path').value

        self.latest_frame = None
        self.latest_da = None
        self.latest_ll = None
        self.latest_dets = []
        self.data_lock = threading.Lock()

        self.latest_jpeg = None
        self.jpeg_lock = threading.Lock()

        self.frame_count = 0
        self.fps_start = time.time()
        self.current_fps = 0.0

        self.vid_writer = None
        if save_video:
            self.vid_writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

        # Subscriptions
        self.create_subscription(Image, '/camera/image_raw', self.on_camera, 1)
        self.create_subscription(Image, '/tlnp/da_mask', self.on_da, 1)
        self.create_subscription(Image, '/tlnp/ll_mask', self.on_ll, 1)
        self.create_subscription(String, '/detections', self.on_detections, 1)

        # Render timer at ~30 Hz
        self.create_timer(1.0 / 30.0, self.render)

        # Start web server
        self._start_web_server()
        self.get_logger().info(f'Viz node ready -http://localhost:{self.web_port}')

    def on_camera(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self.data_lock:
            self.latest_frame = frame

    def on_da(self, msg: Image):
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        with self.data_lock:
            self.latest_da = mask

    def on_ll(self, msg: Image):
        mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        with self.data_lock:
            self.latest_ll = mask

    def on_detections(self, msg: String):
        data = json.loads(msg.data)
        with self.data_lock:
            self.latest_dets = data.get('detections', [])

    def render(self):
        with self.data_lock:
            frame = self.latest_frame
            da = self.latest_da
            ll = self.latest_ll
            dets = list(self.latest_dets)

        if frame is None:
            return

        vis = frame.copy()

        # Overlay segmentation masks
        if da is not None:
            da_resized = cv2.resize(da, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = np.zeros_like(vis)
            overlay[da_resized == 1] = [0, 255, 0]  # Green for drivable area
            mask = da_resized > 0
            vis[mask] = (vis[mask] * 0.6 + overlay[mask] * 0.4).astype(np.uint8)

        if ll is not None:
            ll_resized = cv2.resize(ll, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            vis[ll_resized == 1] = [255, 0, 0]  # Blue for lane lines

        # Draw detections
        for det in dets:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            cls_conf = det['cls_conf']

            color = (0, 255, 0) if cls_conf > 0.8 else (0, 255, 255) if cls_conf > 0.6 else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {cls_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # FPS counter
        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed > 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start = time.time()

        cv2.putText(vis, f'{self.current_fps:.1f} FPS', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize for output
        vis = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_LINEAR)

        if self.vid_writer is not None:
            self.vid_writer.write(vis)

        _, jbuf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        with self.jpeg_lock:
            self.latest_jpeg = jbuf.tobytes()

    def _start_web_server(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(b'''<html><head><title>Driving Pipeline</title>
                    <style>body{margin:0;background:#000;display:flex;justify-content:center;
                    align-items:center;height:100vh;font-family:monospace;color:#0f0}
                    img{max-width:100vw;max-height:100vh}
                    #msg{position:absolute;top:50%;color:#0f0;font-size:24px}</style>
                    </head><body>
                    <div id="msg">Waiting for frames...</div>
                    <img id="vid" src="/stream" onload="document.getElementById('msg').style.display='none'">
                    </body></html>''')
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    while rclpy.ok():
                        with node.jpeg_lock:
                            jpeg = node.latest_jpeg
                        if jpeg is not None:
                            try:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ')
                                self.wfile.write(str(len(jpeg)).encode())
                                self.wfile.write(b'\r\n\r\n')
                                self.wfile.write(jpeg)
                                self.wfile.write(b'\r\n')
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                break
                        time.sleep(0.033)

            def log_message(self, *a):
                pass

        httpd = HTTPServer(('0.0.0.0', self.web_port), Handler)
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def destroy_node(self):
        if self.vid_writer is not None:
            self.vid_writer.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
