"""ROS2 node: TwinLiteNetPlus segmentation (drivable area + lane lines).

Supports three backends (auto-selected in order):
  1. TensorRT engine (.engine) -fastest
  2. ONNX Runtime with CUDA (.onnx) -fast, cross-platform
  3. PyTorch (.pth) -universal fallback

Also supports frame skipping to maintain high effective FPS.
"""

import sys
import os
import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

TLNP_DIR = '/home/akshun-wsl/capstone-tlnp/TwinLiteNetPlus'
sys.path.insert(0, TLNP_DIR)

from demoDataset import letterbox_for_img


class TlnpNode(Node):
    def __init__(self):
        super().__init__('tlnp_node')

        self.declare_parameter('weight_path', os.path.join(TLNP_DIR, 'trained-models/large.pth'))
        self.declare_parameter('config', 'large')
        self.declare_parameter('img_size', 640)
        self.declare_parameter('use_half', True)
        self.declare_parameter('skip_frames', 1)  # process every Nth frame
        self.declare_parameter('engine_path', '')  # TensorRT engine path (auto-detected if empty)
        self.declare_parameter('onnx_path', '')    # ONNX model path (auto-detected if empty)

        weight_path = self.get_parameter('weight_path').value
        config = self.get_parameter('config').value
        self.img_size = self.get_parameter('img_size').value
        self.use_half = self.get_parameter('use_half').value
        self.skip_frames = max(1, self.get_parameter('skip_frames').value)
        engine_path = self.get_parameter('engine_path').value
        onnx_path = self.get_parameter('onnx_path').value

        # Auto-detect engine/onnx paths from weight path
        base = weight_path.rsplit('.', 1)[0]
        if not engine_path:
            engine_path = base + '.engine'
        if not onnx_path:
            onnx_path = base + '.onnx'

        # Select backend
        self.backend = self._load_backend(engine_path, onnx_path, weight_path, config)

        # Warmup
        dummy = torch.zeros((1, 3, 384, 640), device='cuda')
        if self.use_half:
            dummy = dummy.half()
        with torch.no_grad():
            self._infer(dummy)
        self.get_logger().info(f'Warmup complete')

        # Frame skip state
        self.frame_idx = 0
        self.cached_da = None
        self.cached_ll = None

        self.sub = self.create_subscription(Image, '/camera/image_raw', self.on_image, 1)
        self.da_pub = self.create_publisher(Image, '/tlnp/da_mask', 10)
        self.ll_pub = self.create_publisher(Image, '/tlnp/ll_mask', 10)

    def _load_backend(self, engine_path, onnx_path, weight_path, config):
        # Try TensorRT first
        if os.path.exists(engine_path):
            try:
                import tensorrt as trt
                self.trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                with open(engine_path, 'rb') as f:
                    self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
                self.trt_context = self.trt_engine.create_execution_context()
                self.get_logger().info(f'TensorRT engine loaded: {engine_path}')
                return 'tensorrt'
            except Exception as e:
                self.get_logger().warn(f'TensorRT failed ({e}), trying next backend...')

        # Try ONNX Runtime
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
                self.ort_input_name = self.ort_session.get_inputs()[0].name
                active = self.ort_session.get_providers()
                self.get_logger().info(f'ONNX Runtime loaded: {onnx_path} (providers: {active})')
                return 'onnx'
            except Exception as e:
                self.get_logger().warn(f'ONNX Runtime failed ({e}), falling back to PyTorch...')

        # PyTorch fallback
        from model.model import TwinLiteNetPlus
        from argparse import Namespace
        args = Namespace(config=config, onGPU=True)
        self.model = TwinLiteNetPlus(args).cuda()
        if self.use_half:
            self.model = self.model.half()
        self.model.load_state_dict(torch.load(weight_path, map_location='cuda'))
        self.model.eval()
        self.get_logger().info(f'PyTorch model loaded: {weight_path} ({config})')
        return 'pytorch'

    def _infer(self, img_tensor):
        """Run inference on preprocessed tensor, returns (da_out, ll_out)."""
        if self.backend == 'tensorrt':
            return self._infer_trt(img_tensor)
        elif self.backend == 'onnx':
            return self._infer_onnx(img_tensor)
        else:
            return self.model(img_tensor)

    def _infer_trt(self, img_tensor):
        import tensorrt as trt
        stream = torch.cuda.current_stream().cuda_stream
        bindings = []
        outputs = []

        for i in range(self.trt_engine.num_io_tensors):
            name = self.trt_engine.get_tensor_name(i)
            shape = self.trt_context.get_tensor_shape(name)
            dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))

            if self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.trt_context.set_input_shape(name, img_tensor.shape)
                bindings.append(img_tensor.data_ptr())
            else:
                out = torch.empty(tuple(shape), dtype=torch.float16 if self.use_half else torch.float32, device='cuda')
                bindings.append(out.data_ptr())
                outputs.append(out)
                self.trt_context.set_tensor_address(name, out.data_ptr())

        self.trt_context.set_tensor_address(self.trt_engine.get_tensor_name(0), img_tensor.data_ptr())
        self.trt_context.execute_async_v3(stream)
        torch.cuda.synchronize()

        return outputs[0], outputs[1]

    def _infer_onnx(self, img_tensor):
        np_input = img_tensor.cpu().numpy().astype(np.float16 if self.use_half else np.float32)
        results = self.ort_session.run(None, {self.ort_input_name: np_input})
        da_out = torch.from_numpy(results[0]).cuda()
        ll_out = torch.from_numpy(results[1]).cuda()
        return da_out, ll_out

    def on_image(self, msg: Image):
        self.frame_idx += 1

        # Skip frames -reuse cached masks
        if self.skip_frames > 1 and (self.frame_idx % self.skip_frames) != 1:
            if self.cached_da is not None:
                self._publish(msg.header.stamp, self.cached_da, self.cached_ll)
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        # Preprocess
        img, ratio, pad = letterbox_for_img(frame, new_shape=self.img_size, auto=True)
        h0, w0 = frame.shape[:2]
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        img_tensor = img[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).cuda()
        if self.use_half:
            img_tensor = img_tensor.half()
        img_tensor = img_tensor / 255.0

        _, _, height, width = img_tensor.shape
        pad_w, pad_h = int(pad[0]), int(pad[1])
        scale = shapes[1][0][1]

        with torch.no_grad():
            da_out, ll_out = self._infer(img_tensor)

        # Crop padding and resize to original
        da_pred = da_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_mask = torch.nn.functional.interpolate(da_pred, scale_factor=int(1 / scale), mode='bilinear')
        _, da_mask = torch.max(da_mask, 1)
        da_mask = da_mask.squeeze().cpu().numpy().astype(np.uint8)

        ll_pred = ll_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_mask = torch.nn.functional.interpolate(ll_pred, scale_factor=int(1 / scale), mode='bilinear')
        _, ll_mask = torch.max(ll_mask, 1)
        ll_mask = ll_mask.squeeze().cpu().numpy().astype(np.uint8)

        # Cache for frame skipping
        self.cached_da = da_mask
        self.cached_ll = ll_mask

        self._publish(msg.header.stamp, da_mask, ll_mask)

    def _publish(self, stamp, da_mask, ll_mask):
        da_msg = Image()
        da_msg.header = Header(stamp=stamp, frame_id='camera')
        da_msg.height, da_msg.width = da_mask.shape
        da_msg.encoding = 'mono8'
        da_msg.is_bigendian = False
        da_msg.step = da_mask.shape[1]
        da_msg.data = da_mask.tobytes()
        self.da_pub.publish(da_msg)

        ll_msg = Image()
        ll_msg.header = Header(stamp=stamp, frame_id='camera')
        ll_msg.height, ll_msg.width = ll_mask.shape
        ll_msg.encoding = 'mono8'
        ll_msg.is_bigendian = False
        ll_msg.step = ll_mask.shape[1]
        ll_msg.data = ll_mask.tobytes()
        self.ll_pub.publish(ll_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TlnpNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
