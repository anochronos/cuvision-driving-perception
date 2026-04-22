"""Export TwinLiteNetPlus to ONNX and TensorRT for fast inference.

Usage:
    python export_tensorrt.py [--weight PATH] [--config large] [--img-h 384] [--img-w 640]

Outputs:
    <weight_base>.onnx    -portable, works anywhere with onnxruntime
    <weight_base>.engine  -GPU-specific, fastest inference

The .engine file is NOT portable across GPUs. Run this script on each
target machine. The .onnx file IS portable and can be committed to git.
"""

import argparse
import sys
import os
import time

import torch
import numpy as np

TLNP_DIR = '/home/akshun-wsl/capstone-tlnp/TwinLiteNetPlus'
sys.path.insert(0, TLNP_DIR)

from model.model import TwinLiteNetPlus
from argparse import Namespace


def export_onnx(model, onnx_path, img_h, img_w):
    """Export PyTorch model to ONNX."""
    print(f'\n[1/2] Exporting to ONNX: {onnx_path}')
    dummy = torch.randn(1, 3, img_h, img_w, device='cuda').half()

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['da_seg', 'll_seg'],
        opset_version=17,
        dynamic_axes={
            'input': {0: 'batch'},
            'da_seg': {0: 'batch'},
            'll_seg': {0: 'batch'},
        }
    )

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f'  ONNX exported: {size_mb:.1f} MB')

    # Verify
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f'  ONNX verification passed')

    return onnx_path


def export_tensorrt(onnx_path, engine_path, img_h, img_w, fp16=True):
    """Convert ONNX to TensorRT engine."""
    print(f'\n[2/2] Building TensorRT engine: {engine_path}')
    print(f'  This may take a few minutes...')

    try:
        import tensorrt as trt
    except ImportError:
        print('  TensorRT not installed. Skipping engine build.')
        print('  Install with: pip install tensorrt')
        print(f'  The ONNX model at {onnx_path} can still be used with onnxruntime.')
        return None

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f'  Parse error: {parser.get_error(i)}')
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f'  FP16 enabled')

    # Set input shape
    profile = builder.create_optimization_profile()
    profile.set_shape('input',
                      min=(1, 3, img_h, img_w),
                      opt=(1, 3, img_h, img_w),
                      max=(1, 3, img_h, img_w))
    config.add_optimization_profile(profile)

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print('  Engine build failed!')
        return None

    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f'  TensorRT engine built in {elapsed:.1f}s ({size_mb:.1f} MB)')

    return engine_path


def benchmark(model_pytorch, onnx_path, engine_path, img_h, img_w, runs=100):
    """Compare inference speed across backends."""
    print(f'\n{"="*60}')
    print('BENCHMARK COMPARISON')
    print(f'{"="*60}')
    dummy = torch.randn(1, 3, img_h, img_w, device='cuda').half()

    # PyTorch
    for _ in range(10):
        model_pytorch(dummy)
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model_pytorch(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    pt_ms = np.mean(times)
    print(f'  PyTorch FP16:    {pt_ms:.2f} ms ({1000/pt_ms:.0f} FPS)')

    # ONNX Runtime
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path,
                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            np_input = dummy.cpu().numpy()

            for _ in range(10):
                sess.run(None, {'input': np_input})

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                sess.run(None, {'input': np_input})
                times.append((time.perf_counter() - t0) * 1000)
            ort_ms = np.mean(times)
            print(f'  ONNX Runtime:    {ort_ms:.2f} ms ({1000/ort_ms:.0f} FPS)')
        except Exception as e:
            print(f'  ONNX Runtime:    skipped ({e})')

    # TensorRT
    if engine_path and os.path.exists(engine_path):
        try:
            import tensorrt as trt
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

            # Allocate outputs
            da_out = torch.empty((1, 2, img_h, img_w), dtype=torch.float16, device='cuda')
            ll_out = torch.empty((1, 2, img_h, img_w), dtype=torch.float16, device='cuda')

            context.set_input_shape('input', dummy.shape)
            context.set_tensor_address('input', dummy.data_ptr())
            context.set_tensor_address('da_seg', da_out.data_ptr())
            context.set_tensor_address('ll_seg', ll_out.data_ptr())

            stream = torch.cuda.current_stream().cuda_stream
            for _ in range(10):
                context.execute_async_v3(stream)
            torch.cuda.synchronize()

            times = []
            for _ in range(runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                context.execute_async_v3(stream)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            trt_ms = np.mean(times)
            print(f'  TensorRT FP16:   {trt_ms:.2f} ms ({1000/trt_ms:.0f} FPS)')
            print(f'  Speedup vs PT:   {pt_ms/trt_ms:.1f}x')
        except Exception as e:
            print(f'  TensorRT:        skipped ({e})')


def main():
    parser = argparse.ArgumentParser(description='Export TwinLiteNetPlus to ONNX/TensorRT')
    parser.add_argument('--weight', default=os.path.join(TLNP_DIR, 'trained-models/large.pth'))
    parser.add_argument('--config', default='large', choices=['nano', 'small', 'medium', 'large'])
    parser.add_argument('--img-h', type=int, default=384)
    parser.add_argument('--img-w', type=int, default=640)
    parser.add_argument('--no-benchmark', action='store_true')
    args = parser.parse_args()

    base = args.weight.rsplit('.', 1)[0]
    onnx_path = base + '.onnx'
    engine_path = base + '.engine'

    print(f'Model: {args.weight} ({args.config})')
    print(f'Input: {args.img_h}x{args.img_w}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Load PyTorch model
    model_args = Namespace(config=args.config, onGPU=True)
    model = TwinLiteNetPlus(model_args).cuda().half().eval()
    model.load_state_dict(torch.load(args.weight, map_location='cuda'))

    # Export
    export_onnx(model, onnx_path, args.img_h, args.img_w)
    export_tensorrt(onnx_path, engine_path, args.img_h, args.img_w)

    if not args.no_benchmark:
        with torch.no_grad():
            benchmark(model, onnx_path, engine_path, args.img_h, args.img_w)

    print(f'\nFiles:')
    print(f'  ONNX:     {onnx_path}')
    if os.path.exists(engine_path):
        print(f'  TensorRT: {engine_path}')
    print(f'\nAdd to .gitignore: *.engine')


if __name__ == '__main__':
    main()
