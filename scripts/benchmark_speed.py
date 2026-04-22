"""Benchmark inference speed: TwinLiteNetPlus large vs YOLOPX.
TLNP runs in current Python. YOLOPX runs via conda run in yolopx env.
"""
import subprocess
import json
import sys
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

TLNP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLOPX_DIR = '/home/akshun-wsl/capstone-yolopx/YOLOPX'

def bench_tlnp():
    print(f"\n{'='*50}")
    print("Benchmarking TwinLiteNetPlus-Large")
    print(f"{'='*50}")
    sys.path.insert(0, TLNP_DIR)
    os.chdir(TLNP_DIR)
    from argparse import Namespace
    from model.model import TwinLiteNetPlus

    cudnn.benchmark = True
    args = Namespace(config='large', onGPU=True)
    model = TwinLiteNetPlus(args).cuda().half().eval()
    params = sum(p.numel() for p in model.parameters())

    dummy = torch.randn(1, 3, 384, 640).cuda().half()

    # Warmup
    for _ in range(50):
        _ = model(dummy)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    result = {
        "model": "TwinLiteNetPlus-Large",
        "params": params,
        "input_size": "384x640",
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
        "fps": float(1000.0 / np.mean(times)),
        "runs": len(times)
    }
    print(f"  Parameters: {params:,}")
    print(f"  Mean:       {result['mean_ms']:.2f} ms")
    print(f"  Median:     {result['median_ms']:.2f} ms")
    print(f"  Min:        {result['min_ms']:.2f} ms")
    print(f"  Std:        {result['std_ms']:.2f} ms")
    print(f"  FPS:        {result['fps']:.1f}")

    del model
    torch.cuda.empty_cache()
    return result


def bench_yolopx(conda_env='yolopx'):
    print(f"\n{'='*50}")
    print(f"Benchmarking YOLOPX (conda: {conda_env})")
    print(f"{'='*50}")

    bench_code = """
import torch, torch.backends.cudnn as cudnn, time, sys, os, numpy as np, json
sys.path.insert(0, '{yolopx_dir}')
os.chdir('{yolopx_dir}')
from lib.config import cfg
from lib.models import get_net

cudnn.benchmark = True
model = get_net(cfg).cuda().half().eval()
checkpoint = torch.load('weights/epoch-195.pth', map_location='cuda')
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
params = sum(p.numel() for p in model.parameters())

dummy = torch.randn(1, 3, 384, 640).cuda().half()
for _ in range(50):
    _ = model(dummy)
torch.cuda.synchronize()

times = []
for _ in range(200):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(dummy)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

result = dict(model='YOLOPX', params=params, input_size='384x640',
    mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
    min_ms=float(np.min(times)), max_ms=float(np.max(times)),
    std_ms=float(np.std(times)), fps=float(1000.0/np.mean(times)), runs=len(times))
print('YOLOPX_RESULT:' + json.dumps(result))
""".format(yolopx_dir=YOLOPX_DIR)

    # Write temp script to avoid quoting issues
    tmp_script = '/tmp/bench_yolopx.py'
    with open(tmp_script, 'w') as f:
        f.write(bench_code)

    cmd = f'conda run -n {conda_env} --no-capture-output python {tmp_script}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return None

    for line in result.stdout.split('\n'):
        if 'YOLOPX_RESULT:' in line:
            data = json.loads(line.split('YOLOPX_RESULT:')[1])
            print(f"  Parameters: {data['params']:,}")
            print(f"  Mean:       {data['mean_ms']:.2f} ms")
            print(f"  Median:     {data['median_ms']:.2f} ms")
            print(f"  Min:        {data['min_ms']:.2f} ms")
            print(f"  Std:        {data['std_ms']:.2f} ms")
            print(f"  FPS:        {data['fps']:.1f}")
            return data

    print(f"  No result. stdout: {result.stdout[:500]}")
    return None


if __name__ == '__main__':
    yolopx_env = sys.argv[1] if len(sys.argv) > 1 else 'yolopx'

    tlnp = bench_tlnp()
    yolopx = bench_yolopx(yolopx_env)

    if tlnp and yolopx:
        print(f"\n{'='*60}")
        print("SPEED COMPARISON: TwinLiteNetPlus-Large vs YOLOPX")
        print(f"{'='*60}")
        speedup = yolopx['mean_ms'] / tlnp['mean_ms']
        param_ratio = yolopx['params'] / tlnp['params']
        print(f"  {'Model':<25} {'Params':>12} {'Mean ms':>10} {'FPS':>8}")
        print(f"  {'-'*55}")
        print(f"  {'TwinLiteNetPlus-Large':<25} {tlnp['params']:>12,} {tlnp['mean_ms']:>10.2f} {tlnp['fps']:>8.1f}")
        print(f"  {'YOLOPX':<25} {yolopx['params']:>12,} {yolopx['mean_ms']:>10.2f} {yolopx['fps']:>8.1f}")
        print(f"  {'-'*55}")
        if speedup > 1:
            print(f"  TwinLiteNetPlus is {speedup:.2f}x FASTER than YOLOPX")
        else:
            print(f"  TwinLiteNetPlus is {1/speedup:.2f}x SLOWER than YOLOPX")
        print(f"  YOLOPX has {param_ratio:.1f}x more parameters")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Input: 384x640, FP16, 200 runs after 50 warmup")
