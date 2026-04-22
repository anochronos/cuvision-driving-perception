"""
CARLA Autonomous Driving Pipeline
Connects to CARLA simulator and runs real-time segmentation + traffic sign detection
with results streamed to a browser-based web viewer.

Supports three driving modes:
  - Autopilot (default): CARLA AI drives with traffic
  - Manual (--manual): WASD controls from browser
  - Attach (--attach): hooks into an existing manual_control.py vehicle

Usage:
    python carla_pipeline.py --carla-host 172.29.240.1
    python carla_pipeline.py --carla-host 172.29.240.1 --manual
    python carla_pipeline.py --carla-host 172.29.240.1 --attach
    python carla_pipeline.py --carla-host 172.29.240.1 --weather snow --town Town03
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import carla
import cv2
import numpy as np
import torch

# Add parent directory so we can import from segmentation/ and detection/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from segmentation.model.model import TwinLiteNetPlus
from segmentation.src.lane_splitter import (
    split_da_into_lanes, draw_lane_lines,
    TemporalSmoother, DAMaskAccumulator,
)


# Preprocessing & visualization helpers (adapted for single-frame inference) ARE WE SURE?

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True):
    """Resize image to a 32-pixel-multiple rectangle with letterboxing."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def preprocess_frame(frame, img_size=640):
    """Preprocess a BGR frame for TwinLiteNetPlus inference.

    Returns:
        img_tensor: (1, 3, H, W) tensor
        shapes: ((h0, w0), ((h_ratio, w_ratio), (pad_w, pad_h)))
    """
    img, ratio, pad = letterbox(frame, new_shape=img_size, auto=True)
    h, w = img.shape[:2]
    h0, w0 = frame.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img_np = img[:, :, ::-1].transpose(2, 0, 1)
    img_np = np.ascontiguousarray(img_np)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    return img_tensor, shapes


def overlay_segmentation(img, da_mask, ll_mask, output_size=(1280, 720)):
    """Overlay drivable area (green) and lane line (red) masks on an image."""
    color_area = np.zeros((da_mask.shape[0], da_mask.shape[1], 3), dtype=np.uint8)
    color_area[da_mask == 1] = [0, 255, 0]
    color_area[ll_mask == 1] = [255, 0, 0]
    color_seg = color_area[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
    return img


# Web viewer (MJPEG stream with optional WASD manual controls)

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


MANUAL_HTML = b'''<html><head><title>CARLA Pipeline</title>
<style>body{margin:0;background:#000;display:flex;flex-direction:column;align-items:center;
min-height:100vh}img{max-width:100vw;max-height:85vh}
#msg{position:absolute;top:50%;color:#0f0;font-size:24px;font-family:monospace}
#ctrl{color:#0f0;font-family:monospace;padding:10px;text-align:center}
.k{display:inline-block;width:38px;height:38px;border:2px solid #555;text-align:center;
line-height:38px;margin:2px;border-radius:5px;font-size:15px;font-weight:bold}
.k.on{background:#0f0;color:#000;border-color:#0f0}
.row{display:flex;justify-content:center}
.actions{display:flex;justify-content:center;gap:8px;margin-top:8px}
.ak{display:inline-block;padding:4px 12px;border:2px solid #555;border-radius:5px;
font-size:13px;font-weight:bold;cursor:default}
.ak.on{background:#0f0;color:#000;border-color:#0f0}
#help{color:#ccc;font-size:13px;font-weight:bold;margin-top:8px;letter-spacing:0.3px}
#help span{color:#0f0}</style></head><body>
<div id="msg">Waiting for CARLA frames...</div>
<img id="vid" src="/stream" onload="document.getElementById('msg').style.display='none'">
<div id="ctrl">
<div class="row"><span class="k" id="kw">W</span></div>
<div class="row"><span class="k" id="ka">A</span><span class="k" id="ks">S</span>
<span class="k" id="kd">D</span></div>
<div class="actions"><span class="ak" id="kt">T Reset</span>
<span class="ak" id="kp">P Auto</span></div>
<div id="modeLabel" style="color:#0f0;font-size:16px;font-weight:bold;margin-top:6px">MANUAL</div>
<div id="help"><span>W/S</span> throttle/brake &middot; <span>A/D</span> steer &middot;
<span>SPACE</span> handbrake &middot;
<span>T</span> reset &middot; <span>P</span> autopilot</div></div>
<script>
const k={};let auto=false;
document.onkeydown=e=>{
const c=e.key.toLowerCase();
if(c==='t'&&!k.t){fetch('/reset',{method:'POST'})}
if(c==='p'&&!k.p){auto=!auto;fetch('/autopilot',{method:'POST'});
document.getElementById('modeLabel').textContent=auto?'AUTOPILOT':'MANUAL';
document.getElementById('modeLabel').style.color=auto?'#ff0':'#0f0'}
k[c]=1;e.preventDefault()};
document.onkeyup=e=>{k[e.key.toLowerCase()]=0;e.preventDefault()};
setInterval(()=>{
if(auto)return;
const t=k.w?1.0:0,b=k.s?1:0,s=(k.a?-0.7:0)+(k.d?0.7:0);
['w','a','s','d'].forEach(x=>{
const el=document.getElementById('k'+x);
if(el)el.className='k'+(k[x]?' on':'')});
document.getElementById('kt').className='ak'+(k.t?' on':'');
document.getElementById('kp').className='ak'+(auto?' on':'');
fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({throttle:t,brake:b,steer:s,reverse:false,handbrake:!!k[' ']})})
},16);
</script></body></html>'''

NORMAL_HTML = b'''<html><head><title>CARLA Pipeline</title>
<style>body{margin:0;background:#000;display:flex;justify-content:center;
align-items:center;height:100vh}img{max-width:100vw;max-height:100vh}
#msg{position:absolute;top:50%;color:#0f0;font-size:24px;
font-family:monospace}</style></head><body>
<div id="msg">Waiting for CARLA frames...</div>
<img id="vid" src="/stream"
 onload="document.getElementById('msg').style.display='none'">
</body></html>'''

# YO CLEAN THIS HTML FILE UP PLEASE

class WebViewer:
    """MJPEG web viewer with optional manual driving controls."""

    def __init__(self, port=8080, manual=False):
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.manual = manual
        self.control_state = {'throttle': 0, 'steer': 0, 'brake': 0,
                              'reverse': False, 'handbrake': False}
        self.control_lock = threading.Lock()
        self.reset_requested = False
        self.autopilot_toggle = False
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(MANUAL_HTML if viewer.manual else NORMAL_HTML)
                elif self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type',
                                     'multipart/x-mixed-replace; boundary=--frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    while True:
                        with viewer.lock:
                            jpeg = viewer.latest_jpeg
                        if jpeg is not None:
                            try:
                                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                                 b'Content-Length: ')
                                self.wfile.write(str(len(jpeg)).encode())
                                self.wfile.write(b'\r\n\r\n')
                                self.wfile.write(jpeg)
                                self.wfile.write(b'\r\n')
                                self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError):
                                break
                        time.sleep(0.016)

            def do_POST(self):
                if self.path == '/control' and viewer.manual:
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                        with viewer.control_lock:
                            viewer.control_state.update(data)
                    except Exception:
                        pass
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                elif self.path == '/reset' and viewer.manual:
                    viewer.reset_requested = True
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                elif self.path == '/autopilot' and viewer.manual:
                    viewer.autopilot_toggle = True
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *a):
                pass

        self.httpd = ThreadedHTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"[WEB] Viewer at http://localhost:{port}")
        if manual:
            print("[WEB] Manual controls active -- click browser window and use WASD")

    def update(self, frame_bgr):
        _, jbuf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.latest_jpeg = jbuf.tobytes()

    def get_control(self):
        with self.control_lock:
            return dict(self.control_state)

    def stop(self):
        self.httpd.shutdown()


# Weather presets

WEATHER_PRESETS = {
    'clear':     carla.WeatherParameters.ClearNoon,
    'cloudy':    carla.WeatherParameters.CloudyNoon,
    'rain':      carla.WeatherParameters.HardRainNoon,
    'sunset':    carla.WeatherParameters.SoftRainSunset,
    'night':     carla.WeatherParameters(
        sun_altitude_angle=-30.0, cloudiness=20.0,
        precipitation=0.0, fog_density=0.0,
    ),
    'snow': carla.WeatherParameters(
        sun_altitude_angle=50.0, cloudiness=80.0,
        precipitation=80.0, precipitation_deposits=80.0,
        fog_density=30.0, fog_distance=50.0,
        wetness=80.0, wind_intensity=60.0,
    ),
    'fog': carla.WeatherParameters(
        sun_altitude_angle=30.0, cloudiness=60.0,
        fog_density=80.0, fog_distance=20.0,
        fog_falloff=1.0,
    ),
}


# Model loading & inference

def load_segmentation_model(config='large', weight_path=None, img_size=640):
    """Load TwinLiteNetPlus and warm up with a dummy forward pass."""
    if weight_path is None:
        weight_path = os.path.join(PARENT_DIR, 'segmentation', 'weights', 'large.pth')

    class ModelArgs:
        pass
    args = ModelArgs()
    args.config = config

    model = TwinLiteNetPlus(args).cuda().half()
    dummy = torch.zeros((1, 3, img_size, img_size), device='cuda:0')
    _ = model(dummy.half())
    model.load_state_dict(torch.load(weight_path, map_location='cuda'))
    model.eval()
    return model


def run_segmentation(model, img_tensor, shapes):
    """Run TwinLiteNetPlus inference and return DA + LL masks at original resolution."""
    img = img_tensor.cuda().half() / 255.0
    _, _, height, width = img.shape
    pad_w, pad_h = shapes[1][1]
    pad_w, pad_h = int(pad_w), int(pad_h)
    ratio = shapes[1][0][1]

    da_seg_out, ll_seg_out = model(img)

    da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(
        da_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

    ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(
        ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    return da_seg_mask, ll_seg_mask


# CARLA setup

def setup_carla(host, port, town, weather, res_x, res_y, attach=False, manual=False):
    """Connect to CARLA, spawn or attach to vehicle, attach camera."""
    print(f"[CARLA] Connecting to {host}:{port}...")
    client = carla.Client(host, port)
    client.set_timeout(120.0)

    try:
        world = client.get_world()
        print(f"[CARLA] Connected to existing world")
        if not attach and town != 'Town03':
            print(f"[CARLA] Loading {town}...")
            world = client.load_world(town)
    except RuntimeError:
        print(f"[CARLA] Loading {town}...")
        world = client.load_world(town)

    if weather in WEATHER_PRESETS:
        world.set_weather(WEATHER_PRESETS[weather])
        print(f"[CARLA] Weather: {weather}")

    blueprint_library = world.get_blueprint_library()
    traffic_vehicles = []

    if attach:
        actors = world.get_actors().filter('vehicle.*')
        if len(actors) == 0:
            print("[CARLA] ERROR: No vehicles found. Start manual_control.py first.")
            sys.exit(1)
        vehicle = actors[0]
        print(f"[CARLA] Attached to existing vehicle: {vehicle.type_id}")

        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    else:
        # Clean up leftover actors from previous runs
        for a in world.get_actors().filter('sensor.*'):
            a.destroy()
        for a in world.get_actors().filter('vehicle.*'):
            a.destroy()
        print("[CARLA] Cleaned up leftover actors")

        # Reset to async briefly to clear stuck traffic manager state
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        time.sleep(0.5)

        # Synchronous mode at 30 FPS
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 30.0
        world.apply_settings(settings)

        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
        for sp in spawn_points:
            try:
                vehicle = world.spawn_actor(vehicle_bp, sp)
                break
            except RuntimeError:
                continue
        if vehicle is None:
            print("[CARLA] ERROR: Could not spawn vehicle at any point.")
            sys.exit(1)

        if manual:
            print(f"[CARLA] Spawned ego vehicle (MANUAL mode)")
        else:
            vehicle.set_autopilot(True)
            print(f"[CARLA] Spawned ego vehicle (AUTOPILOT)")

        # Spawn traffic NPCs
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        vehicle_bps = blueprint_library.filter('vehicle.*')
        num_traffic = min(30, len(spawn_points) - 1)
        for i in range(1, num_traffic + 1):
            bp = np.random.choice(vehicle_bps)
            try:
                npc = world.spawn_actor(bp, spawn_points[i])
                npc.set_autopilot(True, 8000)
                traffic_vehicles.append(npc)
            except Exception:
                pass
        print(f"[CARLA] Spawned {len(traffic_vehicles)} traffic vehicles")

    # Attach RGB camera to ego vehicle
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(res_x))
    camera_bp.set_attribute('image_size_y', str(res_y))
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.4),
        carla.Rotation(pitch=-5)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print(f"[CARLA] Camera attached ({res_x}x{res_y})")

    return client, world, vehicle, camera, traffic_vehicles


# Main loop

def main():
    parser = argparse.ArgumentParser(description='CARLA Driving Pipeline')

    # CARLA connection
    parser.add_argument('--carla-host', type=str, default='localhost')
    parser.add_argument('--carla-port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town03')
    parser.add_argument('--weather', type=str, default='clear',
                        choices=list(WEATHER_PRESETS.keys()))
    parser.add_argument('--res-x', type=int, default=1280)
    parser.add_argument('--res-y', type=int, default=720)

    # Driving mode
    parser.add_argument('--manual', action='store_true',
                        help='Drive with WASD from web viewer')
    parser.add_argument('--attach', action='store_true',
                        help='Attach to existing vehicle (e.g. manual_control.py)')

    # Web viewer
    parser.add_argument('--web-port', type=int, default=8080)

    # Segmentation
    parser.add_argument('--seg-config', type=str, default='large',
                        choices=['nano', 'small', 'medium', 'large'])
    parser.add_argument('--seg-weight', type=str, default=None)
    parser.add_argument('--img-size', type=int, default=640)

    # Lane splitting
    parser.add_argument('--num-lanes', type=int, default=2)
    parser.add_argument('--no-lane-split', action='store_true')
    parser.add_argument('--cutoff-bottom', type=float, default=0.08)
    parser.add_argument('--smooth-window', type=int, default=51)
    parser.add_argument('--lane-thickness', type=int, default=4)
    parser.add_argument('--lane-color', type=int, nargs=3, default=[0, 255, 255])
    parser.add_argument('--lane-ratio', type=float, default=1.4)

    # Detection
    parser.add_argument('--no-detection', action='store_true')
    parser.add_argument('--det-weight', type=str, default=None)
    parser.add_argument('--cls-weight', type=str, default=None)
    parser.add_argument('--det-imgsz', type=int, default=720)
    parser.add_argument('--det-conf', type=float, default=0.20)
    parser.add_argument('--cls-conf', type=float, default=0.5)

    # Output
    parser.add_argument('--save-video', type=str, default=None)

    args = parser.parse_args()
    args.lane_color = tuple(args.lane_color)

    print("=" * 60)
    print("  CARLA Autonomous Driving Pipeline")
    mode = 'MANUAL' if args.manual else ('ATTACHED' if args.attach else 'AUTOPILOT')
    print(f"  Mode: {mode}")
    print("=" * 60)

    # Load segmentation model
    print(f"\n[SEG] Loading TwinLiteNet+ ({args.seg_config})...")
    seg_model = load_segmentation_model(
        config=args.seg_config, weight_path=args.seg_weight, img_size=args.img_size)
    print("[SEG] Ready.")

    # Load detection pipeline (optional)
    det_pipeline = None
    if not args.no_detection:
        try:
            from detection.two_stage_pipeline import TwoStagePipeline
            det_weight = args.det_weight or os.path.join(
                PARENT_DIR, 'detection', 'weights', 'detector_yolo26n.pt')
            cls_weight = args.cls_weight or os.path.join(
                PARENT_DIR, 'detection', 'classifier_model', 'best_classifier.pt')

            if os.path.exists(det_weight) and os.path.exists(cls_weight):
                det_pipeline = TwoStagePipeline(
                    detector_path=det_weight, classifier_path=cls_weight,
                    det_conf=args.det_conf, cls_conf=args.cls_conf,
                    det_imgsz=args.det_imgsz, device='cuda')
        except Exception as e:
            print(f"[DET] Warning: {e}")

    # Lane splitting state
    lane_smoother = None
    da_accum = None
    if not args.no_lane_split and args.num_lanes > 1:
        lane_smoother = TemporalSmoother(alpha=0.3)
        da_accum = DAMaskAccumulator(alpha=0.25, threshold=0.4)

    # Connect to CARLA
    client, world, vehicle, camera, traffic = setup_carla(
        args.carla_host, args.carla_port, args.town, args.weather,
        args.res_x, args.res_y, attach=args.attach, manual=args.manual)

    # Start web viewer
    web = WebViewer(port=args.web_port, manual=args.manual)

    # Video writer (optional)
    vid_writer = None
    if args.save_video:
        vid_writer = cv2.VideoWriter(
            args.save_video, cv2.VideoWriter_fourcc(*'mp4v'), 30,
            (args.res_x, args.res_y))

    # Frame buffer from CARLA camera callback
    latest_frame = [None]
    frame_lock = threading.Lock()
    frame_count = [0]

    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        with frame_lock:
            latest_frame[0] = array.copy()
            frame_count[0] += 1

    camera.listen(camera_callback)

    print(f"\n[READY] Pipeline running. Open http://localhost:{args.web_port}")
    print("[READY] Ctrl+C to stop.\n")

    processed = 0
    fps_start = time.time()
    current_fps = 0.0
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        with torch.no_grad():
            while True:
                # Handle autopilot toggle from web UI
                if args.manual and web.autopilot_toggle:
                    web.autopilot_toggle = False
                    autopilot_on = not getattr(vehicle, '_autopilot_on', False)
                    vehicle._autopilot_on = autopilot_on
                    vehicle.set_autopilot(autopilot_on)
                    mode = 'AUTOPILOT' if autopilot_on else 'MANUAL'
                    print(f"\n[MODE] Switched to {mode}", flush=True)

                # Handle reset from web UI
                if args.manual and web.reset_requested:
                    web.reset_requested = False
                    spawn_points = world.get_map().get_spawn_points()
                    sp = spawn_points[np.random.randint(len(spawn_points))]
                    vehicle.set_transform(sp)
                    vehicle.apply_control(carla.VehicleControl())
                    print("\n[RESET] Vehicle repositioned", flush=True)

                # Apply manual controls before tick
                if args.manual and not args.attach and not getattr(vehicle, '_autopilot_on', False):
                    cs = web.get_control()
                    control = carla.VehicleControl(
                        throttle=float(cs.get('throttle', 0)),
                        steer=float(cs.get('steer', 0)),
                        brake=float(cs.get('brake', 0)),
                        reverse=bool(cs.get('reverse', False)),
                        hand_brake=bool(cs.get('handbrake', False)),
                    )
                    vehicle.apply_control(control)

                # Advance simulation
                if args.attach:
                    time.sleep(0.033)
                else:
                    world.tick()

                with frame_lock:
                    frame = latest_frame[0]
                    latest_frame[0] = None

                if frame is None:
                    continue

                # Run segmentation and detection in parallel
                img_tensor, shapes = preprocess_frame(frame, args.img_size)
                seg_future = executor.submit(run_segmentation, seg_model, img_tensor, shapes)
                det_future = None
                if det_pipeline is not None:
                    det_future = executor.submit(det_pipeline.process_frame, frame)

                da_seg_mask, ll_seg_mask = seg_future.result()
                detections = det_future.result() if det_future else []

                # Overlay segmentation
                img_vis = overlay_segmentation(
                    frame.copy(), da_seg_mask, ll_seg_mask,
                    output_size=(args.res_x, args.res_y))

                # Draw lane dividers
                if not args.no_lane_split and args.num_lanes > 1:
                    lane_polylines = split_da_into_lanes(
                        da_seg_mask, args.num_lanes,
                        smooth_window=args.smooth_window,
                        cutoff_bottom=args.cutoff_bottom,
                        lane_ratio_threshold=args.lane_ratio,
                        temporal_smoother=lane_smoother,
                        da_accumulator=da_accum)
                    if lane_polylines:
                        vis_h, vis_w = img_vis.shape[:2]
                        da_h, da_w = da_seg_mask.shape
                        sx, sy = vis_w / da_w, vis_h / da_h
                        scaled = [[(int(x * sx), int(y * sy)) for x, y in pl]
                                  for pl in lane_polylines]
                        draw_lane_lines(img_vis, scaled,
                                        color=args.lane_color,
                                        thickness=args.lane_thickness)

                # Draw sign detections
                if detections:
                    img_vis = det_pipeline.draw_detections(img_vis, detections)

                # FPS counter
                processed += 1
                elapsed = time.time() - fps_start
                if elapsed > 1.0:
                    current_fps = processed / elapsed
                    processed = 0
                    fps_start = time.time()

                cur_mode = 'AUTOPILOT' if getattr(vehicle, '_autopilot_on', False) else mode
                cv2.putText(img_vis,
                            f'{current_fps:.0f} FPS | {cur_mode} | {args.weather.upper()} | {args.town}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                web.update(img_vis)
                if vid_writer is not None:
                    vid_writer.write(img_vis)

                print(f"\r{current_fps:.0f} FPS | {cur_mode} | Frame: {frame_count[0]}",
                      end='', flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n[STOP] Cleaning up...")
        camera.stop()
        camera.destroy()
        if not args.attach:
            vehicle.destroy()
            for v in traffic:
                v.destroy()
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        if vid_writer is not None:
            vid_writer.release()
            print(f"[STOP] Video saved to {args.save_video}")
        executor.shutdown(wait=False)
        web.stop()
        print("[STOP] Done.")


if __name__ == '__main__':
    main()
