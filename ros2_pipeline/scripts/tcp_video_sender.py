"""Send video frames over TCP to the driving pipeline, simulating Simulink.

Usage:
    python tcp_video_sender.py <video_path> [--host HOST] [--port PORT] [--fps FPS]
"""

import argparse
import socket
import time
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Send video frames via TCP')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--fps', type=float, default=0, help='Playback FPS (0 = original)')
    parser.add_argument('--loop', action='store_true', help='Loop video')
    parser.add_argument('--resize', default='1280x720', help='WxH to send')
    args = parser.parse_args()

    w, h = map(int, args.resize.split('x'))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = args.fps if args.fps > 0 else orig_fps
    frame_interval = 1.0 / target_fps

    print(f"Video: {args.video}")
    print(f"Frames: {total}, Original FPS: {orig_fps:.1f}, Target FPS: {target_fps:.1f}")
    print(f"Sending {w}x{h} frames to {args.host}:{args.port}")

    # Connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)

    print(f"Connecting to {args.host}:{args.port}...")
    sock.connect((args.host, args.port))
    print("Connected! Sending frames...")

    sent = 0
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            t0 = time.time()

            # Resize to expected dimensions
            frame = cv2.resize(frame, (w, h))

            # Convert to RGB and Fortran-order bytes (matching Simulink format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = frame_rgb.flatten(order='F').tobytes()

            sock.sendall(data)
            sent += 1

            # Rate limit
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if sent % 30 == 0:
                wall = time.time() - start
                print(f"\r  Sent {sent}/{total} frames | {sent/wall:.1f} FPS", end='', flush=True)

    except (BrokenPipeError, ConnectionResetError):
        print(f"\nConnection closed by receiver after {sent} frames")
    except KeyboardInterrupt:
        print(f"\nStopped after {sent} frames")
    finally:
        sock.close()
        cap.release()

    wall = time.time() - start
    print(f"\nDone: {sent} frames in {wall:.1f}s ({sent/wall:.1f} FPS)")


if __name__ == '__main__':
    main()
