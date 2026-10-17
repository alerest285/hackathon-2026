"""Round trip to the so101.v1 server: n pings, then N synthetic infer requests (fake frames, rest pose).

    infra/aws/ssh.sh -N &        # forwards localhost:8000 to the GPU host
    uv run --python 3.13 python ping.py --server ws://localhost:8000 --n 20 --infer 20

Prints the ping round trip, the infer round trip, the server's own total, and the network share
(round trip minus server total). The first infer of a cold server includes caption encoding.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from so101_client import CAMERAS, N_OBS, protocol
from so101_client.cameras import encode_image
from so101_client.fakes import REST_POSE, FakeCameras
from so101_client.history import Window
from so101_client.net import NetClient


def pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def synthetic_window(image_format: str, image_size: int) -> Window:
    pattern = FakeCameras(width=640, height=480)
    images = {cam: [encode_image(pattern.pattern(i, s), image_size, image_format) for s in (0.0, 7 / 30)]
              for i, cam in enumerate(CAMERAS)}
    state = np.tile(np.asarray(REST_POSE, np.float32), (N_OBS, 1))
    return Window(state=state, cmd_hist=state.copy(), images=images, image_format=image_format,
                  frame_t={cam: time.perf_counter() for cam in CAMERAS})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--server", default="ws://localhost:8000")
    p.add_argument("--n", type=int, default=20, help="pings")
    p.add_argument("--infer", type=int, default=10, help="synthetic infer requests")
    p.add_argument("--task", default="pick up the small red disk and place it on the right peg")
    p.add_argument("--image-format", choices=("jpeg", "png", "raw"), default="jpeg")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for one reply")
    a = p.parse_args(argv)

    net = NetClient(a.server)
    hello = net.connect()
    print(f"connected to {a.server}")
    for key in ("protocol", "engine", "checkpoint", "revision", "chunk", "exec", "fps", "guidance", "steps", "warm"):
        print(f"  {key:10s} {hello.get(key)}")
    ok = True
    try:
        if a.n > 0:
            rtts = net.ping(a.n)
            print(f"\nping x{a.n}: p50 {pct(rtts, 50):.1f} ms, p95 {pct(rtts, 95):.1f} ms, max {max(rtts):.1f} ms")
        rtt, server = [], []
        if a.infer > 0:
            window = synthetic_window(a.image_format, a.image_size)
            size = len(protocol.pack(protocol.infer_message(0, 0, a.task, window)))
            print(f"\ninfer x{a.infer}: {a.image_format} {a.image_size}x{a.image_size}, request {size / 1024:.1f} KiB")
            for i in range(a.infer):
                rid = net.submit(window, a.task, i)
                deadline = time.perf_counter() + a.timeout
                reply = None
                while reply is None and time.perf_counter() < deadline:
                    reply = next((r for r in net.poll() if r.id == rid), None)
                    if reply is None:
                        time.sleep(0.001)
                if reply is None:
                    print(f"  {i}: no reply within {a.timeout:.0f} s")
                    ok = False
                    break
                if reply.error:
                    print(f"  {i}: error {reply.error}")
                    ok = False
                    continue
                shape_ok = reply.actions.shape == (42, 6) and np.isfinite(reply.actions).all()
                ok &= shape_ok
                r_ms = (reply.recv_t - reply.sent_t) * 1000
                s_ms = float(reply.timing.get("total_ms", float("nan")))
                rtt.append(r_ms)
                server.append(s_ms)
                print(f"  {i}: round trip {r_ms:7.1f} ms, server {s_ms:7.1f} ms, network {r_ms - s_ms:6.1f} ms, "
                      f"actions {reply.actions.shape}{'' if shape_ok else ' (unexpected)'}, timing {reply.timing}")
        if rtt:
            steady = rtt[1:] or rtt
            steady_server = server[1:] or server
            net_ms = [r - s for r, s in zip(steady, steady_server)]
            print(f"\nround trip p50 {pct(steady, 50):.1f} ms, p95 {pct(steady, 95):.1f} ms (first request excluded)")
            print(f"server total p50 {pct(steady_server, 50):.1f} ms, p95 {pct(steady_server, 95):.1f} ms")
            print(f"network p50 {pct(net_ms, 50):.1f} ms, p95 {pct(net_ms, 95):.1f} ms, "
                  f"share {pct(net_ms, 50) / max(pct(steady, 50), 1e-9):.0%} of the round trip")
    finally:
        net.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
