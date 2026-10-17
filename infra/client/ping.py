# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client",
# ]
# ///
"""Milestone 2 check: ask the served DROID policy for action chunks from any laptop.

    infra/aws/ssh.sh -N &            # forwards localhost:8000 to the GPU
    uv run infra/client/ping.py --n 5 [--npz observation.npz] [--prompt "..."]

Without --npz it sends black frames and zero joints: enough to test the pipe and latency,
not to judge behaviour. Request format: docs/architecture.md (served interface).
Status: written against BFL's docs, not yet run against a live server (task T05).
"""
import argparse, time
import numpy as np
from openpi_client import websocket_client_policy

p = argparse.ArgumentParser()
p.add_argument("--host", default="localhost"); p.add_argument("--port", type=int, default=8000)
p.add_argument("--n", type=int, default=3)
p.add_argument("--npz", help="DROID observation from make_observation.py (images.wrist/left/right, state)")
p.add_argument("--prompt", default="put the marker in the cup")
a = p.parse_args()

if a.npz:
    z = np.load(a.npz)
    obs = {"observation/wrist_image_left": z["images.wrist"], "observation/exterior_image_1_left": z["images.left"],
           "observation/exterior_image_2_left": z["images.right"],
           "observation/joint_position": z["state"][:7].astype(np.float32),
           "observation/gripper_position": np.float32(z["state"][7])}
else:
    obs = {"observation/image": np.zeros((540, 640, 3), np.uint8),
           "observation/joint_position": np.zeros(7, np.float32), "observation/gripper_position": np.float32(0)}
obs["prompt"] = a.prompt

policy = websocket_client_policy.WebsocketClientPolicy(host=a.host, port=a.port)
print("server metadata:", policy.get_server_metadata())
for i in range(a.n):
    t = time.perf_counter(); r = policy.infer(obs); dt = (time.perf_counter() - t) * 1000
    act = np.asarray(r["action"])
    print(f"{i}: {dt:7.1f} ms round trip  action {act.shape}  server_timing {r.get('server_timing')}")
assert act.shape == (32, 8), f"unexpected action shape {act.shape}"
print("PASS")
