"""Mac-side SO-101 client for the so101.v1 policy server (docs/architecture.md, "SO-101 path").

Cameras and the Feetech bus are read at 30 Hz, the last 8 ticks form the window the model reads, chunks are
fetched before the current one runs out and stitched in joint-delta space. The hardware classes import
LeRobot and OpenCV lazily, so the pure parts (history, chunks, protocol) and the fakes work without them.
"""

PROTOCOL = "so101.v1"
STATE_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("scene", "wrist")
N_OBS = 8
CHUNK = 42
EXEC = 32
FPS = 30

__all__ = ["CAMERAS", "CHUNK", "EXEC", "FPS", "N_OBS", "PROTOCOL", "STATE_NAMES"]
