"""so101.v1 wire format (docs/architecture.md, "Wire protocol so101.v1").

One WebSocket per client, binary msgpack frames. Every message is a map with a "type" key: hello, ping,
pong, infer, chunk, error. Arrays travel as {"__nd__": True, "dtype": "<f4" | "|u1", "shape": [...],
"data": <bytes, C order>}. numpy + msgpack only, so the Mac client can import the same codec.
"""

import math
from dataclasses import dataclass

import msgpack
import numpy as np

PROTOCOL = "so101.v1"
CAMERAS = ("scene", "wrist")
STATE_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
IMAGE_FORMATS = ("jpeg", "png", "raw")
N_OBS = 8
DIM = 6
CHUNK = 42
SNAPSHOTS = (0, 7)
MAX_MESSAGE_BYTES = 16 * 2**20
MAX_FRAME_SIDE = 4096
# Fixed-size little-endian numeric dtypes the codec accepts; each field then demands its own dtype.
WIRE_DTYPES = frozenset({"<f2", "<f4", "<f8", "|u1", "|i1", "<u2", "<i2", "<u4", "<i4", "<u8", "<i8", "|b1"})


class ProtocolError(ValueError):
    """A message that does not follow so101.v1. The server answers it with an error message."""


def _default(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.str not in WIRE_DTYPES:
            raise TypeError(f"cannot send array dtype {obj.dtype.str}")
        data = np.ascontiguousarray(obj).tobytes()
        return {"__nd__": True, "dtype": obj.dtype.str, "shape": list(obj.shape), "data": data}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot pack {type(obj).__name__}")


def _hook(obj: dict):
    if obj.get("__nd__") is not True:
        return obj
    dtype, shape, data = obj.get("dtype"), obj.get("shape"), obj.get("data")
    if dtype not in WIRE_DTYPES:
        raise ProtocolError(f"unsupported array dtype {dtype!r}")
    if not isinstance(shape, list) or len(shape) > 8 or not all(is_int(n) and n >= 0 for n in shape):
        raise ProtocolError(f"bad array shape {shape!r}")
    if not isinstance(data, bytes):
        raise ProtocolError("array data must be bytes")
    dt = np.dtype(dtype)
    need = dt.itemsize * math.prod(shape)
    if len(data) != need:
        raise ProtocolError(f"array of shape {shape} {dtype} needs {need} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=dt).reshape(shape)


def packb(msg: dict) -> bytes:
    return msgpack.packb(msg, default=_default, use_bin_type=True)


def unpackb(data) -> dict:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError("expected a binary msgpack frame, got a text frame")
    try:
        msg = msgpack.unpackb(data, object_hook=_hook, raw=False)
    except ProtocolError:
        raise
    except Exception as e:
        raise ProtocolError(f"not a msgpack message: {type(e).__name__}: {e}") from None
    if not isinstance(msg, dict):
        raise ProtocolError(f"a message must be a map, got {type(msg).__name__}")
    return msg


# ---- messages -----------------------------------------------------------------------------------
def hello(**fields) -> dict:
    return {"type": "hello", "protocol": PROTOCOL, **fields}


def ping(t: float) -> dict:
    return {"type": "ping", "t": float(t)}


def pong(t, server_t: float) -> dict:
    return {"type": "pong", "t": t, "server_t": float(server_t)}


def infer(id: int, tick: int, task: str, state, cmd_hist, images: dict, image_format: str) -> dict:
    return {
        "type": "infer",
        "id": int(id),
        "tick": int(tick),
        "task": str(task),
        "state": np.asarray(state, dtype=np.float32),
        "cmd_hist": np.asarray(cmd_hist, dtype=np.float32),
        "images": {cam: list(frames) for cam, frames in images.items()},
        "image_format": image_format,
    }


def chunk(id: int, tick: int, actions, timing: dict) -> dict:
    return {
        "type": "chunk",
        "id": id,
        "tick": tick,
        "actions": np.asarray(actions, dtype=np.float32),
        "timing": {k: round(float(v), 3) for k, v in timing.items()},
    }


def error(id, message: str) -> dict:
    return {"type": "error", "id": id, "error": str(message)}


# ---- infer validation ---------------------------------------------------------------------------
@dataclass
class InferRequest:
    id: int
    tick: int
    task: str
    state: np.ndarray  # (n_obs, 6) float32
    cmd_hist: np.ndarray  # (n_obs, 6) float32
    images: dict  # {camera: [i0, i7]}, bytes for jpeg/png, uint8 (H, W, 3) for raw
    image_format: str


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _window(msg: dict, key: str, n_obs: int, dim: int) -> np.ndarray:
    if key not in msg:
        raise ProtocolError(f"missing field {key!r}")
    value = msg[key]
    if isinstance(value, np.ndarray):
        if value.dtype.str != "<f4":
            raise ProtocolError(f"{key} must be float32 ('<f4'), got {value.dtype.str!r}")
        out = np.array(value, dtype=np.float32)
    else:
        try:
            out = np.array(value, dtype=np.float32)
        except (TypeError, ValueError):
            raise ProtocolError(f"{key} must be a float32 array") from None
    if out.shape != (n_obs, dim):
        raise ProtocolError(f"{key} must have shape ({n_obs}, {dim}), got {tuple(out.shape)}")
    if not np.isfinite(out).all():
        raise ProtocolError(f"{key} has non-finite values")
    return out


def parse_infer(
    msg: dict, *, n_obs: int = N_OBS, dim: int = DIM, cameras=CAMERAS, n_frames: int = len(SNAPSHOTS)
) -> InferRequest:
    """Validate an infer message (fields, shapes, dtypes). Raises ProtocolError with a readable reason."""
    for key in ("id", "tick"):
        if not is_int(msg.get(key)):
            raise ProtocolError(f"{key} must be an integer, got {msg.get(key)!r}")
    task = msg.get("task")
    if not isinstance(task, str):
        raise ProtocolError(f"task must be a string, got {type(task).__name__}")
    state = _window(msg, "state", n_obs, dim)
    cmd_hist = _window(msg, "cmd_hist", n_obs, dim)
    fmt = msg.get("image_format")
    if fmt not in IMAGE_FORMATS:
        raise ProtocolError(f"image_format must be one of {', '.join(IMAGE_FORMATS)}, got {fmt!r}")
    images = msg.get("images")
    if not isinstance(images, dict) or set(images) != set(cameras):
        have = sorted(images) if isinstance(images, dict) else type(images).__name__
        raise ProtocolError(f"images must map exactly {list(cameras)}, got {have}")
    out = {}
    for cam in cameras:
        frames = images[cam]
        if not isinstance(frames, (list, tuple)) or len(frames) != n_frames:
            raise ProtocolError(f"images[{cam!r}] must be a list of {n_frames} frames [i0, i7]")
        for k, frame in enumerate(frames):
            where = f"images[{cam!r}][{k}]"
            if fmt == "raw":
                if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                    raise ProtocolError(f"{where} must be a uint8 array for image_format raw")
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ProtocolError(f"{where} must have shape (H, W, 3), got {tuple(frame.shape)}")
                if not (16 <= frame.shape[0] <= MAX_FRAME_SIDE and 16 <= frame.shape[1] <= MAX_FRAME_SIDE):
                    raise ProtocolError(f"{where} has an unsupported size {frame.shape[:2]}")
            elif not isinstance(frame, bytes) or not frame:
                raise ProtocolError(f"{where} must be non-empty {fmt} bytes")
        out[cam] = list(frames)
    return InferRequest(msg["id"], msg["tick"], task, state, cmd_hist, out, fmt)
