"""so101.v1 wire format, client side (docs/architecture.md, "Wire protocol so101.v1").

One WebSocket per client, binary msgpack frames, no compression. Arrays travel as maps
{"__nd__": True, "dtype": "<f4" | "|u1", "shape": [...], "data": <bytes, C order>}. Every message carries
"type" (hello, ping, pong, infer, chunk, error). The architecture table names the fields but not the
discriminator key, so message_type() also recognises a message by its fields.
"""

from __future__ import annotations

import logging

import msgpack
import numpy as np

from . import CAMERAS, CHUNK, N_OBS, PROTOCOL, STATE_NAMES

log = logging.getLogger(__name__)

IMAGE_FORMATS = ("jpeg", "png", "raw")


def encode_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    if a.dtype.kind in ("O", "V", "c"):
        raise ValueError(f"unsupported dtype {a.dtype}")
    return {"__nd__": True, "dtype": a.dtype.str, "shape": list(a.shape), "data": a.tobytes()}


def decode_array(obj: dict) -> np.ndarray:
    """Read-only view on the message bytes; copy it before writing."""
    return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


def _default(obj):
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot pack {type(obj).__name__}")


def _hook(obj: dict):
    if obj.get("__nd__") is True:
        return decode_array(obj)
    return obj


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=_default, use_bin_type=True)


def unpack(data: bytes) -> dict:
    msg = msgpack.unpackb(data, object_hook=_hook, raw=False, strict_map_key=False)
    if not isinstance(msg, dict):
        raise ValueError("so101.v1 messages are maps")
    return msg


def message_type(msg: dict) -> str:
    kind = msg.get("type")
    if isinstance(kind, str):
        return kind
    if "protocol" in msg:
        return "hello"
    if "actions" in msg:
        return "chunk"
    if "error" in msg:
        return "error"
    if "server_t" in msg:
        return "pong"
    if "state" in msg and "cmd_hist" in msg:
        return "infer"
    if "t" in msg:
        return "ping"
    return "unknown"


def _image(x, image_format: str):
    if image_format == "raw":
        a = np.asarray(x)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError(f"raw images must be uint8 (H, W, 3) RGB, got {a.dtype} {a.shape}")
        return a
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    raise TypeError(f"{image_format} images must be encoded bytes, got {type(x).__name__}")


def infer_message(rid: int, tick: int, task: str, window) -> dict:
    """The infer request for one Window (history.Window)."""
    if window.image_format not in IMAGE_FORMATS:
        raise ValueError(f"image_format must be one of {IMAGE_FORMATS}")
    state = np.asarray(window.state, np.float32)
    cmd = np.asarray(window.cmd_hist, np.float32)
    if state.shape != (N_OBS, len(STATE_NAMES)) or cmd.shape != state.shape:
        raise ValueError(f"state and cmd_hist must be ({N_OBS}, 6), got {state.shape} and {cmd.shape}")
    images = {}
    for cam, pair in window.images.items():
        if len(pair) != 2:
            raise ValueError(f"camera {cam!r} needs exactly two frames [i0, i7]")
        images[cam] = [_image(x, window.image_format) for x in pair]
    return {
        "type": "infer",
        "id": int(rid),
        "tick": int(tick),
        "task": str(task),
        "state": state,
        "cmd_hist": cmd,
        "images": images,
        "image_format": window.image_format,
    }


def ping_message(t: float) -> dict:
    return {"type": "ping", "t": float(t)}


def pong_message(t, server_t: float) -> dict:
    return {"type": "pong", "t": t, "server_t": float(server_t)}


def parse_actions(msg: dict) -> np.ndarray:
    """A chunk's actions as an owned float32 (n, 6) array; raises on a bad shape or non-finite values."""
    actions = msg.get("actions")
    if actions is None:
        raise ValueError("chunk without actions")
    a = np.array(actions, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:  # (1, 42, 6), as predict_action_chunk returns it
        a = a[0]
    if a.ndim != 2 or a.shape[1] != len(STATE_NAMES) or a.shape[0] < 1:
        raise ValueError(f"actions must be (n, 6), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("actions contain non-finite values")
    return a


def check_hello(hello: dict) -> list[str]:
    """Differences between the server's hello and what this client assumes; empty when they agree."""
    notes = []
    if hello.get("protocol") != PROTOCOL:
        notes.append(f"protocol {hello.get('protocol')!r}, client speaks {PROTOCOL!r}")
    if hello.get("chunk") not in (None, CHUNK):
        notes.append(f"chunk {hello.get('chunk')}, client assumes {CHUNK}")
    if hello.get("n_obs") not in (None, N_OBS):
        notes.append(f"n_obs {hello.get('n_obs')}, client assumes {N_OBS}")
    if hello.get("snapshots") not in (None, [0, N_OBS - 1]):
        notes.append(f"snapshots {hello.get('snapshots')}, client sends [0, {N_OBS - 1}]")
    if hello.get("cameras") not in (None, list(CAMERAS)):
        notes.append(f"cameras {hello.get('cameras')}, client sends {list(CAMERAS)}")
    if hello.get("state_names") not in (None, list(STATE_NAMES)):
        notes.append(f"state_names {hello.get('state_names')}, client sends {list(STATE_NAMES)}")
    return notes
