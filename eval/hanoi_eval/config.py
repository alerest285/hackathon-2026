"""Load and check eval/config/hanoi.toml. Unknown keys and bad values are refused with every problem listed."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .hanoi import PLACES, PROMPT_FIELDS, other_place, template_fields


class ConfigError(ValueError):
    """The config file has one or more problems; the message lists them all."""


@dataclass
class AutoEndConfig:
    enabled: bool = True
    signal: str = "state"            # state: measured gripper | cmd: commanded gripper
    closed_is: str = "low"           # which end of the 0..100 gripper range is closed
    closed_threshold: float = 15.0   # at or past this the gripper counts as closed
    open_threshold: float = 25.0     # at or past this it counts as open again
    min_hold_s: float = 0.5
    rest_s: float = 1.0
    rest_speed_deg_s: float = 10.0


@dataclass
class LoopSettings:
    """The fields of so101_client.loop.LoopConfig that come from the config (dry_run comes from the CLI)."""

    mode: str = "prefetch"
    fps: float = 30.0
    period: int = 16
    crossfade: int = 6
    exec_horizon: int = 32
    chunk: int = 42
    step_max_deg: float = 6.0
    spin_margin_ms: float = 6.0
    joint_limits: list | None = None


@dataclass
class RobotSettings:
    port: str = ""
    id: str = "so101_follower"


@dataclass
class CameraSettings:
    scene: int | str = 0
    wrist: int | str = 1
    width: int = 640
    height: int = 480
    fps: int = 30
    image_size: int = 256
    image_format: str = "jpeg"
    jpeg_quality: int = 95


@dataclass
class ServerSettings:
    url: str = "ws://localhost:8000"


@dataclass
class VideoSettings:
    enabled: bool = True
    codec: str = "mp4v"
    fps: float = 30.0


@dataclass
class SanitySettings:
    moves: str = "all"      # all | solution
    repeats: int = 2


@dataclass
class HanoiConfig:
    pieces: list[str] = field(default_factory=lambda: ["large disk", "medium disk", "small disk"])
    places: dict[str, str] = field(
        default_factory=lambda: {"left": "left peg", "middle": "middle peg", "right": "right peg"}
    )
    start: str = "left"
    goal: str = "right"
    prompt: str = "pick up the {piece} and place it on the {place}"
    step_timeout_s: float = 30.0
    mode: str = "eval"                  # eval | demo
    protocol: str = "strict"            # strict | recover
    max_retries: int = 2
    episodes: int = 10
    between_steps: str = "keep_history"  # keep_history | reset_history
    label: str = "zero-shot"
    auto_end: AutoEndConfig = field(default_factory=AutoEndConfig)
    loop: LoopSettings = field(default_factory=LoopSettings)
    robot: RobotSettings = field(default_factory=RobotSettings)
    cameras: CameraSettings = field(default_factory=CameraSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    sanity: SanitySettings = field(default_factory=SanitySettings)

    @property
    def n(self) -> int:
        return len(self.pieces)

    @property
    def aux(self) -> str:
        return other_place(self.start, self.goal, PLACES)

    @property
    def place_order(self) -> tuple[str, ...]:
        return PLACES

    def to_dict(self) -> dict:
        return asdict(self)


SECTIONS = {
    "auto_end": AutoEndConfig,
    "loop": LoopSettings,
    "robot": RobotSettings,
    "cameras": CameraSettings,
    "server": ServerSettings,
    "video": VideoSettings,
    "sanity": SanitySettings,
}

CHOICES = {
    "mode": ("eval", "demo"),
    "protocol": ("strict", "recover"),
    "between_steps": ("keep_history", "reset_history"),
    "auto_end.signal": ("state", "cmd"),
    "auto_end.closed_is": ("low", "high"),
    "loop.mode": ("prefetch", "sync"),
    "cameras.image_format": ("jpeg", "png", "raw"),
    "sanity.moves": ("all", "solution"),
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _build(cls, raw: dict, prefix: str, problems: list[str]):
    """Instantiate a dataclass from a TOML table, collecting unknown keys and type problems."""
    known = {f.name: f for f in fields(cls)}
    defaults = cls()
    kwargs = {}
    for key, value in raw.items():
        name = f"{prefix}{key}"
        if key not in known:
            problems.append(f"unknown key {name!r}")
            continue
        default = getattr(defaults, key)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                problems.append(f"{name} must be true or false, got {value!r}")
                continue
        elif isinstance(default, float):
            if not _is_number(value):
                problems.append(f"{name} must be a number, got {value!r}")
                continue
            value = float(value)
        elif isinstance(default, int) and not isinstance(default, bool) and key not in ("scene", "wrist"):
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"{name} must be an integer, got {value!r}")
                continue
        elif isinstance(default, str) and key not in ("scene", "wrist"):
            if not isinstance(value, str):
                problems.append(f"{name} must be a string, got {value!r}")
                continue
        kwargs[key] = value
    return cls(**kwargs)


def from_dict(raw: dict) -> HanoiConfig:
    """Build and check a config from a parsed TOML document."""
    problems: list[str] = []
    raw = copy.deepcopy(raw)
    top = {k: v for k, v in raw.items() if k not in SECTIONS}
    cfg = _build(HanoiConfig, top, "", problems)
    for name, cls in SECTIONS.items():
        table = raw.get(name, {})
        if not isinstance(table, dict):
            problems.append(f"[{name}] must be a table")
            continue
        setattr(cfg, name, _build(cls, table, f"{name}.", problems))
    problems.extend(check(cfg))
    if problems:
        raise ConfigError("config problems:\n  - " + "\n  - ".join(problems))
    if cfg.loop.joint_limits is not None:
        cfg.loop.joint_limits = [[float(lo), float(hi)] for lo, hi in cfg.loop.joint_limits]
    return cfg


def check(cfg: HanoiConfig) -> list[str]:
    """Value checks across fields. Returns the problems found."""
    problems = []
    if not isinstance(cfg.pieces, list) or not cfg.pieces or not all(isinstance(p, str) and p.strip() for p in cfg.pieces):
        problems.append("pieces must be a non-empty list of names, largest first")
    elif len(set(cfg.pieces)) != len(cfg.pieces):
        problems.append("pieces must have different names")
    if not isinstance(cfg.places, dict) or set(cfg.places) != set(PLACES):
        problems.append(f"[places] must name exactly {list(PLACES)}")
    elif not all(isinstance(v, str) and v.strip() for v in cfg.places.values()):
        problems.append("[places] values must be non-empty strings")
    elif len(set(cfg.places.values())) != len(cfg.places):
        problems.append("[places] values must be different")
    for key in ("start", "goal"):
        if getattr(cfg, key) not in PLACES:
            problems.append(f"{key} must be one of {list(PLACES)}, got {getattr(cfg, key)!r}")
    if cfg.start == cfg.goal:
        problems.append("start and goal must be different places")
    try:
        used = template_fields(cfg.prompt)
    except ValueError as e:
        problems.append(f"prompt is not a valid template: {e}")
    else:
        unknown = [f for f in used if f not in PROMPT_FIELDS]
        if unknown:
            problems.append(f"prompt uses unknown placeholders {unknown}; allowed {list(PROMPT_FIELDS)}")
        for needed in ("piece", "place"):
            if needed not in used:
                problems.append(f"prompt must contain {{{needed}}}")
    for dotted, allowed in CHOICES.items():
        obj = cfg
        *path, last = dotted.split(".")
        for part in path:
            obj = getattr(obj, part)
        if getattr(obj, last) not in allowed:
            problems.append(f"{dotted} must be one of {list(allowed)}, got {getattr(obj, last)!r}")
    if cfg.step_timeout_s <= 0:
        problems.append("step_timeout_s must be > 0")
    if cfg.max_retries < 0:
        problems.append("max_retries must be >= 0")
    if cfg.episodes < 1:
        problems.append("episodes must be >= 1")
    if not cfg.label or any(c in cfg.label for c in "/\\ "):
        problems.append("label must be non-empty without spaces or slashes")
    a = cfg.auto_end
    if a.closed_is == "low" and not a.closed_threshold < a.open_threshold:
        problems.append("auto_end: with closed_is = low, closed_threshold must be below open_threshold")
    if a.closed_is == "high" and not a.closed_threshold > a.open_threshold:
        problems.append("auto_end: with closed_is = high, closed_threshold must be above open_threshold")
    for name in ("closed_threshold", "open_threshold"):
        if not 0.0 <= getattr(a, name) <= 100.0:
            problems.append(f"auto_end.{name} must be within the gripper range 0..100")
    for name in ("min_hold_s", "rest_s", "rest_speed_deg_s"):
        if getattr(a, name) <= 0:
            problems.append(f"auto_end.{name} must be > 0")
    lp = cfg.loop
    if lp.fps <= 0:
        problems.append("loop.fps must be > 0")
    if not 1 <= lp.exec_horizon <= lp.chunk:
        problems.append("loop.exec_horizon must be within 1..loop.chunk")
    if lp.period < 1 or lp.crossfade < 0:
        problems.append("loop.period must be >= 1 and loop.crossfade >= 0")
    if lp.step_max_deg <= 0:
        problems.append("loop.step_max_deg must be > 0")
    if lp.joint_limits is not None:
        ok = (
            isinstance(lp.joint_limits, list)
            and len(lp.joint_limits) == 5
            and all(
                isinstance(pair, list) and len(pair) == 2 and all(_is_number(x) for x in pair) and pair[0] < pair[1]
                for pair in lp.joint_limits
            )
        )
        if not ok:
            problems.append("loop.joint_limits must be 5 pairs [lo, hi] in degrees with lo < hi")
    for name in ("scene", "wrist"):
        value = getattr(cfg.cameras, name)
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            problems.append(f"cameras.{name} must be a camera index or a device path")
    if cfg.cameras.image_size < 16 or not 1 <= cfg.cameras.jpeg_quality <= 100:
        problems.append("cameras.image_size must be >= 16 and cameras.jpeg_quality within 1..100")
    if cfg.video.fps <= 0 or len(cfg.video.codec) != 4:
        problems.append("video.fps must be > 0 and video.codec a fourcc of 4 characters")
    if cfg.sanity.repeats < 1:
        problems.append("sanity.repeats must be >= 1")
    if not cfg.server.url.startswith(("ws://", "wss://")):
        problems.append("server.url must start with ws:// or wss://")
    return problems


def load_config(path: str | Path, overrides: dict | None = None) -> HanoiConfig:
    """Read a TOML config, apply dotted-key overrides (e.g. {"auto_end.rest_s": 0.2}), then check it."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        table = raw
        *path_parts, last = dotted.split(".")
        for part in path_parts:
            table = table.setdefault(part, {})
        table[last] = value
    return from_dict(raw)
