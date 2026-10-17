"""Hanoi sequencer and episode runner for FLUX 3 Action on the SO-101 (task T07, decisions 05 and 06).

It runs in the same process as robot/so101_client: the runner gives the ControlLoop one instruction per
Hanoi move, ends each step on the operator's key, the grasp-release-rest detector or a timeout, and writes
eval/runs/<UTC stamp>-<label>/ (docs/architecture.md, "eval/ output").
"""

from .config import ConfigError, HanoiConfig, load_config
from .hanoi import IllegalMove, Move, TowerState, prompt, sanity_moves, solve
from .step_end import AutoEndDetector

__all__ = [
    "AutoEndDetector",
    "ConfigError",
    "HanoiConfig",
    "IllegalMove",
    "Move",
    "TowerState",
    "load_config",
    "prompt",
    "sanity_moves",
    "solve",
]
