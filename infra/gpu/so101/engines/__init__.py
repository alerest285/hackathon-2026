"""Server engines: flux-action (default), lerobot, fake. Each is imported only when asked for, so the
server and the fake engine run without torch."""

from .base import Engine

NAMES = ("flux-action", "lerobot", "fake")


def make_engine(name: str, **kw) -> Engine:
    if name == "fake":
        from .fake import FakeEngine

        return FakeEngine(**kw)
    if name == "flux-action":
        from .flux_action import FluxActionEngine

        return FluxActionEngine(**kw)
    if name == "lerobot":
        from .lerobot import LeRobotEngine

        return LeRobotEngine(**kw)
    raise ValueError(f"unknown engine {name!r}; choose from {', '.join(NAMES)}")


__all__ = ["NAMES", "Engine", "make_engine"]
