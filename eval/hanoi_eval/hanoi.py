"""Tower of Hanoi: moves, the optimal solution, a legal-move state tracker and the prompt for a move.

Disks are numbered by size, 1 = smallest and n = largest. The config lists piece names largest first,
so disk n is `pieces[0]` and disk 1 is `pieces[-1]`. Places are the logical keys `left`, `middle`, `right`;
the words used in prompts come from the config.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Sequence

PLACES = ("left", "middle", "right")
PROMPT_FIELDS = ("piece", "place", "source")


class IllegalMove(ValueError):
    """A move that breaks the rules from the current state."""


@dataclass(frozen=True)
class Move:
    """Carry disk `disk` from place `src` to place `dst`."""

    disk: int
    src: str
    dst: str

    def to_dict(self) -> dict:
        return {"disk": self.disk, "src": self.src, "dst": self.dst}


def other_place(a: str, b: str, places: Sequence[str] = PLACES) -> str:
    """The third place, given two different ones."""
    rest = [p for p in places if p not in (a, b)]
    if a == b or len(rest) != 1:
        raise ValueError(f"need two different places out of {list(places)}, got {a!r} and {b!r}")
    return rest[0]


def solve(n: int, src: str, dst: str, aux: str) -> list[Move]:
    """The optimal 2**n - 1 moves that carry an n-disk tower from src to dst."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if len({src, dst, aux}) != 3:
        raise ValueError("src, dst and aux must be three different places")
    moves: list[Move] = []

    def carry(k: int, a: str, b: str, c: str) -> None:
        if k == 0:
            return
        carry(k - 1, a, c, b)
        moves.append(Move(k, a, b))
        carry(k - 1, c, b, a)

    carry(n, src, dst, aux)
    return moves


@dataclass(frozen=True)
class TowerState:
    """Disks on each place, bottom to top. Immutable: apply() returns a new state."""

    stacks: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_dict(cls, stacks: dict[str, Sequence[int]]) -> "TowerState":
        state = cls(tuple((place, tuple(int(d) for d in disks)) for place, disks in stacks.items()))
        state.validate()
        return state

    @classmethod
    def tower(cls, n: int, on: str, places: Sequence[str] = PLACES) -> "TowerState":
        """All n disks on one place, largest at the bottom."""
        if on not in places:
            raise ValueError(f"unknown place {on!r}")
        return cls.from_dict({p: (list(range(n, 0, -1)) if p == on else []) for p in places})

    def to_dict(self) -> dict[str, list[int]]:
        return {place: list(disks) for place, disks in self.stacks}

    @property
    def places(self) -> tuple[str, ...]:
        return tuple(place for place, _ in self.stacks)

    @property
    def n(self) -> int:
        return sum(len(disks) for _, disks in self.stacks)

    def disks(self, place: str) -> tuple[int, ...]:
        for p, disks in self.stacks:
            if p == place:
                return disks
        raise KeyError(f"unknown place {place!r}")

    def top(self, place: str) -> int | None:
        disks = self.disks(place)
        return disks[-1] if disks else None

    def validate(self) -> None:
        if len(set(self.places)) != len(self.places):
            raise ValueError("duplicate place")
        seen = sorted(d for _, disks in self.stacks for d in disks)
        if seen != list(range(1, len(seen) + 1)):
            raise ValueError(f"disks must be 1..n, each exactly once, got {seen}")
        for place, disks in self.stacks:
            if any(lower <= upper for lower, upper in zip(disks, disks[1:])):
                raise ValueError(f"{place}: a larger disk sits on a smaller one {list(disks)}")

    def why_illegal(self, move: Move) -> str | None:
        """The reason the move is illegal from this state, or None when it is legal."""
        if move.src not in self.places or move.dst not in self.places:
            return f"unknown place in {move}"
        if move.src == move.dst:
            return "source and destination are the same place"
        top = self.top(move.src)
        if top is None:
            return f"{move.src} is empty"
        if top != move.disk:
            return f"disk {move.disk} is not on top of {move.src} (top is disk {top})"
        under = self.top(move.dst)
        if under is not None and under < move.disk:
            return f"disk {move.disk} cannot go on the smaller disk {under} on {move.dst}"
        return None

    def is_legal(self, move: Move) -> bool:
        return self.why_illegal(move) is None

    def apply(self, move: Move) -> "TowerState":
        """The state after a legal move. Raises IllegalMove otherwise."""
        reason = self.why_illegal(move)
        if reason:
            raise IllegalMove(reason)
        stacks = self.to_dict()
        stacks[move.src].pop()
        stacks[move.dst].append(move.disk)
        return TowerState.from_dict(stacks)

    def legal_moves(self) -> list[Move]:
        moves = []
        for src in self.places:
            disk = self.top(src)
            if disk is None:
                continue
            for dst in self.places:
                move = Move(disk, src, dst)
                if dst != src and self.is_legal(move):
                    moves.append(move)
        return moves

    def is_tower_on(self, place: str) -> bool:
        return self.disks(place) == tuple(range(self.n, 0, -1))

    def describe(self, pieces: Sequence[str], place_names: dict[str, str]) -> str:
        """Human description, bottom to top, e.g. `left peg: large disk, small disk | middle peg: empty`."""
        parts = []
        for place, disks in self.stacks:
            names = ", ".join(piece_name(d, pieces) for d in disks) or "empty"
            parts.append(f"{place_names.get(place, place)}: {names}")
        return " | ".join(parts)


def piece_name(disk: int, pieces: Sequence[str]) -> str:
    """The configured name of a disk. `pieces` is largest first."""
    if not 1 <= disk <= len(pieces):
        raise ValueError(f"disk {disk} out of range for {len(pieces)} pieces")
    return pieces[len(pieces) - disk]


def template_fields(template: str) -> list[str]:
    """Placeholder names used in a prompt template."""
    return [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]


def prompt(move: Move, cfg) -> str:
    """The instruction for one move. `cfg` needs `.prompt`, `.pieces` and `.places` (a HanoiConfig works)."""
    return cfg.prompt.format(
        piece=piece_name(move.disk, cfg.pieces),
        place=cfg.places[move.dst],
        source=cfg.places[move.src],
    )


def all_prompts(cfg) -> list[str]:
    """Every distinct instruction the config can produce, solution order first: every piece onto every place
    (from every other place when the template uses {source}). It covers the solution, the demo's way back and
    the sanity runs, so warming these up on the server leaves no caption to encode mid-episode."""
    moves = solve(len(cfg.pieces), cfg.start, cfg.goal, other_place(cfg.start, cfg.goal))
    moves += [move for move, _ in sanity_moves(len(cfg.pieces), "all", cfg.start, cfg.goal)]
    return list(dict.fromkeys(prompt(move, cfg) for move in moves))


def describe_move(move: Move, cfg) -> str:
    return f"{piece_name(move.disk, cfg.pieces)}: {cfg.places[move.src]} -> {cfg.places[move.dst]}"


def setup_for(move: Move, n: int, places: Sequence[str] = PLACES) -> TowerState:
    """A canonical state from which `move` is legal: the moved disk on top of the larger ones on its
    source, the smaller disks on the third place, the destination empty."""
    aux = other_place(move.src, move.dst, places)
    stacks = {p: [] for p in places}
    stacks[move.src] = list(range(n, move.disk - 1, -1))
    stacks[aux] = list(range(move.disk - 1, 0, -1))
    return TowerState.from_dict(stacks)


def sanity_moves(n: int, which: str, start: str, goal: str, places: Sequence[str] = PLACES) -> list[tuple[Move, TowerState]]:
    """Single moves for the sanity runs, each with the state to set up before it.

    `all`: every disk times every ordered pair of places (n * 6), each from setup_for().
    `solution`: the 2**n - 1 moves of the solution from start to goal, each from its own state in the solution.
    """
    if which == "all":
        out = []
        for disk in range(1, n + 1):
            for src in places:
                for dst in places:
                    if src != dst:
                        move = Move(disk, src, dst)
                        out.append((move, setup_for(move, n, places)))
        return out
    if which == "solution":
        state = TowerState.tower(n, start, places)
        out = []
        for move in solve(n, start, goal, other_place(start, goal, places)):
            out.append((move, state))
            state = state.apply(move)
        return out
    raise ValueError(f"unknown sanity move set {which!r} (all | solution)")
