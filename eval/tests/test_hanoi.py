from pathlib import Path

import pytest

from hanoi_eval.config import load_config
from hanoi_eval.hanoi import (
    IllegalMove,
    Move,
    TowerState,
    other_place,
    piece_name,
    prompt,
    sanity_moves,
    setup_for,
    solve,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "hanoi.toml"


def test_three_disks_take_seven_legal_moves_to_the_goal():
    moves = solve(3, "left", "right", "middle")
    assert len(moves) == 7
    state = TowerState.tower(3, "left")
    for move in moves:
        assert state.is_legal(move), move
        state = state.apply(move)
    assert state.is_tower_on("right")
    assert state.to_dict() == {"left": [], "middle": [], "right": [3, 2, 1]}


@pytest.mark.parametrize("n", range(1, 7))
def test_solution_length_is_two_to_the_n_minus_one(n):
    moves = solve(n, "left", "right", "middle")
    assert len(moves) == 2**n - 1
    state = TowerState.tower(n, "left")
    for move in moves:
        state = state.apply(move)
    assert state.is_tower_on("right")


def test_prompts_come_from_the_template():
    cfg = load_config(CONFIG)
    moves = solve(cfg.n, cfg.start, cfg.goal, cfg.aux)
    texts = [prompt(m, cfg) for m in moves]
    assert texts[0] == "pick up the small disk and place it on the right peg"
    assert texts[1] == "pick up the medium disk and place it on the middle peg"
    assert texts[3] == "pick up the large disk and place it on the right peg"
    assert texts[6] == "pick up the small disk and place it on the right peg"


def test_prompt_can_name_the_source_place():
    cfg = load_config(CONFIG, {"prompt": "move the {piece} from the {source} to the {place}"})
    assert prompt(Move(2, "left", "middle"), cfg) == "move the medium disk from the left peg to the middle peg"


def test_piece_names_are_largest_first():
    pieces = ["large red disk", "medium green disk", "small blue disk"]
    assert piece_name(3, pieces) == "large red disk"
    assert piece_name(1, pieces) == "small blue disk"
    with pytest.raises(ValueError):
        piece_name(4, pieces)


def test_illegal_moves_raise():
    state = TowerState.tower(3, "left")
    with pytest.raises(IllegalMove, match="not on top"):
        state.apply(Move(3, "left", "right"))
    with pytest.raises(IllegalMove, match="empty"):
        state.apply(Move(1, "middle", "right"))
    with pytest.raises(IllegalMove, match="same place"):
        state.apply(Move(1, "left", "left"))
    state = state.apply(Move(1, "left", "right"))
    with pytest.raises(IllegalMove, match="smaller disk"):
        state.apply(Move(2, "left", "right"))


def test_apply_returns_a_new_state():
    state = TowerState.tower(3, "left")
    after = state.apply(Move(1, "left", "middle"))
    assert state.to_dict()["left"] == [3, 2, 1]
    assert after.to_dict() == {"left": [3, 2], "middle": [1], "right": []}


def test_bad_states_are_refused():
    with pytest.raises(ValueError, match="larger disk"):
        TowerState.from_dict({"left": [1, 2], "middle": [3], "right": []})
    with pytest.raises(ValueError, match="exactly once"):
        TowerState.from_dict({"left": [3, 2], "middle": [2], "right": []})


def test_legal_moves_from_the_start():
    moves = TowerState.tower(3, "left").legal_moves()
    assert moves == [Move(1, "left", "middle"), Move(1, "left", "right")]


def test_sanity_all_covers_every_piece_and_ordered_pair():
    items = sanity_moves(3, "all", "left", "right")
    assert len(items) == 18
    assert len({m for m, _ in items}) == 18
    for move, setup in items:
        assert setup.is_legal(move), (move, setup)
    move, setup = items[-1]
    assert move.disk == 3
    assert setup.disks(move.src) == (3,)
    assert setup.disks(other_place(move.src, move.dst)) == (2, 1)


def test_sanity_solution_follows_the_solution_states():
    items = sanity_moves(3, "solution", "left", "right")
    assert [m for m, _ in items] == solve(3, "left", "right", "middle")
    state = TowerState.tower(3, "left")
    for move, setup in items:
        assert setup == state
        state = state.apply(move)


def test_setup_for_puts_smaller_disks_aside():
    setup = setup_for(Move(2, "middle", "left"), 3)
    assert setup.to_dict() == {"left": [], "middle": [3, 2], "right": [1]}


def test_describe_uses_configured_names():
    cfg = load_config(CONFIG)
    text = TowerState.tower(3, "left").describe(cfg.pieces, cfg.places)
    assert text == "left peg: large disk, medium disk, small disk | middle peg: empty | right peg: empty"
