from pathlib import Path

import pytest

from hanoi_eval.config import ConfigError, from_dict, load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "hanoi.toml"


def write(tmp_path, text):
    path = tmp_path / "hanoi.toml"
    path.write_text(text)
    return path


def test_shipped_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.pieces == ["large disk", "medium disk", "small disk"]
    assert cfg.places == {"left": "left peg", "middle": "middle peg", "right": "right peg"}
    assert (cfg.start, cfg.goal, cfg.aux) == ("left", "right", "middle")
    assert cfg.step_timeout_s == 30.0
    assert cfg.mode == "eval" and cfg.protocol == "strict" and cfg.max_retries == 2
    assert cfg.episodes == 10 and cfg.between_steps == "keep_history"
    assert cfg.auto_end.rest_s == 1.0 and cfg.auto_end.closed_is == "low"
    assert cfg.loop.mode == "prefetch" and cfg.loop.period == 16 and cfg.loop.crossfade == 6
    assert cfg.loop.joint_limits is None


def test_defaults_fill_missing_keys(tmp_path):
    cfg = load_config(write(tmp_path, 'pieces = ["big", "small"]\n'))
    assert cfg.n == 2
    assert cfg.prompt == "pick up the {piece} and place it on the {place}"
    assert cfg.auto_end.min_hold_s == 0.5


def test_overrides_use_dotted_keys():
    cfg = load_config(CONFIG, {"episodes": 3, "auto_end.rest_s": 0.25, "loop.mode": "sync", "label": None})
    assert cfg.episodes == 3 and cfg.auto_end.rest_s == 0.25 and cfg.loop.mode == "sync"
    assert cfg.label == "zero-shot"


def test_unknown_keys_are_refused(tmp_path):
    with pytest.raises(ConfigError, match="unknown key 'episode'"):
        load_config(write(tmp_path, "episode = 3\n"))
    with pytest.raises(ConfigError, match="unknown key 'auto_end.closed_treshold'"):
        load_config(write(tmp_path, "[auto_end]\nclosed_treshold = 3.0\n"))


def test_bad_choices_and_types_are_refused(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path, 'mode = "show"\nprotocol = "lenient"\nepisodes = "ten"\n'))
    message = str(e.value)
    assert "mode must be one of" in message
    assert "protocol must be one of" in message
    assert "episodes must be an integer" in message


def test_thresholds_must_match_the_closed_direction():
    with pytest.raises(ConfigError, match="closed_is = low"):
        from_dict({"auto_end": {"closed_is": "low", "closed_threshold": 40.0, "open_threshold": 20.0}})
    cfg = from_dict({"auto_end": {"closed_is": "high", "closed_threshold": 80.0, "open_threshold": 60.0}})
    assert cfg.auto_end.closed_is == "high"
    with pytest.raises(ConfigError, match="closed_is = high"):
        from_dict({"auto_end": {"closed_is": "high", "closed_threshold": 20.0, "open_threshold": 60.0}})


def test_prompt_template_is_checked():
    with pytest.raises(ConfigError, match=r"must contain \{place\}"):
        from_dict({"prompt": "pick up the {piece}"})
    with pytest.raises(ConfigError, match="unknown placeholders"):
        from_dict({"prompt": "pick up the {piece} and place it on the {place} {colour}"})


def test_places_and_endpoints_are_checked():
    with pytest.raises(ConfigError, match="start and goal must be different"):
        from_dict({"start": "left", "goal": "left"})
    with pytest.raises(ConfigError, match=r"\[places\] must name exactly"):
        from_dict({"places": {"left": "a", "right": "b"}})
    with pytest.raises(ConfigError, match="goal must be one of"):
        from_dict({"goal": "centre"})


def test_joint_limits_are_checked():
    limits = [[-100, 100]] * 5
    assert from_dict({"loop": {"joint_limits": limits}}).loop.joint_limits == [[-100.0, 100.0]] * 5
    with pytest.raises(ConfigError, match="joint_limits"):
        from_dict({"loop": {"joint_limits": [[-100, 100]] * 4}})


def test_all_problems_are_listed_at_once():
    with pytest.raises(ConfigError) as e:
        from_dict({"start": "up", "step_timeout_s": -1.0, "pieces": []})
    message = str(e.value)
    assert "start must be one of" in message
    assert "step_timeout_s must be > 0" in message
    assert "pieces must be a non-empty list" in message


def test_empty_threshold_lies_beyond_closed_in_the_closed_direction():
    assert load_config(CONFIG).auto_end.empty_threshold == 3.0
    assert from_dict({}).auto_end.empty_threshold is None  # absent: no miss check
    assert from_dict({"auto_end": {"empty_threshold": 2}}).auto_end.empty_threshold == 2.0
    with pytest.raises(ConfigError, match="empty_threshold must be below closed_threshold"):
        from_dict({"auto_end": {"empty_threshold": 20.0}})
    high = {"closed_is": "high", "closed_threshold": 85.0, "open_threshold": 75.0}
    assert from_dict({"auto_end": {**high, "empty_threshold": 97.0}}).auto_end.empty_threshold == 97.0
    with pytest.raises(ConfigError, match="empty_threshold must be above closed_threshold"):
        from_dict({"auto_end": {**high, "empty_threshold": 3.0}})
    with pytest.raises(ConfigError, match="empty_threshold must be a number"):
        from_dict({"auto_end": {"empty_threshold": "low"}})
