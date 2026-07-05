"""Tests for the engine-owned per-step resume checkpoint (D3/D4).

Cover the round-trip fidelity (a loaded checkpoint equals what was saved), the
atomic-write guarantee (a torn write never surfaces), ``find_last_resumable``
newest-wins selection, and the clear error on a missing checkpoint.
"""

import json

import pytest

from my_coding_agent.engine.checkpoint import (
    Checkpoint,
    CheckpointError,
    checkpoint_path,
    find_last_resumable,
    load_checkpoint,
    save_checkpoint,
)


def _cp(session_id="s1", step=3, messages=None):
    return Checkpoint(
        session_id=session_id,
        step_num=step,
        last_prompt_tokens=1234,
        messages=messages
        or [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
        ],
    )


def test_save_then_load_matches_exactly(tmp_path):
    sdir = tmp_path / "s1"
    cp = _cp()
    save_checkpoint(sdir, cp)
    loaded = load_checkpoint(sdir)
    assert loaded.session_id == cp.session_id
    assert loaded.step_num == cp.step_num
    assert loaded.last_prompt_tokens == cp.last_prompt_tokens
    assert loaded.messages == cp.messages


def test_checkpoint_matches_conversation_state_at_end_of_step(tmp_path):
    # After "step N" the checkpoint holds exactly the messages known at end of N.
    sdir = tmp_path / "s1"
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    save_checkpoint(sdir, _cp(step=5, messages=messages))
    assert load_checkpoint(sdir).messages == messages
    assert load_checkpoint(sdir).step_num == 5


def test_save_is_atomic_no_temp_left_and_valid_json(tmp_path):
    sdir = tmp_path / "s1"
    save_checkpoint(sdir, _cp())
    final = checkpoint_path(sdir)
    assert final.exists()
    # No temp file left behind, and the final file is complete valid JSON.
    assert not list(sdir.glob("*.tmp"))
    json.loads(final.read_text())  # does not raise → not torn


def test_overwrite_replaces_previous_checkpoint(tmp_path):
    sdir = tmp_path / "s1"
    save_checkpoint(sdir, _cp(step=1))
    save_checkpoint(sdir, _cp(step=2))
    assert load_checkpoint(sdir).step_num == 2
    assert not list(sdir.glob("*.tmp"))


def test_load_missing_raises_checkpoint_error(tmp_path):
    with pytest.raises(CheckpointError, match="No checkpoint"):
        load_checkpoint(tmp_path / "nope")


def test_load_unreadable_raises_checkpoint_error(tmp_path):
    sdir = tmp_path / "s1"
    sdir.mkdir()
    checkpoint_path(sdir).write_text("{ not json")
    with pytest.raises(CheckpointError, match="Unreadable"):
        load_checkpoint(sdir)


def test_find_last_resumable_picks_newest(tmp_path):
    import os
    import time

    older = tmp_path / "old"
    newer = tmp_path / "new"
    save_checkpoint(older, _cp(session_id="old"))
    save_checkpoint(newer, _cp(session_id="new"))
    # Force a clearly newer mtime on the second checkpoint.
    now = time.time()
    os.utime(checkpoint_path(older), (now - 100, now - 100))
    os.utime(checkpoint_path(newer), (now, now))
    assert find_last_resumable(tmp_path) == "new"


def test_find_last_resumable_skips_dirs_without_checkpoint(tmp_path):
    (tmp_path / "no_cp").mkdir()
    save_checkpoint(tmp_path / "has_cp", _cp(session_id="has_cp"))
    assert find_last_resumable(tmp_path) == "has_cp"


def test_find_last_resumable_none_when_empty(tmp_path):
    assert find_last_resumable(tmp_path) is None
    assert find_last_resumable(tmp_path / "missing") is None
