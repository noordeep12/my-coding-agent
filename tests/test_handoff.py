"""Tests for ContextHandoff — context-window state transfer.

These cover the percentage computation (including the zero-window edge),
the continuation seed message, and persistence to an isolated tmp_path
workspace so no real .my_coding_agent/ directory is touched.
"""

from my_coding_agent.handoff import ContextHandoff


def _handoff(**overrides):
    base = dict(
        agent_label="Main Agent",
        step_num=3,
        prompt_tokens=750,
        context_window=1000,
        content="progress so far",
    )
    base.update(overrides)
    return ContextHandoff(**base)


# --- context_pct -------------------------------------------------------------


def test_context_pct_normal():
    assert _handoff(prompt_tokens=750, context_window=1000).context_pct == 75.0


def test_context_pct_zero_window_returns_zero():
    # Guard against division by zero.
    assert _handoff(context_window=0).context_pct == 0.0


# --- to_user_message ---------------------------------------------------------


def test_to_user_message_role_and_content():
    msg = _handoff(agent_label="Main Agent", step_num=3).to_user_message()
    assert msg["role"] == "user"
    assert "[Context Reset — Main Agent, step 3, 75.0% context used]" in msg["content"]
    assert "progress so far" in msg["content"]


# --- save --------------------------------------------------------------------


def test_save_writes_file_under_workspace(tmp_path):
    h = _handoff()
    path = h.save(workspace=str(tmp_path))
    out = tmp_path / ".my_coding_agent" / "handoffs"
    written = list(out.glob("*.md"))
    assert len(written) == 1
    assert str(written[0]) == path
    assert h.path == path


def test_save_filename_slugifies_label_and_pads_step(tmp_path):
    h = _handoff(agent_label="Main Agent", step_num=7)
    path = h.save(workspace=str(tmp_path))
    assert "main_agent_step007_" in path


def test_save_content_includes_metrics_and_body(tmp_path):
    h = _handoff(prompt_tokens=750, context_window=1000, content="the body text")
    path = h.save(workspace=str(tmp_path))
    text = (
        tmp_path / ".my_coding_agent" / "handoffs" / path.split("/")[-1]
    ).read_text()
    assert "# Context Handoff" in text
    assert "750 / 1,000 (75.0%)" in text
    assert "the body text" in text
