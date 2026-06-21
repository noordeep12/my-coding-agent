"""Tests for the pipeline trace-tree reconstruction (tree.py) and its capture.

Drive a Recorder through representative runs (router events, tool calls, an LLM
call with reasoning+content, and a delegated subagent) and assert build_trace_tree
produces the expected hierarchical-by-step shape, creators, and metadata. All
against tmp_path; no live LLM.
"""

from my_coding_agent.observability import build_trace_tree, reader
from my_coding_agent.observability.recorder import Recorder


def _resp(content="", reasoning="", tool_calls=None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls or [],
                }
            }
        ]
    }


def _full_step(session_dir, *, parent=None):
    """One agent run: router → main call (reasoning+content) → bash tool."""
    rec = Recorder("main1", session_dir, parent_session_id=parent)
    rec.start("Main Agent", "local-model", 1000)
    rec.record_router("list files", ["bash", "read_file"], "phase1_keyword")
    rec.record_llm_call(
        kind="main",
        call=1,
        latency_s=2.0,
        usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        messages=[
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "list files"},
        ],
        context_window=1000,
        response_data=_resp(content="I'll run ls", reasoning="thinking"),
    )
    rec.before_tool("bash", {"command": "ls"})
    rec.after_tool("bash", {"command": "ls"}, "a.py")
    rec.finish("stop", 1, 5.0)
    return rec


def _build(session_dir):
    session = reader.load_session(session_dir)
    return build_trace_tree(session, {session.session_id: session})


# --- structure ----------------------------------------------------------------


def test_tree_root_is_agent(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    assert root.type == "agent"
    assert root.title == "Agent: Main Agent"
    assert root.metadata["status"] == "success"
    assert root.node_id == "0"


def test_initial_messages_are_children(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    types = [c.type for c in root.children]
    assert types[:2] == ["system_message", "user_message"]


def test_step_has_ordered_children(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    types = [c.type for c in step.children]
    assert types == ["context_manager", "tool_router", "llm_call", "tool_executor"]


def test_llm_call_holds_messages_in_output(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    llm = next(c for c in step.children if c.type == "llm_call")
    assert llm.children == []  # AI messages are folded into output, not children
    assert llm.title == "LLM.chat_completion"  # real class.method executed
    # Output is the raw assistant response message from the server.
    assert llm.metadata["output"]["content"] == "I'll run ls"
    assert llm.metadata["output"]["reasoning_content"] == "thinking"


def test_tool_router_metadata_has_selected(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    router = next(c for c in step.children if c.type == "tool_router")
    assert router.title == "ToolRouter.route_tools"
    assert router.metadata["output"] == ["bash", "read_file"]  # selected subset
    assert router.metadata["phase"] == "phase1_keyword"


def test_router_llm_fallback_nested_as_llm_call(tmp_path):
    rec = Recorder("m", tmp_path)
    rec.start("Main Agent", "local-model", 1000)
    rec.record_router("run", ["bash"], "phase2_llm")  # LLM fallback
    rec.record_llm_call(
        "tool_router",
        1,
        0.2,
        {"prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33},
        [{"role": "user", "content": "pick a tool"}],
        1000,
        _resp(content="bash"),
    )
    rec.record_llm_call(
        "main",
        2,
        1.0,
        {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        [{"role": "user", "content": "run"}],
        1000,
        _resp(content="ok"),
    )
    rec.finish("stop", 1, 1.0)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    router = next(c for c in step.children if c.type == "tool_router")
    assert router.metadata["used_llm_fallback"] is True
    # the fallback LLM call is reflected as a full LLM.chat_completion node
    llm = router.children[0]
    assert llm.type == "llm_call"
    assert llm.title == "LLM.chat_completion"
    assert llm.metadata["input"] == [{"role": "user", "content": "pick a tool"}]
    assert llm.metadata["total_tokens"] == 33
    assert llm.metadata["ctx"]["added"] == 0  # routing doesn't grow the agent window


def test_tool_executor_with_io(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    tool = next(c for c in step.children if c.type == "tool_executor")
    assert tool.title == "ToolExecutor.invoke_tool: bash"  # real class.method + tool
    assert tool.metadata["input"] == {"command": "ls"}
    assert tool.metadata["output"] == "a.py"
    assert tool.metadata["status"] == "success"
    assert tool.metadata["ctx"]["agent_label"] == "Main Agent"


def test_tool_node_flags_structured_failure(tmp_path):
    # bash returns failure as data (no exception); the node must show the error logo.
    rec = Recorder("m", tmp_path)
    rec.start("Main Agent", "local-model", 1000)
    rec.record_llm_call(
        "main",
        1,
        1.0,
        {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        [{"role": "user", "content": "run"}],
        1000,
        _resp(content="ok"),
    )
    rec.before_tool("bash", {"command": "false"})
    rec.after_tool(
        "bash",
        {"command": "false"},
        '{"stdout": "", "stderr": "boom", "exit_code": 1, "ok": false}',
    )
    rec.finish("stop", 1, 1.0)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    tool = next(c for c in step.children if c.type == "tool_executor")
    assert tool.metadata["status"] == "failure"


def test_context_bar_anchors_and_deltas(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    # ContextManager history is re-anchored to the main call's exact prompt tokens.
    cm = next(c for c in step.children if c.type == "context_manager")
    assert cm.metadata["ctx"]["history"] == 100
    assert cm.metadata["ctx"]["window"] == 1000
    # LLM call adds exactly its completion tokens (green, not estimated).
    llm = next(c for c in step.children if c.type == "llm_call")
    assert llm.metadata["ctx"]["added"] == 10
    assert llm.metadata["ctx"]["estimated"] is False
    # Tool result is a length estimate (green, estimated).
    tool = next(c for c in step.children if c.type == "tool_executor")
    assert tool.metadata["ctx"]["added"] > 0
    assert tool.metadata["ctx"]["estimated"] is True


def test_every_node_has_a_status(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    statuses = []

    def walk(n):
        statuses.append(n.metadata.get("status"))
        for c in n.children:
            walk(c)

    walk(root)
    # Normalized: every node (incl. ContextManager / ToolRouter / messages) has one.
    assert statuses  # non-empty
    assert all(s in ("success", "failure", "warning") for s in statuses)


def test_handoff_node_removes_context(tmp_path):
    rec = Recorder("m", tmp_path)
    rec.start("Main Agent", "local-model", 1000)
    rec.record_router("go", ["bash"], "phase1_keyword")
    rec.record_llm_call(
        "main",
        1,
        1.0,
        {"prompt_tokens": 800, "completion_tokens": 5, "total_tokens": 805},
        [{"role": "user", "content": "go"}],
        1000,
        _resp(content="x"),
    )
    rec.record_handoff(1, 800, 80.0, "summary of progress", "/tmp/h.md")
    rec.finish("context_reset", 1, 1.0)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    handoff = [c for c in step.children if c.type == "context_manager"][-1]
    assert handoff.title == "Agent._handle_context_reset"
    assert handoff.metadata["status"] == "warning"
    assert handoff.metadata["ctx"]["removed"] == 800


def test_node_ids_are_unique(tmp_path):
    _full_step(tmp_path)
    root = _build(tmp_path)
    ids = []

    def walk(n):
        ids.append(n.node_id)
        for c in n.children:
            walk(c)

    walk(root)
    assert len(ids) == len(set(ids))


# --- validation node (artifact summarization buffered before the tool call) ---


def test_tool_output_validation_nested_under_tool(tmp_path):
    rec = Recorder("m", tmp_path)
    rec.start("Main Agent", "local-model", 1000)
    rec.record_router("run", ["bash"], "phase1_keyword")
    rec.record_llm_call(
        "main",
        1,
        1.0,
        {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        [{"role": "user", "content": "run"}],
        1000,
        _resp(content="ok"),
    )
    # summarizer call fires during dispatch, before the tool_call event
    rec.record_llm_call(
        "tool_output_summarizer",
        2,
        0.3,
        {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        [],
        1000,
        _resp(content="summary"),
    )
    rec.before_tool("bash", {"command": "big"})
    rec.after_tool("bash", {"command": "big"}, "huge output")
    rec.finish("stop", 1, 1.0)
    root = _build(tmp_path)
    step = next(c for c in root.children if c.type == "step")
    tool = next(c for c in step.children if c.type == "tool_executor")
    val = tool.children[0]
    assert val.type == "tool_output_validation"
    assert val.metadata["function"] == "_summarize_artifact"
    # the summarizer is a real LLM.chat_completion call, nested + reflected
    llm = val.children[0]
    assert llm.type == "llm_call"
    assert llm.title == "LLM.chat_completion"
    assert llm.metadata["output"]["content"] == "summary"
    assert llm.metadata["total_tokens"] == 24
    # side-call: runs on its own conversation, so it adds nothing to the window
    assert llm.metadata["ctx"]["added"] == 0


# --- delegate subagent nesting ------------------------------------------------


def _delegating_parent(root_dir):
    """Parent that delegates to a child, plus the child's own session."""
    parent_dir = root_dir / "main1"
    child_dir = root_dir / "child1"
    # child session first
    child = Recorder("child1", child_dir, parent_session_id="main1")
    child.start("SubAgent", "local-model", 1000)
    child.record_router("explore", ["read_file"], "phase1_keyword")
    child.record_llm_call(
        "main",
        1,
        1.0,
        {"prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33},
        [{"role": "user", "content": "explore"}],
        1000,
        _resp(content="found it"),
    )
    child.finish("stop", 1, 1.0)
    # parent with a delegate tool call linked to the child
    parent = Recorder("main1", parent_dir)
    parent.start("Main Agent", "local-model", 1000)
    parent.record_router("delegate it", ["delegate"], "phase2_llm")
    parent.record_llm_call(
        "main",
        1,
        1.0,
        {"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44},
        [{"role": "user", "content": "delegate it"}],
        1000,
        _resp(content="delegating"),
    )
    parent.before_tool("delegate", {"task": "explore", "context": "x"})
    parent.note_delegate_child("child1")
    parent.after_tool("delegate", {"task": "explore", "context": "x"}, "found it")
    parent.finish("stop", 1, 1.0)


def test_delegate_child_nested_under_tool_call(tmp_path):
    root_dir = tmp_path / ".my_coding_agent"
    root_dir.mkdir()
    _delegating_parent(root_dir)
    by_id = reader.load_sessions_by_id(root_dir)
    parent = by_id["main1"]
    tree = build_trace_tree(parent, by_id)

    step = next(c for c in tree.children if c.type == "step")
    delegate_call = next(c for c in step.children if c.type == "tool_executor")
    assert delegate_call.title == "ToolExecutor.invoke_tool: delegate"
    assert delegate_call.metadata["name"] == "delegate"
    assert delegate_call.metadata["child_session_id"] == "child1"
    child_agent = delegate_call.children[-1]
    assert child_agent.type == "agent"
    assert child_agent.title == "Agent: SubAgent"


def test_delegate_child_only_top_level_session(tmp_path):
    """The child is nested, not also listed as a top-level session."""
    root_dir = tmp_path / ".my_coding_agent"
    root_dir.mkdir()
    _delegating_parent(root_dir)
    by_id = reader.load_sessions_by_id(root_dir)
    roots = [
        s
        for s in by_id.values()
        if not s.parent_session_id or s.parent_session_id not in by_id
    ]
    assert [s.session_id for s in roots] == ["main1"]


def test_delegate_child_session_id_recorded_on_event(tmp_path):
    """note_delegate_child attaches child_session_id to the delegate tool event."""
    rec = Recorder("p", tmp_path)
    rec.before_tool("delegate", {"task": "t", "context": "c"})
    rec.note_delegate_child("kid42")
    rec.after_tool("delegate", {"task": "t", "context": "c"}, "report")
    session = reader.load_session(tmp_path)
    assert session.tool_calls[0].child_session_id == "kid42"
