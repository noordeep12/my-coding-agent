"""Tests for the self-contained HTML viewer generator (report.py).

Exercise the payload build, TreeNode serialization, safe ``</script>`` escaping,
file output, and the CLI entry point — all against tmp_path with no browser.
"""

from my_coding_agent.observability import report
from my_coding_agent.observability.events import TreeNode
from my_coding_agent.observability.recorder import Recorder


def _resp(content="", reasoning=""):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": [],
                }
            }
        ]
    }


def _seed(root_dir, label="Main Agent", sid="sess1"):
    """Write one representative session under ``root_dir``."""
    rec = Recorder(sid, root_dir / sid)
    rec.start(label, "local-model", 1000)
    rec.record_router("list files", ["bash"], "phase1_keyword")
    rec.record_llm_call(
        "main",
        1,
        2.0,
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        1000,
        _resp(content="running ls", reasoning="thinking"),
    )
    rec.before_tool("bash", {"command": "ls"})
    rec.after_tool("bash", {"command": "ls"}, "a.py")
    rec.finish("stop", 1, 5.0)


# --- TreeNode serialization ---------------------------------------------------


def test_tree_node_to_dict_roundtrip():
    root = TreeNode(type="agent", title="Agent: X", node_id="0")
    child = root.add(TreeNode(type="step", title="Step 1"))
    child.add(TreeNode(type="llm_call", title="LLM", metadata={"total_tokens": 5}))
    d = root.to_dict()
    assert d["type"] == "agent"
    assert "creator" not in d  # the harness-managed model has no per-node creator
    assert d["children"][0]["title"] == "Step 1"
    assert d["children"][0]["children"][0]["metadata"]["total_tokens"] == 5


# --- payload ------------------------------------------------------------------


def test_build_payload_lists_top_level_session(tmp_path):
    _seed(tmp_path)
    payload = report.build_payload(tmp_path)
    assert len(payload["sessions"]) == 1
    s = payload["sessions"][0]
    assert s["session_id"] == "sess1"
    assert s["ok"] is True
    assert s["tree"]["type"] == "agent"


def test_build_payload_empty_when_no_sessions(tmp_path):
    assert report.build_payload(tmp_path / "missing") == {"sessions": []}


def _seed_rich(root_dir, sid="rich"):
    """Two snapshot-bearing calls + a repeated bash call (loop + diff present)."""
    rec = Recorder(sid, root_dir / sid)
    rec.start("Main Agent", "local-model", 1000)
    rec.record_llm_call(
        "main",
        1,
        2.0,
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
        1000,
        _resp(content="running ls"),
    )
    rec.before_tool("bash", {"command": "ls"})
    rec.after_tool("bash", {"command": "ls"}, "a.py")
    rec.record_llm_call(
        "main",
        2,
        3.0,
        {"prompt_tokens": 250, "completion_tokens": 20, "total_tokens": 270},
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "running ls"},
        ],
        1000,
        _resp(content="running ls again"),
    )
    rec.before_tool("bash", {"command": "ls"})
    rec.after_tool("bash", {"command": "ls"}, "a.py")
    rec.finish("stop", 2, 8.0)


def test_build_payload_includes_summary_chips(tmp_path):
    _seed_rich(tmp_path)
    s = report.build_payload(tmp_path)["sessions"][0]
    assert set(s["summary"]) == {"total_tokens", "est_cost_usd", "failures"}
    assert s["summary"]["total_tokens"] == 380  # 110 + 270
    assert s["summary"]["est_cost_usd"] == 0.0  # local model is free
    assert s["summary"]["failures"] == 0


def test_build_payload_includes_analytics_views(tmp_path):
    _seed_rich(tmp_path)
    a = report.build_payload(tmp_path)["sessions"][0]["analytics"]
    assert a["context_series"]["prompt_tokens"] == [100, 250]
    assert any(r["step"] == "bash" and r["calls"] == 2 for r in a["bottlenecks"])
    loop = next(f for f in a["loops"] if f["kind"] == "tool")
    assert loop["count"] == 2  # bash(ls) called twice


# --- rendering ----------------------------------------------------------------


def test_render_html_embeds_data_and_is_self_contained(tmp_path):
    _seed(tmp_path)
    html = report.render_html(report.build_payload(tmp_path))
    assert html.startswith("<!doctype html>")
    assert "window.__OBS__ =" in html
    assert "__OBSERVABILITY_DATA__" not in html  # token was substituted
    assert "http://" not in html and "https://" not in html  # no external refs


def test_render_html_embeds_analytics_and_overview_panel(tmp_path):
    _seed_rich(tmp_path)
    html = report.render_html(report.build_payload(tmp_path))
    assert "Session Overview" in html  # synthetic overview node + renderer
    assert "renderOverview" in html
    for key in ("context_series", "bottlenecks", "loops", "summary"):
        assert key in html  # derived views embedded in the payload JSON


def test_render_html_escapes_script_close():
    payload = {"sessions": [{"label": "x</script><b>", "tree": {}}]}
    html = report.render_html(payload)
    # the raw closer must not appear inside the embedded data
    body = html.split("window.__OBS__ =")[1]
    assert "</script><b>" not in body.split("</script>")[0]
    assert "<\\/script>" in html


# --- file + CLI ---------------------------------------------------------------


def test_write_report_creates_file(tmp_path):
    _seed(tmp_path)
    out = report.write_report(tmp_path, tmp_path / "viewer.html")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_write_report_default_path(tmp_path):
    _seed(tmp_path)
    out = report.write_report(tmp_path)
    assert out == tmp_path / "viewer.html"
    assert out.exists()


def test_main_writes_without_opening_browser(tmp_path, monkeypatch):
    _seed(tmp_path)
    opened = []
    monkeypatch.setattr(report.webbrowser, "open", lambda url: opened.append(url))
    out = report.main(
        ["--root", str(tmp_path), "--out", str(tmp_path / "v.html"), "--no-open"]
    )
    assert out.exists()
    assert opened == []  # --no-open suppressed the browser


def test_main_opens_browser_by_default(tmp_path, monkeypatch):
    _seed(tmp_path)
    opened = []
    monkeypatch.setattr(report.webbrowser, "open", lambda url: opened.append(url))
    report.main(["--root", str(tmp_path), "--out", str(tmp_path / "v.html")])
    assert len(opened) == 1
    assert opened[0].startswith("file://")
