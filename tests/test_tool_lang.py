"""Tests for the tool-result language-hint resolver (``tool_execution.lang``).

Locks the heuristic that produces ``metadata.lang = {command, output, error}``:
tool name drives the command box, a ``read_file`` extension (else a JSON sniff)
drives the output box, and the error box is always ``text``. Unknown content
falls back to ``text``.
"""

from my_coding_agent.engine.tool_execution.envelope import build_tool_result
from my_coding_agent.engine.tool_execution.lang import resolve_lang


def _env(output: str = "", error: str | None = None):
    return build_tool_result("t", True, output, error)


def test_bash_command_is_shell():
    lang = resolve_lang("bash", {"command": "ls -la"}, _env("total 0"))
    assert lang["command"] == "shell"


def test_non_bash_command_is_json_when_args_is_dict():
    lang = resolve_lang("read_file", {"file_path": "/x/a.txt"}, _env("hello"))
    assert lang["command"] == "json"


def test_read_py_output_is_python():
    lang = resolve_lang("read_file", {"file_path": "/x/main.py"}, _env("import os\n"))
    assert lang["output"] == "python"


def test_read_json_file_output_is_json_by_extension():
    lang = resolve_lang("read_file", {"file_path": "/x/data.json"}, _env("not json"))
    assert lang["output"] == "json"


def test_json_looking_output_is_json_by_sniff():
    lang = resolve_lang("bash", {"command": "cat x"}, _env('{"a": 1}'))
    assert lang["output"] == "json"


def test_plain_output_falls_back_to_text():
    lang = resolve_lang("bash", {"command": "echo hi"}, _env("hi there"))
    assert lang["output"] == "text"


def test_unknown_extension_falls_back_to_text():
    lang = resolve_lang("read_file", {"file_path": "/x/notes.md"}, _env("# title"))
    assert lang["output"] == "text"


def test_read_without_file_path_degrades_to_sniff_then_text():
    # No file_path: JSON-looking output still sniffs to json; plain text stays text.
    assert resolve_lang("read_file", {}, _env("[1, 2]"))["output"] == "json"
    assert resolve_lang("read_file", {}, _env("plain"))["output"] == "text"


def test_error_box_is_always_text():
    lang = resolve_lang("bash", {"command": "false"}, _env("", "boom"))
    assert lang["error"] == "text"


def test_hint_values_are_within_the_closed_set():
    lang = resolve_lang("read_file", {"file_path": "/x/a.py"}, _env('{"k": 1}', "e"))
    assert set(lang) == {"command", "output", "error"}
    assert set(lang.values()) <= {"python", "shell", "json", "text"}
