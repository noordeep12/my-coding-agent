"""Tests for the docstring/signature parsing in tools.py.

function_to_json and its helper parsers convert a Python method into an
OpenAI-compatible tool definition. These are pure string/signature functions —
the highest-value, lowest-cost units to verify.
"""

import inspect

from my_coding_agent.tools import (
    ToolsRegistry,
    _parse_args_section,
    _parse_tags_section,
    _strip_args_section,
    _strip_tags_section,
    function_to_json,
    tool,
)

# --- _parse_tags_section -----------------------------------------------------


def test_parse_tags_section_basic():
    doc = "Do a thing.\n\nTags:\n    shell, bash, run\n"
    assert _parse_tags_section(doc) == ["shell", "bash", "run"]


def test_parse_tags_section_lowercases_and_strips():
    doc = "x.\n\nTags:\n    Shell , BASH ,  Run\n"
    assert _parse_tags_section(doc) == ["shell", "bash", "run"]


def test_parse_tags_section_empty_doc():
    assert _parse_tags_section("") == []


def test_parse_tags_section_no_tags():
    assert _parse_tags_section("Just a description, no tags here.") == []


# --- _parse_args_section -----------------------------------------------------


def test_parse_args_section_single_param():
    doc = "Summary.\n\nArgs:\n    command: Shell command to run.\n"
    assert _parse_args_section(doc) == {"command": "Shell command to run."}


def test_parse_args_section_multiline_continuation():
    doc = (
        "Summary.\n\n"
        "Args:\n"
        "    command: Shell command to run.\n"
        "        Example: 'ls -la'\n"
    )
    assert _parse_args_section(doc) == {
        "command": "Shell command to run. Example: 'ls -la'"
    }


def test_parse_args_section_multiple_params():
    doc = (
        "Summary.\n\n"
        "Args:\n"
        "    task: What to do.\n"
        "    context: Background info.\n"
    )
    assert _parse_args_section(doc) == {
        "task": "What to do.",
        "context": "Background info.",
    }


def test_parse_args_section_empty_doc():
    assert _parse_args_section("") == {}


def test_parse_args_section_no_args():
    assert _parse_args_section("Just a summary line.") == {}


# --- strip helpers -----------------------------------------------------------


def test_strip_tags_section_removes_tags():
    doc = "Description here.\n\nTags:\n    a, b, c\n"
    assert _strip_tags_section(doc) == "Description here."


def test_strip_args_section_removes_args_and_tags():
    doc = (
        "Top description.\n\n"
        "Tags:\n    a, b\n\n"
        "Args:\n    p: a param.\n"
    )
    assert _strip_args_section(doc) == "Top description."


# --- function_to_json --------------------------------------------------------


def test_function_to_json_on_bash_method():
    result = function_to_json(ToolsRegistry.bash)
    assert result["type"] == "function"
    fn = result["function"]
    assert fn["name"] == "bash"
    # self must be excluded; command must be present and required.
    assert "self" not in fn["parameters"]["properties"]
    assert fn["parameters"]["properties"]["command"]["type"] == "string"
    assert fn["parameters"]["required"] == ["command"]
    assert "shell" in result["tags"]
    assert fn["description"]  # non-empty top description


def test_function_to_json_required_excludes_defaulted_params():
    def sample(a: str, b: int = 3):
        """Do it.

        Args:
            a: first.
            b: second.
        """

    result = function_to_json(sample)
    assert result["function"]["parameters"]["required"] == ["a"]


def test_function_to_json_type_mapping():
    def sample(flag: bool, count: int, ratio: float, items: list):
        """Do it."""

    props = function_to_json(sample)["function"]["parameters"]["properties"]
    assert props["flag"]["type"] == "boolean"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["items"]["type"] == "array"


def test_function_to_json_unannotated_defaults_to_string():
    def sample(x):
        """Do it."""

    props = function_to_json(sample)["function"]["parameters"]["properties"]
    assert props["x"]["type"] == "string"


def test_function_to_json_no_docstring():
    def sample(x: str):
        pass

    result = function_to_json(sample)
    assert result["function"]["description"] == ""
    assert result["tags"] == []


def test_tool_is_alias_for_function_to_json():
    def sample(x: str):
        """Do it."""

    assert tool(sample) == function_to_json(sample)


def test_function_to_json_matches_actual_signature():
    """Required params must match the method's real non-default params."""
    result = function_to_json(ToolsRegistry.write_file)
    sig = inspect.signature(ToolsRegistry.write_file)
    expected_required = [
        name
        for name, p in sig.parameters.items()
        if name not in ("self", "cls") and p.default is inspect.Parameter.empty
    ]
    assert result["function"]["parameters"]["required"] == expected_required
