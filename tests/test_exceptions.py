"""Tests for the custom exception hierarchy (CONTRIBUTE.md §29)."""

import pytest

from my_coding_agent import (
    APIResponseError,
    MyCodingAgentError,
    PathTraversalError,
    ToolDefinitionError,
)
from my_coding_agent.tool_registry import ToolRegistry as ToolsRegistry
from my_coding_agent.tool_registry import function_to_json


@pytest.mark.parametrize(
    "subclass",
    [PathTraversalError, ToolDefinitionError, APIResponseError],
)
def test_base_catches_subclasses(subclass):
    """Every subclass is catchable via the package base exception."""
    with pytest.raises(MyCodingAgentError):
        raise subclass("boom")


@pytest.mark.parametrize(
    "subclass",
    [PathTraversalError, ToolDefinitionError, APIResponseError],
)
def test_subclasses_are_value_errors(subclass):
    """Subclasses remain ValueErrors so recoverable-error handling is unchanged."""
    assert issubclass(subclass, ValueError)


def test_hint_stored_and_accessible():
    """The hint keyword is stored on the exception and defaults to None."""
    err = MyCodingAgentError("msg", hint="try this")
    assert str(err) == "msg"
    assert err.hint == "try this"
    assert MyCodingAgentError("msg").hint is None


def test_path_traversal_raise_site(tmp_path):
    """_resolve_in_base raises PathTraversalError with a workspace hint."""
    reg = ToolsRegistry(base_dir=str(tmp_path))
    with pytest.raises(PathTraversalError, match="Path traversal detected") as exc:
        reg.read_file("../../etc/passwd")
    assert exc.value.hint == "Use a path inside the workspace."


def test_tool_definition_raise_site(mocker):
    """function_to_json raises ToolDefinitionError when a signature is uninspectable."""

    def fn():
        pass  # pragma: no cover

    mocker.patch(
        "my_coding_agent.tool_registry.converter.inspect.signature",
        side_effect=ValueError("no signature"),
    )
    with pytest.raises(ToolDefinitionError, match="Failed to get signature"):
        function_to_json(fn)
