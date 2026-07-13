"""Identity checks for exceptions relocated to their raising domains (issue #180)."""

from my_coding_agent import APIResponseError as root_api_error
from my_coding_agent import ToolDefinitionError as root_tool_def_error
from my_coding_agent.engine.llm.errors import APIResponseError as llm_api_error
from my_coding_agent.engine.tool_registry.exceptions import (
    ToolDefinitionError as registry_tool_def_error,
)
from my_coding_agent.utils.exceptions import APIResponseError as utils_api_error
from my_coding_agent.utils.exceptions import ToolDefinitionError as utils_tool_def_error


def test_tool_definition_error_same_class_across_import_paths():
    assert utils_tool_def_error is registry_tool_def_error
    assert root_tool_def_error is registry_tool_def_error


def test_api_response_error_same_class_across_import_paths():
    assert utils_api_error is llm_api_error
    assert root_api_error is llm_api_error


def test_tool_definition_error_is_value_error():
    assert issubclass(registry_tool_def_error, ValueError)


def test_api_response_error_is_value_error():
    assert issubclass(llm_api_error, ValueError)
