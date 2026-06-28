"""Tool registry shape — the OpenAI tool definition JSON structure."""

# Keys in an OpenAI-compatible tool definition object.
TOOL_DEF_TYPE = "type"  # always "function"
TOOL_DEF_FUNCTION = "function"  # nested object with name/description/parameters
TOOL_DEF_TAGS = "tags"  # list of keyword strings for phase-1 routing

# Keys inside the nested "function" object.
FUNC_NAME = "name"
FUNC_DESCRIPTION = "description"
FUNC_PARAMETERS = "parameters"

# Keys inside the "parameters" object.
PARAMS_TYPE = "type"  # always "object"
PARAMS_PROPERTIES = "properties"
PARAMS_REQUIRED = "required"
