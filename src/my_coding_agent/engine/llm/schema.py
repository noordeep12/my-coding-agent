"""LLM request/response shape — constants for call kinds and usage fields."""

# Recognized chat-completion call kinds (tagged per call for token accounting).
CALL_KIND_MAIN = "main"
CALL_KIND_HANDOFF = "handoff"
CALL_KIND_REPORT = "report"
CALL_KIND_TOOL_ROUTER = "tool_router"
CALL_KIND_TOOL_OUTPUT_SUMMARIZER = "tool_output_summarizer"
CALL_KIND_TOOL_ARG_CORRECTION = "tool_arg_correction"
CALL_KIND_ARTIFACT_QUERY = "artifact_query"

# Usage dict keys returned by the server.
USAGE_PROMPT = "prompt_tokens"
USAGE_COMPLETION = "completion_tokens"
USAGE_TOTAL = "total_tokens"
