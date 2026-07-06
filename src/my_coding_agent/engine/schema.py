"""Engine event type constants — session lifecycle and LLM/tool/handoff call tags."""

SESSION_START = "session_start"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
HANDOFF = "handoff"
REPORT = "report"
SESSION_END = "session_end"

# Which of the three cost-divergent paths produced a subagent's report
# (report-cost-provenance D1): verbatim hand-back costs zero LLM calls,
# summarizer/fallback each cost exactly one `report`-kind LLM call.
REPORT_SOURCE_VERBATIM = "verbatim"
REPORT_SOURCE_SUMMARIZER = "summarizer"
REPORT_SOURCE_FALLBACK = "fallback"

# Run-resilience (D6): an unrecoverable LLM failure ends the run with a
# classified stop reason of the form ``llm_failure_<classification>`` (the
# hyphen in a classification becomes an underscore for a clean identifier),
# so the CLI can recognize a resumable failure without a raw traceback.
STOP_REASON_LLM_FAILURE_PREFIX = "llm_failure_"


def llm_failure_stop_reason(classification: str) -> str:
    """Return the ``stop_reason`` for an unrecoverable LLM failure class."""
    return STOP_REASON_LLM_FAILURE_PREFIX + classification.replace("-", "_")
