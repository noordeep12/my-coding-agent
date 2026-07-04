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
