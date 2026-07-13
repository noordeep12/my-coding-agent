"""LLM request/response shape — constants for call kinds and usage fields."""

# Recognized chat-completion call kinds (tagged per call for token accounting).
CALL_KIND_MAIN = "main"
CALL_KIND_HANDOFF = "handoff"
CALL_KIND_REPORT = "report"
CALL_KIND_TOOL_ARG_CORRECTION = "tool_arg_correction"
CALL_KIND_ARTIFACT_QUERY = "artifact_query"
CALL_KIND_JUDGE = "judge"

# Usage dict keys returned by the server.
USAGE_PROMPT = "prompt_tokens"
USAGE_COMPLETION = "completion_tokens"
USAGE_TOTAL = "total_tokens"

# ── Failure classification (run-resilience D1) ────────────────────────────────
# Every chat-completion failure is classified as exactly one of these at the
# single choke point (``LLM.chat_completion``). ``transport`` and ``http-status``
# 5xx/429 are retryable; ``malformed-body`` and other 4xx are not.
CLASSIFICATION_TRANSPORT = "transport"
CLASSIFICATION_HTTP_STATUS = "http-status"
CLASSIFICATION_MALFORMED_BODY = "malformed-body"

# Resilience tuning (run-resilience D2). Fast retries are the existing transport
# window inside ``_request_with_retry``; the patient phase probes with capped
# exponential backoff until the server answers or the outage tolerance is hit.
DEFAULT_OUTAGE_TOLERANCE_S = 300.0  # 5 min — server-restart / model-reload scale
PATIENT_BACKOFF_CAP_S = 30.0  # ceiling on a single patient-phase wait
OUTAGE_TOLERANCE_ENV = "MCA_LLM_OUTAGE_TOLERANCE_S"
