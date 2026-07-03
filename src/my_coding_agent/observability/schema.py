"""Observability record shape — the JSONL event row structure.

Every event row written to ``events.jsonl`` shares these top-level keys.
Module-specific event types are defined in their own schema.py files:
- engine/schema.py  — SESSION_START, LLM_CALL, TOOL_CALL, HANDOFF, SESSION_END
- pipeline/schema.py — ROUTER
"""

# Required key present in every JSONL event row.
EVENT_TYPE = "type"

# Optional timing key present in most event rows.
EVENT_STARTED_AT = "started_at"

# Keys of an "anomaly" row (``Recorder.record_anomaly``), kept here so
# consumers (e.g. the viewer) can reference the contract without importing
# the recorder module itself.
ANOMALY_KIND = "kind"
ANOMALY_STREAK_ID = "streak_id"
ANOMALY_SIGNATURE = "signature"
ANOMALY_TOOL_NAME = "tool_name"
ANOMALY_STREAK_LEN = "streak_len"
ANOMALY_TOKENS_SPENT = "tokens_spent"
ANOMALY_STEP = "step"

# The one detector kind implemented in this change; the field is an open
# taxonomy so future detectors (exact-repeat loops, cost growth) add new
# values without a schema change.
ANOMALY_KIND_FAILURE_STREAK = "failure_streak"
