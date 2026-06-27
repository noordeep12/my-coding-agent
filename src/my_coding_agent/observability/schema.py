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
