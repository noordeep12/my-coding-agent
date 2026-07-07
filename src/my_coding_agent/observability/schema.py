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

# Keys of a "refusal" row (``Recorder.record_refusal``) and of the
# ``metadata.refusal`` structure carried on a refused tool call's envelope, kept
# here so consumers (e.g. the viewer) can reference the contract without
# importing the recorder or tool_execution modules directly.
REFUSAL_RULE_ID = "rule_id"
REFUSAL_REASON = "reason"
REFUSAL_REFERENCES = "references"
REFUSAL_REFERENCE_STANDARD_ID = "standard_id"
REFUSAL_REFERENCE_URL = "url"
REFUSAL_SAFER_ALTERNATIVE = "safer_alternative"
REFUSAL_TOOL_NAME = "tool_name"
REFUSAL_COMMAND = "command"
REFUSAL_STEP = "step"

# Keys of an "egress" row (``Recorder.record_egress``) and of the
# ``metadata.egress`` structure carried on a blocked ``fetch_web`` call's
# envelope, kept here so consumers (e.g. the viewer) can reference the
# contract without importing the recorder or engine.egress modules directly.
EGRESS_HOST = "host"
EGRESS_MATCHED_LIST = "matched_list"
EGRESS_REASON = "reason"
EGRESS_TOOL_NAME = "tool_name"
EGRESS_STEP = "step"

# First-layer/limitation note carried on every refusal envelope (issue #130):
# the gate is a high-signal textual screen, not a complete boundary — it can
# be evaded by obfuscation (base64, ${IFS}, eval). OS-level enforcement (the
# #25 sandbox) is what covers what text-matching cannot; this key exists so
# the model/operator never infers completeness from a refusal alone.
REFUSAL_POSTURE_NOTE = "posture_note"
POSTURE_NOTE_TEXT = (
    "This gate is a high-signal textual screen, not a complete boundary — "
    "obfuscated commands can evade it. OS-level enforcement (sandbox), when "
    "active, is what covers what text-matching cannot."
)

# Keys of the run-level "protection posture" recorded once per session on the
# session_start row (issue #130): whether the run had OS-level enforcement
# (the #25 sandbox) or only this textual gate. Absent on pre-change traces.
POSTURE = "posture"
POSTURE_SANDBOXED = "sandboxed"
POSTURE_SCREENED_ONLY = "screened_only"


# Keys of a "skill_index" row (``Recorder.record_skill_index``, issue #19),
# emitted once per session start / continuation when a skill index is placed
# into the opening user message. Absent entirely from a skill-free run.
SKILL_INDEX_NAMES = "names"
SKILL_INDEX_COUNT = "count"
SKILL_INDEX_CHARS = "chars"
SKILL_INDEX_TIER = "tier"

# Keys of a machine-wide "resources" object, optionally attached to timed
# events (``llm_call``, ``tool_call``, ``summarizer``) and to the run-level
# rollup in ``session_data.json`` (node-resource-monitoring). Absent means
# "not captured" — legacy rows and hosts without a working sampler simply
# omit the key.
RESOURCES = "resources"
RESOURCES_MACHINE_WIDE = "machine_wide"
RESOURCES_SAMPLE_COUNT = "sample_count"
# Gauge resources: {"avg": float, "peak": float} over the window.
RESOURCES_RAM_PCT = "ram_pct"
RESOURCES_CPU_PCT = "cpu_pct"
RESOURCES_GPU_PCT = "gpu_pct"
RESOURCES_GAUGE_AVG = "avg"
RESOURCES_GAUGE_PEAK = "peak"
# Counter resources: byte-count delta across the window.
RESOURCES_NET_BYTES = "net_bytes"
RESOURCES_DISK_BYTES = "disk_bytes"
