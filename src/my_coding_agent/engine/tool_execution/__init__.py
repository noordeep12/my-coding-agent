"""Tool-call dispatch for one assistant message.

Defines ``ToolExecutor``, constructed per message: it parses and validates each
raw tool call, applies argument aliases and strips unknown kwargs, dispatches
through the ``ToolRegistry``, and offloads oversized outputs: each is written to
a per-artifact file on disk and replaced in the result by a bounded preview (an
excerpt plus skim guidance). It makes no LLM calls itself; the LLM client is held
only for the session log path and the observability recorder.
"""

import contextvars
import hashlib
import inspect
import json
import re
import subprocess
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ...observability import current_session_id
from ...observability.recorder import _now
from ...observability.schema import (
    EGRESS_HOST,
    EGRESS_MATCHED_LIST,
    EGRESS_REASON,
    EXFIL_CATEGORY,
    EXFIL_TOOL_NAME,
    HOOK_NAME,
    HOOK_OUTCOME_BLOCKED,
    HOOK_OUTCOME_FIRED,
    HOOK_REASON,
    POSTURE_NOTE_TEXT,
    PROVENANCE_KIND_MARK,
    PROVENANCE_KIND_REDUCTION_REFUSAL,
    REFUSAL_POSTURE_NOTE,
    REFUSAL_REASON,
    REFUSAL_REFERENCE_STANDARD_ID,
    REFUSAL_REFERENCE_URL,
    REFUSAL_REFERENCES,
    REFUSAL_RULE_ID,
    REFUSAL_SAFER_ALTERNATIVE,
)
from ...utils import get_logger
from .. import egress, exfil, provenance
from ..hooks import Hooks
from ..hooks.schema import (
    BLOCKED_BY_HOOK_REASON,
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    HookContext,
    HookResult,
    HookSpec,
)
from . import args as arg_prep
from . import policy
from .concurrency import is_parallel_safe, max_tool_concurrency
from .envelope import (
    build_tool_result,
    result_envelope,
    validate_tool_result,
)
from .lang import resolve_lang
from .output import (
    MAX_TOOL_OUTPUT_CHARS,
    PREVIEW_MAX_CHARS,
    build_stream_preview,
    validate_tool_output,
)
from .records import call_record, error_record
from .schema import TOOL_SCHEMA_VERSION

if TYPE_CHECKING:
    from ..llm import LLM

__all__ = [
    "ToolExecutor",
    "ToolRegistry",
    "MAX_TOOL_OUTPUT_CHARS",
    "TOOL_SCHEMA_VERSION",
    "build_tool_result",
    "validate_tool_result",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve ``ToolRegistry`` so it stays part of this module's public
    surface (``__all__``) without an eager import — see ``ToolExecutor.__init__``
    for why that import must be deferred (breaks a cycle with tool_registry)."""
    if name == "ToolRegistry":
        from ..tool_registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Exceptions a tool may raise that are surfaced as an ``ok:false`` result rather
# than re-raised. Anything not in this tuple hard-stops the agent loop.
_RECOVERABLE_EXCEPTIONS = (
    TypeError,  # wrong arg names / types
    ValueError,  # bad arg values
    FileNotFoundError,  # wrong path
    json.JSONDecodeError,  # malformed tool arguments
    subprocess.TimeoutExpired,  # belt-and-suspenders (bash catches this itself)
)

# Data contract, envelope builders, output post-processing, and argument prep
# live in the sibling modules schema / envelope / output / args; the executor
# below composes them. The envelope builders (build/validate/normalize) and the
# truncation limit (MAX_TOOL_OUTPUT_CHARS) are imported above.

# Matches a stored per-stream artifact filename, as written by
# ``_write_artifact_file`` / read by ``artifact_file_path`` (tool_registry).
_ARTIFACT_FILENAME_RE = re.compile(
    r"^(?P<tool_call_id>[A-Za-z0-9_-]+)\.(?P<stream>stdout|stderr)\.txt$"
)


def _iter_stored_artifacts(
    session_id: str | None,
) -> list[tuple[str, str, str]]:
    """Return ``(tool_call_id, stream, content)`` for every artifact on disk
    in this run's session, newest first. Empty when there is no session or
    artifacts directory yet. No LLM, no caching beyond the OS page cache —
    the run's own artifact count and size are small and bounded.
    """
    from ..tool_registry import artifact_file_path  # lazy: avoids a cycle

    # A placeholder id purely to resolve the artifacts directory shared with
    # the write side; it is never itself read or written here.
    marker = artifact_file_path(session_id, "_dir_marker")
    if marker is None:
        return []
    directory = marker.parent
    if not directory.is_dir():
        return []
    files = sorted(
        directory.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    artifacts = []
    for f in files:
        match = _ARTIFACT_FILENAME_RE.match(f.name)
        if not match:
            continue
        try:
            content = f.read_text()
        except OSError:
            continue
        artifacts.append((match["tool_call_id"], match["stream"], content))
    return artifacts


def _find_duplicate(session_id: str | None, text: str) -> dict[str, Any] | None:
    """Return a duplicate-of descriptor if ``text`` matches an already-stored
    artifact, or ``None``. Exact hash match first (byte-identical read-back),
    then containment (``text`` a contiguous substring of a stored artifact of
    equal or larger size, newest first) — needed because ``bash`` rstrips its
    streams, so a read-back of an artifact with a trailing newline differs by
    exactly that stripped whitespace. Deterministic, no LLM call.
    """
    if not text:
        return None
    stored = _iter_stored_artifacts(session_id)
    if not stored:
        return None
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    for tool_call_id, stream, content in stored:
        if hashlib.sha256(content.encode()).hexdigest() == text_hash:
            return {
                "tool_call_id": tool_call_id,
                "stream": stream,
                "offset": 0,
                "length": len(text),
            }
    for tool_call_id, stream, content in stored:
        if len(content) < len(text):
            continue
        offset = content.find(text)
        if offset != -1:
            return {
                "tool_call_id": tool_call_id,
                "stream": stream,
                "offset": offset,
                "length": len(text),
            }
    return None


def _build_duplicate_notice(duplicate: dict[str, Any], path: str | None) -> str:
    """Build the agent-facing pointer for a deduplicated stream — no excerpt,
    just enough to retrieve the exact bytes deterministically."""
    location = f" (on disk at {path})" if path else ""
    return (
        f"[This output is already stored — it duplicates tool_call_id="
        f"'{duplicate['tool_call_id']}' {duplicate['stream']}{location}, "
        f"at byte offset {duplicate['offset']}, length {duplicate['length']}. "
        "No new artifact was created. Retrieve the exact bytes with "
        f'read_tool_artifact(tool_call_id="{duplicate["tool_call_id"]}", '
        f"start={duplicate['offset']}, length={duplicate['length']}).]"
    )


# One parsed tool call, before dispatch. ``error`` is set (and ``func_name`` /
# ``args`` may be partial) only when parsing failed; otherwise both are present.
_PreparedCall = namedtuple("_PreparedCall", "tool_call_id func_name args error")


def _plan_groups(prepared: list["_PreparedCall"]) -> list[list["_PreparedCall"]]:
    """Partition parsed calls into ordered execution groups.

    A maximal run of *contiguous* parallel-safe calls becomes one group (the
    executor overlaps it); every other call — a parse error, or one whose
    effects cannot be proven read-only — is its own singleton group and runs
    inline in sequence. Group order matches call order, so a non-overlappable
    call acts as a barrier: nothing after it starts until it finishes, and it
    starts only once everything before it has finished. This preserves the exact
    observable ordering of the sequential path for every call that is not itself
    part of a read-only overlap.
    """
    groups: list[list[_PreparedCall]] = []
    run: list[_PreparedCall] = []
    for item in prepared:
        safe = item.error is None and is_parallel_safe(item.func_name, item.args)
        if safe:
            run.append(item)
            continue
        if run:
            groups.append(run)
            run = []
        groups.append([item])
    if run:
        groups.append(run)
    return groups


class ToolExecutor:
    """Dispatch the tool calls in one assistant message.

    Constructed per message: holds that message's ``tool_calls`` plus the running
    ``tool_messages`` / ``tool_records`` it fills and the ``tool_artifacts`` it
    offloads. It owns no LLM calls — the LLM client is kept only for the session
    log path and the observability recorder (``llm._recorder``).

    The agent's available ``tools`` are forwarded to the ``ToolRegistry`` so
    toolset-aware tools (notably ``delegate``, which spawns a subagent with the
    parent toolset minus ``delegate``) can read them. Omitting ``tools`` leaves
    the registry with an empty toolset. The ``llm`` client is forwarded the same
    way so ``read_tool_artifact`` can make its bounded extraction call.
    """

    def __init__(
        self,
        message: dict[str, Any],
        llm: "LLM",
        tools: list[dict[str, Any]] | None = None,
        skills: dict[str, Any] | None = None,
        loaded_skills: set[str] | None = None,
        step_num: int = 0,
    ) -> None:
        # Imported lazily (not at module level) to avoid a circular import:
        # tool_registry reads its size-threshold constants from
        # tool_execution.schema, so tool_execution can't eagerly import
        # tool_registry back at module load time.
        from ..tool_registry import ToolRegistry

        self.tool_calls = message.get("tool_calls", []) or []
        self.tool_messages: list[dict[str, Any]] = []
        self.tool_records: list[dict[str, Any]] = []
        self.tool_artifacts: dict = {}
        self.llm = llm
        # Current pipeline step number, carried only to attribute a refusal
        # event (record_refusal) to the step it happened in; defaults to 0 for
        # callers (mostly tests) that construct an executor outside a run.
        self.step_num = step_num
        # Developer-configured lifecycle hooks (issue #129), loaded per executor —
        # zero-config (no MCA_HOOKS_CONFIG) yields an empty registry, so a
        # hook-free run behaves byte-identically to before this seam existed.
        self.hooks = Hooks.load()
        self.logger = get_logger(self.__class__.__name__)
        # ``skills``/``loaded_skills`` flow from RunContext so ``use_skill`` can
        # lazily load a body and dedup repeats; the loaded-set is shared by
        # reference across the run's per-message registries so dedup persists.
        self.registry = ToolRegistry(
            artifacts=self.tool_artifacts,
            tools=tools or [],
            llm=llm,
            skills=skills,
            loaded_skills=loaded_skills,
            step_num=step_num,
        )

    def run(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Dispatch every tool call, filling ``tool_messages`` / ``tool_records``.

        Calls are parsed, then partitioned into ordered groups (:func:`_plan_groups`):
        a contiguous run of provably read-only calls runs concurrently (bounded
        pool), while every other call runs inline in sequence. Each call still
        runs the three phases — before → call → after — and parse failures
        short-circuit to an error result. Results are appended strictly in call
        order regardless of finish order, so the conversation stays coherent.
        Returns the two lists for convenience; they are also attributes.
        """
        self.logger.tool("dispatch: %d tool call(s)", len(self.tool_calls))
        prepared = [
            _PreparedCall(*arg_prep.parse_tool_call(tc)) for tc in self.tool_calls
        ]
        for group in _plan_groups(prepared):
            if len(group) > 1:
                self._dispatch_parallel(group)
            else:
                self._dispatch_one(group[0])
        return self.tool_messages, self.tool_records

    def _dispatch_one(self, item: "_PreparedCall") -> None:
        """Run one call through the sequential before → call → after path.

        Unchanged from the pre-concurrency path: a parse failure short-circuits
        to an error result; otherwise the recorder times the call via its own
        pending-slot. This is the path for every non-overlappable call and for a
        read-only call that has no read-only neighbour to overlap with.
        """
        if item.error is not None:
            self._append_parse_error(item)
            return
        # parse_tool_call guarantees func_name/args are set when error is None.
        assert item.func_name is not None and item.args is not None
        args = self.before_tool_call(item.func_name, item.args)
        self.logger.tool("%s → %s(%s)", item.tool_call_id, item.func_name, args)
        hook_firings: list[tuple[HookSpec, HookResult]] = []
        raw, failure = self.invoke_tool(
            item.tool_call_id, item.func_name, args, hook_firings
        )
        content, status, record = self.after_tool_call(
            item.tool_call_id,
            item.func_name,
            args,
            raw,
            failure,
            hook_firings=hook_firings,
        )
        self._append_result(item.tool_call_id, content, status, record)

    def _dispatch_parallel(self, group: list["_PreparedCall"]) -> None:
        """Overlap a run of read-only calls, then process results in call order.

        Only the tool *invocation* (the I/O-bound work) overlaps: argument prep,
        output offloading, envelope building, and every recorder emit stay on the
        main thread, in call order, so the shared artifact store, session files,
        and recorder capture state are never touched concurrently. Each worker
        captures its own true start/end so the recorded per-call latency reflects
        the isolated call, not the group's wall-clock. Non-recoverable exceptions
        propagate (``future.result``) exactly as in the sequential path.
        """
        prepared_args = [
            self._prepare_args(item.func_name, item.args) for item in group
        ]
        workers = min(max_tool_concurrency(), len(group))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # copy_context() per task: each worker runs under the parent's
            # contextvars (session id, recorder) without sharing one Context.
            futures = [
                pool.submit(
                    contextvars.copy_context().run,
                    self._invoke_timed,
                    item.tool_call_id,
                    item.func_name,
                    args,
                )
                for item, args in zip(group, prepared_args)
            ]
        for item, args, future in zip(group, prepared_args, futures):
            raw, failure, start_mono, end_mono, started_at, hook_firings = (
                future.result()
            )
            content, status, record = self.after_tool_call(
                item.tool_call_id,
                item.func_name,
                args,
                raw,
                failure,
                timing=(start_mono, end_mono, started_at),
                hook_firings=hook_firings,
            )
            self._append_result(item.tool_call_id, content, status, record)

    def _invoke_timed(
        self, tool_call_id: str, func_name: str, args: dict
    ) -> tuple[Any, dict | None, float, float, str, list[tuple[HookSpec, HookResult]]]:
        """Worker body: invoke one tool, bracketed by its own true timing.

        Returns ``(raw, failure, start_mono, end_mono, started_at, hook_firings)``
        — the monotonic bracket bounds the call's real duration for
        latency/resource accounting, ``started_at`` is its wall-clock start in
        the recorder's format, and ``hook_firings`` are this call's ``PreToolUse``
        firings (returned, not recorded — recording stays on the main thread so
        overlap never races the recorder). Runs no recorder or artifact-store
        code; those stay on the main thread so overlap never races shared state.
        """
        started_at = _now()
        start_mono = time.monotonic()
        hook_firings: list[tuple[HookSpec, HookResult]] = []
        raw, failure = self.invoke_tool(tool_call_id, func_name, args, hook_firings)
        return raw, failure, start_mono, time.monotonic(), started_at, hook_firings

    def _append_parse_error(self, item: "_PreparedCall") -> None:
        """Append the error result for a call that failed to parse."""
        name = item.func_name or "<unknown>"
        env = build_tool_result(name, False, "", item.error, {"reason": "parse_error"})
        env["metadata"]["lang"] = resolve_lang(name, {}, env)
        self.tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "content": json.dumps(validate_tool_result(env), default=str),
                "status": "error",
            }
        )
        self.tool_records.append(error_record(name, {}, item.tool_call_id, item.error))

    def _append_result(
        self, tool_call_id: str, content: str, status: str, record: dict
    ) -> None:
        """Append one dispatched call's tool message and record, in call order."""
        self.tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "status": status,
            }
        )
        self.tool_records.append(record)

    def _prepare_args(self, func_name: str, args: dict) -> dict:
        """Alias-remap args and strip unknown kwargs (no recorder side effect).

        The pure argument-prep half of :meth:`before_tool_call`, split out so the
        concurrent path can prepare args on the main thread without stamping the
        recorder's single pending-slot (which it bypasses via explicit timing).
        """
        args = arg_prep.apply_arg_aliases(func_name, args)
        args = arg_prep.strip_unknown_args(func_name, args)
        self.logger.tool("before %s(%s) [after alias remapping]", func_name, args)
        return args

    def before_tool_call(self, func_name: str, args: dict) -> dict:
        """Before the call: alias-remap args, strip unknown kwargs, stamp recorder.

        Returns the prepared args. The recorder (if any) stamps the call's start
        time for latency accounting.
        """
        args = self._prepare_args(func_name, args)
        if self.llm._recorder is not None:
            self.llm._recorder.before_tool(func_name, args)
        return args

    def _pre_dispatch_gate(
        self,
        func_name: str,
        args: dict,
        hook_firings: list[tuple[HookSpec, HookResult]] | None,
    ) -> dict | None:
        """Run every no-execution gate ahead of dispatch; return the first
        failure descriptor, or ``None`` when every gate allows the call.

        Order: ``PreToolUse`` hooks, then exfiltration, refusal, egress, and
        provenance-reduction — see :meth:`invoke_tool`'s docstring for why
        each one is checked in this order and covers both dispatch paths.
        """
        pre_ctx = HookContext(
            event=EVENT_PRE_TOOL_USE,
            session_id=current_session_id.get() or "",
            step=self.step_num,
            tool_name=func_name,
            args=args,
        )
        firings = self.hooks.fire(EVENT_PRE_TOOL_USE, pre_ctx)
        if hook_firings is not None:
            hook_firings.extend(firings)
        blocking = next((result for _, result in firings if result.blocked), None)
        if blocking is not None:
            blocking_spec = next(spec for spec, result in firings if result.blocked)
            return {
                "reason": BLOCKED_BY_HOOK_REASON,
                "hook_name": blocking_spec.name,
                "block_reason": blocking.reason,
            }

        category = exfil.evaluate(func_name, args)
        if category is not None:
            return {"reason": "exfil_blocked", "category": category}

        refusal = policy.evaluate(func_name, args)
        if refusal is not None:
            return {"reason": "refused", "refusal": refusal}

        block = egress.evaluate(func_name, args)
        if block is not None:
            return {"reason": "egress_blocked", "block": block}

        reduction = provenance.check_reduction(func_name, args)
        if reduction is not None:
            return {"reason": "reduced", "reduction": reduction}

        return None

    def invoke_tool(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        hook_firings: list[tuple[HookSpec, HookResult]] | None = None,
    ) -> tuple[Any, dict | None]:
        """The call step only: invoke ``func_name(**args)`` against the registry.

        Returns ``(raw_result, failure)`` — the tool's raw return value (str or
        artifact tuple) and ``None`` on success, or ``None`` and a
        ``{"reason", "error"}`` descriptor on a handled failure. No retries and no
        LLM: a wrong-argument call fails directly. Non-recoverable exceptions
        re-raise. Turning the raw result into the envelope is
        :meth:`after_tool_call`'s job.

        Fired first, before any dispatch: every ``PreToolUse`` hook matching
        this call (issue #129). A hook that returns a block decision never
        reaches any other gate, ``getattr(self.registry, ...)``, or
        ``subprocess.run`` — it short-circuits to a ``reason:
        "blocked_by_hook"`` descriptor, one of the no-execution failure kinds
        alongside ``exfil_blocked``/``refused``/``egress_blocked``/``reduced``/
        ``not_found``/``wrong_args``/``raised``. When ``hook_firings`` is given
        (a caller-owned list), every fired spec/result pair is appended to it
        so the caller can record them later — this method itself makes no
        recorder call, matching the gates below.

        After that, a secret-exfiltration match on an egress tool's outbound
        payload (:mod:`engine.exfil`, e.g. ``fetch_web``'s ``url``) short-
        circuits to a ``reason: "exfil_blocked"`` descriptor before anything
        else runs, since a blocked egress never needs a bash-specific or
        destination check. Local reads are unaffected — only egress tools are
        evaluated. Next, a dangerous ``bash`` command line matching
        :mod:`policy` never reaches ``getattr(self.registry, ...)`` or
        ``subprocess.run`` — it short-circuits to a ``reason: "refused"``
        descriptor. A ``fetch_web`` call whose destination matches
        :mod:`engine.egress`'s known-malicious blocklist is checked next and
        short-circuits the same way with ``reason: "egress_blocked"``. Next, a
        build/install command subject to the untrusted-content capability
        reduction (:mod:`provenance`) short-circuits to a ``reason: "reduced"``
        descriptor. This is the single dispatch choke point both the
        sequential and concurrent paths funnel through, so every gate covers
        both by construction (and subagents, which share the same executor).
        No gate makes a recorder call itself — that happens in
        :meth:`after_tool_call`, on the main thread, in call order.
        """
        gate_failure = self._pre_dispatch_gate(func_name, args, hook_firings)
        if gate_failure is not None:
            return None, gate_failure

        if not hasattr(self.registry, func_name):
            self.logger.error("not found: '%s' is not registered", func_name)
            valid = [n for n in dir(type(self.registry)) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            return None, {"reason": "not_found", "error": err}

        try:
            return getattr(self.registry, func_name)(**args), None
        except TypeError as exc:  # wrong arguments — surfaced as a failure, no retry
            sig = inspect.signature(getattr(type(self.registry), func_name))
            self.logger.error("wrong args %s → %s: %s", tool_call_id, func_name, exc)
            err = (
                f"Error: wrong arguments for '{func_name}': {exc}. "
                f"Expected: {func_name}{sig}"
            )
            return None, {"reason": "wrong_args", "error": err}
        except Exception as exc:
            if not isinstance(exc, _RECOVERABLE_EXCEPTIONS):
                self.logger.error(
                    "non-recoverable error %s → %s: %s", tool_call_id, func_name, exc
                )
                raise
            self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
            err = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
            return None, {"reason": "raised", "error": err}

    def _build_refusal_result(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        refusal: policy.Refusal,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Build the ``ok:false`` envelope for a policy refusal (returns ``env,
        status, record``), and — since this runs from :meth:`after_tool_call`,
        the executor's ordered main-thread recorder path — emit the WARNING log
        line and the passive ``refusal`` event alongside it. The gate that made
        the decision (:mod:`policy`) stays recorder-free.
        """
        command = args.get("command", "") if func_name == "bash" else str(args)
        references = [
            {
                REFUSAL_REFERENCE_STANDARD_ID: ref.standard_id,
                REFUSAL_REFERENCE_URL: ref.url,
            }
            for ref in refusal.references
        ]
        ref_text = "; ".join(f"{r['standard_id']} ({r['url']})" for r in references)
        error_text = (
            f"Refused (not a failure): {command!r} — {refusal.reason} "
            f"Reference: {ref_text}. Safer alternative: {refusal.safer_alternative} "
            f"Note: {POSTURE_NOTE_TEXT}"
        )
        metadata = {
            "reason": "refused",
            "refusal": {
                REFUSAL_RULE_ID: refusal.rule_id,
                REFUSAL_REASON: refusal.reason,
                REFUSAL_REFERENCES: references,
                REFUSAL_SAFER_ALTERNATIVE: refusal.safer_alternative,
                REFUSAL_POSTURE_NOTE: POSTURE_NOTE_TEXT,
            },
        }
        env = build_tool_result(func_name, False, "", error_text, metadata)
        status = "error"
        record = error_record(func_name, args, tool_call_id, error_text)

        self.logger.warning(
            "refused %s → %s(%s): rule=%s",
            tool_call_id,
            func_name,
            command,
            refusal.rule_id,
        )
        if self.llm._recorder is not None:
            self.llm._recorder.record_refusal(
                tool_name=func_name,
                command=command,
                rule_id=refusal.rule_id,
                reason=refusal.reason,
                references=references,
                safer_alternative=refusal.safer_alternative,
                step=self.step_num,
            )
        return env, status, record

    def _build_egress_blocked_result(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        block: "egress.EgressBlock",
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Build the ``ok:false`` envelope for an egress block (returns ``env,
        status, record``), mirroring :meth:`_build_refusal_result`: emits the
        WARNING log line and the passive ``egress`` event alongside it. The
        gate that made the decision (:mod:`engine.egress`) stays recorder-free.
        """
        error_text = (
            f"Blocked (not a failure): destination {block.host!r} — {block.reason}"
        )
        metadata = {
            "reason": "egress_blocked",
            "egress": {
                EGRESS_HOST: block.host,
                EGRESS_MATCHED_LIST: block.matched_list,
                EGRESS_REASON: block.reason,
            },
        }
        env = build_tool_result(func_name, False, "", error_text, metadata)
        status = "error"
        record = error_record(func_name, args, tool_call_id, error_text)

        self.logger.warning(
            "egress blocked %s → %s: host=%s list=%s",
            tool_call_id,
            func_name,
            block.host,
            block.matched_list,
        )
        if self.llm._recorder is not None:
            self.llm._recorder.record_egress(
                tool_name=func_name,
                host=block.host,
                matched_list=block.matched_list,
                reason=block.reason,
                step=self.step_num,
            )
        return env, status, record

    def _build_blocked_by_hook_result(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        hook_name: str,
        block_reason: str | None,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Build the ``ok:false`` envelope for a ``PreToolUse`` hook block.

        Mirrors :meth:`_build_refusal_result`'s shape: model-facing prose in
        ``error``, structured facts in ``metadata.hook_block`` so a consumer
        can tell "blocked by hook" from "raised"/"refused" without parsing
        prose. Recording of the block (via ``_record_hook_firings``) happens
        in :meth:`after_tool_call`'s ordered path, not here.
        """
        error_text = (
            f"Blocked by hook (not a failure): {func_name!r} call vetoed by "
            f"hook {hook_name!r}: {block_reason}"
        )
        metadata = {
            "reason": BLOCKED_BY_HOOK_REASON,
            "hook_block": {HOOK_NAME: hook_name, HOOK_REASON: block_reason},
        }
        env = build_tool_result(func_name, False, "", error_text, metadata)
        status = "error"
        record = error_record(func_name, args, tool_call_id, error_text)
        self.logger.warning(
            "blocked by hook %s → %s(): hook=%s", tool_call_id, func_name, hook_name
        )
        return env, status, record

    def _build_exfil_result(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        category: str,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Build the ``ok:false`` envelope for an exfiltration-guard block
        (returns ``env, status, record``), and emit the WARNING log line and
        the passive ``exfil`` event alongside it — follows
        :meth:`_build_refusal_result`'s template. Names only the matched
        category, never the secret value itself (design.md decision 3).
        """
        error_text = (
            f"Blocked (not a failure): outbound payload for '{func_name}' "
            f"matches a known-sensitive category ({category}) and was not "
            "sent. Do not attempt to transmit this content off the machine."
        )
        metadata = {
            "reason": "exfil_blocked",
            "exfil": {EXFIL_TOOL_NAME: func_name, EXFIL_CATEGORY: category},
        }
        env = build_tool_result(func_name, False, "", error_text, metadata)
        status = "error"
        record = error_record(func_name, args, tool_call_id, error_text)

        self.logger.warning(
            "exfil blocked %s → %s: category=%s", tool_call_id, func_name, category
        )
        if self.llm._recorder is not None:
            self.llm._recorder.record_exfil(
                tool_name=func_name,
                category=category,
                step=self.step_num,
            )
        return env, status, record

    def _build_reduction_result(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        reduction: provenance.Reduction,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Build the ``ok:false`` envelope for a crossed capability-reduction
        boundary (returns ``env, status, record``), mirroring
        :meth:`_build_refusal_result`. Emits the passive ``provenance`` event
        (kind ``reduction_refusal``) alongside it.
        """
        command = args.get("command", "") if func_name == "bash" else str(args)
        error_text = (
            f"Refused (reduced capability, not a failure): {command!r} — "
            f"{reduction.reason} Safer alternative: {reduction.safer_alternative}"
        )
        metadata = {
            "reason": "reduced",
            "reduction": {
                "rule_id": reduction.rule_id,
                "reason": reduction.reason,
                "safer_alternative": reduction.safer_alternative,
            },
        }
        env = build_tool_result(func_name, False, "", error_text, metadata)
        status = "error"
        record = error_record(func_name, args, tool_call_id, error_text)

        self.logger.warning(
            "reduced %s → %s(%s): rule=%s",
            tool_call_id,
            func_name,
            command,
            reduction.rule_id,
        )
        if self.llm._recorder is not None:
            self.llm._recorder.record_provenance(
                kind=PROVENANCE_KIND_REDUCTION_REFUSAL,
                tool_name=func_name,
                reason=reduction.reason,
                step=self.step_num,
            )
        return env, status, record

    def _apply_provenance(
        self, func_name: str, args: dict, env: dict[str, Any]
    ) -> None:
        """Demarcate a freshly-tagged untrusted result and update the
        freshly-cloned-repo state from a completed ``bash`` call — the two
        pieces of run-scoped state :func:`provenance.check_reduction` reads.
        Called only from the success path of :meth:`after_tool_call`.
        """
        if env["ok"] and env["metadata"].get("provenance") == provenance.UNTRUSTED:
            provenance.note_untrusted_content()
            env["output"] = provenance.demarcate(env["output"])
            if self.llm._recorder is not None:
                self.llm._recorder.record_provenance(
                    kind=PROVENANCE_KIND_MARK,
                    tool_name=func_name,
                    reason=f"{func_name} result tagged untrusted at ingestion",
                    step=self.step_num,
                )
        if func_name == "bash":
            provenance.note_bash_command(args.get("command", ""), env["ok"])

    def _record_hook_firings(
        self,
        firings: list[tuple[HookSpec, HookResult]],
        tool_name: str | None,
    ) -> None:
        """Record each ``(spec, result)`` firing as a passive ``hook`` event.

        The recorder never participates in the hook decision — it only
        appends what the mechanism (``engine.hooks``) already decided.
        """
        if self.llm._recorder is None:
            return
        for spec, result in firings:
            self.llm._recorder.record_hook(
                event=spec.event,
                hook_name=spec.name,
                outcome=HOOK_OUTCOME_BLOCKED if result.blocked else HOOK_OUTCOME_FIRED,
                step=self.step_num,
                tool_name=tool_name,
                reason=result.reason,
            )

    def _build_failure_result(
        self, tool_call_id: str, func_name: str, args: dict, failure: dict
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Dispatch a ``failure`` descriptor to its envelope builder by ``reason``."""
        reason = failure["reason"]
        if reason == "refused":
            return self._build_refusal_result(
                tool_call_id, func_name, args, failure["refusal"]
            )
        if reason == BLOCKED_BY_HOOK_REASON:
            return self._build_blocked_by_hook_result(
                tool_call_id,
                func_name,
                args,
                failure["hook_name"],
                failure["block_reason"],
            )
        if reason == "exfil_blocked":
            return self._build_exfil_result(
                tool_call_id, func_name, args, failure["category"]
            )
        if reason == "egress_blocked":
            return self._build_egress_blocked_result(
                tool_call_id, func_name, args, failure["block"]
            )
        if reason == "reduced":
            return self._build_reduction_result(
                tool_call_id, func_name, args, failure["reduction"]
            )
        env = build_tool_result(
            func_name, False, "", failure["error"], {"reason": reason}
        )
        status = "error"
        record = error_record(func_name, args, tool_call_id, failure["error"])
        return env, status, record

    def _build_success_result(
        self, tool_call_id: str, func_name: str, args: dict, raw_result: Any
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Turn a successful raw tool return into (env, status, record)."""
        is_artifact = isinstance(raw_result, tuple) and len(raw_result) == 2
        preview: dict[str, Any] | None = None
        error: str | None = None
        duplicate_of: dict[str, Any] | None = None
        if is_artifact:
            _, artifact = raw_result
            self.tool_artifacts[tool_call_id] = artifact
            result, error, preview, duplicate_of = self._offload_streams(
                tool_call_id, artifact
            )
        else:
            result = raw_result
        if not isinstance(result, str):
            result = str(result)
        pre_len = len(result)
        result = validate_tool_output(
            result, func_name, self.llm._session_log_path, is_summary=is_artifact
        )
        is_truncated = not is_artifact and len(result) < pre_len

        self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
        env = result_envelope(
            func_name,
            result,
            is_artifact,
            is_truncated,
            tool_call_id,
            self.tool_artifacts.get(tool_call_id),
            preview=preview,
            error=error,
            duplicate_of=duplicate_of,
        )
        status, record = call_record(
            func_name, args, tool_call_id, env, is_artifact, is_truncated
        )
        self._apply_provenance(func_name, args, env)
        return env, status, record

    def after_tool_call(
        self,
        tool_call_id: str,
        func_name: str,
        args: dict,
        raw_result: Any,
        failure: dict | None,
        timing: tuple[float, float, str] | None = None,
        hook_firings: list[tuple[HookSpec, HookResult]] | None = None,
    ) -> tuple[str, str, dict]:
        """Turn the tool's raw return (or failure) into (content, status, record).

        On failure, builds the error envelope from the ``{reason, error}``
        descriptor. On success, offloads artifact tuples (writing each to a
        per-artifact file and replacing it with a bounded preview — no LLM),
        coerces to str, truncates, and normalizes into the canonical envelope.
        Serializes, then lets the recorder capture the final agent-facing content.

        ``timing`` is supplied only by the concurrent path — a
        ``(start_mono, end_mono, started_at)`` bracket the worker captured — so
        the recorded latency reflects the isolated call rather than the shared
        pending-slot, which overlap would otherwise race. ``None`` keeps the
        sequential path's recorder timing exactly as before.
        """

        def capture(content: str, ok: bool, error: str | None) -> str:
            """Let the observability recorder (if any) emit the tool event."""
            if self.llm._recorder is not None:
                self.llm._recorder.after_tool(
                    func_name, args, content, ok, error, timing=timing
                )
            return content

        if failure is not None:
            env, status, record = self._build_failure_result(
                tool_call_id, func_name, args, failure
            )
        else:
            env, status, record = self._build_success_result(
                tool_call_id, func_name, args, raw_result
            )

        self._record_hook_firings(hook_firings or [], func_name)
        post_ctx = HookContext(
            event=EVENT_POST_TOOL_USE,
            session_id=current_session_id.get() or "",
            step=self.step_num,
            tool_name=func_name,
            args=args,
            result=env,
        )
        post_firings = self.hooks.fire(EVENT_POST_TOOL_USE, post_ctx)
        self._record_hook_firings(post_firings, func_name)

        env["metadata"]["lang"] = resolve_lang(func_name, args, env)
        serialized = json.dumps(validate_tool_result(env), default=str)
        return capture(serialized, env["ok"], env["error"]), status, record

    def _offload_streams(
        self, tool_call_id: str, artifact: dict[str, Any]
    ) -> tuple[str, str | None, dict[str, Any], dict[str, Any]]:
        """Bound each output stream of an offloaded command artifact independently.

        Returns ``(output, error, preview, duplicate_of)``: ``output`` is the
        composed stdout (bounded preview, duplicate pointer, or inline), ``error``
        is the composed stderr (same three shapes, or ``None`` when empty),
        ``preview`` maps each freshly-offloaded stream to its descriptor, and
        ``duplicate_of`` maps each deduplicated stream to its descriptor. A stream
        appears in at most one of the two maps.
        """
        preview: dict[str, Any] = {}
        duplicate_of: dict[str, Any] = {}
        output, out_desc, out_dup = self._offload_stream(
            tool_call_id, "stdout", artifact.get("stdout") or ""
        )
        if out_desc is not None:
            preview["stdout"] = out_desc
        if out_dup is not None:
            duplicate_of["stdout"] = out_dup
        error, err_desc, err_dup = self._offload_stream(
            tool_call_id, "stderr", artifact.get("stderr") or ""
        )
        if err_desc is not None:
            preview["stderr"] = err_desc
        if err_dup is not None:
            duplicate_of["stderr"] = err_dup
        return output, (error or None), preview, duplicate_of

    def _offload_stream(
        self, tool_call_id: str, stream: str, text: str
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
        """Return ``(field_value, preview_descriptor, duplicate_descriptor)`` for
        one output stream.

        Small streams (within the preview budget) are inlined with no descriptor
        and no file. Larger streams are checked against this run's already-stored
        artifacts (deterministic, no LLM): a duplicate (byte-identical or
        contained) skips the file write and preview entirely, returning a pointer
        instead; a novel stream is written to a per-stream file and replaced with
        a bounded excerpt + skim guidance.
        """
        if len(text) <= PREVIEW_MAX_CHARS:
            return text, None, None
        session_id = current_session_id.get()
        duplicate = _find_duplicate(session_id, text)
        if duplicate is not None:
            from ..tool_registry import artifact_file_path  # lazy: avoids a cycle

            original_path = artifact_file_path(
                session_id, duplicate["tool_call_id"], duplicate["stream"]
            )
            notice = _build_duplicate_notice(
                duplicate, str(original_path) if original_path else None
            )
            return notice, None, duplicate
        path = self._write_artifact_file(tool_call_id, stream, text)
        field_value, preview_desc = build_stream_preview(text, path)
        return field_value, preview_desc, None

    def _write_artifact_file(
        self, tool_call_id: str, stream: str, text: str
    ) -> str | None:
        """Write a stream's full content to its per-run file so bash can skim it.

        The file lives at
        ``.my_coding_agent/<session>/artifacts/<tool_call_id>.<stream>.txt`` and
        persists for the run, so a later step can inspect it with bash text tools.
        Returns the path, or ``None`` when the session directory or id is
        unavailable (e.g. unit tests invoking the executor without an agent run),
        or when the write itself fails (full disk / permissions) — a failed write
        is logged and downgraded to "no on-disk copy" so offloading continues
        rather than aborting the run.
        """
        from ..tool_registry import artifact_file_path  # lazy: avoids a cycle

        path = artifact_file_path(current_session_id.get(), tool_call_id, stream)
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        except OSError as exc:
            # A full disk or bad permissions must not abort the run: offloading
            # and the preview continue without an on-disk copy (the preview
            # guidance falls back to read_tool_artifact when the path is None).
            self.logger.warning(
                "artifact write failed for %s (%s) at %s: %s",
                tool_call_id,
                stream,
                path,
                exc,
            )
            return None
        return str(path)
