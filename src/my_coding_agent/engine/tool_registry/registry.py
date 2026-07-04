"""Tool implementations exposed to the agent.

Each public method of ``ToolRegistry`` is a callable tool the LLM can invoke.
The ``ToolExecutor`` normalizes every return value into the canonical envelope —
tools therefore stay simple and need not know about the schema.
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import html2text
import httpx

from ...observability.recorder import (
    current_agent_node,
    current_recorder,
    current_session_id,
)
from ...utils import get_logger
from ...utils.exceptions import PathTraversalError
from ...utils.parsing import extract_message
from ..llm.schema import CALL_KIND_ARTIFACT_QUERY
from ..tool_execution.schema import (
    ARTICLE_FETCH_MAX_CHARS,
    ARTIFACT_THRESHOLD,
    RANGE_MAX_CHARS,
)

if TYPE_CHECKING:
    from ..llm import LLM

logger = get_logger(__name__)

# ARTIFACT_THRESHOLD (large-output boundary) and ARTICLE_FETCH_MAX_CHARS (fetch
# sanity cap) are centrally configured in tool_execution.schema, imported above.

# Extraction budgets for read_tool_artifact (chars, ~4 chars/token estimate).
_CHARS_PER_TOKEN = 4
EXTRACTION_OUTPUT_TOKEN_BUDGET = 800  # bounds a single read_tool_artifact return
EXTRACTION_OUTPUT_MAX_CHARS = EXTRACTION_OUTPUT_TOKEN_BUDGET * _CHARS_PER_TOKEN
EXTRACTION_CHUNK_MAX_CHARS = 16_000  # per-call input budget for one scan chunk

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# A tool_call_id doubles as a per-artifact filename; restrict it to a safe set so
# a crafted id can never traverse out of the session's artifacts directory.
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def artifact_file_path(
    session_id: str | None, tool_call_id: str, stream: str = "stdout"
) -> Path | None:
    """Return the on-disk path for a tool call's per-stream artifact file, or None.

    Single source of truth for the per-artifact path scheme
    ``.my_coding_agent/<session>/artifacts/<tool_call_id>.<stream>.txt``, shared by
    the write side (executor) and the read side (``read_tool_artifact``) so the two
    can never drift apart. Each output stream (``stdout``/``stderr``) is offloaded
    to its own file so a large stream in either channel can be skimmed.

    Returns None when there is no session id or the id is unsafe as a filename
    (would traverse out of the artifacts directory). Performs no filesystem
    I/O — callers create the directory and read/write the file.

    Args:
        session_id: The current session id, or None outside an agent run.
        tool_call_id: The id whose artifact file path is requested.
        stream: The output stream this file holds — ``stdout`` or ``stderr``.

    Returns:
        The artifact file path, or None when it cannot be safely constructed.
    """
    if not session_id or not _SAFE_ARTIFACT_ID.match(tool_call_id):
        return None
    return (
        Path(".my_coding_agent")
        / session_id
        / "artifacts"
        / f"{tool_call_id}.{stream}.txt"
    )


class ToolRegistry:
    """Hold the callable tools exposed to the agent.

    Each public method is a tool the LLM can invoke: ``bash`` runs a shell command,
    ``read_file``/``write_file`` access the workspace (confined to ``base_dir`` to
    block path traversal), ``read_article`` fetches a URL as markdown,
    ``read_tool_artifact`` queries a previously stored large output for a
    query-scoped, bounded extract, and ``delegate`` spawns a read-only subagent.
    Large outputs (``bash`` streams, file reads, web fetches) are offloaded to
    the per-run artifact store instead of being returned inline.
    """

    def __init__(
        self,
        artifacts: dict | None = None,
        tools: list | None = None,
        base_dir: str | None = None,
        llm: "LLM | None" = None,
    ):
        self._artifacts = artifacts if artifacts is not None else {}
        self._tools = tools if tools is not None else []
        # Workspace root that read_file/write_file must stay within. Defaults
        # to the current working directory; override per deployment if a
        # different root applies.
        self._base_dir = (
            Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        )
        # Injected by ToolExecutor (same pattern as `tools` above) so
        # read_tool_artifact can make its bounded extraction call. None outside
        # an agent run (unit tests, standalone registry) — extraction then
        # degrades to a bounded head excerpt.
        self._llm = llm

    def _resolve_in_base(self, file_path: str) -> Path:
        """Resolve file_path against the workspace base, rejecting any escape.

        Relative paths are resolved under the base; absolute paths are allowed
        only when they fall inside the base. Raises PathTraversalError on
        path traversal.
        """
        candidate = Path(file_path)
        target = candidate if candidate.is_absolute() else self._base_dir / candidate
        target = target.resolve()
        if not target.is_relative_to(self._base_dir):
            raise PathTraversalError(
                f"Path traversal detected: '{file_path}' resolves outside "
                f"the workspace base '{self._base_dir}'.",
                hint="Use a path inside the workspace.",
            )
        return target

    def bash(self, command: str, timeout: int = 60) -> "str | tuple[None, dict]":
        """Run a shell command and return stdout, stderr, exit_code, and ok as JSON.
        Use for running tests, installing packages, git operations, or any shell task.
        The 'ok' field is true when exit_code is 0.

        Note: shell=True is intentional — this tool is a first-class shell execution
        surface that must support pipes, redirections, builtins, and compound commands.

        Tags:
            shell, bash, execute, run, command, git, test, install, terminal

        Args:
            command: Shell command to run. Use absolute paths where possible.
                Example: 'ls -la' or 'git status'
            timeout: Seconds before the command is killed. Defaults to 60.
        """
        try:
            # shell=True is required here: this bash tool is a first-class execution
            # surface for the LLM coding agent. It must support shell features such as
            # pipes (`|`), redirections (`>`), builtins (`cd`), and compound commands
            # (`&&`, `;`). Splitting on whitespace and passing a list would break these
            # core use cases. The command originates from the LLM (not from raw,
            # unmediated user string interpolation), and shell=True on this surface is
            # a documented, intentional design decision — not an incidental subprocess
            # call. CONTRIBUTE.md §32 is acknowledged; this is the approved exception.
            result = subprocess.run(  # nosec B602  # noqa: S602
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "stdout": "",
                    "stderr": f"Error: command timed out after {timeout}s",
                    "exit_code": -1,
                    "ok": False,
                }
            )
        stdout = result.stdout.rstrip()
        stderr = result.stderr.rstrip()
        full = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "ok": result.returncode == 0,
        }
        if len(stdout) + len(stderr) > ARTIFACT_THRESHOLD:
            return None, full  # dispatcher offloads it
        return json.dumps(full)

    def read_tool_artifact(
        self,
        tool_call_id: str,
        query: str | None = None,
        start: int | None = None,
        length: int | None = None,
    ) -> str:
        """Query, or exactly slice, a previously offloaded large tool output.

        Two bounded, mutually exclusive modes:

        - Query mode (default): pass ``query`` for a bounded extract relevant to
          it — never the whole stored content. Call it as many times as you need
          with different queries.
        - Range mode: pass ``start`` (and optionally ``length``) for an exact,
          verbatim byte slice — no LLM call, deterministic. This is the only mode
          that works on content with little or no line structure (e.g. a single
          giant JSON line), and is what a ``duplicate_of`` pointer's offset/length
          feed directly into.

        Bash text tools (grep/rg, sed, awk, jq, head/tail, wc) over the on-disk
        artifact file remain available as a secondary path when you already know
        the shape of what you're looking for and the content has line structure.

        Tags:
            artifact, output, result, retrieve, query, search, range

        Args:
            tool_call_id: The tool_call_id from a previous call whose output was
                offloaded. Example: 'call_abc123'
            query: Natural-language description of what you need from the stored
                output. Required unless ``start`` is given. Example: 'the
                traceback line naming the failing assertion'
            start: Byte offset (0-based) for exact verbatim range retrieval. When
                given, ``query`` is ignored and the exact slice is returned.
            length: Number of bytes to return from ``start``. Defaults to, and is
                capped at, the per-call budget (RANGE_MAX_CHARS). Only used with
                ``start``.
        """
        if start is not None:
            text = self._load_artifact_text(tool_call_id)
            if text is None:
                return f"Error: no artifact found for tool_call_id '{tool_call_id}'"
            return self._range_slice(text, start, length)
        if not query or not query.strip():
            return (
                "Error: 'query' is required and must be non-empty unless 'start' "
                "is given for byte-range retrieval. Example: "
                f'read_tool_artifact(tool_call_id="{tool_call_id}", '
                'query="the error message near the end of the output")'
            )
        text = self._load_artifact_text(tool_call_id)
        if text is None:
            return f"Error: no artifact found for tool_call_id '{tool_call_id}'"
        if self._llm is None:
            return self._head_excerpt(tool_call_id, text)
        return self._extract(tool_call_id, text, query)

    def _range_slice(self, text: str, start: int, length: int | None) -> str:
        """Return an exact, verbatim byte-range slice of ``text``, capped at
        RANGE_MAX_CHARS, prefixed by a one-line range/total header. No LLM call.
        """
        total = len(text)
        if start < 0 or start >= total:
            return (
                f"Error: 'start' {start} is out of range — stored content is "
                f"{total} bytes."
            )
        requested = length if length is not None else RANGE_MAX_CHARS
        bounded = max(0, min(requested, RANGE_MAX_CHARS, total - start))
        end = start + bounded
        return f"[range {start}-{end} of {total} bytes]\n{text[start:end]}"

    def _load_artifact_text(self, tool_call_id: str) -> str | None:
        """Return the full stored text for tool_call_id, or None if nothing is
        stored. Prefers the on-disk per-stream files (persist for the whole run);
        falls back to the in-memory store (same step, or no session dir). Bash
        artifacts' stdout and stderr are concatenated so a query can match either.
        """
        session_id = current_session_id.get()
        parts = []
        found = False
        for stream in ("stdout", "stderr"):
            path = artifact_file_path(session_id, tool_call_id, stream)
            if path is not None and path.exists():
                found = True
                text = path.read_text()
                if text:
                    parts.append(text)
        if found:
            return "\n".join(parts)
        artifact = self._artifacts.get(tool_call_id)
        if artifact is None:
            return None
        if isinstance(artifact, str):
            return artifact
        parts = [artifact.get("stdout") or "", artifact.get("stderr") or ""]
        return "\n".join(p for p in parts if p)

    def _artifact_path_hint(self, tool_call_id: str) -> str | None:
        """Return the on-disk artifact path for tool_call_id, if one exists."""
        session_id = current_session_id.get()
        for stream in ("stdout", "stderr"):
            path = artifact_file_path(session_id, tool_call_id, stream)
            if path is not None and path.exists():
                return str(path)
        return None

    def _head_excerpt(self, tool_call_id: str, text: str) -> str:
        """Bounded degradation when extraction cannot run: a head excerpt of the
        stored text plus guidance pointing at the on-disk file — never the full
        content, and the run continues rather than aborting."""
        excerpt = text[:EXTRACTION_OUTPUT_MAX_CHARS]
        path = self._artifact_path_hint(tool_call_id)
        hint = (
            f" Full output on disk at {path} — skim it with bash text tools "
            "(grep/rg, sed, awk, jq, head/tail, wc)."
            if path
            else ""
        )
        return (
            f"{excerpt}\n\n[Extraction unavailable — showing a bounded excerpt "
            f"of the stored output instead of the full content.{hint}]"
        )

    def _extract(self, tool_call_id: str, text: str, query: str) -> str:
        """Query-scoped extraction: single-call fast path for within-budget
        artifacts, sequential chunked scan for larger ones. Accumulates relevant
        extracts across chunks, stopping early once the output budget fills so
        the whole artifact stays reachable without ever returning it whole.
        """
        chunks = [
            text[i : i + EXTRACTION_CHUNK_MAX_CHARS]
            for i in range(0, len(text), EXTRACTION_CHUNK_MAX_CHARS)
        ] or [""]
        collected: list[str] = []
        remaining = EXTRACTION_OUTPUT_MAX_CHARS
        for chunk in chunks:
            if remaining <= 0:
                break
            extract = self._extract_chunk(chunk, query)
            if extract is None:
                return self._head_excerpt(tool_call_id, text)
            if extract.strip().upper() != "NOT FOUND":
                collected.append(extract)
                remaining -= len(extract)
        if not collected:
            return f"No content relevant to '{query}' was found in the stored output."
        result = "\n\n---\n\n".join(collected)
        return result[:EXTRACTION_OUTPUT_MAX_CHARS]

    def _extract_chunk(self, chunk: str, query: str) -> str | None:
        """Make one bounded extraction call over a chunk. Returns the model's
        (cleaned) response, or None when the call itself fails (degrades the
        whole retrieval to the head-excerpt fallback)."""
        prompt = (
            "/no_think\n"
            "You are extracting information from a stored tool output on behalf "
            "of an AI coding agent that cannot see the whole output. Given the "
            "query and a chunk of that output, quote the exact passages relevant "
            "to the query, verbatim. If nothing in this chunk is relevant, "
            "respond with exactly: NOT FOUND\n\n"
            f"Query: {query}\n\n"
            f"Chunk:\n{chunk}"
        )
        try:
            assert self._llm is not None
            resp = self._llm.chat_completion(
                [{"role": "user", "content": prompt}],
                tools=[],
                kind=CALL_KIND_ARTIFACT_QUERY,
                max_tokens=EXTRACTION_OUTPUT_TOKEN_BUDGET,
            )
            content = extract_message(resp).get("content") or ""
        except Exception as exc:
            logger.warning("artifact_query extraction failed: %s", exc)
            return None
        return _THINK_RE.sub("", content).strip()

    def delegate(self, task: str, context: str) -> str:
        """Delegate a focused exploration or research task to a subagent.
        Provide 'context' with the relevant background the subagent needs (file
        paths, goal of the main task, key names/symbols). The subagent starts
        fresh with only that context, reads files, runs targeted bash commands,
        and returns a structured report. Use when understanding a file or
        codebase section would crowd the main context.

        Tags:
        delegate, subagent, explore, analyze, file, code, read, understand, investigate

        Args:
            task: What the subagent should do. Example: 'Read llm.py and explain
                how before_tool_call hooks work'
            context: Relevant background from the main agent. Include file paths,
                goal, and key names. Example: 'We are adding a hook to the agent
                loop. Relevant files: agent.py, llm.py at /abs/path/'
        """
        # Lazy import — avoids a circular dependency (agent → tools → registry).
        from my_coding_agent.engine.agent import DEFAULT_MAX_STEPS, AgentNode
        from my_coding_agent.engine.llm import OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL
        from my_coding_agent.pipeline.schema import CLEAN_FINISH_REASONS

        subagent_tools = [t for t in self._tools if t["function"]["name"] != "delegate"]
        now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M (%Z)")
        system_prompt = (
            "You are a focused subagent working for a main coding assistant. "
            "You receive a task and context, and you have the same tools as the "
            "main agent. Use tools when needed — read files, run targeted bash "
            "commands, fetch URLs for web/research, and gather context — then "
            "write a clear, structured report. The task may be code or file "
            "exploration, web research, or context gathering. Do NOT modify any "
            "files. Be concise — the main agent only needs the key findings.\n\n"
            "Every tool returns JSON: "
            '{"schema_version", "tool", "ok", "output", "error", "metadata"}. '
            "When `ok` is true read `output`; when false read `error` (and "
            "`metadata`) to recover.\n\n"
            f"Working directory: {os.getcwd()}\n"
            f"Current date and time: {now}"
        )
        agent = AgentNode(
            api_url=OMLX_API_URL,
            api_key=OMLX_API_KEY,
            model=OMLX_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{task}"},
            ],
            tools=subagent_tools,
            label="SubAgent",
            needs_handback=True,
        )
        agent.execute(max_steps=DEFAULT_MAX_STEPS)
        # Link this subagent to the delegate tool call in the parent's trace tree.
        parent_recorder = current_recorder.get()
        if parent_recorder is not None:
            parent_recorder.note_delegate_child(agent.session_id)
        # On a clean finish the final assistant turn is already the report — hand
        # it back verbatim (zero extra LLM calls). On a cutoff the last message
        # drops the final tool results, so ContextSummarizerNode synthesized a
        # report from the full conversation in-pipeline (handback_report);
        # generate_report() remains the out-of-pipeline fallback (aborted runs,
        # or a clean finish whose final turn carries no usable text).
        report = None
        if agent.stop_reason in CLEAN_FINISH_REASONS:
            report = agent.final_assistant_text()
        if not (report and report.strip()):
            report = agent.handback_report
        if report and report.strip():
            agent.recorder.record_report(report)
        else:
            report = agent.generate_report()
            # execute() already saved session_data.json before this
            # out-of-pipeline report call ran; re-save so the child's
            # persisted totals include the report's tokens (D4).
            agent._save_session_data(DEFAULT_MAX_STEPS)
        # Hand the completed child's usage summary up to the parent so it can
        # accumulate its rollup without re-reading the child's files (D3).
        parent_node = current_agent_node.get()
        if parent_node is not None:
            parent_node.add_child_usage(agent._usage_summary())
        return report

    def read_file(self, file_path: str) -> str | tuple[None, dict]:
        """Read and return the full contents of a file at the given file_path.
        Use to inspect source code, configs, or any text file before editing.
        Large files are offloaded to the artifact store with a bounded preview
        instead of flooding the context.

        Tags:
            file, filesystem, read, inspect, source, code, config

        Args:
            file_path: Absolute path to the file to read. Example: '/path/to/file.py'
        """
        target = self._resolve_in_base(file_path)
        try:
            content = target.read_text()
        except FileNotFoundError:
            return f"Error: file not found: {file_path}"
        except Exception as e:
            return f"Error reading {file_path}: {e}"
        if len(content) > ARTIFACT_THRESHOLD:
            return None, {"stdout": content, "ok": True}  # dispatcher offloads it
        return content

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file at file_path, creating parent directories if needed.
        Use to create new files or overwrite existing ones.

        Tags:
            file, filesystem, write, create, edit, save

        Args:
            file_path: Absolute path where the file should be written.
                Example: '/path/to/file.py'
            content: Full text content to write. Overwrites any existing file.
        """
        target = self._resolve_in_base(file_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return f"Written {len(content)} bytes to {file_path}"
        except Exception as e:
            return f"Error writing {file_path}: {e}"

    @staticmethod
    def read_article(url: str, timeout: float = 15.0) -> str | tuple[None, dict]:
        """Fetch any text URL and return its content.

        HTML responses (``text/html``, ``application/xhtml+xml``) are converted
        to clean markdown, as for an article, blog post, or documentation page.
        Every other text response (JSON, plain text, XML, ...) is returned
        verbatim — the served body is never reshaped — so this is also the tool
        for fetching a JSON API endpoint. The result's ``metadata`` always
        discloses the served content type and whether a transform was applied
        (``html-to-markdown`` or ``none``). Non-text content types (images,
        binaries, PDFs) are rejected with an explicit error. Large bodies are
        offloaded to the artifact store with a bounded preview instead of
        flooding the context.

        Tags:
            web, url, article, fetch, http, browse, documentation, link, json, api

        Args:
            url: Full URL of the page to fetch. Example: 'https://example.com/article'
            timeout: Seconds before the request is abandoned. Defaults to 15.0.
        """
        try:
            resp = httpx.get(
                url,
                follow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            media_type = content_type.split(";", 1)[0].strip().lower()

            if (
                media_type
                and not media_type.startswith("text/")
                and media_type
                not in (
                    "application/xhtml+xml",
                    "application/json",
                    "application/xml",
                )
            ):
                return f"Error: unsupported content type '{media_type}' for {url}"

            is_html = media_type in ("text/html", "application/xhtml+xml")
            if is_html:
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                text = h.handle(resp.text)
                transform = "html-to-markdown"
            else:
                text = resp.text
                transform = "none"

            metadata: dict = {
                "content_type": media_type or "unknown",
                "transform": transform,
            }

            truncated = len(text) > ARTICLE_FETCH_MAX_CHARS
            if truncated:
                # Sanity cap on a pathological page — guards fetch size, not fidelity
                # within it (the kept portion still offloads losslessly below).
                if is_html:
                    text = (
                        text[:ARTICLE_FETCH_MAX_CHARS]
                        + f"\n\n[...truncated — article exceeds "
                        f"{ARTICLE_FETCH_MAX_CHARS} chars]"
                    )
                else:
                    # Verbatim path: truncation is disclosed in metadata only —
                    # no text is appended into a machine-readable body.
                    text = text[:ARTICLE_FETCH_MAX_CHARS]
                    metadata["truncated"] = True

            return None, {"stdout": text, "ok": True, "metadata": metadata}
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} fetching {url}"
        except Exception as e:
            return f"Error fetching {url}: {e}"
