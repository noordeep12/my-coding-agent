"""Tool implementations exposed to the agent.

Each public method of ``ToolRegistry`` is a callable tool the LLM can invoke.
The ``ToolExecutor`` normalizes every return value into the canonical envelope —
tools therefore stay simple and need not know about the schema.
"""

import json
import re
import subprocess
from pathlib import Path

import html2text
import httpx

from ...observability.recorder import current_recorder, current_session_id
from ...utils.exceptions import PathTraversalError

# Single source of truth for the large-tool-output boundary (chars). bash output
# above this triggers artifact separation; tool_execution.MAX_TOOL_OUTPUT_CHARS
# aliases this.
ARTIFACT_THRESHOLD = 8_000

# A tool_call_id doubles as a per-artifact filename; restrict it to a safe set so
# a crafted id can never traverse out of the session's artifacts directory.
_SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def artifact_file_path(session_id: str | None, tool_call_id: str) -> Path | None:
    """Return the on-disk path for a tool call's artifact file, or None.

    Single source of truth for the per-artifact path scheme
    ``.my_coding_agent/<session>/artifacts/<tool_call_id>.txt``, shared by the
    write side (executor) and the read side (``read_tool_artifact``) so the two
    can never drift apart.

    Returns None when there is no session id or the id is unsafe as a filename
    (would traverse out of the artifacts directory). Performs no filesystem
    I/O — callers create the directory and read/write the file.

    Args:
        session_id: The current session id, or None outside an agent run.
        tool_call_id: The id whose artifact file path is requested.

    Returns:
        The artifact file path, or None when it cannot be safely constructed.
    """
    if not session_id or not _SAFE_ARTIFACT_ID.match(tool_call_id):
        return None
    return Path(".my_coding_agent") / session_id / "artifacts" / f"{tool_call_id}.txt"


class ToolRegistry:
    """Hold the callable tools exposed to the agent.

    Each public method is a tool the LLM can invoke: ``bash`` runs a shell command,
    ``read_file``/``write_file`` access the workspace (confined to ``base_dir`` to
    block path traversal), ``read_article`` fetches a URL as markdown,
    ``read_tool_artifact`` retrieves a previously stored large output, and
    ``delegate`` spawns a read-only subagent. Large outputs are stored as artifacts
    and summarized for the model rather than returned inline.
    """

    def __init__(
        self,
        artifacts: dict | None = None,
        tools: list | None = None,
        base_dir: str | None = None,
    ):
        self._artifacts = artifacts if artifacts is not None else {}
        self._tools = tools if tools is not None else []
        # Workspace root that read_file/write_file must stay within. Defaults
        # to the current working directory; override per deployment if a
        # different root applies.
        self._base_dir = (
            Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        )

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
            return None, full  # dispatcher will generate an LLM summary
        return json.dumps(full)

    def read_tool_artifact(self, tool_call_id: str) -> str:
        """Return the full stored output for a previous tool call by its id.

        Prefer the preview and skim the on-disk artifact file with bash text tools
        (grep/rg, sed, awk, jq, head/tail, wc) — that keeps only what you need in
        context. Use this tool only when you deliberately need the whole output.

        Tags:
            artifact, output, result, retrieve

        Args:
            tool_call_id: The tool_call_id from a previous call whose output was
                offloaded. Example: 'call_abc123'
        """
        # The per-artifact file persists for the whole run, so this works from any
        # step after the one that created it (unlike the per-step in-memory store).
        path = artifact_file_path(current_session_id.get(), tool_call_id)
        if path is not None and path.exists():
            return path.read_text()
        # Fallback: in-memory store (same step, or when no session dir exists).
        artifact = self._artifacts.get(tool_call_id)
        if artifact is None:
            return f"Error: no artifact found for tool_call_id '{tool_call_id}'"
        return json.dumps(artifact) if not isinstance(artifact, str) else artifact

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

        subagent_tools = [t for t in self._tools if t["function"]["name"] != "delegate"]
        system_prompt = (
            "You are a focused subagent working for a main coding assistant. "
            "You receive a task and context, and you have the same tools as the "
            "main agent. Use tools when needed — read files, run targeted bash "
            "commands, fetch URLs for web/research, and gather context — then "
            "write a clear, structured report. The task may be code or file "
            "exploration, web research, or context gathering. Do NOT modify any "
            "files. Be concise — the main agent only needs the key findings."
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
        )
        agent.execute(max_steps=DEFAULT_MAX_STEPS)
        # Link this subagent to the delegate tool call in the parent's trace tree.
        parent_recorder = current_recorder.get()
        if parent_recorder is not None:
            parent_recorder.note_delegate_child(agent.session_id)
        # Return an LLM-summarized final report of the whole subagent conversation
        # (recorded as a distinct report node) instead of a reverse-scan of the
        # last assistant message, which drops the final tool results whenever the
        # subagent is cut off mid-progress at its step ceiling.
        return agent.generate_report()

    def read_file(self, file_path: str) -> str:
        """Read and return the full contents of a file at the given file_path.
        Use to inspect source code, configs, or any text file before editing.

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
    def read_article(url: str, timeout: float = 15.0) -> str:
        """Fetch a web page and return its content as clean markdown.
        Returns at most ~6 000 tokens. Use when the user provides a URL or link
        to an article, blog post, or documentation page.

        Tags:
            web, url, article, fetch, http, browse, documentation, link

        Args:
            url: Full URL of the web page to fetch. Example: 'https://example.com/article'
            timeout: Seconds before the request is abandoned. Defaults to 15.0.
        """
        MAX_CHARS = 24_000  # ~6 000 tokens; prevents context explosion from large pages
        try:
            resp = httpx.get(
                url,
                follow_redirects=True,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            text = h.handle(resp.text)
            if len(text) > MAX_CHARS:
                text = (
                    text[:MAX_CHARS]
                    + f"\n\n[...truncated — article exceeds {MAX_CHARS} chars]"
                )
            return text
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} fetching {url}"
        except Exception as e:
            return f"Error fetching {url}: {e}"
