"""Tool implementations exposed to the agent.

Each public method of ``ToolRegistry`` is a callable tool the LLM can invoke.
The ``ToolExecutor`` normalizes every return value into the canonical envelope —
tools therefore stay simple and need not know about the schema.
"""

import json
import subprocess
from pathlib import Path

import html2text
import httpx

from ..observability.recorder import current_recorder
from ..utils.exceptions import PathTraversalError

# Single source of truth for the large-tool-output boundary (chars). bash output
# above this triggers artifact separation; tool_execution.MAX_TOOL_OUTPUT_CHARS
# aliases this.
ARTIFACT_THRESHOLD = 8_000


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
        Use this when a bash or read_file result was summarized and you need
        the complete content.

        Tags:
            artifact, output, result, retrieve

        Args:
            tool_call_id: The tool_call_id from a previous call whose output
                was summarized. Example: 'call_abc123'
        """
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
        from my_coding_agent.agent import (
            Agent,  # lazy import — avoids circular dependency
        )
        from my_coding_agent.llm import OMLX_API_KEY, OMLX_API_URL, OMLX_MODEL

        subagent_tools = [t for t in self._tools if t["function"]["name"] != "delegate"]
        system_prompt = (
            "You are a focused code exploration subagent. You receive a task "
            "and context from the main agent. Read files, run targeted bash "
            "commands, understand what is asked, and write a clear structured "
            "report. Do NOT modify any files. Be concise — the main agent only "
            "needs the key findings."
        )
        agent = Agent(
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
        messages = agent.run(max_steps=5)
        # Link this subagent to the delegate tool call in the parent's trace tree.
        parent_recorder = current_recorder.get()
        if parent_recorder is not None:
            parent_recorder.note_delegate_child(agent.session_id)
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                content: str = msg["content"]
                return content
        return "(subagent produced no report)"

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
