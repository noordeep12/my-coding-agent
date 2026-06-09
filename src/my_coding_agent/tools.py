import inspect
import json
import re
import subprocess
import html2text
import httpx
from pathlib import Path


def _parse_tags_section(docstring: str) -> list[str]:
    """Extract tags from a Google-style Tags: section (comma-separated on one line)."""
    if not docstring:
        return []
    m = re.search(r"\bTags:\s*\n\s*(.+)", docstring)
    if not m:
        return []
    return [t.strip().lower() for t in m.group(1).split(",") if t.strip()]


def _strip_tags_section(docstring: str) -> str:
    """Return the docstring with the Tags: section removed."""
    return re.sub(r"\s*\bTags:\s*\n\s*.+", "", docstring, flags=re.DOTALL).strip()


def _parse_args_section(docstring: str) -> dict[str, str]:
    """Extract {param: description} from a Google-style Args: section."""
    if not docstring:
        return {}
    m = re.search(r"\bArgs:\s*\n(.*?)(?:\n\s*\n\S|\Z)", docstring, re.DOTALL)
    if not m:
        return {}
    block = m.group(1)
    # Detect indent of first param line to handle any indentation level.
    first_param = re.search(r"^(\s+)\w+:", block, re.MULTILINE)
    if not first_param:
        return {}
    param_indent = len(first_param.group(1))
    continuation_indent = param_indent + 1  # any deeper line continues the description
    result: dict[str, str] = {}
    current_param: str | None = None
    current_lines: list[str] = []
    for line in block.splitlines():
        if not line.strip():
            continue
        stripped = line.lstrip()
        line_indent = len(line) - len(stripped)
        if line_indent == param_indent:
            param_match = re.match(r"(\w+):\s*(.*)", stripped)
            if param_match:
                if current_param:
                    result[current_param] = " ".join(current_lines).strip()
                current_param = param_match.group(1)
                current_lines = [param_match.group(2)]
                continue
        if current_param and line_indent >= continuation_indent:
            current_lines.append(stripped)
    if current_param:
        result[current_param] = " ".join(current_lines).strip()
    return result


def _strip_args_section(docstring: str) -> str:
    """Return the docstring with the Args: and Tags: sections removed (used as top-level description)."""
    cleaned = re.sub(r"\s*\bArgs:\s*\n.*", "", docstring, flags=re.DOTALL)
    cleaned = re.sub(r"\s*\bTags:\s*\n\s*.+", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def function_to_json(func) -> dict:
    """Convert a Python function into an OpenAI-compatible tool definition dict."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {e}")

    docstring = inspect.cleandoc(func.__doc__ or "")
    param_descriptions = _parse_args_section(docstring)
    tags = _parse_tags_section(docstring)
    top_description = _strip_args_section(docstring)

    parameters = {}
    required = []
    for param in signature.parameters.values():
        if param.name in ("self", "cls"):
            continue
        param_type = type_map.get(param.annotation, "string")
        entry: dict = {"type": param_type}
        if param.name in param_descriptions:
            entry["description"] = param_descriptions[param.name]
        parameters[param.name] = entry
        if (
            param.default is inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ):
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": top_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
        "tags": tags,
    }


def tool(func) -> dict:
    """Decorator/converter: turn a Python function into an LLM tool definition."""
    return function_to_json(func)


ARTIFACT_THRESHOLD = 2_000  # chars; bash output above this triggers artifact separation


class ToolsRegistry:

    def __init__(self, artifacts: dict | None = None):
        self._artifacts = artifacts if artifacts is not None else {}

    def bash(self, command: str) -> "str | tuple[None, dict]":
        """Run a shell command and return stdout, stderr, exit_code, and ok as JSON.
        Use for running tests, installing packages, git operations, or any shell task.
        The 'ok' field is true when exit_code is 0.

        Tags:
            shell, bash, execute, run, command, git, test, install, terminal

        Args:
            command: Shell command to run. Use absolute paths where possible.
                Example: 'ls -la' or 'git status'
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return json.dumps({
                "stdout":    "",
                "stderr":    "Error: command timed out after 60s",
                "exit_code": -1,
                "ok":        False,
            })
        full = {
            "stdout":    result.stdout.rstrip(),
            "stderr":    result.stderr.rstrip(),
            "exit_code": result.returncode,
            "ok":        result.returncode == 0,
        }
        if len(full["stdout"]) + len(full["stderr"]) > ARTIFACT_THRESHOLD:
            return None, full  # dispatcher will generate an LLM summary
        return json.dumps(full)

    def read_tool_artifact(self, tool_call_id: str) -> str:
        """Return the full stored output for a previous tool call identified by tool_call_id.
        Use this when a bash or read_file result was summarized and you need the complete content.

        Tags:
            artifact, output, result, retrieve

        Args:
            tool_call_id: The tool_call_id from a previous call whose output was summarized.
                Example: 'call_abc123'
        """
        artifact = self._artifacts.get(tool_call_id)
        if artifact is None:
            return f"Error: no artifact found for tool_call_id '{tool_call_id}'"
        return json.dumps(artifact) if not isinstance(artifact, str) else artifact

    def read_file(self, file_path: str) -> "str | tuple[None, dict]":
        """Read and return the full contents of a file at the given file_path.
        Use to inspect source code, configs, or any text file before editing.

        Tags:
            file, filesystem, read, inspect, source, code, config

        Args:
            file_path: Absolute path to the file to read. Example: '/path/to/file.py'
        """
        try:
            content = Path(file_path).read_text()
        except FileNotFoundError:
            return f"Error: file not found: {file_path}"
        except Exception as e:
            return f"Error reading {file_path}: {e}"
        if len(content) > ARTIFACT_THRESHOLD:
            return None, {"file_path": file_path, "content": content, "size": len(content)}
        return content

    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file at file_path, creating parent directories if needed.
        Use to create new files or overwrite existing ones.

        Tags:
            file, filesystem, write, create, edit, save

        Args:
            file_path: Absolute path where the file should be written.
                Example: '/path/to/file.py'
            content: Full text content to write. Overwrites any existing file.
        """
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Written {len(content)} bytes to {file_path}"
        except Exception as e:
            return f"Error writing {file_path}: {e}"

    @staticmethod
    def read_article(url: str) -> str:
        """Fetch a web page and return its content as clean markdown (max ~6 000 tokens).
        Use when the user provides a URL or link to an article, blog post, or documentation page.

        Tags:
            web, url, article, fetch, http, browse, documentation, link

        Args:
            url: Full URL of the web page to fetch. Example: 'https://example.com/article'
        """
        MAX_CHARS = 24_000  # ~6 000 tokens; prevents context explosion from large pages
        try:
            resp = httpx.get(
                url,
                follow_redirects=True,
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            text = h.handle(resp.text)
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + f"\n\n[...truncated — article exceeds {MAX_CHARS} chars]"
            return text
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} fetching {url}"
        except Exception as e:
            return f"Error fetching {url}: {e}"
