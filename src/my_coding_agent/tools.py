import inspect
import json
import subprocess
import html2text
import httpx
from pathlib import Path


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

    parameters = {}
    required = []
    for param in signature.parameters.values():
        if param.name in ("self", "cls"):
            continue
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}
        if (
            param.default is inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ):
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def tool(func) -> dict:
    """Decorator/converter: turn a Python function into an LLM tool definition."""
    return function_to_json(func)


class ToolsRegistry:

    @staticmethod
    def bash(command: str) -> str:
        """Run a shell command and return stdout, stderr, exit_code, and ok as JSON.
        Use for running tests, installing packages, git operations, or any shell task.
        The 'ok' field is true when exit_code is 0.
        Example: bash(command='ls -la') or bash(command='git status')"""
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
        return json.dumps({
            "stdout":    result.stdout.rstrip(),
            "stderr":    result.stderr.rstrip(),
            "exit_code": result.returncode,
            "ok":        result.returncode == 0,
        })

    @staticmethod
    def read_file(file_path: str) -> str:
        """Read and return the full contents of a file at the given file_path.
        Use to inspect source code, configs, or any text file before editing.
        Example: read_file(file_path='/path/to/file.py')"""
        try:
            return Path(file_path).read_text()
        except FileNotFoundError:
            return f"Error: file not found: {file_path}"
        except Exception as e:
            return f"Error reading {file_path}: {e}"

    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file at file_path, creating parent directories if needed.
        Use to create new files or overwrite existing ones.
        Example: write_file(file_path='/path/to/file.py', content='print(\"hello\")')"""
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
        Use when the user provides a URL or link to an article, blog post, or documentation page."""
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
