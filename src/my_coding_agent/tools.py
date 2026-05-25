import inspect
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
        """Run a shell command and return its stdout, stderr, and exit code.
        Use for running tests, installing packages, git operations, or any shell task."""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        parts = []
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout.rstrip()}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr.rstrip()}")
        parts.append(f"exit_code: {result.returncode}")
        return "\n".join(parts)

    @staticmethod
    def read_file(path: str) -> str:
        """Read and return the full contents of a file at the given path.
        Use to inspect source code, configs, or any text file before editing."""
        try:
            return Path(path).read_text()
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except Exception as e:
            return f"Error reading {path}: {e}"
    
    @staticmethod
    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories if needed.
        Use to create new files or overwrite existing ones."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"

    @staticmethod
    def read_article(url: str) -> str:
        """Fetch a web page and return its content as clean markdown.
        Use when the user provides a URL or link to an article, blog post, or documentation page."""
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
            return h.handle(resp.text)
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} fetching {url}"
        except Exception as e:
            return f"Error fetching {url}: {e}"
