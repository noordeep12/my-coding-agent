import httpx
import inspect
import json
import subprocess
import time

from .logger import get_logger
from .tools import ToolsRegistry
from .utils import extract_message, parse_tool_args
from httpx import Response

OMLX_API_URL = "http://127.0.0.1:8321/v1"
OMLX_API_KEY = "changeme"
OMLX_MODEL = "Qwen3.6-35B-A3B-4bit"

MAX_TOOL_OUTPUT_CHARS = 8_000


class LLM:
    def __init__(
        self,
        api_url=OMLX_API_URL,
        api_key=OMLX_API_KEY,
        model=OMLX_MODEL
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.logger = get_logger(self.__class__.__name__)
        self.setup_session()
        self.available_models()
        self._session_log_path: str | None = None  # set by Agent after session dir is created

    def setup_session(self) -> None:
        self.session = httpx.Client()
        self.session.headers.update(
            {
                "content-type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            }
        )
        self.session.timeout = 30.0

    def available_models(self) -> list:
        resp = self.session.get(self.api_url + "/models")
        data = resp.json().get("data", [])
        models = [m["id"] for m in data]
        self.logger.api("Models: %s", models)
        DEFAULT_CONTEXT_WINDOW = 131_072  # 128k fallback
        self.context_window = DEFAULT_CONTEXT_WINDOW
        for m in data:
            if m["id"] == self.model:
                self.context_window = (
                    m.get("context_length")
                    or m.get("max_context_length")
                    or m.get("context_window")
                    or DEFAULT_CONTEXT_WINDOW
                )
                break
        self.logger.api("Context window for %s: %d tokens", self.model, self.context_window)
        return models

    def chat_completion(self, messages, tools=None) -> Response:
        self.logger.api("→ POST %s", self.api_url + "/chat/completions")
        self.logger.debug("Request body: %s", json.dumps({"model": self.model, "messages": messages, "tools": tools or []}, indent=4))

        resp = self.session.post(
            self.api_url + "/chat/completions",
            json={"model": self.model, "messages": messages, "tools": tools or []},
        )
        self.logger.api("← %s (%d bytes)", resp.status_code, len(resp.content))
        try:
            data = resp.json()
        except Exception as exc:
            raise ValueError(
                f"API returned non-JSON response (HTTP {resp.status_code}): {exc}. "
                f"Body prefix: {resp.text[:200]!r}"
            ) from exc
        self.logger.debug("Response body: %s", json.dumps(data, indent=4))

        try:
            choices = data.get("choices", [])
        except Exception as exc:
            self.logger.error("Failed to parse choices: %s", exc)
            choices = []

        for choice in choices:
            message = choice.get("message", {})
            reasoning = message.get("reasoning_content") or ""
            content   = message.get("content") or ""
            if reasoning:
                self.logger.llm("Reasoning: \n\n%s\n", reasoning)
            if content:
                self.logger.llm("Content: \n\n%s\n", content)
        return resp


    # Exceptions the LLM can recover from — returned as error content, not re-raised.
    # Anything not in this tuple hard-stops the agent loop via re-raise.
    _RECOVERABLE_EXCEPTIONS = (
        TypeError,                    # wrong arg names / types — LLM can fix
        ValueError,                   # bad arg values — LLM can fix
        FileNotFoundError,            # wrong path — LLM can fix
        json.JSONDecodeError,         # malformed tool arguments — LLM can fix
        subprocess.TimeoutExpired,    # belt-and-suspenders (bash catches this itself)
    )

    # Max retries for the inner arg-correction loop before falling back to error result.
    _MAX_ARG_RETRIES: int = 3

    # Known parameter aliases: maps wrong arg name → correct arg name per tool.
    # Handles recurring model hallucinations (e.g. bash(path=) instead of bash(command=)).
    _ARG_ALIASES: dict[str, dict[str, str]] = {
        "bash":       {"path": "command", "cmd": "command", "script": "command", "shell": "command"},
        "read_file":  {"path": "file_path", "filename": "file_path", "filepath": "file_path"},
        "write_file": {"path": "file_path", "filename": "file_path", "filepath": "file_path"},
    }

    def _validate_tool_output(self, result: str, func_name: str) -> str:
        if not result.strip():
            return "(tool returned empty output)"
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            log_hint = (
                f" Use read_file(file_path='{self._session_log_path}') to inspect the full output."
                if self._session_log_path
                else ""
            )
            self.logger.warning(
                "tool output truncated: %s returned %d chars (limit %d)",
                func_name, len(result), MAX_TOOL_OUTPUT_CHARS,
            )
            result = (
                result[:MAX_TOOL_OUTPUT_CHARS]
                + f"\n[output truncated at {MAX_TOOL_OUTPUT_CHARS} chars —"
                + f" full output is in the session log.{log_hint}]"
            )
        if func_name == "bash":
            try:
                json.loads(result.split("\n[output truncated")[0])
            except json.JSONDecodeError:
                self.logger.warning("bash tool returned non-JSON output")
        return result

    def execute_tool_calls(self, message) -> tuple[list, list]:
        """Returns (tool_messages, call_records).

        Success record: {"name": str, "args": dict, "ok": True}
        Failure record: {"name": str, "args": dict, "ok": False, "error": str}

        Recoverable errors (TypeError, ValueError, FileNotFoundError, json.JSONDecodeError,
        subprocess.TimeoutExpired) are returned to the LLM as error content so it can self-correct.
        All other exceptions are re-raised to hard-stop the agent loop.
        """
        tool_calls = message.get("tool_calls", []) or []
        messages = []
        records = []
        registry = ToolsRegistry()

        self.logger.tool("dispatch: %d tool call(s)", len(tool_calls))

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "unknown_id")

            tool_type = tool_call.get("type")
            if tool_type is None:
                self.logger.warning("skip %s — malformed tool call: missing 'type' field", tool_call_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Error: malformed tool call — missing 'type' field",
                    "status": "error",
                })
                continue

            if tool_type != "function":
                self.logger.warning("skip %s — type '%s' not supported", tool_call_id, tool_type)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: tool type '{tool_type}' is not supported",
                    "status": "error",
                })
                continue

            func_block = tool_call.get("function")
            func_name = func_block.get("name") if func_block else None
            if not func_name:
                self.logger.warning("skip %s — malformed tool call: missing 'function.name'", tool_call_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Error: malformed tool call — missing 'function.name'",
                    "status": "error",
                })
                continue

            try:
                args = parse_tool_args(func_block.get("arguments", {}))
            except json.JSONDecodeError as exc:
                err = f"Error: could not parse tool arguments as JSON: {exc}"
                self.logger.error("malformed args %s → %s: %s", tool_call_id, func_name, exc)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": err,
                    "status": "error",
                })
                records.append({"name": func_name, "args": {}, "ok": False, "error": str(exc)})
                continue

            # Remap known wrong parameter names before dispatch.
            aliases = self._ARG_ALIASES.get(func_name, {})
            for wrong, correct in aliases.items():
                if wrong in args and correct not in args:
                    self.logger.warning("arg alias: %s(%s=) → %s(%s=)", func_name, wrong, func_name, correct)
                    args[correct] = args.pop(wrong)

            self.logger.tool("%s → %s(%s)", tool_call_id, func_name, args)

            if not hasattr(registry, func_name):
                self.logger.error("not found: '%s' is not registered", func_name)
                valid = [n for n in dir(ToolsRegistry) if not n.startswith("_")]
                err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": err,
                    "status": "error",
                })
                records.append({"name": func_name, "args": args, "ok": False, "error": f"tool '{func_name}' not found"})
                continue

            status = "success"
            result = None
            sig = inspect.signature(getattr(ToolsRegistry, func_name))
            last_exc: Exception | None = None

            for attempt in range(self._MAX_ARG_RETRIES + 1):
                try:
                    result = getattr(registry, func_name)(**args)
                    if not isinstance(result, str):
                        result = str(result)
                    result = self._validate_tool_output(result, func_name)
                    self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
                    records.append({"name": func_name, "args": args, "ok": True})
                    last_exc = None
                    break
                except TypeError as exc:
                    last_exc = exc
                    self.logger.error(
                        "arg error %s → %s (attempt %d/%d): %s",
                        tool_call_id, func_name, attempt + 1, self._MAX_ARG_RETRIES, exc,
                    )
                    if attempt == self._MAX_ARG_RETRIES:
                        break
                    # Inner correction loop: ask the model to fix args without advancing step counter.
                    correction_prompt = (
                        f"Tool '{func_name}' was called with wrong arguments: {exc}. "
                        f"Expected signature: {func_name}{sig}. "
                        f"Please call '{func_name}' again with the correct arguments."
                    )
                    correction_messages = list(getattr(self, "messages", [])) + [
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        {"role": "tool", "tool_call_id": tool_call_id, "content": f"Error: {exc}"},
                        {"role": "user", "content": correction_prompt},
                    ]
                    correction_resp = self.chat_completion(
                        correction_messages,
                        tools=getattr(self, "tools", None),
                    )
                    correction_msg = extract_message(correction_resp)
                    new_calls = correction_msg.get("tool_calls") or []
                    # Find the corrected call for this tool; fall back to original on mismatch.
                    corrected = next(
                        (c for c in new_calls if c.get("function", {}).get("name") == func_name),
                        None,
                    )
                    if not corrected:
                        self.logger.warning("correction attempt %d: model did not return a %s call", attempt + 1, func_name)
                        break
                    try:
                        args = parse_tool_args(corrected.get("function", {}).get("arguments", {}))
                    except json.JSONDecodeError:
                        self.logger.warning("correction attempt %d: could not parse corrected args", attempt + 1)
                        break
                    # Apply aliases to the corrected args too.
                    aliases = self._ARG_ALIASES.get(func_name, {})
                    for wrong, correct in aliases.items():
                        if wrong in args and correct not in args:
                            args[correct] = args.pop(wrong)
                    self.logger.tool("corrected args (attempt %d): %s(%s)", attempt + 1, func_name, args)
                except Exception as exc:
                    if not isinstance(exc, self._RECOVERABLE_EXCEPTIONS):
                        self.logger.error("non-recoverable tool error %s → %s: %s", tool_call_id, func_name, exc)
                        raise
                    last_exc = exc
                    result = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                    self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
                    records.append({"name": func_name, "args": args, "ok": False, "error": str(exc)})
                    status = "error"
                    break

            if last_exc is not None and status != "error":
                # TypeError exhausted all retries.
                result = (
                    f"Error: wrong arguments for '{func_name}' after {self._MAX_ARG_RETRIES} correction "
                    f"attempt(s): {last_exc}. Expected signature: {func_name}{sig}"
                )
                records.append({"name": func_name, "args": args, "ok": False, "error": str(last_exc)})
                status = "error"

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result, "status": status})

        return messages, records
