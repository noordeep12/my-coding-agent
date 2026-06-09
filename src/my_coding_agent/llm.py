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
        model=OMLX_MODEL,
        before_tool_call=None,
        after_tool_call=None,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.logger = get_logger(self.__class__.__name__)
        self.setup_session()
        self.available_models()
        self._session_log_path: str | None = None  # set by Agent after session dir is created
        self.tool_artifacts: dict = {}
        self._before_hook = before_tool_call or (lambda name, args: args)
        self._after_hook = after_tool_call or (lambda name, args, result: result)

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

    def _validate_tool_output(self, result: str, func_name: str, is_summary: bool = False) -> str:
        if not result.strip():
            return "(tool returned empty output)"
        if not is_summary and len(result) > MAX_TOOL_OUTPUT_CHARS:
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
        if func_name == "bash" and not is_summary:
            try:
                json.loads(result.split("\n[output truncated")[0])
            except json.JSONDecodeError:
                self.logger.warning("bash tool returned non-JSON output")
        return result

    def _summarize_artifact(self, artifact: dict, func_name: str, tool_call_id: str) -> str:
        prompt = (
            f"Summarize the following `{func_name}` tool output concisely for an AI coding agent. "
            "Include: exit status, key findings, any errors, and what the agent needs to know to continue its task. "
            "Be factual and brief — 3 to 8 sentences max.\n\n"
            f"Output:\n{json.dumps(artifact, indent=2)[:12_000]}"
        )
        try:
            resp = self.chat_completion([{"role": "user", "content": prompt}], tools=[])
            summary = extract_message(resp).get("content") or ""
        except Exception as exc:
            self.logger.warning("artifact summarization failed: %s", exc)
            summary = json.dumps({
                "exit_code": artifact.get("exit_code"),
                "ok": artifact.get("ok"),
                "stdout_chars": len(artifact.get("stdout", "")),
                "stderr_chars": len(artifact.get("stderr", "")),
            })
        return (
            summary.strip()
            + f'\n[Full output stored — call read_tool_artifact(tool_call_id="{tool_call_id}") to retrieve it.]'
        )

    def parse_tool_call(self, tool_call: dict) -> tuple[str, str | None, dict | None, str | None]:
        """Parse and validate a raw tool_call dict from the LLM response.

        Returns (tool_call_id, func_name, args, error).
        error is None on success; func_name is set on JSON-parse failure for record creation.
        """
        tool_call_id = tool_call.get("id", "unknown_id")

        tool_type = tool_call.get("type")
        if tool_type is None:
            self.logger.warning("skip %s — malformed tool call: missing 'type' field", tool_call_id)
            return tool_call_id, None, None, "Error: malformed tool call — missing 'type' field"
        if tool_type != "function":
            self.logger.warning("skip %s — type '%s' not supported", tool_call_id, tool_type)
            return tool_call_id, None, None, f"Error: tool type '{tool_type}' is not supported"

        func_block = tool_call.get("function")
        func_name = func_block.get("name") if func_block else None
        if not func_name:
            self.logger.warning("skip %s — malformed tool call: missing 'function.name'", tool_call_id)
            return tool_call_id, None, None, "Error: malformed tool call — missing 'function.name'"

        try:
            args = parse_tool_args(func_block.get("arguments", {}))
        except json.JSONDecodeError as exc:
            self.logger.error("malformed args %s → %s: %s", tool_call_id, func_name, exc)
            return tool_call_id, func_name, None, f"Error: could not parse tool arguments as JSON: {exc}"

        return tool_call_id, func_name, args, None

    def _apply_arg_aliases(self, func_name: str, args: dict) -> dict:
        """Remap known wrong parameter names to their correct names for func_name."""
        for wrong, correct in self._ARG_ALIASES.get(func_name, {}).items():
            if wrong in args and correct not in args:
                self.logger.warning("arg alias: %s(%s=) → %s(%s=)", func_name, wrong, func_name, correct)
                args[correct] = args.pop(wrong)
        return args

    def before_tool_call(self, func_name: str, args: dict) -> dict | None:
        """Runs before every tool dispatch: alias-remap args, then apply the user hook.

        Returns the (possibly modified) args to proceed, or None to skip the call.
        """
        args = self._apply_arg_aliases(func_name, args)
        return self._before_hook(func_name, args)

    def after_tool_call(self, func_name: str, args: dict, result: str) -> str:
        """Runs after every tool dispatch: apply the user hook to the result."""
        return self._after_hook(func_name, args, result)

    def _dispatch_tool(self, registry: ToolsRegistry, func_name: str, args: dict, tool_call_id: str) -> str:
        """Call func_name(**args), handle artifact tuples, coerce to string, and validate output."""
        result = getattr(registry, func_name)(**args)
        artifact = None
        if isinstance(result, tuple) and len(result) == 2:
            _, artifact = result
            self.tool_artifacts[tool_call_id] = artifact
            result = self._summarize_artifact(artifact, func_name, tool_call_id)
        if not isinstance(result, str):
            result = str(result)
        return self._validate_tool_output(result, func_name, is_summary=(artifact is not None))

    def _correct_args(
        self, func_name: str, args: dict, exc: Exception, sig, tool_call: dict, tool_call_id: str, attempt: int
    ) -> dict | None:
        """Ask the LLM to fix wrong args after a TypeError. Returns corrected args or None on failure."""
        correction_messages = list(getattr(self, "messages", [])) + [
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call_id, "content": f"Error: {exc}"},
            {"role": "user", "content": (
                f"Tool '{func_name}' was called with wrong arguments: {exc}. "
                f"Expected signature: {func_name}{sig}. "
                f"Please call '{func_name}' again with the correct arguments."
            )},
        ]
        correction_resp = self.chat_completion(correction_messages, tools=getattr(self, "tools", None))
        corrected = next(
            (c for c in (extract_message(correction_resp).get("tool_calls") or [])
             if c.get("function", {}).get("name") == func_name),
            None,
        )
        if not corrected:
            self.logger.warning("correction attempt %d: model did not return a %s call", attempt + 1, func_name)
            return None
        try:
            args = parse_tool_args(corrected.get("function", {}).get("arguments", {}))
        except json.JSONDecodeError:
            self.logger.warning("correction attempt %d: could not parse corrected args", attempt + 1)
            return None
        args = self._apply_arg_aliases(func_name, args)
        self.logger.tool("corrected args (attempt %d): %s(%s)", attempt + 1, func_name, args)
        return args

    def invoke_tool(
        self, tool_call_id: str, func_name: str, args: dict, registry: ToolsRegistry, tool_call: dict
    ) -> tuple[str, str, dict]:
        """Dispatch func_name with arg-correction retries. Returns (result, status, record)."""
        if not hasattr(registry, func_name):
            self.logger.error("not found: '%s' is not registered", func_name)
            valid = [n for n in dir(ToolsRegistry) if not n.startswith("_")]
            err = f"Error: tool '{func_name}' not found. Available tools: {valid}"
            return err, "error", {"name": func_name, "args": args, "ok": False, "error": f"tool '{func_name}' not found"}

        sig = inspect.signature(getattr(ToolsRegistry, func_name))

        for attempt in range(self._MAX_ARG_RETRIES + 1):
            try:
                result = self._dispatch_tool(registry, func_name, args, tool_call_id)
                self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
                return result, "success", {"name": func_name, "args": args, "ok": True}

            except TypeError as wrong_args_exc: # wrong arguments — attempt correction with the LLM
                self.logger.error("wrong args %s → %s (attempt %d/%d): %s", tool_call_id, func_name, attempt + 1, self._MAX_ARG_RETRIES, wrong_args_exc)
                retries_exhausted = attempt == self._MAX_ARG_RETRIES
                corrected_args = None if retries_exhausted else self._correct_args(func_name, args, wrong_args_exc, sig, tool_call, tool_call_id, attempt)
                if corrected_args is None:
                    err = f"Error: wrong arguments for '{func_name}' after {attempt + 1} attempt(s): {wrong_args_exc}. Expected: {func_name}{sig}"
                    return err, "error", {"name": func_name, "args": args, "ok": False, "error": str(wrong_args_exc)}
                args = corrected_args

            except Exception as exc: # other errors — log and return as error result (don't re-raise, to allow the agent to keep going)
                if not isinstance(exc, self._RECOVERABLE_EXCEPTIONS):
                    self.logger.error("non-recoverable error %s → %s: %s", tool_call_id, func_name, exc)
                    raise
                self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
                err = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                return err, "error", {"name": func_name, "args": args, "ok": False, "error": str(exc)}

    def execute_tool_calls(self, message) -> tuple[list, list]:
        """Dispatch all tool calls in message, returning (tool_messages, call_records).

        Success record: {"name": str, "args": dict, "ok": True}
        Failure record: {"name": str, "args": dict, "ok": False, "error": str}
        """
        tool_calls = message.get("tool_calls", []) or []
        messages, records = [], []
        registry = ToolsRegistry(artifacts=self.tool_artifacts)
        self.logger.tool("dispatch: %d tool call(s)", len(tool_calls))

        for tool_call in tool_calls:
            # Parse and validate the raw tool call first, to catch issues before invoking any tools.
            tool_call_id, func_name, args, error = self.parse_tool_call(tool_call)
            if error:
                status = "error"
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": error, "status": status})
                if func_name:
                    records.append({"name": func_name, "args": {}, "ok": False, "error": error})
                continue
            
            # Run the before_tool_call hook, which can modify args or return None to skip the call.
            args = self.before_tool_call(func_name, args)
            if args is None:
                self.logger.tool("skip %s → %s (before_tool_call)", tool_call_id, func_name)
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": "(tool call skipped)", "status": "skipped"})
                records.append({"name": func_name, "args": {}, "ok": False, "error": "skipped"})
                continue
            
            # Invoke the tool with retries for argument correction, and handle any exceptions.
            self.logger.tool("%s → %s(%s)", tool_call_id, func_name, args)
            result, status, record = self.invoke_tool(tool_call_id, func_name, args, registry, tool_call)

            # Post-process the result before sending it back to the LLM.
            result = self.after_tool_call(func_name, args, result)
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result, "status": status})
            records.append(record)

        return messages, records
