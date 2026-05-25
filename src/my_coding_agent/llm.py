import httpx
import time
import json

from .logger import get_logger
from .tools import ToolsRegistry
from .utils import parse_tool_args
from httpx import Response

OMLX_API_URL = "http://127.0.0.1:8321/v1"
OMLX_API_KEY = "changeme"
OMLX_MODEL = "Qwen3.6-35B-A3B-4bit"


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
        data = resp.json()
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


    # Known parameter aliases: maps wrong arg name → correct arg name per tool.
    # Handles recurring model hallucinations (e.g. bash(path=) instead of bash(command=)).
    _ARG_ALIASES: dict[str, dict[str, str]] = {
        "bash": {"path": "command", "cmd": "command", "script": "command", "shell": "command"},
    }

    def execute_tool_calls(self, message) -> tuple[list, list]:
        """Returns (tool_messages, call_records) where each record is {"name": str, "args": dict, "ok": bool}."""
        tool_calls = message.get("tool_calls", []) or []
        messages = []
        records = []
        registry = ToolsRegistry()

        self.logger.tool("dispatch: %d tool call(s)", len(tool_calls))

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "unknown_id")

            if tool_call["type"] != "function":
                self.logger.warning("skip %s — type '%s' not supported", tool_call_id, tool_call["type"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: tool type '{tool_call['type']}' is not supported",
                })
                continue

            func_name = tool_call["function"]["name"]
            args = parse_tool_args(tool_call["function"]["arguments"])

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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: tool '{func_name}' not found. Available tools: {valid}",
                })
                records.append({"name": func_name, "args": args, "ok": False})
                continue

            try:
                result = getattr(registry, func_name)(**args)
                if not isinstance(result, str):
                    result = str(result)
                self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
                records.append({"name": func_name, "args": args, "ok": True})
            except TypeError as exc:
                import inspect as _inspect
                sig = _inspect.signature(getattr(ToolsRegistry, func_name))
                result = f"Error: wrong arguments for '{func_name}': {exc}. Expected signature: {func_name}{sig}"
                self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
                records.append({"name": func_name, "args": args, "ok": False})
            except Exception as exc:
                result = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)
                records.append({"name": func_name, "args": args, "ok": False})

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

        return messages, records
