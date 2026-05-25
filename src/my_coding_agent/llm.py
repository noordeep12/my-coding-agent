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
        self.context_window = None
        for m in data:
            if m["id"] == self.model:
                self.context_window = (
                    m.get("context_length")
                    or m.get("max_context_length")
                    or m.get("context_window")
                )
                break
        if self.context_window:
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
                self.logger.llm("Reasoning: %s\n\n", reasoning)
            if content:
                self.logger.llm("Content: %s\n\n", content)
        return resp


    def execute_tool_calls(self, message) -> list:
        tool_calls = message.get("tool_calls", []) or []
        messages = []
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

            self.logger.tool("%s → %s(%s)", tool_call_id, func_name, args)

            if not hasattr(registry, func_name):
                self.logger.error("not found: '%s' is not registered", func_name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: tool '{func_name}' not found in ToolsRegistry",
                })
                continue

            try:
                result = getattr(registry, func_name)(**args)
                if not isinstance(result, str):
                    result = str(result)
                self.logger.tool("%s → %s: %s", tool_call_id, func_name, result)
            except Exception as exc:
                result = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                self.logger.error("error %s → %s: %s", tool_call_id, func_name, exc)

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

        return messages
