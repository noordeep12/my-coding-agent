import httpx
import time
import json

from colorama import Fore, Style

from ._logging import get_logger
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
        self.logger.info("Models: %s", models)
        # capture context window for the active model
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
            self.logger.info("Context window for %s: %d tokens", self.model, self.context_window)
        return models

    def chat_completion(self, messages, tools=None) -> Response:
        self.logger.info("Request sent to %s", self.api_url + "/chat/completions")
        
        body = {"model": self.model, "messages": messages, "tools": tools or []}
        self.logger.debug("Request body: %s", json.dumps(body, indent=4))

        # self.logger.debug("Request body: %s", body)
        resp = self.session.post(
            self.api_url + "/chat/completions",
            json=body,
        )
        self.logger.info("Received response: %s (%d bytes)", resp.status_code, len(resp.content))
        self.logger.debug("Response content: %s", json.dumps(resp.json(), indent=4))
        # debbuging highlight reasoning content in response for better visibility
        choices = resp.json().get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")
            if content:
                self.logger.info(
                    "%s[LLM content]\n%s\n%s%s%s\n",
                    Fore.CYAN, Style.RESET_ALL, Fore.CYAN, content, Style.RESET_ALL
                )
            if reasoning_content:
                self.logger.info(
                    "%s[LLM reasoning]\n%s\n%s%s%s\n",
                    Fore.YELLOW, Style.RESET_ALL, Fore.YELLOW, reasoning_content, Style.RESET_ALL
                )
        return resp


    def execute_tool_calls(self, message) -> list:
        TOOL_COLOR = Fore.MAGENTA
        tool_calls = message.get("tool_calls", [])
        tool_calls = tool_calls or []
        messages = []
        registry = ToolsRegistry()

        self.logger.debug(
            "%s[Tool dispatch] found %d tool call(s) to execute%s",
            TOOL_COLOR, len(tool_calls), Style.RESET_ALL,
        )

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "unknown_id")

            if tool_call["type"] != "function":
                self.logger.warning(
                    "%s[Tool skip] %s — type '%s' is not supported%s",
                    TOOL_COLOR, tool_call_id, tool_call["type"], Style.RESET_ALL,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: tool type '{tool_call['type']}' is not supported",
                })
                continue

            func_name = tool_call["function"]["name"]
            args = parse_tool_args(tool_call["function"]["arguments"])

            self.logger.info(
                "%s[Tool call] %s → %s(%s)%s",
                TOOL_COLOR, tool_call_id, func_name, args, Style.RESET_ALL,
            )

            if not hasattr(registry, func_name):
                error_msg = f"Error: tool '{func_name}' not found in ToolsRegistry"
                self.logger.error(
                    "%s[Tool not found] %s — '%s' is not registered. Returning error to LLM.%s",
                    TOOL_COLOR, tool_call_id, func_name, Style.RESET_ALL,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                })
                continue

            try:
                result = getattr(registry, func_name)(**args)
                if not isinstance(result, str):
                    result = str(result)
                self.logger.info(
                    "%s[Tool result] %s → %s returned: %s%s",
                    TOOL_COLOR, tool_call_id, func_name, result, Style.RESET_ALL,
                )
            except Exception as exc:
                result = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                self.logger.error(
                    "%s[Tool error] %s → %s raised: %s%s",
                    TOOL_COLOR, tool_call_id, func_name, exc, Style.RESET_ALL,
                )

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

        return messages
