import httpx

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
        models = [m["id"] for m in resp.json().get("data", [])]
        self.logger.info("Models: %s", models)
        return models

    def chat_completion(self, messages, tools=None) -> Response:
        self.logger.info("Request sent to %s", self.api_url + "/chat/completions")
        body = {"model": self.model, "messages": messages, "tools": tools or []}
        # self.logger.debug("Request body: %s", body)
        resp = self.session.post(
            self.api_url + "/chat/completions",
            json=body,
        )
        self.logger.info("Received response: %s (%d bytes)", resp.status_code, len(resp.content))
        self.logger.info("Response content: %s", resp.json())
        # debbuging highlight reasoning content in response for better visibility
        choices = resp.json().get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")
            if content:
                self.logger.info("LLM response content:\n%s", content)
            if reasoning_content:
                self.logger.info("LLM response reasoning content:\n%s", reasoning_content)
        return resp


    def execute_tool_calls(self, message) -> list:
        from colorama import Fore, Style
        TOOL_COLOR = Fore.MAGENTA
        tool_calls = message.get("tool_calls", [])
        tool_calls = tool_calls or []
        messages = []
        registry = ToolsRegistry()

        self.logger.debug(
            "%s[tool dispatch]%s found %d tool call(s) to execute",
            TOOL_COLOR, Style.RESET_ALL, len(tool_calls)
        )

        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "unknown_id")

            if tool_call["type"] != "function":
                self.logger.warning(
                    "%s[tool skip]%s %s — type '%s' is not supported",
                    TOOL_COLOR, Style.RESET_ALL, tool_call_id, tool_call["type"]
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
                "%s[tool call]%s %s → %s(%s)",
                TOOL_COLOR, Style.RESET_ALL, tool_call_id, func_name, args
            )

            if not hasattr(registry, func_name):
                error_msg = f"Error: tool '{func_name}' not found in ToolsRegistry"
                self.logger.error(
                    "%s[tool not found]%s %s — '%s' is not registered. Returning error to LLM.",
                    TOOL_COLOR, Style.RESET_ALL, tool_call_id, func_name
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
                    "%s[tool result]%s %s → %s returned: %s",
                    TOOL_COLOR, Style.RESET_ALL, tool_call_id, func_name, result
                )
            except Exception as exc:
                result = f"Error: tool '{func_name}' raised {type(exc).__name__}: {exc}"
                self.logger.error(
                    "%s[tool error]%s %s → %s raised: %s",
                    TOOL_COLOR, Style.RESET_ALL, tool_call_id, func_name, exc
                )

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

        return messages
