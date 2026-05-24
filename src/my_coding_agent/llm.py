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
        self.logger.debug("Response content: %s", resp.json())
        return resp


    def execute_tool_calls(self, message) -> list:
        tool_calls = message.get("tool_calls", [])
        tool_calls = tool_calls or []
        messages = []
        registry = ToolsRegistry()
        for tool_call in tool_calls:
            if tool_call["type"] != "function":
                self.logger.warning("Non-function tool calls are not supported")
                continue
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            args = parse_tool_args(args)
            # red color for tool calls to make them stand out in logs
            self.logger.warning(f"\033[91mExecuting tool: {func_name} with args {args}\033[0m")
            if not hasattr(registry, func_name):
                self.logger.warning("Tool function %s not found in ToolsRegistry", func_name)
                continue
            result = getattr(registry, func_name)(**args)
            # ensure result is a string for now, could be extended to support more complex types with better formatting
            if not isinstance(result, str):
                result = str(result)
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})
        return messages
