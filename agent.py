import json
import logging
import time

import httpx

OMLX_API_URL = "http://127.0.0.1:8321/v1"  # OpenAI API compatible
OMLX_API_KEY = "changeme"
OMLX_MODEL = "Qwen3.6-35B-A3B-4bit"


def tool(func):
    """Decorator that converts a Python function into an LLM tool definition."""
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    arg: {"type": "string"}
                    for arg in func.__code__.co_varnames[: func.__code__.co_argcount]
                    if arg not in ("self", "cls")
                },
                "required": [
                    arg
                    for arg in func.__code__.co_varnames[: func.__code__.co_argcount]
                    if arg not in ("self", "cls") and not arg.startswith("*")
                ],
            },
        },
    }


class ToolsRegistry:
    def get_weather(self, location):
        """Get the current weather for a location."""
        return f"The current weather in {location} is sunny with a temperature of 25 degrees Celsius."


class LLM:
    def __init__(self, api_url=OMLX_API_URL, api_key=OMLX_API_KEY, model=OMLX_MODEL):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self._setup_logger()
        self._setup_session()
        self.available_models()

    def _setup_session(self):
        self.session = httpx.Client()
        self.session.headers.update(
            {
                "content-type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            }
        )
        self.session.timeout = 30.0
        self.logger.info("HTTP session initialized")

    def _setup_logger(self):
        self.logger = logging.getLogger("LLM")
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def available_models(self):
        resp = self.session.get(self.api_url + "/models")
        models = [m["id"] for m in resp.json().get("data", [])]
        self.logger.debug("Available models: %s", models)
        return models

    def chat_completion(self, messages, tools=None):
        self.logger.info("Sending request to %s", self.api_url + "/chat/completions")
        resp = self.session.post(
            self.api_url + "/chat/completions",
            json={"model": self.model, "messages": messages, "tools": tools or []},
        )
        self.logger.info("Response: %s (%d bytes)", resp.status_code, len(resp.content))
        return resp.json().get("choices", [{}])[0].get("message", {})

    def execute_tool_calls(self, message):
        tool_calls = message.get("tool_calls") or []
        results = []
        registry = ToolsRegistry()
        for tool_call in tool_calls:
            if tool_call["type"] != "function":
                self.logger.warning("Non-function tool calls are not supported")
                continue
            func_name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            args = json.loads(args) if isinstance(args, str) else args
            self.logger.info("Calling tool: %s(%s)", func_name, args)
            if not hasattr(registry, func_name):
                self.logger.warning("Tool %s not found in ToolsRegistry", func_name)
                continue
            result = getattr(registry, func_name)(**args)
            results.append(
                {"role": "tool", "tool_call_id": tool_call["id"], "content": result}
            )
        return results


def run_agent(user_query: str):
    llm = LLM()
    tools = [tool(ToolsRegistry.get_weather)]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using the tools you have. "
                "Available tools: get_weather(location)."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    # Step 1: ask the LLM
    message = llm.chat_completion(messages, tools=tools)
    messages.append(message)
    time.sleep(1)

    # Step 2: execute any tool calls
    tool_results = llm.execute_tool_calls(message)
    messages.extend(tool_results)

    # Step 3: get final response (only needed if tools were called)
    if tool_results:
        time.sleep(1)
        message = llm.chat_completion(messages)

    return message.get("content", "")


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is the weather in San Francisco?"
    print(run_agent(query))
