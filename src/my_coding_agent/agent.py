from .llm import LLM, OMLX_API_URL, OMLX_API_KEY, OMLX_MODEL
from .utils import extract_message, extract_finish_reason, extract_usage
from ._logging import get_logger

from httpx import Response

import time

class Agent(LLM):

    def __init__(
        self, 
        api_url=OMLX_API_URL, 
        api_key=OMLX_API_KEY, 
        model=OMLX_MODEL, 
        messages=[], 
        tools=[]
    ):
        super().__init__(api_url, api_key, model)
        self.messages = messages or []
        self.tools = tools or []
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Agent initialized with API URL: %s, Model: %s", api_url, model)

    def add_message(self, message) -> None:
        self.messages.append(message)
        self.logger.debug("Added message (total: %d): %s", len(self.messages), message)
    
    def step(self) -> Response:
        # 1. Send current messages to LLM and get response
        resp = self.chat_completion(self.messages, tools=self.tools)
        message = extract_message(resp)
        self.add_message(message)

        # 2. Check if there are tool calls in the response,
        # execute them and add results back to messages
        tool_messages = self.execute_tool_calls(message)
        tool_messages = tool_messages or []
        for tool_message in tool_messages:
            self.add_message(tool_message)

        # 3. Send another request to LLM with tool results for final response
        final_resp = self.chat_completion(self.messages)
        return final_resp

    def run(self, max_steps=5):
        self.logger.info("Agent run started with max_steps: %d", max_steps)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        step_num = 0
        while True:
            self.logger.info("----------------------------------------------------------------")
            self.logger.info("----------------------------------------------------------------   STEP %d/%d", step_num+1, max_steps)
            self.logger.info("----------------------------------------------------------------")
            resp = self.step()
            # accumulate usage from the final response of each step
            usage = extract_usage(resp)
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
            self.logger.info(
                "[step %d] tokens — prompt: %d, completion: %d, total: %d",
                step_num + 1, total_usage["prompt_tokens"], total_usage["completion_tokens"], total_usage["total_tokens"]
            )
            message = extract_message(resp)
            self.add_message(message)
            finish_reason = extract_finish_reason(resp)
            if finish_reason in ('stop', 'exit', 'quit'):
                self.logger.warning("Agent run stopped by LLM response: %s", finish_reason)
                break
            if step_num >= max_steps:
                self.logger.warning("Agent run stopped after reaching max_steps: %d", max_steps)
                break
            step_num += 1
        self.logger.info("Agent run completed with %d steps", step_num)
        self.logger.info(
            "[total] tokens — prompt: %d, completion: %d, total: %d",
            total_usage["prompt_tokens"], total_usage["completion_tokens"], total_usage["total_tokens"]
        )
        return self.messages
