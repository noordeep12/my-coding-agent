from .llm import LLM, OMLX_API_URL, OMLX_API_KEY, OMLX_MODEL
from .utils import extract_message, extract_finish_reason, extract_usage
from .logger import get_logger, print_banner

from httpx import Response

import time

class Agent(LLM):

    def __init__(
        self, 
        api_url=OMLX_API_URL, 
        api_key=OMLX_API_KEY, 
        model=OMLX_MODEL, 
        messages=[], 
        tools=[],
        label="Agent",
    ):
        super().__init__(api_url, api_key, model)
        self.messages = messages or []
        self.tools = tools or []
        self.logger = get_logger(self.__class__.__name__)
        print_banner(model=self.model, tools=self.tools, context_window=self.context_window)
        self.logger.info("%s initialized with %d messages and %d tools", label, len(self.messages), len(self.tools))
    
    def add_message(self, message) -> None:
        self.messages.append(message)
        message_type = message.get("role", "unknown").upper()
        self.logger.info("%s message added to the conversation", message_type)
    
    def run(self, max_steps=5):
        self.logger.info("Agent run started with max_steps: %d", max_steps)
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        step_num = 0
        while True:
            self.logger.info("----------------------------------------------------------------")
            self.logger.info("----------------------------------------------------------------   STEP %d/%d", step_num+1, max_steps)
            self.logger.info("----------------------------------------------------------------")

            # 1. Send current messages to LLM and get response
            resp = self.chat_completion(self.messages, tools=self.tools)
            message = extract_message(resp)
            self.add_message(message) # assistant message

            # 2. Check if there are tool calls in the response,
            # execute them and add results back to messages
            tool_messages = self.execute_tool_calls(message)
            tool_messages = tool_messages or []
            for tool_message in tool_messages:
                self.add_message(tool_message) # tool message
            
            # Usage
            usage = extract_usage(resp)
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
            prompt     = total_usage["prompt_tokens"]
            completion = total_usage["completion_tokens"]
            total      = total_usage["total_tokens"]
            ctx        = self.context_window
            if ctx:
                pct = total / ctx * 100
                ctx_str = f" / {ctx} ({pct:.1f}% used)"
            else:
                ctx_str = ""
            self.logger.info(
                "Step %d tokens — prompt: %d, completion: %d, total: %d%s",
                step_num + 1, prompt, completion, total, ctx_str,
            )

            # Finish conditions
            finish_reason = extract_finish_reason(resp)
            if finish_reason in ('stop', 'exit', 'quit'):
                self.logger.info("Agent run stopped by LLM response: %s", finish_reason)
                break
            if step_num >= max_steps:
                self.logger.info("Agent run stopped after reaching max_steps: %d", max_steps)
                break
            step_num += 1
        
        self.logger.info("Agent run completed with %d steps", step_num)
        self.logger.info(
            "Total tokens — prompt: %d, completion: %d, total: %d%s",
            total_usage["prompt_tokens"], total_usage["completion_tokens"], total_usage["total_tokens"], ctx_str,
        )
        return self.messages
