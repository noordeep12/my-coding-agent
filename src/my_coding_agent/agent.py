from .llm import LLM, OMLX_API_URL, OMLX_API_KEY, OMLX_MODEL
from .utils import extract_message, extract_finish_reason, extract_usage
from .logger import get_logger, print_banner, print_run_summary

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
        # run stats — reset at the start of each run()
        self.step_num = 0
        self.stop_reason = "max_steps"
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_records: list = []
        self.elapsed_seconds: float = 0.0
        print_banner(model=self.model, tools=self.tools, context_window=self.context_window)
        self.logger.info("%s initialized with %d messages and %d tools", label, len(self.messages), len(self.tools))
    
    def add_message(self, message) -> None:
        self.messages.append(message)
        message_type = message.get("role", "unknown").upper()
        self.logger.info("%s message added to the conversation", message_type)
    
    def run(self, max_steps=5):
        # reset stats for this run
        self.step_num = 0
        self.stop_reason = "max_steps"
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_records = []
        t_start = time.monotonic()

        self.logger.info("Agent run started with max_steps: %d", max_steps)
        while True:
            self.logger.info("----------------------------------------------------------------")
            self.logger.info("----------------------------------------------------------------   STEP %d/%d", self.step_num + 1, max_steps)
            self.logger.info("----------------------------------------------------------------")

            # 1. Send current messages to LLM and get response
            resp = self.chat_completion(self.messages, tools=self.tools)
            message = extract_message(resp)
            self.add_message(message)

            # 2. Execute tool calls and add results back to messages
            tool_messages, records = self.execute_tool_calls(message)
            self.tool_records.extend(records)
            for tool_message in (tool_messages or []):
                self.add_message(tool_message)

            # Accumulate usage
            usage = extract_usage(resp)
            for key in self.total_usage:
                self.total_usage[key] += usage.get(key, 0)
            ctx = self.context_window
            ctx_str = f" / {ctx} ({self.total_usage['total_tokens'] / ctx * 100:.1f}% used)" if ctx else ""
            self.logger.info(
                "Step %d tokens — prompt: %d, completion: %d, total: %d%s",
                self.step_num + 1,
                self.total_usage["prompt_tokens"],
                self.total_usage["completion_tokens"],
                self.total_usage["total_tokens"],
                ctx_str,
            )

            # Finish conditions
            finish_reason = extract_finish_reason(resp)
            if finish_reason in ('stop', 'exit', 'quit'):
                self.stop_reason = finish_reason
                break
            if self.step_num >= max_steps:
                self.stop_reason = "max_steps"
                break
            self.step_num += 1

        self.elapsed_seconds = time.monotonic() - t_start
        print_run_summary(
            steps=self.step_num + 1,
            max_steps=max_steps,
            stop_reason=self.stop_reason,
            prompt_tokens=self.total_usage["prompt_tokens"],
            completion_tokens=self.total_usage["completion_tokens"],
            total_tokens=self.total_usage["total_tokens"],
            context_window=self.context_window,
            elapsed_seconds=self.elapsed_seconds,
            tool_records=self.tool_records,
        )
        return self.messages
