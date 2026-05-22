from .llm import LLM, OMLX_API_URL, OMLX_API_KEY, OMLX_MODEL
from .utils import extract_message, extract_finish_reason

class Agent(LLM):

    def __init__(self, api_url=OMLX_API_URL, api_key=OMLX_API_KEY, model=OMLX_MODEL, messages=[], tools=[]):
        super().__init__(api_url, api_key, model)
        self.messages = messages
        self.tools = tools
        self.logger.info("Agent initialized with API URL: %s, Model: %s", api_url, model)

    def add_message(self, message):
        self.messages.append(message)
        self.logger.debug("Added message: %s", message)

    def step(self):
        # 1. Send current messages to LLM and get response
        resp = self.chat_completion(self.messages, tools=self.tools)
        message = extract_message(resp)
        self.add_message(message)

        # 2. Check if there are tool calls in the response, 
        # execute them and add results back to messages
        results = self.execute_tool_calls(message)
        for result in results:
            self.add_message(result)

        # 3. Send another request to LLM with tool results for final response
        final_resp = self.chat_completion(self.messages)
        return final_resp

    def run(self, max_steps=5):
        self.logger.info("Agent run started with max_steps: %d", max_steps)
        step_num = 0
        while True:
            self.logger.info("Running Agent step %d/%d", step_num+1, max_steps)
            resp = self.step()
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
        self.logger.info("Agent run completed with final messages: %s", self.messages)
        return self.messages
