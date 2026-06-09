from .llm import LLM, OMLX_API_URL, OMLX_API_KEY, OMLX_MODEL
from .utils import extract_message, extract_finish_reason, extract_usage
from .logger import get_logger, print_banner, print_run_summary, attach_session_log, detach_session_log
from .handoff import ContextHandoff

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

_HANDOFF_PROMPT = (
    "CONTEXT RESET REQUIRED: your context window is nearly full. "
    "Before the reset, write a structured handoff so the continuation agent can pick up exactly where you left off.\n\n"
    "Include ALL of the following:\n"
    "1. **Original task** — what was asked\n"
    "2. **Progress** — what you have completed so far (be specific)\n"
    "3. **Files created/modified** — list each file and what it contains\n"
    "4. **Key decisions** — important choices made and why\n"
    "5. **Remaining work** — exactly what still needs to be done, in order\n"
    "6. **Critical context** — any state the next agent must know to continue correctly\n\n"
    "Be exhaustive. This will be the ONLY context the continuation agent starts with."
)


class Agent(LLM):

    def __init__(
        self,
        api_url=OMLX_API_URL,
        api_key=OMLX_API_KEY,
        model=OMLX_MODEL,
        messages=[],
        tools=[],
        label="Agent",
        context_reset_threshold: float = 0.75,
        before_tool_call=None,
        after_tool_call=None,
    ):
        super().__init__(api_url, api_key, model, before_tool_call, after_tool_call)
        self.label = label
        self.messages = messages or []
        self.tools = tools or []
        self.context_reset_threshold = context_reset_threshold
        self.logger = get_logger(self.__class__.__name__)
        self.session_id = uuid.uuid4().hex[:12]
        self.started_at = datetime.now().isoformat(timespec="seconds")
        _log_path = Path(".my_coding_agent") / self.session_id / "stderr.log"
        self._session_log_path = str(_log_path)
        self._session_log_handler = attach_session_log(_log_path)
        # run stats — reset at the start of each run()
        self.step_num = 0
        self.stop_reason = "max_steps"
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_records: list = []
        self.handoff_records: list = []  # one entry per context reset that fired
        self.elapsed_seconds: float = 0.0
        print_banner(
            label=self.label,
            model=self.model,
            tools=self.tools,
            context_window=self.context_window,
            n_messages=len(self.messages),
            context_reset_threshold=self.context_reset_threshold,
            session_id=self.session_id,
        )
        self.logger.info("%s initialized with %d messages and %d tools", label, len(self.messages), len(self.tools))
    
    def add_message(self, message) -> None:
        self.messages.append(message)
        message_type = message.get("role", "unknown").upper()
        self.logger.info("%s message added to the conversation", message_type)

    def _generate_handoff(self, step_num: int, prompt_tokens: int) -> ContextHandoff:
        """Ask the LLM to summarize current state, persist handoff, return it."""
        handoff_messages = self.messages + [{"role": "user", "content": _HANDOFF_PROMPT}]
        self.logger.info("Generating context handoff summary...")
        resp = self.chat_completion(handoff_messages, tools=[])
        content = extract_message(resp).get("content", "") or ""
        handoff = ContextHandoff(
            agent_label=self.label,
            step_num=step_num,
            prompt_tokens=prompt_tokens,
            context_window=self.context_window,
            content=content,
        )
        path = handoff.save()
        self.logger.info("Handoff saved → %s", path)
        return handoff

    def _print_summary(self, max_steps: int) -> None:
        last_message = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_message = msg["content"]
                break
        print_run_summary(
            steps=self.step_num + 1,
            max_steps=max_steps,
            stop_reason=self.stop_reason,
            prompt_tokens=self.last_prompt_tokens,
            completion_tokens=self.total_usage["completion_tokens"],
            total_tokens=self.total_usage["total_tokens"],
            context_window=self.context_window,
            elapsed_seconds=self.elapsed_seconds,
            tool_records=self.tool_records,
            handoff_records=self.handoff_records,
            agent_name=self.label,
            last_message=last_message,
        )

    def _save_session_data(self, max_steps: int) -> None:
        last_message = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_message = msg["content"]
                break
        data = {
            "session_id": self.session_id,
            "agent_label": self.label,
            "model": self.model,
            "started_at": self.started_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "steps": self.step_num + 1,
            "max_steps": max_steps,
            "stop_reason": self.stop_reason,
            "total_usage": self.total_usage,
            "context_window": self.context_window,
            "context_reset_threshold": self.context_reset_threshold,
            "tool_records": self.tool_records,
            "handoff_records": self.handoff_records,
            "last_message": last_message,
        }
        out = Path(".my_coding_agent") / self.session_id / "session_data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        self.logger.info("Session data saved → %s", out)
        if self.tool_artifacts:
            artifacts_out = out.parent / "tool_artifacts.json"
            artifacts_out.write_text(json.dumps(self.tool_artifacts, indent=2))
            self.logger.info("Tool artifacts saved → %s", artifacts_out)

    def run(self, max_steps=5):
        # reset stats for this run
        self.step_num = 0
        self.stop_reason = "max_steps"
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_records = []
        self.handoff_records = []
        self.last_prompt_tokens = 0  # actual value from last API response
        t_start = time.monotonic()

        self.logger.info("Agent run started with max_steps: %d", max_steps)
        while True:
            self.logger.info("----------------------------------------------------------------")
            self.logger.info("----------------------------------------------------------------   STEP %d/%d", self.step_num + 1, max_steps)
            self.logger.info("----------------------------------------------------------------")

            # Pre-flight context check using actual tokens reported by the API in the
            # previous step. Step 1 falls back to a character estimate (no prior data).
            if self.context_window:
                ctx_tokens = self.last_prompt_tokens or len(json.dumps(self.messages)) // 2
                ctx_pct = ctx_tokens / self.context_window

                if ctx_pct >= 1.0:
                    # Hard stop — context fully exhausted, no room to generate handoff.
                    self.stop_reason = "context_limit"
                    self.logger.warning(
                        "Context limit reached: %d / %d tokens (%.1f%%). Stopping.",
                        ctx_tokens, self.context_window, ctx_pct * 100,
                    )
                    break

                elif ctx_pct >= self.context_reset_threshold:
                    # Threshold reached — generate handoff and continue in a fresh context.
                    self.logger.warning(
                        "Context reset threshold reached: %.1f%% used (%d / %d tokens). "
                        "Generating handoff and spawning continuation.",
                        ctx_pct * 100, ctx_tokens, self.context_window,
                    )
                    handoff = self._generate_handoff(self.step_num, ctx_tokens)
                    self.handoff_records.append({
                        "step":       self.step_num,
                        "ctx_tokens": ctx_tokens,
                        "ctx_pct":    ctx_pct * 100,
                        "threshold":  self.context_reset_threshold * 100,
                        "path":       handoff.path,
                        "reason":     f"prompt_tokens {ctx_tokens:,} >= {self.context_reset_threshold*100:.0f}% of {self.context_window:,}",
                    })
                    self.stop_reason = "context_reset"
                    self.elapsed_seconds = time.monotonic() - t_start
                    self._save_session_data(max_steps)
                    self._print_summary(max_steps)
                    detach_session_log(self._session_log_handler)

                    # Spawn continuation: system prompt + handoff as the only context.
                    system_messages = [m for m in self.messages if m.get("role") == "system"]
                    continuation = Agent(
                        api_url=self.api_url,
                        api_key=self.api_key,
                        model=self.model,
                        messages=system_messages + [handoff.to_user_message()],
                        tools=self.tools,
                        label=f"{self.label} (cont.)",
                        context_reset_threshold=self.context_reset_threshold,
                        before_tool_call=self._before_hook,
                        after_tool_call=self._after_hook,
                    )
                    remaining_steps = max_steps - self.step_num
                    return continuation.run(max_steps=max(remaining_steps, 1))

                elif ctx_pct >= 0.6:
                    self.logger.warning(
                        "Context at %.1f%% (%d / %d tokens) — reset at %.0f%%.",
                        ctx_pct * 100, ctx_tokens, self.context_window,
                        self.context_reset_threshold * 100,
                    )

            # Route: pick the relevant subset of tools for this step's context.
            last_user_content = next(
                (m.get("content", "") or "" for m in reversed(self.messages) if m.get("role") == "user"),
                "",
            )
            routed_tools = self.route_tools(last_user_content, self.tools)
            resp = self.chat_completion(self.messages, tools=routed_tools)
            message = extract_message(resp)
            if not message:
                self.logger.error("Step %d: API returned empty message — skipping step", self.step_num + 1)
                self.step_num += 1
                continue
            self.add_message(message)

            # Execute tool calls and add results back to messages
            tool_messages, records = self.execute_tool_calls(message)
            self.tool_records.extend(records)
            for tool_message in (tool_messages or []):
                self.add_message(tool_message)

            # Track usage — log per-step actuals, not cumulative sums
            usage = extract_usage(resp)
            step_prompt     = usage.get("prompt_tokens", 0)
            step_completion = usage.get("completion_tokens", 0)
            step_total      = usage.get("total_tokens", 0)
            self.last_prompt_tokens = step_prompt
            for key in self.total_usage:
                self.total_usage[key] += usage.get(key, 0)

            ctx = self.context_window
            ctx_str = f" / {ctx:,} ({step_prompt / ctx * 100:.1f}% ctx used)" if ctx else ""
            self.logger.info(
                "Step %d tokens — prompt: %d, completion: %d, total: %d%s",
                self.step_num + 1,
                step_prompt,
                step_completion,
                step_total,
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
        self._save_session_data(max_steps)
        self._print_summary(max_steps)
        detach_session_log(self._session_log_handler)
        return self.messages
