"""PromptStageNode — a declaratively-configured stage in a custom workflow graph.

Backs the ``pipeline`` section of a declarative eval-run config (issue #228):
each config-declared stage becomes one ``PromptStageNode``, so a YAML file can
describe a multi-stage workflow (e.g. a generator/evaluator loop) without any
Python code, reusing the engine's existing conditional-transition machinery
(``pipeline.graph.Pipeline`` + ``pipeline.schema.Transition``).

Each stage owns its own private conversation (``ctx.node_threads[self.name]``)
instead of reading and appending to one conversation shared by every stage —
an evaluator must judge the generator's draft without inheriting the
generator's own system prompt or internal reasoning, and a generator revising
after rejection needs the evaluator's feedback, not the evaluator's persona.
The only thing that crosses between stages is plain output text
(``ctx.node_outputs``), and only when a stage explicitly opts in via
``receives_from``.
"""

from __future__ import annotations

from ...engine.llm import parsing as llm_parsing
from ...utils import get_logger
from ..context import RunContext
from ..node import BaseNode

_logger = get_logger(__name__)


class PromptStageNode(BaseNode):
    """One workflow-graph stage: its own thread, its own call, its own decision.

    Every call is tagged with ``kind=self.name`` (the node's own name) so the
    Trace Explorer labels and captures the full input/output of each stage
    distinctly — see ``observability.recorder.FULL_PAYLOAD_KINDS``. Because
    each stage calls the LLM with its own private thread (not the shared
    ``ctx.messages``), that captured input is exactly — and only — what this
    stage itself has seen.

    Behavior is driven entirely by construction args, no subclassing needed:

    - ``system_prompt``: seeds this stage's own thread once, on its first
      call — this stage's persona, never shared with any other stage.
    - ``seed_task``: seeds this stage's thread once, right after
      ``system_prompt`` — the overall task text (``run.task``), given only to
      the pipeline's entry stage.
    - ``receives_from``: another node's name whose latest output
      (``ctx.node_outputs``) is appended to this stage's own thread as a
      plain user message before its own ``prompt`` — the sole channel by
      which stages exchange information. ``None`` means this stage only ever
      sees its own history and ``prompt``.
    - ``prompt``: appended as a new user message before every call.
    - ``accept_if_contains``: ``None`` makes this a pure generator stage — it
      always signals ``CONTINUE`` to the next node in pipeline order. Set,
      it makes this a decision stage — a case-insensitive substring match in
      the reply signals ``STOP``; no match signals ``JUMP`` to
      ``jump_target`` (which must be a declared ``Transition`` source==this
      node's name).
    - ``jump_target``: the node name to jump back to on a non-matching
      decision-stage reply. Required when ``accept_if_contains`` is set.
    """

    def __init__(
        self,
        name: str,
        prompt: str,
        system_prompt: str | None = None,
        seed_task: str | None = None,
        receives_from: str | None = None,
        accept_if_contains: str | None = None,
        jump_target: str | None = None,
    ) -> None:
        self.name = name
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._seed_task = seed_task
        self._receives_from = receives_from
        self._accept_if_contains = accept_if_contains
        self._jump_target = jump_target

    def _own_thread(self, ctx: RunContext) -> list[dict[str, str]]:
        """Return this stage's private thread, seeding it on first use."""
        thread = ctx.node_threads.setdefault(self.name, [])
        if not thread:
            if self._system_prompt:
                thread.append({"role": "system", "content": self._system_prompt})
            if self._seed_task:
                thread.append({"role": "user", "content": self._seed_task})
        return thread

    def run(self, ctx: RunContext) -> None:
        ctx.step_num += 1
        thread = self._own_thread(ctx)

        if self._receives_from is not None:
            incoming = ctx.node_outputs.get(self._receives_from)
            if incoming is not None:
                thread.append({"role": "user", "content": incoming})
        thread.append({"role": "user", "content": self._prompt})

        resp = ctx.llm.chat_completion(thread, kind=self.name)
        ctx.last_response = resp

        message = llm_parsing.extract_message(resp)
        if not message:
            _logger.error(
                "Step %d (%s): API returned empty message — skipping",
                ctx.step_num,
                self.name,
            )
            ctx.signal = "CONTINUE"
            return
        thread.append(message)
        content = message.get("content") or ""
        ctx.node_outputs[self.name] = content
        # Mirror the reply into the run-level audit trail: the sole source of
        # ``_final_output`` (eval scoring) and the resume checkpoint — never
        # read back as LLM input by any stage, which always uses its own
        # private thread above.
        ctx.messages.append(message)

        if self._accept_if_contains is None:
            ctx.signal = "CONTINUE"
            return

        if self._accept_if_contains.upper() in content.upper():
            ctx.stop_reason = "stop"
            ctx.signal = "STOP"
            return

        if self._jump_target is None:
            raise ValueError(
                f"PromptStageNode {self.name!r} has accept_if_contains set but "
                "no jump_target — a decision stage must declare one"
            )
        ctx.signal = "JUMP"
        ctx.jump_target = self._jump_target
