"""Node protocol and BaseNode — the unit of pipeline composition."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .context import RunContext


@runtime_checkable
class Node(Protocol):
    """A single executable stage in the pipeline."""

    name: str

    def run(self, ctx: RunContext) -> None:
        """Execute this stage, reading and writing ``ctx`` in place."""
        ...


class BaseNode:
    """Convenience base; subclasses set ``name`` and override ``run``."""

    name: str = ""

    def run(self, ctx: RunContext) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")
