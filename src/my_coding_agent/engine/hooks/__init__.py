"""Lifecycle hook registry — the deterministic seam for #129.

Defines ``Hooks``: a registry of developer-configured ``HookSpec``s that fires
in-process Python callables at four lifecycle events (``PreToolUse``,
``PostToolUse``, ``SessionStart``, ``SessionEnd``), independent of model
output. Hooks are loaded from a developer-owned JSON file on disk (path from
``MCA_HOOKS_CONFIG``, read at call time like ``MCA_TOOL_MAX_CONCURRENCY``/
``MCA_DISABLE_DANGEROUS_COMMAND_GATE``); the model has no path to this surface.
A run with no config set loads zero hooks, so behavior is byte-identical to
before this module existed (zero-config parity, issue #129).

Firing is fault-tolerant by default: a hook that raises is caught, logged, and
treated as ``allow`` — a buggy observability hook must never dead-lock a run.
A security hook that wants fail-closed owns that choice in its own callback
(catch its own errors and return a block).
"""

import importlib
import json
import os
from pathlib import Path
from typing import Any

from ...utils import get_logger
from .schema import HOOK_EVENTS, HookContext, HookResult, HookSpec

__all__ = ["HOOKS_CONFIG_ENV_VAR", "Hooks"]

HOOKS_CONFIG_ENV_VAR = "MCA_HOOKS_CONFIG"

logger = get_logger("hooks")


class Hooks:
    """A registry of ``HookSpec``s, matched by event and (for tool events) tool name."""

    def __init__(self, specs: list[HookSpec] | None = None) -> None:
        self._specs = specs or []

    @classmethod
    def load(cls) -> "Hooks":
        """Build a registry from the developer config named by ``MCA_HOOKS_CONFIG``.

        No env var, or a missing/unreadable file, yields an empty registry
        (zero-config parity) rather than failing the run.
        """
        path = os.environ.get(HOOKS_CONFIG_ENV_VAR)
        if not path:
            return cls()
        try:
            raw = Path(path).read_text(encoding="utf-8")
            entries = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("hooks config %r unreadable, skipping: %s", path, exc)
            return cls()
        if not isinstance(entries, list):
            logger.warning("hooks config %r must be a JSON array, skipping", path)
            return cls()
        specs = [spec for entry in entries if (spec := cls._parse_entry(entry))]
        return cls(specs)

    @staticmethod
    def _parse_entry(entry: Any) -> HookSpec | None:
        """Parse one config entry into a ``HookSpec``, or ``None`` if malformed."""
        if not isinstance(entry, dict):
            logger.warning("hooks config entry is not an object, skipping: %r", entry)
            return None
        name = entry.get("name")
        event = entry.get("event")
        target = entry.get("callable")
        tool = entry.get("tool")
        if not isinstance(name, str) or not name:
            logger.warning("hooks config entry missing 'name', skipping: %r", entry)
            return None
        if event not in HOOK_EVENTS:
            logger.warning(
                "hooks config entry %r has unknown event %r, skipping", name, event
            )
            return None
        if not isinstance(target, str) or ":" not in target:
            logger.warning(
                "hooks config entry %r missing 'module:attr' callable, skipping", name
            )
            return None
        if tool is not None and not isinstance(tool, str):
            logger.warning(
                "hooks config entry %r has non-string 'tool', skipping", name
            )
            return None
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
            callback = getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "hooks config entry %r callable %r could not be loaded, skipping: %s",
                name,
                target,
                exc,
            )
            return None
        if not callable(callback):
            logger.warning(
                "hooks config entry %r callable %r is not callable", name, target
            )
            return None
        return HookSpec(name=name, event=event, callback=callback, tool=tool)

    def fire(self, event: str, ctx: HookContext) -> list[tuple[HookSpec, HookResult]]:
        """Fire every spec matching ``event`` (and ``ctx.tool_name``), in order.

        Returns each fired ``(spec, result)`` pair for the caller to record. A
        ``PreToolUse`` block short-circuits: hooks after the first block are
        not fired. A raising hook is caught, logged, and counted as ``allow``.
        """
        firings: list[tuple[HookSpec, HookResult]] = []
        for spec in self._specs:
            if spec.event != event:
                continue
            if spec.tool is not None and ctx.tool_name != spec.tool:
                continue
            try:
                result = spec.callback(ctx)
            except Exception as exc:  # fail-open: a buggy hook must not kill the run
                logger.warning("hook %r raised, treating as allow: %s", spec.name, exc)
                result = HookResult.allow()
            if result is None:
                result = HookResult.allow()
            firings.append((spec, result))
            if result.blocked:
                break
        return firings
