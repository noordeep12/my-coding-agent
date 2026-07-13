"""Conformance test for the project-wide error hierarchy (CONTRIBUTE.md §29).

Every exception class defined under `src/my_coding_agent/` must inherit
`MyCodingAgentError`, so a caller catching the base handles every library
failure. Standalone `Exception` subclasses in domain packages are a red flag.
"""

import importlib
import inspect
import pkgutil

import my_coding_agent
from my_coding_agent.utils.exceptions import MyCodingAgentError


def _iter_exception_classes():
    for module_info in pkgutil.walk_packages(
        my_coding_agent.__path__, prefix=f"{my_coding_agent.__name__}."
    ):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if not issubclass(obj, Exception):
                continue
            yield obj


def test_all_library_exceptions_inherit_the_base():
    offenders = [
        obj
        for obj in _iter_exception_classes()
        if obj is not MyCodingAgentError and not issubclass(obj, MyCodingAgentError)
    ]
    assert not offenders, (
        "Standalone Exception subclasses found (must inherit MyCodingAgentError): "
        f"{[f'{c.__module__}.{c.__qualname__}' for c in offenders]}"
    )
