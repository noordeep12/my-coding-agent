"""Sphinx configuration for the my-coding-agent API documentation.

The build runs under ``-W`` (warnings-as-errors) in CI per CONTRIBUTE.md §40.
Docstrings are Google-style (CONTRIBUTE.md §39), parsed by ``napoleon``.
"""

from importlib.metadata import version as _pkg_version

project = "my-coding-agent"
author = "Noordeep Singh"
copyright = "2026, Noordeep Singh"  # noqa: A001  (Sphinx requires this name)

# Single-source the version from package metadata (consistent with G-13).
release = _pkg_version("my-coding-agent")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Document members in source order so the API reads top-to-bottom.
autodoc_member_order = "bysource"

templates_path: list[str] = []
exclude_patterns = ["_build"]

html_theme = "alabaster"
