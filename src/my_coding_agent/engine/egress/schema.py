"""Config for the network egress filter — deterministic, env-driven.

Mirrors ``tool_execution.policy``'s opt-out env-var pattern: read at call time
(not import time) so the CLI flag and a shell-exported var behave identically,
whichever is in effect when the filter runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Opt-out switch (on by default): set to any value other than ""/"0"/"false"
# to disable the egress filter for the process.
DISABLE_ENV_VAR = "MCA_DISABLE_EGRESS_FILTER"
# Select the blocklist source (see SOURCES below). Defaults to DEFAULT_SOURCE.
SOURCE_ENV_VAR = "MCA_EGRESS_FILTER_SOURCE"
# Override the local blocklist cache file path.
CACHE_PATH_ENV_VAR = "MCA_EGRESS_CACHE_PATH"

# hagezi Threat-Intelligence-Feeds: primary source (GPL-3.0, daily,
# plain-domain format, aggregates ~60 sources incl. URLhaus/phishing feeds).
# abuse.ch URLhaus: secondary source (CC0, malware-distribution hostfile).
SOURCES = {
    "hagezi": "https://raw.githubusercontent.com/hagezi/dns-blocklists/main/domains/tif.txt",
    "urlhaus": "https://urlhaus.abuse.ch/downloads/hostfile/",
}
DEFAULT_SOURCE = "hagezi"

DEFAULT_CACHE_PATH = Path.home() / ".my_coding_agent" / "egress_cache" / "blocklist.txt"
DEFAULT_REFRESH_CADENCE_HOURS = 24.0


@dataclass(frozen=True)
class EgressConfig:
    """Egress filter configuration for one check."""

    enabled: bool = True
    source: str = DEFAULT_SOURCE
    cache_path: Path = DEFAULT_CACHE_PATH
    refresh_cadence_hours: float = DEFAULT_REFRESH_CADENCE_HOURS

    @classmethod
    def from_env(cls) -> EgressConfig:
        """Build config from environment, applying the same defaults as above."""
        raw_disabled = os.environ.get(DISABLE_ENV_VAR, "")
        enabled = raw_disabled.strip().lower() in ("", "0", "false")
        source = os.environ.get(SOURCE_ENV_VAR, DEFAULT_SOURCE)
        cache_path = Path(os.environ.get(CACHE_PATH_ENV_VAR, str(DEFAULT_CACHE_PATH)))
        return cls(enabled=enabled, source=source, cache_path=cache_path)


def is_filter_disabled() -> bool:
    """Return ``True`` when the egress filter is disabled for this process."""
    raw = os.environ.get(DISABLE_ENV_VAR, "")
    return raw.strip().lower() not in ("", "0", "false")
