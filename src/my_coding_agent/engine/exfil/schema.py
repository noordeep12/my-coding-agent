"""Config and constants for the secret-exfiltration guard (issue #127).

Kept separate from ``__init__.py`` (the detection logic) so the config shape
and category taxonomy can be referenced without importing the pattern set
itself — same split as ``tool_execution.schema`` / ``tool_execution.policy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sensitivity categories a match can fall under. An open taxonomy: new
# categories append here without touching the ``ok:false`` contract.
CATEGORY_ENV_FILE = "env_file"
CATEGORY_SSH_KEY = "ssh_key"
CATEGORY_CLOUD_CREDENTIALS = "cloud_credentials"
CATEGORY_NETRC = "netrc"
CATEGORY_KEY_FILE = "key_file"
CATEGORY_SECRET_NAMED_PATH = "secret_named_path"  # pragma: allowlist secret # noqa: S105 -- exfil category label, not a credential
CATEGORY_PRIVATE_KEY_CONTENT = "private_key_content"  # pragma: allowlist secret
CATEGORY_TOKEN_CONTENT = "token_content"  # noqa: S105 -- exfil category label, not a credential


@dataclass(frozen=True)
class ExfilConfig:
    """Runtime configuration for the exfiltration guard.

    ``enabled=False`` must make egress behavior byte-identical to a build
    without the guard at all (design.md decision 4 / spec "off-parity").
    """

    enabled: bool = True
    sensitive_path_patterns: tuple[str, ...] = field(default_factory=tuple)
    content_signature_patterns: tuple[str, ...] = field(default_factory=tuple)
