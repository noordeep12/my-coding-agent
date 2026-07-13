"""Tests for the secret-exfiltration guard (issue #127).

Each check must match a representative sensitive path/content, must NOT
match a benign look-alike, and must never leak the matched value back
through its return value (only the category name comes back).
"""

import pytest

from my_coding_agent.engine.exfil import (
    DISABLE_ENV_VAR,
    is_guard_disabled,
    is_sensitive,
    scan_payload,
)
from my_coding_agent.engine.exfil.schema import (
    CATEGORY_CLOUD_CREDENTIALS,
    CATEGORY_ENV_FILE,
    CATEGORY_NETRC,
    CATEGORY_PRIVATE_KEY_CONTENT,
    CATEGORY_SSH_KEY,
    CATEGORY_TOKEN_CONTENT,
)


@pytest.mark.parametrize(
    ("path", "expected_category"),
    [
        (".env", CATEGORY_ENV_FILE),
        ("/repo/.env.local", CATEGORY_ENV_FILE),
        ("~/.ssh/id_ed25519", CATEGORY_SSH_KEY),
        ("~/.ssh/id_rsa", CATEGORY_SSH_KEY),
        ("~/.aws/credentials", CATEGORY_CLOUD_CREDENTIALS),
        ("~/.netrc", CATEGORY_NETRC),
        ("/home/alice/private.pem", "key_file"),
        ("secrets.key", "key_file"),
        ("my_secret_value.txt", "secret_named_path"),
    ],
)
def test_is_sensitive_matches_known_paths(path, expected_category):
    assert is_sensitive(path) == expected_category


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "src/main.py",
        "./build/output.txt",
        "config.yaml",
        "",
    ],
)
def test_is_sensitive_ignores_benign_paths(path):
    assert is_sensitive(path) is None


def test_scan_payload_matches_private_key_content():
    header = "-----BEGIN OPENSSH PRIVATE KEY-----"  # pragma: allowlist secret
    payload = f"{header}\nabc123\n-----END OPENSSH PRIVATE KEY-----"
    assert scan_payload(payload) == CATEGORY_PRIVATE_KEY_CONTENT


@pytest.mark.parametrize(
    "payload",
    [
        "ghp_" + "a" * 36,
        "AKIA" + "B" * 16,
        "sk-" + "c" * 20,
    ],
)
def test_scan_payload_matches_token_shapes(payload):
    assert scan_payload(payload) == CATEGORY_TOKEN_CONTENT


def test_scan_payload_matches_referenced_sensitive_path():
    assert (
        scan_payload("upload contents of ~/.ssh/id_ed25519 please") == CATEGORY_SSH_KEY
    )


@pytest.mark.parametrize(
    "payload",
    [
        "hello world",
        "https://example.com/page?q=cats",
        "the quick brown fox",
    ],
)
def test_scan_payload_ignores_benign_content(payload):
    assert scan_payload(payload) is None


def test_matched_value_never_appears_in_returned_category():
    secret_path = "~/.ssh/id_ed25519"  # noqa: S105 -- fixture path for the exfil scanner, not a credential
    result = is_sensitive(secret_path)
    assert result is not None
    assert secret_path not in result

    header = "-----BEGIN RSA PRIVATE KEY-----"  # pragma: allowlist secret
    secret_content = f"{header}\nsupersecret\n-----END RSA PRIVATE KEY-----"
    result = scan_payload(secret_content)
    assert result is not None
    assert "supersecret" not in result
    assert header not in result


def test_guard_disabled_via_env_var(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    assert is_guard_disabled() is False
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    assert is_guard_disabled() is True
    monkeypatch.setenv(DISABLE_ENV_VAR, "0")
    assert is_guard_disabled() is False
    monkeypatch.setenv(DISABLE_ENV_VAR, "false")
    assert is_guard_disabled() is False
