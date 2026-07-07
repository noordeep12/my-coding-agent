"""Network egress filter — deny-known-bad, allow-unknown.

Checks an outbound destination the harness controls (today: ``fetch_web``'s
URL) against an actively-maintained, open-source blocklist of known-malicious
domains before the connection proceeds. This is a known-bad layer on top of
whatever default-deny sandboxing exists elsewhere — it never claims to catch a
novel/unlisted host, only ones the security community already catalogued.

The blocklist is fetched to a local cache and refreshed on a documented
cadence; offline-tolerant by design — a stale or unreachable source falls
back to the last-good cache, and a wholly absent cache runs the filter *open*
(with a warning) rather than blocking all outbound work. :func:`evaluate` is
the single entry point the executor calls before dispatch, mirroring
``tool_execution.policy.evaluate``'s shape.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import httpx

from .schema import SOURCES, EgressConfig

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class EgressBlock:
    """The decision returned when a destination matches the blocklist."""

    host: str
    matched_list: str
    reason: str


def _parse_blocklist(text: str) -> set[str]:
    """Parse a blocklist body into a lowercased domain set.

    Accepts both a plain-domain-per-line format (hagezi TIF) and a hostfile
    format (``0.0.0.0 bad.example`` / ``127.0.0.1 bad.example``, URLhaus),
    ignoring blank lines and ``#`` comments.
    """
    domains: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        domain = (parts[-1] if len(parts) > 1 else parts[0]).strip().lower()
        if domain and domain != "localhost":
            domains.add(domain)
    return domains


def _fetch_blocklist(source: str) -> set[str] | None:
    """Fetch and parse the named source's blocklist; ``None`` on any failure."""
    url = SOURCES.get(source)
    if url is None:
        logger.warning("egress: unknown blocklist source %r", source)
        return None
    try:
        resp = httpx.get(url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        logger.info("egress: blocklist refresh from %r failed: %s", source, exc)
        return None
    domains = _parse_blocklist(resp.text)
    return domains or None


def _load_cache(cache_path: Path) -> set[str] | None:
    try:
        text = cache_path.read_text()
    except OSError:
        return None
    return _parse_blocklist(text) or None


def _save_cache(cache_path: Path, domains: set[str]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(sorted(domains)) + "\n")
    except OSError:
        logger.warning("egress: failed to write blocklist cache at %s", cache_path)


def _cache_is_stale(cache_path: Path, cadence_hours: float) -> bool:
    try:
        mtime = cache_path.stat().st_mtime
    except OSError:
        return True
    return (time.time() - mtime) > cadence_hours * 3600


def load_blocklist(config: EgressConfig) -> set[str]:
    """Return the current blocklist domain set, offline-tolerant.

    Uses the local cache when it is fresh; otherwise refreshes from
    ``config.source``, falling back to the last-good cache when the refresh
    fails, and to an empty set (filter runs open, with a warning) when there
    is no usable cache at all.
    """
    if not _cache_is_stale(config.cache_path, config.refresh_cadence_hours):
        cached = _load_cache(config.cache_path)
        if cached is not None:
            return cached

    fetched = _fetch_blocklist(config.source)
    if fetched is not None:
        _save_cache(config.cache_path, fetched)
        return fetched

    cached = _load_cache(config.cache_path)
    if cached is not None:
        logger.warning(
            "egress: blocklist refresh failed, using last-good cache at %s",
            config.cache_path,
        )
        return cached

    logger.warning(
        "egress: no blocklist cache available and refresh failed — "
        "filter running open (no hosts blocked) until a refresh succeeds"
    )
    return set()


def _host_matches(host: str, domains: set[str]) -> bool:
    """``True`` when ``host`` or one of its parent domains is in ``domains``."""
    host = host.lower().rstrip(".")
    labels = host.split(".")
    return any(".".join(labels[i:]) in domains for i in range(len(labels)))


def is_blocked(host: str, config: EgressConfig | None = None) -> EgressBlock | None:
    """Return an :class:`EgressBlock` when ``host`` is on the blocklist.

    ``None`` when the filter is disabled, the blocklist is unavailable
    (filter-open), or the host is not present. Matching is host/parent-domain
    based (``a.b.evil.com`` matches an ``evil.com`` blocklist entry).
    """
    config = config or EgressConfig.from_env()
    if not config.enabled:
        return None
    domains = load_blocklist(config)
    if not domains:
        return None
    if _host_matches(host, domains):
        return EgressBlock(
            host=host,
            matched_list=config.source,
            reason=(
                f"host {host!r} is present on the {config.source} "
                "known-malicious domain blocklist"
            ),
        )
    return None


def evaluate(func_name: str, args: dict) -> EgressBlock | None:
    """Return an :class:`EgressBlock` when this call's destination is blocked.

    Only ``fetch_web`` is evaluated today (interface deliberately takes
    ``func_name`` so a future sandbox-egress-allowance integration slots in
    without a signature change). Local and deterministic except for the
    periodic, cached blocklist refresh — no per-call network beyond that.
    """
    if func_name != "fetch_web":
        return None
    url = args.get("url", "")
    if not url:
        return None
    host = urllib.parse.urlparse(url).hostname
    if not host:
        return None
    return is_blocked(host)
