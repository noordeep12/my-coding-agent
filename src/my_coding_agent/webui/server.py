"""Localhost HTTP server for the unified web UI shell.

Serves a persistent-nav single-page shell at ``/`` that mounts the existing
Trace Explorer and Eval Dashboard (reusing their render helpers, not
re-implementing them) under ``/traces`` and ``/evals``. State (last-visited
route, per-tab selection) is persisted to the local SQLite store
(`store.py`) and restored on load.

Entry point::

    my-coding-agent-webui [--port 7474] [--dir .my_coding_agent]
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import click

from ..viewer.evals_server import eval_dashboard_html, handle_eval_api_route
from ..viewer.reader import list_sessions, load_session
from ..viewer.server import _check_vendor_assets, _full_html
from ..viewer.sumcheck import check_tree
from .admin import admin_html, masked_llm_settings, save_llm_settings
from .evals_config import handle_eval_config_route
from .store import Store, default_db_path

logger = logging.getLogger(__name__)

_SID_RE = re.compile(r"^[0-9a-f]{8,64}$")

#: Tab-registration contract: route -> nav label. "admin" is mounted by #155.
NAV_TABS: tuple[tuple[str, str], ...] = (
    ("traces", "Traces"),
    ("evals", "Evals"),
    ("admin", "Admin"),
)
_DEFAULT_TAB = "traces"

_UI_STATE_KEY = "shell"

#: A small subset of the vendored bundles — the shell chrome (nav + router)
#: needs Preact/hooks/htm only, not CodeMirror/markdown-it (those stay inside
#: the mounted Trace Explorer iframe, which vendors its own full set).
_SHELL_VENDOR_FILES = ("preact.min.js", "hooks.umd.js", "htm.umd.js")
_SHELL_VENDOR_TOKEN = "/*__VENDOR__*/"

# ruff: noqa: E501
SHELL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>my-coding-agent</title>
<style>
:root{
  --bg:#ffffff; --bg2:#f5f5f7; --panel:#fbfbfd; --line:#e5e5ea;
  --text:#1d1d1f; --muted:#86868b; --accent:#0071e3; --accent-soft:#e8f1fd;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#1c1c1e; --bg2:#000000; --panel:#232326; --line:#3a3a3c;
    --text:#f5f5f7; --muted:#98989d; --accent:#0a84ff; --accent-soft:#0a3d66;
  }
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:var(--font);background:var(--bg2);color:var(--text);display:flex;flex-direction:column}
#app{display:flex;flex-direction:column;height:100vh;min-width:0}
.topbar{display:flex;align-items:center;gap:16px;height:52px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line);flex:none}
.brand{font-weight:600;font-size:14px;white-space:nowrap}
.nav{display:flex;gap:4px;min-width:0;overflow-x:auto}
.nav button{font-family:var(--font);font-size:12px;font-weight:500;color:var(--muted);background:transparent;border:none;border-radius:8px;padding:6px 12px;cursor:pointer;white-space:nowrap}
.nav button:hover{color:var(--text);background:var(--bg2)}
.nav button.on{color:var(--accent);background:var(--accent-soft)}
.content{flex:1;min-height:0;min-width:0;overflow:auto}
.content iframe{border:none;width:100%;height:100%;display:block}
</style>
</head>
<body>
<div id="app"></div>
<script>/*__VENDOR__*/</script>
<script>
'use strict';
const { h, render } = window.preact;
const { useState, useEffect, useRef, useCallback } = window.preactHooks;
const html = window.htm.bind(h);

const TABS = __NAV_TABS__;
const DEFAULT_TAB = __DEFAULT_TAB__;

function tabUrl(route, selection){
  const sel = selection[route] || {};
  const params = new URLSearchParams();
  if(route === 'traces' && sel.session) params.set('session', sel.session);
  if(route === 'evals' && sel.view) params.set('view', sel.view);
  const qs = params.toString();
  return '/' + route + (qs ? ('?' + qs) : '');
}

// Cross-tab navigation: a mounted tab asks the shell to switch tabs via
// postMessage, same channel used for selection sync.
function useCrossTabNavigate(setRoute){
  useEffect(()=>{
    const onMessage = e=>{
      const d = e.data;
      if(!d || d.type !== 'mca:navigate' || !d.tab) return;
      setRoute(d.tab);
    };
    window.addEventListener('message', onMessage);
    return ()=>window.removeEventListener('message', onMessage);
  },[]);
}

// Tab-registration contract: each entry in TABS is [route, navLabel]; a route
// renders via an <iframe> mounting the existing standalone page at that path.
function App(){
  const [route, setRoute] = useState(DEFAULT_TAB);
  const [selection, setSelection] = useState({});
  const loaded = useRef(false);
  const saveTimer = useRef(null);

  useEffect(()=>{
    fetch('/api/webui/state').then(r=>r.json()).then(s=>{
      if(s && s.route && TABS.some(t=>t[0]===s.route)) setRoute(s.route);
      if(s && s.selection && typeof s.selection === 'object') setSelection(s.selection);
    }).catch(()=>{}).finally(()=>{ loaded.current = true; });
  },[]);

  const scheduleSave = useCallback((nextRoute, nextSelection)=>{
    if(saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(()=>{
      fetch('/api/webui/state', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({route: nextRoute, selection: nextSelection}),
      }).catch(()=>{});
    }, 300);
  },[]);

  useEffect(()=>{ if(loaded.current) scheduleSave(route, selection); },[route, selection]);

  useCrossTabNavigate(setRoute);

  useEffect(()=>{
    const onMessage = e=>{
      const d = e.data;
      if(!d || d.type !== 'mca:selection' || !d.tab) return;
      setSelection(prev=>{
        const sel = Object.assign({}, prev[d.tab] || {});
        if(d.session) sel.session = d.session;
        if(d.view) sel.view = d.view;
        return Object.assign({}, prev, {[d.tab]: sel});
      });
    };
    window.addEventListener('message', onMessage);
    return ()=>window.removeEventListener('message', onMessage);
  },[]);

  return html`
    <div id="app">
      <div class="topbar">
        <div class="brand">my-coding-agent</div>
        <div class="nav">
          ${TABS.map(([id, label])=>html`
            <button key=${id} class=${id===route?'on':''} onClick=${()=>setRoute(id)}>${label}</button>
          `)}
        </div>
      </div>
      <div class="content">
        <iframe src=${tabUrl(route, selection)}></iframe>
      </div>
    </div>
  `;
}

render(html`<${App}/>`, document.getElementById('app'));
</script>
</body>
</html>"""


@lru_cache(maxsize=1)
def _shell_vendor_js() -> str:
    vendor_dir = Path(__file__).parent.parent / "viewer" / "_vendor"
    return "\n".join(
        (vendor_dir / name).read_text(encoding="utf-8") for name in _SHELL_VENDOR_FILES
    )


@lru_cache(maxsize=1)
def _shell_html() -> str:
    return (
        SHELL_HTML.replace(_SHELL_VENDOR_TOKEN, _shell_vendor_js())
        .replace("__NAV_TABS__", json.dumps([list(t) for t in NAV_TABS]))
        .replace("__DEFAULT_TAB__", json.dumps(_DEFAULT_TAB))
    )


def _serve_shell(handler: _WebUIHandler) -> None:
    handler._send_html(_shell_html())


def _serve_traces(handler: _WebUIHandler) -> None:
    handler._send_html(_full_html())


def _serve_evals(handler: _WebUIHandler) -> None:
    handler._send_html(eval_dashboard_html("/evals") or "")


def _serve_admin(handler: _WebUIHandler) -> None:
    handler._send_html(admin_html())


def _serve_sessions_api(handler: _WebUIHandler) -> None:
    handler._send_json(list_sessions(handler.base_dir))


def _serve_webui_state_api(handler: _WebUIHandler) -> None:
    handler._send_json(handler.store.get_ui_state(_UI_STATE_KEY) or {})


def _serve_admin_settings_api(handler: _WebUIHandler) -> None:
    handler._send_json(masked_llm_settings(handler.store))


#: Static GET routes (no path params), dispatched by exact-path lookup so
#: `do_GET` stays a flat table lookup rather than a long if/elif chain.
_STATIC_GET_HANDLERS: dict[str, Any] = {
    "/": _serve_shell,
    "/traces": _serve_traces,
    "/evals": _serve_evals,
    "/admin": _serve_admin,
    "/api/sessions": _serve_sessions_api,
    "/api/webui/state": _serve_webui_state_api,
    "/api/admin/settings": _serve_admin_settings_api,
}


class _WebUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the unified shell: shell page, mounted tabs, API."""

    base_dir: Path  # session/eval data root, set before serve_forever()
    store: Store  # set before serve_forever()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        static = _STATIC_GET_HANDLERS.get(path)
        if static is not None:
            static(self)
            return
        if not self._dispatch_get_api(path):
            self._send_json({"error": "not found"}, status=404)

    def _dispatch_get_api(self, path: str) -> bool:
        if path.startswith("/api/evals/config"):
            return self._handle_eval_config("GET")
        if path.startswith("/api/evals/"):
            return handle_eval_api_route(self, path, self.base_dir.resolve() / "evals")
        match = re.fullmatch(r"/api/session/([^/]+)", path)
        if match:
            self._handle_session(match.group(1))
            return True
        return False

    def do_POST(self) -> None:
        self._dispatch_write("POST")

    def do_PUT(self) -> None:
        self._dispatch_write("PUT")

    def do_DELETE(self) -> None:
        self._dispatch_write("DELETE")

    def _dispatch_write(self, method: str) -> None:
        path = self.path.split("?")[0]
        if path.startswith("/api/evals/config"):
            if not self._handle_eval_config(method):
                self._send_json({"error": "not found"}, status=404)
            return
        payload = self._read_json_body()
        if payload is None:
            return
        if method == "POST" and path == "/api/webui/state":
            if not isinstance(payload, dict):
                self._send_json({"error": "invalid payload"}, status=400)
                return
            self.store.set_ui_state(_UI_STATE_KEY, payload)
            self._send_json({"ok": True})
            return
        if method == "POST" and path == "/api/admin/settings":
            if not isinstance(payload, dict):
                self._send_json({"error": "invalid payload"}, status=400)
                return
            save_llm_settings(self.store, payload)
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, status=404)

    def _read_json_body(self) -> Any | None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except ValueError:
            self._send_json({"error": "invalid json"}, status=400)
            return None

    def _handle_eval_config(self, method: str) -> bool:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?")[0]
        return handle_eval_config_route(
            self,
            method,
            path,
            raw,
            evals_root=self.base_dir.resolve() / "evals",
            sessions_root=self.base_dir.resolve(),
            store=self.store,
        )

    def _handle_session(self, session_id: str) -> None:
        if not _SID_RE.match(session_id):
            self._send_json({"error": "invalid session id"}, status=400)
            return
        base = self.base_dir.resolve()
        candidate = (base / session_id).resolve()
        if not candidate.is_relative_to(base):
            self._send_json({"error": "invalid session id"}, status=400)
            return
        events_path = candidate / "events.jsonl"
        try:
            session = load_session(events_path)
            self._send_json(dataclasses.asdict(session))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error loading session %s: %s", events_path, exc)
            self._send_json({"error": str(exc)}, status=500)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)


def run_server(
    host: str = "127.0.0.1",
    port: int = 7474,
    base_dir: Path | None = None,
) -> None:
    """Start the unified web UI shell server (blocks until Ctrl-C)."""
    _check_vendor_assets()
    base = base_dir or Path(".my_coding_agent")
    store = Store(default_db_path(base))
    _WebUIHandler.base_dir = base
    _WebUIHandler.store = store
    server = HTTPServer((host, port), _WebUIHandler)
    click.echo(f"my-coding-agent web UI → http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.server_close()
        store.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


@click.command()
@click.option("--port", default=7474, show_default=True, help="TCP port to listen on.")
@click.option(
    "--dir",
    "sessions_dir",
    default=".my_coding_agent",
    show_default=True,
    help="Root directory containing session subdirectories.",
)
@click.option(
    "--check",
    "check_session_id",
    default=None,
    help=(
        "Run the deterministic, LLM-free sum-check on this session id "
        "(and its delegated subtree) and exit — no server is started."
    ),
)
def _cli(port: int, sessions_dir: str, check_session_id: str | None) -> None:
    """Launch the unified web UI shell on localhost.

    Opens http://localhost:PORT in your browser. Press Ctrl-C to stop.
    """
    if check_session_id is not None:
        results = check_tree(Path(sessions_dir), check_session_id)
        failed = False
        for result in results:
            if result.status == "fail":
                failed = True
            label = result.status.upper()
            suffix = f": {'; '.join(result.reasons)}" if result.reasons else ""
            click.echo(f"{label} {result.session_id}{suffix}")
        sys.exit(1 if failed else 0)
    run_server(port=port, base_dir=Path(sessions_dir))
