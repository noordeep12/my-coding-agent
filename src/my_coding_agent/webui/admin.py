"""LLM connection settings: resolution, persistence, and the Admin tab page.

Settings persist as a single row in the shared `items` table (`store.py`,
`table_name="llm_settings"`, `id="default"`) via the generic CRUD seam, so no
schema bump is needed. Resolution order is persisted setting -> environment
variable -> documented default (`engine/llm`'s own env resolution stays
unchanged; this module supplies a value on top of it for interface-launched
work).
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..engine.llm import DEFAULT_HTTP_TIMEOUT, LLM
from .store import Store

_TABLE_NAME = "llm_settings"
_ITEM_ID = "default"

#: Field -> (env var name, documented default). Mirrors `engine/llm`'s own
#: env resolution (`OMLX_API_URL`/`OMLX_API_KEY`/`OMLX_MODEL`) but reads the
#: environment live rather than importing its import-time-frozen constants,
#: so a saved setting or env change takes effect without a process restart.
_ENV_FIELDS: dict[str, tuple[str, str]] = {
    "api_url": ("OMLX_API_URL", "http://127.0.0.1:8321/v1"),
    "api_key": ("OMLX_API_KEY", "changeme"),
    "model": ("OMLX_MODEL", "Qwen3.6-35B-A3B-6bit"),
}
_FIELDS: tuple[str, ...] = ("api_url", "api_key", "model", "timeout")

_MASK = "********"


def resolve_llm_settings(store: Store) -> dict[str, Any]:
    """Resolve connection settings: persisted -> env var -> documented default."""
    saved = store.get_item(_TABLE_NAME, _ITEM_ID) or {}
    resolved: dict[str, Any] = {}
    for field, (env_var, default) in _ENV_FIELDS.items():
        value = saved.get(field)
        resolved[field] = (
            value if value not in (None, "") else os.environ.get(env_var, default)
        )
    timeout = saved.get("timeout")
    resolved["timeout"] = timeout if timeout not in (None, "") else DEFAULT_HTTP_TIMEOUT
    return resolved


def save_llm_settings(store: Store, payload: dict[str, Any]) -> None:
    """Persist only the recognized fields present (and non-empty) in *payload*."""
    saved = store.get_item(_TABLE_NAME, _ITEM_ID) or {}
    for field in _FIELDS:
        if field in payload and payload[field] not in (None, ""):
            saved[field] = payload[field]
    if store.get_item(_TABLE_NAME, _ITEM_ID) is None:
        store.create_item(_TABLE_NAME, _ITEM_ID, saved)
    else:
        store.update_item(_TABLE_NAME, _ITEM_ID, saved)


def masked_llm_settings(store: Store) -> dict[str, Any]:
    """Resolved settings with the API key masked for display."""
    resolved = resolve_llm_settings(store)
    resolved["api_key"] = _MASK if resolved.get("api_key") else ""
    return resolved


def build_llm_client(store: Store, **overrides: Any) -> LLM:
    """Construct an `LLM` client from resolved settings.

    Interface-launched eval runs call this instead of `LLM()` directly,
    so a saved setting takes effect without a process restart.
    """
    resolved = resolve_llm_settings(store)
    resolved.update(overrides)
    return LLM(
        api_url=resolved["api_url"],
        api_key=resolved["api_key"],
        model=resolved["model"],
        timeout=resolved["timeout"],
    )


# ── Admin tab page ──────────────────────────────────────────────────────────

# ruff: noqa: E501
ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin — my-coding-agent</title>
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
body{font-family:var(--font);background:var(--bg2);color:var(--text);padding:32px}
.card{max-width:520px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px}
h1{font-size:16px;margin-bottom:16px}
label{display:block;font-size:12px;color:var(--muted);margin:12px 0 4px}
input{width:100%;font-family:var(--font);font-size:13px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--text)}
.row{display:flex;gap:8px;align-items:center}
.row input{flex:1}
button{font-family:var(--font);font-size:12px;font-weight:500;padding:6px 12px;border-radius:8px;border:none;cursor:pointer}
.reveal{background:var(--bg2);color:var(--muted)}
.save{margin-top:20px;background:var(--accent);color:#fff}
.msg{font-size:12px;color:var(--muted);margin-top:10px;min-height:14px}
</style>
</head>
<body>
<div class="card">
<h1>LLM Connection Settings</h1>
<form id="f">
  <label for="api_url">API base URL</label>
  <input id="api_url" name="api_url" autocomplete="off">

  <label for="model">Model id</label>
  <input id="model" name="model" autocomplete="off">

  <label for="api_key">API key</label>
  <div class="row">
    <input id="api_key" name="api_key" type="password" autocomplete="off">
    <button type="button" class="reveal" id="reveal">Reveal</button>
  </div>

  <label for="timeout">Request timeout (seconds)</label>
  <input id="timeout" name="timeout" type="number" step="0.5" min="0">

  <button type="submit" class="save">Save</button>
  <div class="msg" id="msg"></div>
</form>
</div>
<script>
const f = document.getElementById('f');
const msg = document.getElementById('msg');
const keyInput = document.getElementById('api_key');
let keyIsMasked = false;

function loadSettings(){
  fetch('/api/admin/settings').then(r=>r.json()).then(s=>{
    f.api_url.value = s.api_url || '';
    f.model.value = s.model || '';
    f.timeout.value = s.timeout != null ? s.timeout : '';
    keyInput.value = s.api_key || '';
    keyIsMasked = !!s.api_key;
    keyInput.type = 'password';
  });
}
loadSettings();

document.getElementById('reveal').addEventListener('click', ()=>{
  if(keyInput.type === 'password'){
    keyInput.type = 'text';
  } else {
    keyInput.type = 'password';
  }
});

keyInput.addEventListener('input', ()=>{ keyIsMasked = false; });

f.addEventListener('submit', e=>{
  e.preventDefault();
  const payload = {
    api_url: f.api_url.value,
    model: f.model.value,
    timeout: f.timeout.value ? Number(f.timeout.value) : null,
  };
  if(!keyIsMasked) payload.api_key = keyInput.value;
  fetch('/api/admin/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }).then(r=>r.json()).then(()=>{
    msg.textContent = 'Saved.';
    loadSettings();
    setTimeout(()=>{ msg.textContent = ''; }, 2000);
  }).catch(()=>{ msg.textContent = 'Save failed.'; });
});
</script>
</body>
</html>"""


def admin_html() -> str:
    """Return the Admin tab's HTML page."""
    return ADMIN_HTML


def handle_admin_api_route(handler: Any, path: str, method: str, store: Store) -> bool:
    """Handle `/api/admin/settings` GET/POST. Returns True if handled."""
    if path != "/api/admin/settings":
        return False
    if method == "GET":
        handler._send_json(masked_llm_settings(store))
        return True
    if method == "POST":
        length = int(handler.headers.get("Content-Length", "0"))
        raw = handler.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except ValueError:
            handler._send_json({"error": "invalid json"}, status=400)
            return True
        if not isinstance(payload, dict):
            handler._send_json({"error": "invalid payload"}, status=400)
            return True
        save_llm_settings(store, payload)
        handler._send_json({"ok": True})
        return True
    return False
