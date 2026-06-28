"""Localhost HTTP server for the Trace Explorer.

Serves the single-page browser UI at ``/`` and JSON API routes under ``/api/``.
Uses only the Python stdlib ``http.server`` module — no new runtime dependencies.

Entry point::

    my-coding-agent-traces [--port 7474] [--dir .my_coding_agent]
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import click

from .reader import list_sessions, load_session

logger = logging.getLogger(__name__)

# ── Security: allow only hex session IDs (UUID-style without dashes) ──────────
_SID_RE = re.compile(r"^[0-9a-f]{8,64}$")

# ── Embedded single-page HTML ─────────────────────────────────────────────────
# ruff: noqa: E501
EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trace Explorer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'JetBrains Mono',monospace,sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;display:grid;grid-template-rows:48px 1fr;overflow:hidden}
/* ── top bar ── */
#topbar{background:#161b22;border-bottom:1px solid #30363d;display:flex;align-items:center;gap:12px;padding:0 16px;font-size:13px}
#topbar select{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:4px 8px;font-size:12px;cursor:pointer}
#session-meta{color:#8b949e;font-size:11px;flex:1}
#session-meta b{color:#58a6ff}
/* ── main split ── */
#main{display:grid;grid-template-columns:42% 1fr;overflow:hidden}
/* ── graph pane ── */
#graph-pane{border-right:1px solid #30363d;overflow:hidden;position:relative;background:#0d1117}
#graph-svg{width:100%;height:100%;cursor:grab}
#graph-svg:active{cursor:grabbing}
/* ── SVG nodes ── */
.node-g{cursor:pointer}
.node-g text{pointer-events:none;user-select:none}
.node-label{font-size:10px;fill:#8b949e;text-anchor:middle}
.node-abbr{font-size:13px;font-weight:700;fill:#fff;text-anchor:middle;dominant-baseline:central}
.selected-ring{stroke:#58a6ff;stroke-width:3;fill:none;pointer-events:none}
.loop-ring{stroke:#f0883e;stroke-width:2;fill:none;opacity:.9;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.9;r:28}50%{opacity:.4;r:31}}
/* ── detail pane ── */
#detail-pane{overflow-y:auto;padding:0;background:#161b22}
#detail-header{padding:16px 18px;border-bottom:1px solid #30363d;background:#21262d}
#detail-header h2{font-size:14px;color:#e6edf3;margin-bottom:4px}
#detail-header .meta{font-size:11px;color:#8b949e;display:flex;gap:16px;flex-wrap:wrap}
#detail-header .meta span b{color:#c9d1d9}
.badge{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:700;margin-right:6px}
.badge-session{background:#1f4e8c;color:#58a6ff}
.badge-step{background:#1c2a1e;color:#3fb950}
.badge-router{background:#3d2b00;color:#f0883e}
.badge-llm_call{background:#1a3a1a;color:#56d364}
.badge-tool_call{background:#2d1f5e;color:#bc8cff}
.badge-handoff{background:#3d1a1a;color:#f85149}
.badge-session_end{background:#1f2430;color:#8b949e}
.loop-badge{background:#3d2000;color:#f0883e;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:6px}
/* ── accordion sections ── */
.section{border-bottom:1px solid #21262d}
.section-hdr{padding:10px 18px;font-size:12px;font-weight:600;color:#8b949e;cursor:pointer;display:flex;justify-content:space-between;align-items:center;user-select:none}
.section-hdr:hover{background:#21262d;color:#c9d1d9}
.section-hdr .chevron{transition:transform .15s}
.section.open .section-hdr .chevron{transform:rotate(90deg)}
.section-body{padding:0 18px 14px;display:none;font-size:11px}
.section.open .section-body{display:block}
/* ── kv table ── */
.kv-table{width:100%;border-collapse:collapse;margin-top:6px}
.kv-table td{padding:3px 6px;vertical-align:top;border-bottom:1px solid #21262d}
.kv-table td:first-child{color:#8b949e;white-space:nowrap;width:38%;padding-right:10px}
.kv-table td:last-child{color:#e6edf3;word-break:break-all}
pre.json-block{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:10px;overflow-x:auto;white-space:pre-wrap;word-break:break-all;font-size:10.5px;color:#e6edf3;margin-top:6px;max-height:320px;overflow-y:auto}
.copy-btn{float:right;background:#21262d;border:1px solid #30363d;color:#8b949e;padding:2px 7px;border-radius:4px;cursor:pointer;font-size:10px}
.copy-btn:hover{color:#c9d1d9}
/* ── empty / loading states ── */
#detail-empty{padding:40px;text-align:center;color:#8b949e;font-size:12px}
</style>
</head>
<body>
<div id="topbar">
  <select id="session-select" onchange="loadSession(this.value)"></select>
  <div id="session-meta">Select a session to explore</div>
</div>
<div id="main">
  <div id="graph-pane">
    <svg id="graph-svg">
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#30363d"/>
        </marker>
      </defs>
      <g id="graph-root"></g>
    </svg>
  </div>
  <div id="detail-pane">
    <div id="detail-empty">Click a node to inspect it</div>
  </div>
</div>

<script>
'use strict';
// ── state ──
let sessions=[], cur=null, selectedId=null;
let panX=20, panY=20, scale=1, dragging=false, dragStart={x:0,y:0};

// ── boot ──
async function boot(){
  sessions = await apiFetch('/api/sessions');
  const sel = document.getElementById('session-select');
  sessions.forEach(s=>{
    const o=document.createElement('option');
    o.value=s.session_id;
    o.textContent=`${s.label||s.session_id} — ${s.model||'?'} — ${(s.started_at||'').slice(0,16)}`;
    sel.appendChild(o);
  });
  if(sessions.length) loadSession(sessions[0].session_id);
}

async function loadSession(sid){
  cur = await apiFetch('/api/session/'+sid);
  selectedId=null;
  panX=20; panY=20; scale=1;
  renderGraph();
  renderSessionMeta();
  showEmpty();
}

async function apiFetch(url){
  const r=await fetch(url);
  if(!r.ok) throw new Error(r.status+' '+url);
  return r.json();
}

// ── session meta bar ──
function renderSessionMeta(){
  if(!cur) return;
  const a=cur.analytics||{};
  const cost=a.cost_usd!=null?'$'+a.cost_usd.toFixed(4):'—';
  const tok=(a.total_tokens||0).toLocaleString();
  document.getElementById('session-meta').innerHTML=
    `<b>${escHtml(cur.label)}</b> &nbsp;·&nbsp; ${escHtml(cur.model||'?')} &nbsp;·&nbsp; `+
    `steps: <b>${escHtml(String(cur.steps))}</b> &nbsp;·&nbsp; tokens: <b>${tok}</b> &nbsp;·&nbsp; `+
    `cost: <b>${cost}</b>`+
    (a.loop_count?` &nbsp;·&nbsp; <span style="color:#f0883e">⚠ ${a.loop_count} loop(s)</span>`:'');
}

// ── SVG graph ──
const SVG_NS='http://www.w3.org/2000/svg';

function renderGraph(){
  const root=document.getElementById('graph-root');
  root.innerHTML='';
  if(!cur) return;
  const edgeG=svgEl('g'); root.appendChild(edgeG);
  const nodeG=svgEl('g'); root.appendChild(nodeG);
  cur.edges.forEach(([a,b])=>edgeG.appendChild(makeEdge(a,b)));
  Object.values(cur.nodes).forEach(n=>nodeG.appendChild(makeNodeG(n)));
  applyTransform();
}

function makeEdge(fromId, toId){
  const a=cur.nodes[fromId], b=cur.nodes[toId];
  if(!a||!b) return svgEl('g');
  const mx=(a.x+b.x)/2;
  const p=svgEl('path');
  p.setAttribute('d',`M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`);
  p.setAttribute('stroke','#30363d');
  p.setAttribute('fill','none');
  p.setAttribute('stroke-width','1.5');
  p.setAttribute('marker-end','url(#arr)');
  return p;
}

function makeNodeG(node){
  const g=svgEl('g');
  g.classList.add('node-g');
  g.setAttribute('transform',`translate(${node.x},${node.y})`);
  g.setAttribute('data-id',node.id);
  if(node.loop_flag){
    const ring=svgEl('circle'); ring.setAttribute('r','28'); ring.classList.add('loop-ring'); g.appendChild(ring);
  }
  g.appendChild(makeShape(node));
  const abbr=svgEl('text'); abbr.classList.add('node-abbr'); abbr.setAttribute('y','0'); abbr.textContent=nodeAbbr(node.type); g.appendChild(abbr);
  const lbl=svgEl('text'); lbl.classList.add('node-label'); lbl.setAttribute('y','34'); lbl.textContent=truncate(node.label,22); g.appendChild(lbl);
  g.addEventListener('click',e=>{e.stopPropagation(); selectNode(node.id);});
  return g;
}

function makeShape(node){
  const c=node.color;
  if(node.shape==='diamond'){
    const p=svgEl('polygon'); p.setAttribute('points','0,-24 24,0 0,24 -24,0'); p.setAttribute('fill',c); p.setAttribute('rx','3'); return p;
  }
  if(node.shape==='circle'){
    const ci=svgEl('circle'); ci.setAttribute('r','22'); ci.setAttribute('fill',c); return ci;
  }
  if(node.shape==='square'){
    const r=svgEl('rect'); r.setAttribute('x','-20'); r.setAttribute('y','-20'); r.setAttribute('width','40'); r.setAttribute('height','40'); r.setAttribute('rx','5'); r.setAttribute('fill',c); return r;
  }
  // rect (step / session / handoff)
  const r=svgEl('rect'); r.setAttribute('x','-42'); r.setAttribute('y','-16'); r.setAttribute('width','84'); r.setAttribute('height','32'); r.setAttribute('rx','6'); r.setAttribute('fill',c); return r;
}

function nodeAbbr(type){
  return {session:'S',step:'St',router:'TR',llm_call:'LC',tool_call:'TD',handoff:'CP',session_end:'E',token_tracking:'TT',finish_check:'FC'}[type]||'?';
}

function truncate(s,n){return s&&s.length>n?s.slice(0,n-1)+'…':s||'';}

// ── selection ──
function selectNode(id){
  selectedId=id;
  document.querySelectorAll('.selected-ring').forEach(e=>e.remove());
  const g=document.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if(g){
    const ring=svgEl('circle'); ring.setAttribute('r','28'); ring.classList.add('selected-ring'); g.prepend(ring);
  }
  renderDetail(cur.nodes[id]);
}

// ── detail panel ──
function renderDetail(node){
  if(!node){showEmpty();return;}
  const dp=document.getElementById('detail-pane');
  dp.innerHTML='';
  dp.appendChild(makeDetailHeader(node));
  dp.appendChild(makeSection('Inputs', node.inputs, true));
  dp.appendChild(makeSection('Outputs', node.outputs, true));
  dp.appendChild(makeSection('Attributes', node.attributes, true));
}

function makeDetailHeader(node){
  const div=document.createElement('div'); div.id='detail-header';
  const badge=`<span class="badge badge-${node.type}">${node.type}</span>`;
  const loop=node.loop_flag?'<span class="loop-badge">⚠ loop</span>':'';
  const h2=document.createElement('h2'); h2.innerHTML=badge+loop+' '+escHtml(node.label); div.appendChild(h2);
  const meta=document.createElement('div'); meta.className='meta';
  const a=node.attributes||{};
  if(a.started_at) meta.innerHTML+=`<span>⏱ <b>${escHtml(String(a.started_at))}</b></span>`;
  if(a.latency_s!=null) meta.innerHTML+=`<span>⚡ <b>${escHtml(String(a.latency_s))}s</b></span>`;
  if(a.prompt_tokens!=null) meta.innerHTML+=`<span>📥 <b>${(a.prompt_tokens||0).toLocaleString()} tok</b></span>`;
  if(a.completion_tokens!=null) meta.innerHTML+=`<span>📤 <b>${(a.completion_tokens||0).toLocaleString()} tok</b></span>`;
  if(a.ctx_pct!=null) meta.innerHTML+=`<span>🪟 <b>${escHtml(String(a.ctx_pct))}%</b> ctx</span>`;
  div.appendChild(meta);
  return div;
}

function makeSection(title, data, startOpen){
  const sec=document.createElement('div'); sec.className='section'+(startOpen?' open':'');
  const hdr=document.createElement('div'); hdr.className='section-hdr';
  hdr.innerHTML=`<span>${title}</span><span class="chevron">▶</span>`;
  hdr.addEventListener('click',()=>sec.classList.toggle('open'));
  sec.appendChild(hdr);
  const body=document.createElement('div'); body.className='section-body';
  body.appendChild(renderData(data));
  sec.appendChild(body);
  return sec;
}

function renderData(data){
  if(data==null) return txt('—');
  if(typeof data==='string') return renderString(data);
  if(Array.isArray(data)) return renderString(JSON.stringify(data,null,2));
  if(typeof data==='object') return renderKV(data);
  return txt(String(data));
}

function renderKV(obj){
  const keys=Object.keys(obj);
  if(!keys.length) return txt('(empty)');
  const tbl=document.createElement('table'); tbl.className='kv-table';
  keys.forEach(k=>{
    const tr=document.createElement('tr');
    const td1=document.createElement('td'); td1.textContent=k;
    const td2=document.createElement('td');
    const v=obj[k];
    if(v==null){td2.style.color='#8b949e'; td2.textContent='null';}
    else if(typeof v==='string'&&v.length>80) td2.appendChild(makeJsonBlock(v));
    else if(typeof v==='object') td2.appendChild(makeJsonBlock(JSON.stringify(v,null,2)));
    else{td2.textContent=String(v);}
    tr.appendChild(td1); tr.appendChild(td2); tbl.appendChild(tr);
  });
  return tbl;
}

function renderString(s){
  if(!s||s==='') return txt('(empty)');
  return makeJsonBlock(s);
}

function makeJsonBlock(content){
  const wrap=document.createElement('div');
  const btn=document.createElement('button'); btn.className='copy-btn'; btn.textContent='copy';
  btn.addEventListener('click',()=>navigator.clipboard.writeText(content).then(()=>{btn.textContent='✓';setTimeout(()=>btn.textContent='copy',1500);}));
  const pre=document.createElement('pre'); pre.className='json-block'; pre.textContent=content;
  wrap.appendChild(btn); wrap.appendChild(pre);
  return wrap;
}

function txt(s){const sp=document.createElement('span');sp.style.color='#8b949e';sp.textContent=s;return sp;}

function showEmpty(){
  document.getElementById('detail-pane').innerHTML='<div id="detail-empty">Click a node to inspect it</div>';
}

function escHtml(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── pan + zoom ──
const svg=document.getElementById('graph-svg');
svg.addEventListener('mousedown',e=>{dragging=true;dragStart={x:e.clientX-panX,y:e.clientY-panY};});
window.addEventListener('mousemove',e=>{if(dragging){panX=e.clientX-dragStart.x;panY=e.clientY-dragStart.y;applyTransform();}});
window.addEventListener('mouseup',()=>dragging=false);
svg.addEventListener('wheel',e=>{
  const delta=e.deltaY<0?1.1:0.91;
  scale=Math.max(0.15,Math.min(5,scale*delta));
  applyTransform(); e.preventDefault();
},{passive:false});
svg.addEventListener('click',()=>{selectedId=null;document.querySelectorAll('.selected-ring').forEach(e=>e.remove());showEmpty();});

function applyTransform(){
  document.getElementById('graph-root').setAttribute('transform',`translate(${panX},${panY}) scale(${scale})`);
}

function svgEl(tag){return document.createElementNS(SVG_NS,tag);}

boot();
</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────


class _TraceHandler(BaseHTTPRequestHandler):
    """Minimal HTTP request handler for the Trace Explorer API."""

    base_dir: Path  # set as class attribute before serve_forever()

    def do_GET(self) -> None:
        """Dispatch GET requests to the appropriate handler.

        Routes:
            ``/``                        → embedded HTML viewer
            ``/api/sessions``            → session index JSON
            ``/api/session/{session_id}``→ full trace JSON
        """
        path = self.path.split("?")[0]
        if path == "/":
            self._send_html()
        elif path == "/api/sessions":
            self._send_json(list_sessions(self.base_dir))
        else:
            match = re.fullmatch(r"/api/session/([^/]+)", path)
            if match:
                self._handle_session(match.group(1))
            else:
                self._send_json({"error": "not found"}, status=404)

    def _handle_session(self, session_id: str) -> None:
        """Load and return one session as JSON.

        Validates *session_id* against an alphanumeric pattern to prevent path
        traversal attacks (CONTRIBUTE.md §32).

        Args:
            session_id: The raw session ID from the URL path.
        """
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
        """Serialise *data* and write a JSON response.

        Args:
            data: JSON-serialisable value.
            status: HTTP status code.
        """
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        """Write the embedded HTML viewer as the response body."""
        body = EMBEDDED_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route stdlib HTTP request logs to the module logger at DEBUG level."""
        logger.debug(fmt, *args)


# ── Server runner ─────────────────────────────────────────────────────────────


def run_server(
    host: str = "127.0.0.1",
    port: int = 7474,
    base_dir: Path | None = None,
) -> None:
    """Start the Trace Explorer HTTP server (blocks until Ctrl-C).

    Args:
        host: Bind address; defaults to localhost.
        port: TCP port to listen on.
        base_dir: Root directory containing session subdirectories.
            Defaults to ``.my_coding_agent`` under the current working directory.
    """
    _TraceHandler.base_dir = base_dir or Path(".my_coding_agent")
    server = HTTPServer((host, port), _TraceHandler)
    click.echo(f"Trace Explorer → http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.server_close()


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
def _cli(port: int, sessions_dir: str) -> None:
    """Launch the Trace Explorer on localhost.

    Opens http://localhost:PORT in your browser. Press Ctrl-C to stop.
    """
    run_server(port=port, base_dir=Path(sessions_dir))
