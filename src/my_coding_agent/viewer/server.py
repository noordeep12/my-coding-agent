"""Localhost HTTP server for the Trace Explorer.

Serves the single-page browser UI at ``/`` and JSON API routes under ``/api/``.
Uses only the Python stdlib ``http.server`` module — no new *runtime* Python
dependencies.  The UI is a Preact + htm app; those libraries are vendored
offline under ``viewer/_vendor/`` and injected inline into the page (no CDN).

Entry point::

    my-coding-agent-traces [--port 7474] [--dir .my_coding_agent]
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import click

from ..utils.exceptions import MyCodingAgentError
from .reader import list_sessions, load_session

logger = logging.getLogger(__name__)

# ── Security: allow only hex session IDs (UUID-style without dashes) ──────────
_SID_RE = re.compile(r"^[0-9a-f]{8,64}$")

# ── Vendored UI libraries (offline, no CDN) — see viewer/_vendor/README.md ─────
_VENDOR_DIR = Path(__file__).parent / "_vendor"
_VENDOR_FILES = ("preact.min.js", "hooks.umd.js", "htm.umd.js")
_VENDOR_TOKEN = "/*__VENDOR__*/"

# ── Embedded single-page HTML (Apple-minimalist Preact UI) ────────────────────
# ruff: noqa: E501
EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace Explorer</title>
<style>
:root{
  --bg:#ffffff; --bg2:#f5f5f7; --panel:#fbfbfd; --line:#e5e5ea;
  --text:#1d1d1f; --muted:#86868b; --accent:#0071e3; --accent-soft:#e8f1fd;
  --pos:#1a7f37; --pos-bg:#e7f6ec; --neg:#d70015; --neg-bg:#fdeaec;
  --amber:#b25000; --sub:#8b7bd8; --sub-soft:#efeafb; --radius:12px;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Monaco,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:var(--font);background:var(--bg2);color:var(--text);font-size:13px;-webkit-font-smoothing:antialiased;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#app{flex:1;min-height:0;display:flex;flex-direction:column}
.empty{padding:48px;text-align:center;color:var(--muted);font-size:13px}
.muted{color:var(--muted)}
.warn{color:var(--amber)}

/* ── top bar ── */
.topbar{display:flex;align-items:center;gap:14px;height:52px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.crumbs{display:flex;align-items:center;gap:8px;flex:1;min-width:0;font-size:14px}
.crumb-root{color:var(--muted);font-weight:500}
.sep{color:var(--line)}
.crumb-cur{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40%}
.sid{font-family:var(--mono);font-size:11px;color:var(--muted);background:var(--bg2);border:1px solid var(--line);border-radius:6px;padding:2px 8px;cursor:pointer;white-space:nowrap}
.sid:hover{color:var(--accent);border-color:var(--accent)}
.sess-select{font-family:var(--font);font-size:12px;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:6px 10px;max-width:340px;cursor:pointer;outline:none}
.sess-select:focus{border-color:var(--accent)}

/* ── toolbar ── */
.toolbar{display:flex;align-items:center;justify-content:space-between;height:44px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.tb-title{font-size:13px;font-weight:600;color:var(--text)}
.filter-btn{font-family:var(--font);font-size:12px;font-weight:500;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer;display:flex;align-items:center;gap:6px}
.filter-btn:hover{border-color:var(--accent);color:var(--accent)}
.filter-btn.on{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.badge-count{background:var(--accent);color:#fff;border-radius:9px;font-size:10px;padding:0 6px;line-height:16px}
.filters{display:flex;flex-wrap:wrap;gap:8px;padding:10px 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.chip{display:flex;align-items:center;gap:6px;font-family:var(--font);font-size:12px;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:16px;padding:4px 12px;cursor:pointer}
.chip.off{opacity:.4;text-decoration:line-through}
.chip-dot{width:8px;height:8px;border-radius:50%}

/* ── stats strip ── */
.stats{display:flex;gap:18px;align-items:center;padding:8px 20px;background:var(--panel);border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}
.stats b{color:var(--text);font-weight:600}

/* ── main split ── */
.main{flex:1;display:grid;grid-template-columns:minmax(280px,38%) 1fr;min-height:0}
.rail{overflow-y:auto;min-height:0;border-right:1px solid var(--line);background:var(--panel);padding:14px 12px}
.detail{overflow-y:auto;min-height:0;background:var(--bg)}

/* ── shared node dot / tags ── */
.row-dot{width:11px;height:11px;border-radius:50%;flex:none}
.row-dot.sm{width:8px;height:8px}
.loop-tag{font-size:10px;font-weight:600;color:var(--amber);background:#fff3e6;border-radius:5px;padding:1px 6px}

/* ── tree ── */
.tree{display:flex;flex-direction:column;gap:1px}
.agroup-body{display:flex;flex-direction:column;gap:1px;padding-left:16px}
.agroup.sub>.agroup-body{border-left:2px solid var(--sub);margin-left:9px;padding-left:7px}
.agent-head{display:flex;align-items:center;gap:9px;padding:7px 12px;border-radius:8px;cursor:pointer}
.agent-head:hover{background:var(--bg2)}
.agent-head.sel{background:var(--accent-soft)}
.agent-name{font-weight:600;font-size:12px}
.agroup.sub>.agent-head .agent-name{color:var(--sub)}
.sub-tag{font-size:10px;font-weight:600;color:var(--sub);background:var(--sub-soft);border-radius:5px;padding:1px 6px}
.tleaf{display:flex;align-items:center;gap:9px;padding:7px 12px;border-radius:8px;cursor:pointer}
.tleaf:hover{background:var(--bg2)}
.tleaf.sel{background:var(--accent-soft)}
.tleaf-name{font-weight:500;font-size:12px}
.tleaf-sub{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.tleaf-sub.neg{color:var(--neg)}
.twist{display:inline-block;transition:transform .12s;color:var(--muted);font-size:10px}
.twist.open{transform:rotate(90deg)}

/* ── detail ── */
.dwrap{display:flex;flex-direction:column}
.dhead{padding:20px 22px 16px;border-bottom:1px solid var(--line)}
.dbadge{display:inline-block;color:#fff;font-size:11px;font-weight:700;border-radius:7px;padding:3px 9px;margin-right:8px;vertical-align:middle}
.dhead h2{font-size:18px;font-weight:600;margin-top:12px;word-break:break-word}
.dmeta{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--muted)}

/* ── context window card ── */
.ctxcard{margin:16px 22px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius)}
.ctxcard.sub{border-left:3px solid var(--sub)}
.ctx-sub{font-weight:600;font-size:11px;color:var(--sub)}
.ctx-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:9px}
.ctx-label{font-weight:600;font-size:12px}
.ctx-figs{font-family:var(--mono);font-size:12px;color:var(--muted)}
.ctx-bar{display:flex;height:9px;background:var(--line);border-radius:5px;overflow:hidden}
.ctx-seg{height:100%}
.ctx-fill{height:100%;border-radius:5px;background:var(--accent)}
.ctx-legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:11px}
.bd-li{display:flex;align-items:center;gap:6px;font-size:11px}
.bd-sw{width:9px;height:9px;border-radius:3px;display:inline-block}

/* ── sections ── */
.section{border-bottom:1px solid var(--line)}
.shead{display:flex;align-items:center;gap:8px;padding:12px 22px;font-weight:600;font-size:12px;cursor:pointer;user-select:none}
.shead:hover{background:var(--bg2)}
.sbody{padding:0 22px 16px}
.kv{width:100%;border-collapse:collapse;font-size:12px}
.kv td{padding:5px 8px;vertical-align:top;border-bottom:1px solid var(--bg2)}
.kk{color:var(--muted);white-space:nowrap;width:34%;font-family:var(--mono);font-size:11px}
.kv-v{word-break:break-word}
.jb{position:relative;margin-top:2px}
.jb pre{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:11px 12px;font-family:var(--mono);font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto}
.copy{position:absolute;top:7px;right:7px;font-family:var(--font);font-size:10px;color:var(--muted);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:2px 8px;cursor:pointer;opacity:0;transition:opacity .12s}
.jb:hover .copy{opacity:1}
.copy:hover{color:var(--accent);border-color:var(--accent)}
.jb.cmd pre{border-left:3px solid var(--accent)}
.jb.out pre{border-left:3px solid #c7c7cc}
.jb.err pre{border-left:3px solid var(--neg);color:var(--neg);background:var(--neg-bg)}

/* ── tool result / llm body (rendered inside .sbody) ── */
.toolres{display:flex;flex-direction:column}
.tr-head{display:flex;align-items:center;gap:9px;padding:0 0 2px}
.tr-tool{font-family:var(--mono);font-size:12px;font-weight:600;background:var(--bg2);border:1px solid var(--line);border-radius:6px;padding:2px 9px}
.tr-badge{font-size:11px;font-weight:600;border-radius:6px;padding:2px 9px}
.tr-badge.ok{color:var(--pos);background:var(--pos-bg)}
.tr-badge.err{color:var(--neg);background:var(--neg-bg)}
.tr-block{padding:12px 0 0}
.tr-block:first-child{padding-top:0}
.tr-block.muted{font-size:12px}
.tr-label{font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px;text-transform:lowercase}
.tr-label.err{color:var(--neg)}
.tcall{margin-top:10px}
.tc-top{margin-bottom:6px}

/* scrollbars */
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:#d6d6db;border-radius:5px}
::-webkit-scrollbar-thumb:hover{background:#bcbcc4}
</style>
</head>
<body>
<div id="app"></div>
<script>/*__VENDOR__*/</script>
<script>
'use strict';
const { h, render } = window.preact;
const { useState, useEffect, useRef, useMemo, useCallback } = window.preactHooks;
const html = window.htm.bind(h);

// ── node type metadata ──
const TYPE_META = {
  session:        { name:'Session Start',     dot:'#0a84ff' },
  router:         { name:'Tool Routing',      dot:'#ff9f0a' },
  llm_call:       { name:'LLM Call',          dot:'#30b350' },
  tool_call:      { name:'Tool Dispatch',     dot:'#a05cf0' },
  handoff:        { name:'Context Preflight', dot:'#ff453a' },
  report:         { name:'Subagent Report',   dot:'#5e5ce6' },
  token_tracking: { name:'Token Tracking',    dot:'#1aa3c4' },
  finish_check:   { name:'Finish Check',      dot:'#caa400' },
  session_end:    { name:'Session End',       dot:'#8e8e93' },
};
const meta = t => TYPE_META[t] || { name:t, dot:'#8e8e93' };
const fmtNum = n => (n == null ? '—' : Number(n).toLocaleString('en-US'));

async function getJSON(url){ const r = await fetch(url); if(!r.ok) throw new Error(r.status+' '+url); return r.json(); }

// ── app ──
function App(){
  const [sessions,setSessions] = useState([]);
  const [sid,setSid]           = useState(null);
  const [data,setData]         = useState(null);
  const [sel,setSel]           = useState(null);
  const [hidden,setHidden]     = useState(()=>new Set());
  const [showFilters,setShowFilters] = useState(false);
  const [collapsed,setCollapsed]     = useState(()=>new Set());

  useEffect(()=>{ getJSON('/api/sessions').then(s=>{ setSessions(s); if(s.length) setSid(s[0].session_id); }); },[]);
  useEffect(()=>{ if(!sid) return; setData(null); setSel(null); getJSON('/api/session/'+sid).then(setData); },[sid]);

  const visibleIds = useMemo(()=>{
    if(!data) return [];
    return data.order.filter(id=>{ const n=data.nodes[id]; return n && !hidden.has(n.type); });
  },[data,hidden]);

  useEffect(()=>{
    if(data && visibleIds.length && (!sel || !visibleIds.includes(sel))) setSel(visibleIds[0]);
  },[data,visibleIds]);

  // Hold a live ref to the visible ids so the (once-bound) key handler and the
  // functional setSel updater always see the current list — this keeps rapid
  // auto-repeat key presses from reading a stale selection before re-render.
  const visRef = useRef(visibleIds);
  visRef.current = visibleIds;

  const move = useCallback(dir=>{
    setSel(prev=>{
      const ids = visRef.current;
      if(!ids.length) return prev;
      const i = ids.indexOf(prev);
      return ids[Math.max(0, Math.min(ids.length-1, (i<0?0:i+dir)))];
    });
  },[]);

  useEffect(()=>{
    const onKey = e=>{
      if(e.target && /^(input|select|textarea)$/i.test(e.target.tagName)) return;
      if(e.key==='ArrowDown'||e.key==='j'){ e.preventDefault(); move(1); }
      else if(e.key==='ArrowUp'||e.key==='k'){ e.preventDefault(); move(-1); }
    };
    window.addEventListener('keydown',onKey);
    return ()=>window.removeEventListener('keydown',onKey);
  },[move]);

  if(!sessions.length) return html`<div class="empty">No sessions found. Run the agent, then reload.</div>`;

  return html`
    <${Header} sessions=${sessions} sid=${sid} data=${data} onSession=${setSid}/>
    <${Toolbar} showFilters=${showFilters} setShowFilters=${setShowFilters}
                hidden=${hidden} setHidden=${setHidden} data=${data}/>
    ${data ? html`<${Stats} data=${data}/>` : null}
    <div class="main">
      <div class="rail">
        ${!data ? html`<div class="empty">Loading…</div>`
          : html`<${Tree} data=${data} hidden=${hidden} sel=${sel} onSel=${setSel}
                          collapsed=${collapsed} setCollapsed=${setCollapsed}/>`}
      </div>
      <div class="detail">
        ${data && sel && data.nodes[sel]
          ? html`<${Detail} node=${data.nodes[sel]} mainAgent=${data.session_id}/>`
          : html`<div class="empty">Select a node to inspect how it processes the RunContext.</div>`}
      </div>
    </div>
  `;
}

function Header({sessions,sid,data,onSession}){
  const label = data ? data.label : '…';
  return html`
    <header class="topbar">
      <div class="crumbs">
        <span class="crumb-root">Traces</span>
        <span class="sep">›</span>
        <span class="crumb-cur">${label}</span>
        ${sid ? html`<span class="sid" title="Click to copy session id"
                        onClick=${()=>navigator.clipboard && navigator.clipboard.writeText(sid)}>${sid}</span>` : null}
      </div>
      <select class="sess-select" value=${sid||''} onChange=${e=>{ onSession(e.target.value); e.target.blur(); }}>
        ${sessions.map(s=>html`<option key=${s.session_id} value=${s.session_id}>
          ${(s.label||s.session_id)} · ${s.model||'?'} · ${(s.started_at||'').slice(0,16)}</option>`)}
      </select>
    </header>`;
}

function Toolbar({showFilters,setShowFilters,hidden,setHidden,data}){
  const types = data ? [...new Set(data.order.map(id=>data.nodes[id] && data.nodes[id].type).filter(Boolean))] : [];
  const toggle = t=>{ const n=new Set(hidden); n.has(t)?n.delete(t):n.add(t); setHidden(n); };
  return html`
    <div class="toolbar">
      <span class="tb-title">Pipeline tree</span>
      <button class=${'filter-btn'+(showFilters?' on':'')} onClick=${()=>setShowFilters(!showFilters)}>
        Filters${hidden.size ? html`<span class="badge-count">${hidden.size}</span>` : null}
      </button>
    </div>
    ${showFilters ? html`
      <div class="filters">
        ${types.map(t=>html`<button key=${t} class=${'chip'+(hidden.has(t)?' off':'')} onClick=${()=>toggle(t)}>
          <span class="chip-dot" style=${{background:meta(t).dot}}></span>${meta(t).name}</button>`)}
      </div>` : null}
  `;
}

function Stats({data}){
  const a = data.analytics || {};
  const cost = a.cost_usd!=null ? '$'+Number(a.cost_usd).toFixed(4) : '—';
  return html`<div class="stats">
    <span><b>${data.model||'?'}</b></span>
    <span><b>${data.steps}</b> steps</span>
    <span><b>${fmtNum(a.total_tokens||0)}</b> tokens</span>
    <span><b>${cost}</b></span>
    ${a.loop_count ? html`<span class="warn">⚠ ${a.loop_count} loop(s)</span>` : null}
    ${data.stop_reason ? html`<span class="muted">stop: ${data.stop_reason}</span>` : null}
  </div>`;
}

const ROLE_META = {
  system:   {label:'system',    color:'#6f8fd6'},
  user:     {label:'user',      color:'#4fb6a8'},
  assistant:{label:'assistant', color:'#e0995e'},
  tool:     {label:'tool',      color:'#a87bd4'},
};
const ROLE_ORDER = ['system','user','assistant','tool'];

// Cryptic router phase codes → human-readable labels.
const PHASE_LABELS = {
  phase1_keyword:'keyword-matching',
  phase1_baseline:'baseline (all tools)',
  phase2_llm:'llm-routing',
};
const phaseLabel = p => PHASE_LABELS[p] || p;

// Short "+N role" summary of what a node added to the context window.
function addedText(cs){
  if(!cs) return '';
  if(cs.removed) return '−'+fmtNum(cs.removed);
  const a = cs.added||{};
  const roles = ROLE_ORDER.filter(r=>a[r]);
  return roles.map(r=>'+'+fmtNum(a[r])+' '+ROLE_META[r].label).join('  ');
}

function Tree({data,hidden,sel,onSel,collapsed,setCollapsed}){
  // Keep the selected node reachable: expand its owning agent if collapsed.
  useEffect(()=>{
    const n = data.nodes[sel];
    if(n && collapsed.has(n.agent)){ const c=new Set(collapsed); c.delete(n.agent); setCollapsed(c); }
  },[sel]);

  // Build a nested forest from each node's depth: a `session` node opens an
  // agent group whose members are the following deeper nodes (recursive).
  const visible = data.order.map(id=>data.nodes[id])
    .filter(n=>n && (n.type==='session' || !hidden.has(n.type)));
  let i = 0;
  const build = minDepth=>{
    const out=[];
    while(i<visible.length){
      const n = visible[i];
      if(n.depth < minDepth) break;
      if(n.type==='session'){ i++; out.push({node:n, children:build(n.depth+1)}); }
      else { i++; out.push({node:n, children:null}); }
    }
    return out;
  };
  const forest = build(0);
  const toggle = a=>{ const c=new Set(collapsed); c.has(a)?c.delete(a):c.add(a); setCollapsed(c); };

  return html`<div class="tree">
    <${TreeNodes} entries=${forest} data=${data} sel=${sel} onSel=${onSel}
                  collapsed=${collapsed} toggle=${toggle}/>
  </div>`;
}

function TreeNodes({entries,data,sel,onSel,collapsed,toggle}){
  return html`${entries.map(e=> e.children!=null
    ? html`<${AgentGroup} key=${e.node.id} node=${e.node} kids=${e.children} data=${data}
              sel=${sel} onSel=${onSel} collapsed=${collapsed} toggle=${toggle}/>`
    : html`<${TreeLeaf} key=${e.node.id} node=${e.node} sel=${sel} onSel=${onSel}/>`)}`;
}

function AgentGroup({node,kids,data,sel,onSel,collapsed,toggle}){
  const isMain = node.agent===data.session_id;
  const open = !collapsed.has(node.agent);
  const name = isMain ? node.label : 'Subagent '+node.agent.slice(0,8);
  return html`<div class=${'agroup'+(isMain?'':' sub')}>
    <div class=${'agent-head'+(node.id===sel?' sel':'')}>
      <span class=${'twist'+(open?' open':'')} onClick=${()=>toggle(node.agent)}>▸</span>
      <span class="row-dot sm" style=${{background:meta(node.type).dot}}></span>
      <span class="agent-name" onClick=${()=>onSel(node.id)}>${name}</span>
      ${isMain ? null : html`<span class="sub-tag">subagent</span>`}
    </div>
    ${open ? html`<div class="agroup-body">
      <${TreeNodes} entries=${kids} data=${data} sel=${sel} onSel=${onSel}
                    collapsed=${collapsed} toggle=${toggle}/>
    </div>` : null}
  </div>`;
}

function TreeLeaf({node,sel,onSel}){
  const ref = useRef(), selected = node.id===sel, m = meta(node.type);
  useEffect(()=>{ if(selected && ref.current) ref.current.scrollIntoView({block:'nearest'}); },[selected]);
  const summary = addedText(node.ctx_state);
  return html`<div ref=${ref} class=${'tleaf'+(selected?' sel':'')} onClick=${()=>onSel(node.id)}>
    <span class="row-dot sm" style=${{background:m.dot}}></span>
    <span class="tleaf-name">${m.name}</span>
    <span class=${'tleaf-sub'+(node.ctx_state&&node.ctx_state.removed?' neg':'')}>${summary || '—'}</span>
  </div>`;
}

// A display copy of attributes with cryptic codes mapped to friendly labels.
function displayAttrs(node){
  const a = Object.assign({}, node.attributes||{});
  if(a.phase) a.phase = phaseLabel(a.phase);
  return a;
}

function Detail({node,mainAgent}){
  const m = meta(node.type), a = node.attributes||{};
  const subAgent = node.agent && node.agent!==mainAgent ? node.agent : null;
  return html`<div class="dwrap">
    <div class="dhead">
      <span class="dbadge" style=${{background:m.dot}}>${m.name}</span>
      ${subAgent ? html`<span class="sub-tag">subagent ${subAgent.slice(0,8)}</span>` : null}
      ${node.loop_flag ? html`<span class="loop-tag">loop</span>` : null}
      <h2>${node.label}</h2>
      <div class="dmeta">
        ${a.started_at ? html`<span>🕘 ${String(a.started_at).slice(11,19) || a.started_at}</span>` : null}
        ${a.latency_s!=null ? html`<span>⚡ ${a.latency_s}s</span>` : null}
        ${a.step ? html`<span>Step ${a.step}</span>` : null}
        ${node.type==='router' && a.phase ? html`<span>🧭 ${phaseLabel(a.phase)}</span>` : null}
      </div>
    </div>
    <${CtxCard} cs=${node.ctx_state} agent=${subAgent}/>
    <${Section} title="Outputs" data=${node.outputs} body=${outputsBody(node)} open=${true}/>
    <${Section} title="Inputs" data=${node.inputs} open=${true}/>
    <${Section} title="Attributes" data=${displayAttrs(node)} open=${false}/>
  </div>`;
}

// Rich body for the Outputs section: tool results and LLM responses get
// purpose-built views; everything else returns null to use the generic DataView.
function outputsBody(node){
  if(node.type==='tool_call') return html`<${ToolResult} node=${node}/>`;
  if(node.type==='llm_call')  return html`<${LlmOutputs} node=${node}/>`;
  if(node.type==='report')    return html`<${ReportOutput} node=${node}/>`;
  return null;
}

function ReportOutput({node}){
  const content = (node.outputs && node.outputs.content) || '';
  return html`<div class="toolres">
    ${content ? html`<div class="tr-block"><div class="tr-label">report</div>
      <${LogBlock} text=${content}/></div>`
      : html`<div class="tr-block muted">No report content recorded.</div>`}
  </div>`;
}

function parseToolResult(raw){
  if(raw==null) return null;
  if(typeof raw==='object') return raw;
  try { return JSON.parse(raw); } catch(e){ return {output:String(raw)}; }
}

// Render an args dict as the command/path that was actually executed.
function cmdText(args){
  if(args && typeof args==='object'){
    if(typeof args.command==='string') return args.command;
    const keys = Object.keys(args);
    if(keys.length===1 && typeof args[keys[0]]==='string') return args[keys[0]];
    return JSON.stringify(args, null, 2);
  }
  return args==null ? '' : String(args);
}

function ToolResult({node}){
  const r = parseToolResult(node.outputs && node.outputs.result) || {};
  const cmd = cmdText(node.inputs && node.inputs.args);
  return html`<div class="toolres">
    <div class="tr-head">
      ${r.tool ? html`<span class="tr-tool">${r.tool}</span>` : null}
      ${r.ok===true ? html`<span class="tr-badge ok">✓ success</span>`
        : r.ok===false ? html`<span class="tr-badge err">✗ error</span>` : null}
    </div>
    ${cmd ? html`<div class="tr-block"><div class="tr-label">command</div>
      <${LogBlock} text=${cmd} kind="cmd"/></div>` : null}
    ${r.output ? html`<div class="tr-block"><div class="tr-label">output</div>
      <${LogBlock} text=${r.output} kind="out"/></div>` : null}
    ${r.error ? html`<div class="tr-block"><div class="tr-label err">error</div>
      <${LogBlock} text=${r.error} kind="err"/></div>` : null}
    ${(!cmd && !r.output && !r.error) ? html`<div class="tr-block muted">No output recorded.</div>` : null}
  </div>`;
}

function LlmOutputs({node}){
  const o = node.outputs || {};
  const calls = o.tool_calls || [];
  return html`<div class="toolres">
    ${o.content ? html`<div class="tr-block"><div class="tr-label">response</div>
      <${LogBlock} text=${o.content}/></div>` : null}
    ${o.reasoning ? html`<div class="tr-block"><div class="tr-label">reasoning</div>
      <${LogBlock} text=${o.reasoning}/></div>` : null}
    ${calls.length ? html`<div class="tr-block"><div class="tr-label">tool calls · ${calls.length}</div>
      <${ToolCalls} calls=${calls}/></div>` : null}
    ${(!o.content && !o.reasoning && !calls.length) ? html`<div class="tr-block muted">No output recorded.</div>` : null}
  </div>`;
}

function ToolCalls({calls}){
  return html`${calls.map((c,i)=>{
    const fn = c.function || {};
    let args = fn.arguments;
    try { args = JSON.parse(fn.arguments); } catch(e){}
    return html`<div key=${c.id||i} class="tcall">
      <div class="tc-top"><span class="tr-tool">${fn.name||'?'}</span></div>
      <${LogBlock} text=${cmdText(args)} kind="cmd"/>
    </div>`;
  })}`;
}

function LogBlock({text,kind}){
  const [copied,setCopied] = useState(false);
  const copy = ()=>{ if(navigator.clipboard) navigator.clipboard.writeText(text)
    .then(()=>{ setCopied(true); setTimeout(()=>setCopied(false),1200); }); };
  return html`<div class=${'jb '+(kind||'')}>
    <button class="copy" onClick=${copy}>${copied?'✓ copied':'copy'}</button>
    <pre>${text}</pre>
  </div>`;
}

function CtxCard({cs,agent}){
  if(!cs || cs.tokens==null) return null;
  const comp = cs.composition || {};
  const total = cs.tokens || 0;
  const segs = ROLE_ORDER.filter(r=>comp[r]);
  return html`<div class=${'ctxcard'+(agent?' sub':'')}>
    <div class="ctx-top">
      <span class="ctx-label">Context window${agent ? html` <span class="ctx-sub">· subagent ${agent.slice(0,8)}</span>` : null}</span>
      <span class="ctx-figs">${fmtNum(total)}${cs.window ? ' / '+fmtNum(cs.window) : ''}${cs.pct!=null ? ' · '+cs.pct+'%' : ''}</span>
    </div>
    <div class="ctx-bar">
      ${segs.length
        ? segs.map(r=>html`<div key=${r} class="ctx-seg"
            style=${{width:(comp[r]/total*100)+'%',background:ROLE_META[r].color}} title=${ROLE_META[r].label}></div>`)
        : html`<div class="ctx-fill" style=${{width:Math.min(100,cs.pct||0)+'%'}}></div>`}
    </div>
    ${segs.length ? html`<div class="ctx-legend">
      ${segs.map(r=>html`<span key=${r} class="bd-li">
        <span class="bd-sw" style=${{background:ROLE_META[r].color}}></span>
        ${ROLE_META[r].label} <span class="muted">${fmtNum(comp[r])}</span>
      </span>`)}
    </div>` : null}
  </div>`;
}

function Section({title,data,open,body}){
  const [o,setO] = useState(open);
  const isEmpty = body==null && (data==null
    || (typeof data==='object' && !Array.isArray(data) && Object.keys(data).length===0)
    || (Array.isArray(data) && !data.length)
    || (typeof data==='string' && !data.length));
  return html`<div class=${'section'+(o?' open':'')}>
    <div class="shead" onClick=${()=>setO(!o)}>
      <span class=${'twist'+(o?' open':'')}>▸</span> ${title}
      ${isEmpty ? html`<span class="muted" style="font-weight:400">— empty</span>` : null}
    </div>
    ${o && !isEmpty ? html`<div class="sbody">${body!=null ? body : html`<${DataView} data=${data}/>`}</div>` : null}
  </div>`;
}

function DataView({data}){
  if(data==null) return html`<span class="muted">—</span>`;
  if(typeof data==='string') return html`<${JsonBlock} text=${data}/>`;
  if(Array.isArray(data)) return html`<${JsonBlock} text=${JSON.stringify(data,null,2)}/>`;
  if(typeof data==='object'){
    const keys = Object.keys(data);
    return html`<table class="kv">${keys.map(k=>html`<tr key=${k}>
      <td class="kk">${k}</td><td class="kv-v"><${Value} v=${data[k]}/></td></tr>`)}</table>`;
  }
  return html`<span>${String(data)}</span>`;
}

function Value({v}){
  if(v==null) return html`<span class="muted">null</span>`;
  if(typeof v==='object') return html`<${JsonBlock} text=${JSON.stringify(v,null,2)}/>`;
  if(typeof v==='string' && v.length>80) return html`<${JsonBlock} text=${v}/>`;
  return html`<span>${String(v)}</span>`;
}

function JsonBlock({text}){
  const [copied,setCopied] = useState(false);
  const copy = ()=>{ if(navigator.clipboard) navigator.clipboard.writeText(text)
    .then(()=>{ setCopied(true); setTimeout(()=>setCopied(false),1200); }); };
  return html`<div class="jb">
    <button class="copy" onClick=${copy}>${copied?'✓ copied':'copy'}</button>
    <pre>${text}</pre>
  </div>`;
}

render(html`<${App}/>`, document.getElementById('app'));
</script>
</body>
</html>"""


def _check_vendor_assets() -> None:
    """Fail fast if any vendored UI library is missing (CONTRIBUTE.md §11/§29).

    Called at server startup so a broken install surfaces immediately rather
    than as a 500 on the first page load.

    Raises:
        MyCodingAgentError: If one or more files in ``_VENDOR_FILES`` are absent.
    """
    missing = [name for name in _VENDOR_FILES if not (_VENDOR_DIR / name).is_file()]
    if missing:
        raise MyCodingAgentError(
            f"Trace Explorer UI assets missing from {_VENDOR_DIR}: {', '.join(missing)}",
            hint="Reinstall the package (e.g. `uv sync`) to restore the vendored UI libraries.",
        )


@lru_cache(maxsize=1)
def _vendor_js() -> str:
    """Concatenate the vendored Preact/hooks/htm sources, load order preserved.

    Returns:
        The three UMD bundles joined with newlines, ready to inline into a
        ``<script>`` element.
    """
    return "\n".join(
        (_VENDOR_DIR / name).read_text(encoding="utf-8") for name in _VENDOR_FILES
    )


@lru_cache(maxsize=1)
def _full_html() -> str:
    """Return the page with the vendored libraries inlined.

    Cached so the vendor files are read from disk only once per process.

    Returns:
        The complete HTML document served at ``/``.
    """
    return EMBEDDED_HTML.replace(_VENDOR_TOKEN, _vendor_js())


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
        """Write the embedded HTML viewer (with vendored libs) as the response."""
        body = _full_html().encode()
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

    Raises:
        MyCodingAgentError: If the vendored UI assets are missing.
    """
    _check_vendor_assets()
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
