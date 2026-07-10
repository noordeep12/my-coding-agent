"""Localhost HTTP server and render helpers for the Trace Explorer.

Serves the single-page browser UI at ``/`` and JSON API routes under ``/api/``.
Uses only the Python stdlib ``http.server`` module — no new *runtime* Python
dependencies.  The UI is a Preact + htm app; those libraries are vendored
offline under ``viewer/_vendor/`` and injected inline into the page (no CDN).

``run_server`` and the render helpers below are also imported by
``my_coding_agent.webui`` to mount the Trace Explorer into the unified shell;
the standalone ``my-coding-agent-traces`` console script has been retired
(superseded by ``my-coding-agent-webui``).
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
from .evals_server import eval_dashboard_html, handle_eval_api_route
from .reader import list_sessions, load_session

logger = logging.getLogger(__name__)

# ── Security: allow only hex session IDs (UUID-style without dashes) ──────────
_SID_RE = re.compile(r"^[0-9a-f]{8,64}$")

# ── Vendored UI libraries (offline, no CDN) — see viewer/_vendor/README.md ─────
_VENDOR_DIR = Path(__file__).parent / "_vendor"
_VENDOR_FILES = (
    "preact.min.js",
    "hooks.umd.js",
    "htm.umd.js",
    "codemirror.bundle.js",
    "markdown-it.bundle.js",
    "dompurify.bundle.js",
)
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
.stats-breakdown{display:flex;flex-direction:column;gap:6px;padding:8px 20px;background:var(--panel);border-bottom:1px solid var(--line)}
.bd-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:12px}

/* ── main split ── */
.main{flex:1;display:grid;grid-template-columns:minmax(280px,38%) 1fr;min-height:0}
.rail{overflow-y:auto;min-height:0;border-right:1px solid var(--line);background:var(--panel);padding:14px 12px}
.detail{overflow-y:auto;min-height:0;background:var(--bg)}

/* ── shared node dot / tags ── */
.row-dot{width:11px;height:11px;border-radius:50%;flex:none}
.row-dot.sm{width:8px;height:8px}
.loop-tag{font-size:10px;font-weight:600;color:var(--amber);background:#fff3e6;border-radius:5px;padding:1px 6px}
.anomaly-tag{font-size:10px;font-weight:600;color:var(--neg);background:var(--neg-bg);border-radius:5px;padding:1px 6px}
.refusal-tag{font-size:10px;font-weight:600;color:#fff;background:var(--neg);border-radius:5px;padding:1px 6px}
.posture-tag{font-size:10px;font-weight:600;border-radius:5px;padding:1px 6px}
.posture-tag.sandboxed{color:var(--pos);background:var(--pos-bg)}
.posture-tag.screened-only{color:var(--amber);background:#fff3e6}

/* ── tree ── */
.tree{display:flex;flex-direction:column;gap:1px}
/* Every group draws a guide line down its left edge, connecting it to each of
   its direct children (a parent→child link, not just indentation); a small
   horizontal tick on each child row reaches back to that line. Subagent
   groups keep the line in the sub accent color; everything else (a step's own
   pipeline, or a tool's nested LLM call, e.g. read_tool_artifact→artifact_query)
   uses a neutral one. */
.agroup-body{display:flex;flex-direction:column;gap:1px;margin-left:9px;padding-left:16px;
  border-left:1.5px solid var(--line);position:relative}
.agroup.sub>.agroup-body{border-left-color:var(--sub)}
.agent-head,.tleaf{position:relative}
.agroup-body>.agroup>.agent-head::before,
.agroup-body>.tleaf::before{
  content:'';position:absolute;left:-16px;top:50%;width:13px;height:0;
  border-top:1.5px solid var(--line);pointer-events:none}
.agroup-body>.agroup.sub>.agent-head::before{border-top-color:var(--sub)}
.agent-head{display:flex;align-items:center;gap:9px;padding:7px 12px;border-radius:8px;cursor:pointer}
.agent-head:hover{background:var(--bg2)}
.agent-head.sel{background:var(--accent-soft)}
.agent-name{font-weight:600;font-size:12px}
.agroup.sub>.agent-head .agent-name{color:var(--sub)}
.sub-tag{font-size:10px;font-weight:600;color:var(--sub);background:var(--sub-soft);border-radius:5px;padding:1px 6px}
.tleaf{display:flex;align-items:center;gap:9px;padding:7px 12px;border-radius:8px;cursor:pointer}
.tleaf:hover{background:var(--bg2)}
.tleaf.sel{background:var(--accent-soft)}
.tleaf-name{font-weight:500;font-size:12px;flex:none}
.tleaf-badges{display:flex;gap:5px;align-items:center;flex:none;overflow:hidden}
.tleaf-sub{font-family:var(--mono);font-size:11px;color:var(--muted);text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:40px}

/* Detail-panel Section accordion (Outputs/Inputs/Attributes). */
.twist{display:inline-block;transition:transform .12s;color:var(--muted);font-size:10px}
.twist.open{transform:rotate(90deg)}

/* ── detail ── */
.dwrap{display:flex;flex-direction:column}
.dhead{padding:20px 22px 16px;border-bottom:1px solid var(--line)}
.dhead-top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.dbadge{display:inline-block;color:#fff;font-size:14px;font-weight:700;border-radius:8px;padding:5px 12px;vertical-align:middle}
.badge-row{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
/* uniform node badges (detail header + tree rows) */
.nbadge{font-size:11px;font-weight:600;border-radius:6px;padding:2px 9px;white-space:nowrap}
.nbadge.sm{font-size:10px;padding:1px 6px;border-radius:5px}
.nbadge.name{font-family:var(--mono);background:var(--bg2);border:1px solid var(--line);color:var(--text)}
.nbadge.ok{color:var(--pos);background:var(--pos-bg)}
.nbadge.err{color:var(--neg);background:var(--neg-bg)}
.nbadge.lat{color:var(--muted);background:var(--bg2)}
.nbadge.res{color:var(--muted);background:var(--bg2)}
.nbadge.ts{color:var(--muted);background:var(--bg2)}
.nbadge.step{color:var(--muted);background:var(--bg2)}
.nbadge.art{color:var(--sub);background:var(--sub-soft)}
.nbadge.skill{color:var(--sub);background:var(--sub-soft)}
.nbadge.trunc{color:var(--amber);background:#fdf1e5}
.nbadge.phase{color:var(--muted);background:var(--bg2)}
.nbadge.count{color:var(--muted);background:var(--bg2)}
.nbadge.stop{color:var(--muted);background:var(--bg2)}

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
.ctx-delta{display:flex;flex-wrap:wrap;gap:14px;margin-top:9px;padding-top:9px;border-top:1px solid var(--line);font-size:11px;align-items:center}
.ctx-delta-add{color:var(--pos)}
.ctx-delta-rem{color:var(--neg)}
.ctx-delta-rem.clickable{cursor:pointer;text-decoration:underline dotted}
.retire-overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center;z-index:1000}
.retire-modal{background:var(--bg);border:1px solid var(--line);border-radius:10px;max-width:820px;width:90%;max-height:80vh;overflow:auto;padding:20px}
.retire-modal-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.retire-modal-title{font-weight:600;font-size:14px}
.retire-close{cursor:pointer;background:none;border:1px solid var(--line);border-radius:6px;padding:2px 10px;font-size:12px;color:var(--muted);font-family:var(--font)}
.retire-close:hover{color:var(--accent);border-color:var(--accent)}
.retire-item{border:1px solid var(--line);border-radius:8px;margin-bottom:14px;overflow:hidden}
.retire-item:last-child{margin-bottom:0}
.retire-item-head{font-size:11px;color:var(--muted);padding:8px 12px;border-bottom:1px solid var(--line);font-family:var(--mono)}
.retire-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.03em;padding:8px 12px 0;color:var(--muted)}
.retire-before,.retire-after{font-family:var(--mono);font-size:11px;padding:8px 12px 12px;white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto;margin:0}
.retire-before{background:var(--neg-bg);color:var(--neg)}
.retire-after{background:var(--pos-bg);color:var(--pos)}

/* ── sections ── */
.section{border-bottom:1px solid var(--line)}
.shead{display:flex;align-items:center;gap:8px;padding:12px 22px;font-weight:600;font-size:12px;cursor:pointer;user-select:none}
.shead:hover{background:var(--bg2)}
.sbody{padding:0 22px 16px}
/* ── code box: mini CodeMirror viewer (offline, read-only) ── */
.cb{margin-top:2px;background:var(--bg2);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.cb-bar{display:flex;align-items:center;gap:8px;padding:5px 8px;border-bottom:1px solid var(--line);min-height:30px}
.cb-crumbs{flex:1;min-width:0;display:flex;align-items:center;flex-wrap:wrap;gap:2px;font-family:var(--mono);font-size:10px;overflow:hidden}
.cb-crumb{color:var(--sub);cursor:pointer;padding:1px 3px;border-radius:4px;white-space:nowrap}
.cb-crumb:hover{background:var(--sub-soft)}
.cb-sep{color:var(--line);margin:0 1px}
.cb-crumbs.muted{color:var(--muted);cursor:default}
.cb-actions{display:flex;align-items:center;gap:5px;flex:none}
.cb-btn{font-family:var(--font);font-size:10px;color:var(--muted);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:2px 8px;cursor:pointer}
.cb-btn:hover{color:var(--accent);border-color:var(--accent)}
.cb-editor .cm-editor{max-height:340px}
.cb-editor .cm-scroller{overflow:auto}

/* ── rendered markdown (mini markdown-it + DOMPurify preview) ── */
.md-render{padding:10px 12px;font-size:12.5px;line-height:1.6;max-height:340px;overflow:auto}
.md-render>*:first-child{margin-top:0}
.md-render>*:last-child{margin-bottom:0}
.md-render h1,.md-render h2,.md-render h3,.md-render h4,.md-render h5,.md-render h6{margin:14px 0 6px;font-weight:600;line-height:1.3}
.md-render h1{font-size:1.5em}
.md-render h2{font-size:1.3em}
.md-render h3{font-size:1.15em}
.md-render p{margin:8px 0}
.md-render ul,.md-render ol{margin:8px 0;padding-left:1.6em}
.md-render li{margin:2px 0}
.md-render blockquote{margin:8px 0;padding:2px 12px;border-left:3px solid var(--line);color:var(--muted)}
.md-render a{color:var(--accent);text-decoration:none}
.md-render a:hover{text-decoration:underline}
.md-render code{font-family:var(--mono);font-size:.92em;background:var(--bg2);border-radius:4px;padding:1px 5px}
.md-render pre{margin:8px 0;overflow-x:auto}
.md-render pre code{background:transparent;padding:0;border-radius:0}
.md-render table{display:block;max-width:100%;overflow-x:auto;border-collapse:collapse;margin:8px 0}
.md-render th,.md-render td{border:1px solid var(--line);padding:4px 10px;text-align:left}
.md-render th{background:var(--bg2);font-weight:600}
.md-render hr{border:none;border-top:1px solid var(--line);margin:12px 0}
.md-render img{max-width:100%}
.md-render .cm-editor{max-height:none;border:1px solid var(--line);border-radius:6px}

/* ── tool result / llm body (rendered inside .sbody) ── */
.toolres{display:flex;flex-direction:column}
.tr-block{padding:12px 0 0}
.tr-block:first-child{padding-top:0}
.tr-block.muted{font-size:12px}
.tr-label{font-size:11px;font-weight:600;color:var(--muted);margin-bottom:6px;text-transform:lowercase}
.tr-label.err{color:var(--neg)}
.refusal-refs{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:4px}
.refusal-refs a{color:var(--accent);font-size:12px;text-decoration:none}
.refusal-refs a:hover{text-decoration:underline}

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

// ── CodeMirror 6 (vendored, offline) — read-only editor theme for content boxes ──
const CM = window.CM6;
const CB_THEME = CM.EditorView.theme({
  '&':{fontSize:'11px',backgroundColor:'transparent',color:'var(--text)'},
  '.cm-content':{fontFamily:'var(--mono)',padding:'8px 0'},
  '.cm-scroller':{lineHeight:'1.6',maxHeight:'340px'},
  '.cm-gutters':{backgroundColor:'transparent',border:'none',color:'#b8b8bf'},
  '.cm-activeLine':{backgroundColor:'#eef4ff66'},
  '.cm-activeLineGutter':{backgroundColor:'transparent',color:'var(--muted)'},
  '.cm-foldGutter span':{color:'#b8b8bf'},
  '&.cm-focused':{outline:'none'},
  '.cm-selectionMatch':{backgroundColor:'#fff3a8'},
  '.cm-searchMatch':{backgroundColor:'#fff3a8',outline:'1px solid #e0c200'},
  '.cm-searchMatch-selected':{backgroundColor:'#ffd400'},
  '.cm-panels':{backgroundColor:'var(--bg)',color:'var(--text)',borderTop:'1px solid var(--line)'},
  '.cm-panel input':{fontFamily:'var(--font)',fontSize:'11px'},
});

// ── markdown-it (vendored, offline) — html:false so embedded HTML in the
// source is escaped rather than emitted as live markup; this is layer (a) of
// the XSS defense (issue #112). CommonMark + tables preset, no linkify/typography.
const _md = new window.markdownit({html:false, linkify:false, typography:false});

// Parse *src* to HTML and sanitize it with DOMPurify before it is ever
// assigned to innerHTML — layer (b) of the XSS defense. This is the single
// insertion path for rendered markdown; no raw content bypasses sanitize().
function renderMarkdownHTML(src){
  const raw = _md.render(String(src==null?'':src));
  return window.DOMPurify.sanitize(raw);
}

// Fenced code blocks come out of markdown-it as <pre><code class="language-x">.
// Replace each with a small read-only CodeMirror view (same theme as CodeBox)
// when the language is one CM6 already knows (json/python/shell); unsupported
// languages are left as the plain <pre><code> markup DOMPurify let through,
// which already reads as code (monospace, distinct background).
function highlightFences(container){
  if(!container || !CM) return;
  container.querySelectorAll('pre > code').forEach(code=>{
    const m = /language-(\\w+)/.exec(code.className||'');
    const lang = m ? m[1] : null;
    const exts = [
      CM.EditorState.readOnly.of(true),
      CM.syntaxHighlighting(CM.defaultHighlightStyle,{fallback:true}),
      CM.EditorView.lineWrapping,
      CB_THEME,
    ];
    if(lang==='json') exts.push(CM.json());
    else if(lang==='python' && CM.python) exts.push(CM.python());
    else if(['shell','bash','sh'].includes(lang) && CM.shell) exts.push(CM.shell());
    else return;
    const pre = code.parentElement;
    const host = document.createElement('div');
    new CM.EditorView({state: CM.EditorState.create({doc:code.textContent, extensions:exts}), parent: host});
    pre.replaceWith(host);
  });
}

// Rendered-markdown view for free-text content, with a per-box Rendered ⇄ Raw
// toggle. Rendered is the default (issue #112); Raw reuses CodeBox verbatim
// so the byte-exact source, copy/find/line-numbers stay unchanged.
function MarkdownBox({value, lang:hint}){
  const [raw,setRaw] = useState(false);
  const text = typeof value==='string' ? value : JSON.stringify(value,null,2);
  const out = useMemo(()=>renderMarkdownHTML(text), [text]);
  const host = useRef(null);

  useEffect(()=>{
    if(raw || !host.current) return;
    host.current.innerHTML = out;
    highlightFences(host.current);
  }, [raw, out]);

  return html`<div class="cb">
    <div class="cb-bar">
      <div class="cb-crumbs muted">markdown</div>
      <div class="cb-actions">
        <button class="cb-btn" onClick=${()=>setRaw(!raw)}>${raw?'rendered':'raw'}</button>
      </div>
    </div>
    ${raw ? html`<${CodeBox} value=${value} lang=${hint}/>` : html`<div class="md-render" ref=${host}></div>`}
  </div>`;
}

// Content-dispatch wrapper: free-text (non-JSON, non-code-hint) strings get
// the rendered-markdown treatment; JSON and explicitly hinted (python/shell/
// json) values render exactly as before via CodeBox.
function ContentBox({value, lang:hint}){
  const {lang} = useMemo(()=>toDoc(value, hint), [value, hint]);
  if(lang==='text') return html`<${MarkdownBox} value=${value} lang=${hint}/>`;
  return html`<${CodeBox} value=${value} lang=${hint}/>`;
}

// Walk the JSON syntax tree from the caret to the root, building a clickable
// breadcrumb of property names and array indices (VS Code style). Best-effort:
// any parse hiccup just yields a shorter path.
function jsonPathAt(state){
  try{
    const tree = CM.syntaxTree(state);
    const pos = state.selection.main.head;
    const parts = [];
    let child = null;
    const VALUE = ['Object','Array','String','Number','True','False','Null'];
    for(let cur=tree.resolveInner(pos,-1); cur; child=cur, cur=cur.parent){
      if(cur.name==='Property'){
        const pn = cur.getChild('PropertyName');
        if(pn){
          let nm = state.sliceDoc(pn.from,pn.to);
          try{ nm = JSON.parse(nm); }catch(e){}
          parts.unshift({label:String(nm), from:cur.from});
        }
      } else if(cur.name==='Array' && child){
        let idx=-1;
        for(let ch=cur.firstChild; ch; ch=ch.nextSibling){
          if(!VALUE.includes(ch.name)) continue;
          idx++;
          if(ch.from===child.from){ break; }
        }
        if(idx>=0) parts.unshift({label:'['+idx+']', from:child.from});
      }
    }
    return parts;
  }catch(e){ return []; }
}

// Coerce any input/output value into a document + language for the CodeBox.
// An explicit `hint` from the backend (python/shell/json/text) wins; otherwise
// objects/arrays and JSON-looking strings become pretty JSON and everything else
// stays raw text.
function toDoc(value, hint){
  if(hint && hint!=='json' && hint!=='text'){
    const text = typeof value==='string' ? value : JSON.stringify(value,null,2);
    return {text, lang:hint};
  }
  if(typeof value==='string'){
    const t = value.trim();
    if(hint==='json' || (t && (t[0]==='{' || t[0]==='['))){
      try{ return {text:JSON.stringify(JSON.parse(value),null,2), lang:'json'}; }catch(e){}
    }
    return {text:value, lang:hint||'text'};
  }
  return {text:JSON.stringify(value,null,2), lang:'json'};
}

// ── node type metadata ──
const TYPE_META = {
  session:        { name:'Session Start',     dot:'#0a84ff' },
  router:         { name:'Tool Routing',      dot:'#ff9f0a' },
  llm_call:       { name:'LLM Call',          dot:'#30b350' },
  tool_call:      { name:'Tool Dispatch',     dot:'#a05cf0' },
  handoff:        { name:'Context Guard',     dot:'#ff453a' },
  report:         { name:'Subagent Report',   dot:'#5e5ce6' },
  summarizer:     { name:'Context Summarizer',dot:'#d9a800' },
  finalize_step:  { name:'Finalize Step',     dot:'#1aa3c4' },
  anomaly:        { name:'Anomaly Detect',    dot:'#d70015' },
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

  // Restore-where-you-were: an embedding shell (webui) may pass the
  // previously selected session via ?session=<id>; that selection wins over
  // "most recent session" the first time the list loads.
  const initialSid = useMemo(()=>{
    try{ return new URLSearchParams(window.location.search).get('session'); }catch(e){ return null; }
  },[]);
  useEffect(()=>{ getJSON('/api/sessions').then(s=>{
    setSessions(s);
    if(!s.length) return;
    const found = initialSid && s.some(x=>x.session_id===initialSid);
    setSid(found ? initialSid : s[0].session_id);
  }); },[]);
  useEffect(()=>{ if(!sid) return; setData(null); setSel(null); getJSON('/api/session/'+sid).then(setData); },[sid]);
  // Notify an embedding shell of the current selection so it can persist it.
  useEffect(()=>{
    if(!sid) return;
    try{ window.parent && window.parent.postMessage({type:'mca:selection', tab:'traces', session:sid}, '*'); }catch(e){}
  },[sid]);

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
  const [open,setOpen] = useState(false);
  const a = data.analytics || {};
  const cost = a.cost_usd!=null ? '$'+Number(a.cost_usd).toFixed(4) : '—';
  const byKind = a.by_kind || {};
  const byAgent = a.by_agent || {};
  const rr = a.resource_rollup;
  const projectedCosts = a.projected_costs || {};
  const hasBreakdown = Object.keys(byKind).length>0 || Object.keys(byAgent).length>0 || !!rr || Object.keys(projectedCosts).length>0;
  return html`<div>
    <div class="stats">
      <span><b>${data.model||'?'}</b></span>
      <span><b>${data.steps}</b> steps</span>
      <span><b>${fmtNum(a.total_tokens||0)}</b> tokens</span>
      <span><b>${cost}</b></span>
      ${data.posture==='sandboxed' ? html`<span class="posture-tag sandboxed">🔒 sandboxed</span>` : null}
      ${data.posture==='screened_only' ? html`<span class="posture-tag screened-only">🛡 screened only</span>` : null}
      ${a.loop_count ? html`<span class="warn">⚠ ${a.loop_count} loop(s)</span>` : null}
      ${a.anomaly_count ? html`<span class="warn">⚠ ${a.anomaly_count} anomaly(s)</span>` : null}
      ${a.refusal_count ? html`<span class="warn">🛑 ${a.refusal_count} refused</span>` : null}
      ${a.skill_offered_count!=null ? html`<span class="muted">🧠 ${a.skill_offered_count} offered · ${a.skill_loaded_count} loaded</span>` : null}
      ${data.stop_reason ? html`<span class="muted">stop: ${data.stop_reason}</span>` : null}
      ${hasBreakdown ? html`<button class="filter-btn" onClick=${()=>setOpen(!open)}>
        Breakdown</button>` : null}
    </div>
    ${open && hasBreakdown ? html`<div class="stats-breakdown">
      ${Object.keys(byKind).length ? html`<div class="bd-row">
        <span class="muted">by kind:</span>
        ${Object.entries(byKind).map(([k,v])=>html`<span key=${k} class="chip">
          ${k}: ${fmtNum(v.total_tokens)}</span>`)}
      </div>` : null}
      ${Object.keys(byAgent).length ? html`<div class="bd-row">
        <span class="muted">by agent:</span>
        ${Object.entries(byAgent).map(([sid,v])=>html`<span key=${sid} class="chip">
          ${sid.slice(0,8)}: ${fmtNum(v.tokens)} tok · ${v.call_count} calls${
            v.elapsed_s!=null ? ' · '+Number(v.elapsed_s).toFixed(1)+'s' : ''}</span>`)}
      </div>` : null}
      ${Object.keys(projectedCosts).length ? html`<div class="bd-row">
        <span class="muted">projected on:</span>
        ${Object.entries(projectedCosts).map(([m,v])=>html`<span key=${m} class="chip">
          ${m}: $${Number(v).toFixed(4)}</span>`)}
      </div>` : null}
      ${rr ? html`<div class="bd-row">
        <span class="muted">machine (run):</span>
        <span class="chip">ram avg ${rr.ram_pct.avg}% peak ${rr.ram_pct.peak}%</span>
        <span class="chip">cpu avg ${rr.cpu_pct.avg}% peak ${rr.cpu_pct.peak}%</span>
        ${rr.gpu_pct ? html`<span class="chip">gpu avg ${rr.gpu_pct.avg}% peak ${rr.gpu_pct.peak}%</span>` : null}
        <span class="chip">net ${(rr.net_bytes/1048576).toFixed(1)} MB</span>
        <span class="chip">disk ${(rr.disk_bytes/1048576).toFixed(1)} MB</span>
      </div>` : null}
    </div>` : null}
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

// "+N role" / "−N role" segments of what a node added to (and retired from)
// the context window — rendered as separate colored spans in the tree row.
function addedParts(cs){
  if(!cs) return {added:'', removed:''};
  const a = cs.added||{};
  const addedRoles = ROLE_ORDER.filter(r=>a[r]);
  const added = addedRoles.length
    ? addedRoles.map(r=>'+'+fmtNum(a[r])+' '+ROLE_META[r].label).join('  ')
    : '';
  let removed = '';
  if(cs.removed){
    const rem = cs.removed_by_role||{};
    const remRoles = ROLE_ORDER.filter(r=>rem[r]);
    removed = remRoles.length
      ? remRoles.map(r=>'−'+fmtNum(rem[r])+' '+ROLE_META[r].label).join('  ')
      : '−'+fmtNum(cs.removed);
  }
  return {added, removed};
}

// ISO timestamp → HH:MM:SS (best-effort; falls back to the raw value).
const fmtTime = s => s ? (String(s).slice(11,19) || String(s)) : null;

// The main-badge title for a node: its class label with any "(detail)" suffix
// stripped — the detail (tool name / llm kind) is surfaced as its own badge, so
// it is not duplicated in the title.
const nodeTitle = node => (node.label||'').replace(/\\s*\\(.*\\)\\s*$/, '');

// A bash tool_call executes multi-line code when its recorded args carry a
// non-empty `stdin` or a `command` containing a newline — detected from the
// trace alone, so traces recorded before `stdin` existed still badge via the
// newline-in-command signal.
function isMultilineBashCall(node){
  const args = (node.inputs && node.inputs.args) || {};
  if(typeof args.stdin==='string' && args.stdin.length>0) return true;
  return typeof args.command==='string' && args.command.includes('\\n');
}

// Uniform badge descriptors for a node, ordered by importance left→right: the
// most meaningful and colored badges (identity, status, colored type signals)
// lead; housekeeping (latency, timestamp, step) trails. Each is {t: label,
// c: css-class}; only badges whose data exists are emitted.
function nodeBadges(node){
  const a = node.attributes||{}, b = [];
  const r = node.type==='tool_call' ? (parseToolResult(node.outputs && node.outputs.result)||{}) : null;
  // 1. identity — what this node is
  if(r && (a.name||r.tool)) b.push({t:a.name||r.tool, c:'name'});
  else if(node.type==='llm_call' && a.kind && a.kind!=='main') b.push({t:a.kind, c:'name'});
  else if(node.type==='session' && a.model) b.push({t:a.model, c:'name'});
  else if(node.type==='session_end' && a.stop_reason) b.push({t:'stop: '+a.stop_reason, c:'stop'});
  else if(node.type==='anomaly' && a.tool_name) b.push({t:a.tool_name, c:'name'});
  // 2. status — colored success/error
  if(r && r.ok===true) b.push({t:'✓ success', c:'ok'});
  else if(r && r.ok===false) b.push({t:'✗ error', c:'err'});
  // 3. colored type signals
  if(r && r.metadata && r.metadata.artifact===true) b.push({t:'📦 artifact', c:'art'});
  if(r && r.metadata && r.metadata.truncated===true) b.push({t:'✂️ truncated', c:'trunc'});
  if(node.type==='llm_call' && a.capped===true) b.push({t:'✂️ cut at '+a.max_tokens+'-token cap', c:'trunc'});
  if(node.type==='tool_call' && a.name==='bash' && isMultilineBashCall(node)) b.push({t:'📜 multi-line', c:'art'});
  if(node.type==='tool_call' && a.name==='use_skill') b.push({t:'🧠 skill', c:'skill'});
  if(node.type==='router' && a.phase) b.push({t:'🧭 '+phaseLabel(a.phase), c:'phase'});
  if(node.type==='finalize_step' && a.finish_reason) b.push({t:'finish: '+a.finish_reason, c:'phase'});
  if(node.type==='finalize_step' && a.signal) b.push({t:'signal: '+a.signal, c:'phase'});
  if(node.type==='report'){
    if(a.source==='verbatim') b.push({t:'🆓 free', c:'ok'});
    else if(a.source==='summarizer'||a.source==='fallback') b.push({t:'💰 paid', c:'trunc'});
    else b.push({t:'❔ unknown', c:'phase'});
  }
  // 4. counts (neutral type signals)
  if(node.type==='router'){
    const sel = node.outputs && node.outputs.selected;
    if(Array.isArray(sel)) b.push({t:sel.length+' tool'+(sel.length===1?'':'s'), c:'count'});
  }
  if(node.type==='llm_call'){
    const tc = node.outputs && node.outputs.tool_calls;
    if(Array.isArray(tc) && tc.length) b.push({t:tc.length+' call'+(tc.length===1?'':'s'), c:'count'});
  }
  if(node.type==='session_end' && a.steps!=null) b.push({t:a.steps+' steps', c:'count'});
  if(node.type==='anomaly' && a.streak_len!=null) b.push({t:a.streak_len+' fails', c:'count'});
  // 5. latency — de-emphasized (neutral)
  const lat = a.latency_s!=null ? a.latency_s : (node.type==='session_end' ? a.elapsed_s : null);
  if(lat!=null) b.push({t:'⚡ '+lat+'s', c:'lat'});
  // machine-wide resource figures for this node's window (node-resource-monitoring)
  // — detail-panel only (not TREE_BADGE), so the tree rows stay uncluttered.
  if(a.resources){
    const r = a.resources;
    const cpu = r.cpu_pct ? r.cpu_pct.avg+'%' : '?';
    const ram = r.ram_pct ? r.ram_pct.avg+'%' : '?';
    const gpu = r.gpu_pct ? ' gpu '+r.gpu_pct.avg+'%' : '';
    b.push({t:'🖥 cpu '+cpu+' ram '+ram+gpu, c:'res'});
  }
  // 6. timestamp (neutral)
  const ts = fmtTime(a.started_at); if(ts) b.push({t:'🕘 '+ts, c:'ts'});
  // 7. step — least important
  if(a.step) b.push({t:'Step '+a.step, c:'step'});
  return b;
}

// Compact subset for the tree row: the glanceable badges only (timestamp/step
// are redundant in the ordered tree, so they are dropped to save width).
const TREE_BADGE = new Set(['name','ok','err','lat','art','trunc','phase','count','skill']);
const treeBadges = node => nodeBadges(node).filter(x=>TREE_BADGE.has(x.c));

function Tree({data,hidden,sel,onSel,collapsed,setCollapsed}){
  // Keep the selected node reachable: expand every ancestor group (walking the
  // real parent_id chain — a delegate's subagent session, or an LLM call
  // nested under the tool that made it, e.g. read_tool_artifact's
  // artifact_query extraction) that is currently collapsed.
  useEffect(()=>{
    const toOpen = [];
    let n = data.nodes[sel];
    while(n && n.parent_id){
      n = data.nodes[n.parent_id];
      if(n) toOpen.push(n.id);
    }
    if(toOpen.some(id=>collapsed.has(id))){
      const c = new Set(collapsed);
      toOpen.forEach(id=>c.delete(id));
      setCollapsed(c);
    }
  },[sel]);

  // Build a real forest from each node's `parent_id` (not just depth/order):
  // any node can have children — a delegate's subagent session root nests
  // under its `delegate` tool_call, an LLM call a tool makes internally
  // (e.g. read_tool_artifact's artifact_query) nests under that tool's node,
  // and a triggered ContextSummarizerNode nests under its triggering node
  // (finalize_step / context_guard) with its own LLM call beneath it.
  const visibleIds = data.order.filter(id=>{
    const n = data.nodes[id];
    return n && (n.type==='session' || !hidden.has(n.type));
  });
  const visible = new Set(visibleIds);
  const childrenOf = new Map();
  for(const id of visibleIds){
    const pid = data.nodes[id].parent_id;
    const key = visible.has(pid) ? pid : null;
    if(!childrenOf.has(key)) childrenOf.set(key, []);
    childrenOf.get(key).push(id);
  }
  const build = parentId => (childrenOf.get(parentId)||[]).map(id=>{
    const kids = childrenOf.get(id);
    return {node:data.nodes[id], children: kids && kids.length ? build(id) : null};
  });
  const forest = build(null);
  const toggle = id=>{ const c=new Set(collapsed); c.has(id)?c.delete(id):c.add(id); setCollapsed(c); };

  return html`<div class="tree">
    <${TreeNodes} entries=${forest} data=${data} sel=${sel} onSel=${onSel}
                  collapsed=${collapsed} toggle=${toggle}/>
  </div>`;
}

function TreeNodes({entries,data,sel,onSel,collapsed,toggle}){
  return html`${entries.map(e=> e.children!=null
    ? html`<${TreeGroup} key=${e.node.id} node=${e.node} kids=${e.children} data=${data}
              sel=${sel} onSel=${onSel} collapsed=${collapsed} toggle=${toggle}/>`
    : html`<${TreeLeaf} key=${e.node.id} node=${e.node} sel=${sel} onSel=${onSel}/>`)}`;
}

// Any node with children renders as a collapsible group — a `session` node
// (delegate's subagent, badged "Subagent <id>") or any other node whose
// dispatch nested LLM calls under it (e.g. a `tool_call` for
// read_tool_artifact nesting its artifact_query extraction call). Collapse
// state is keyed by the node's own id, so nested groups toggle independently.
function TreeGroup({node,kids,data,sel,onSel,collapsed,toggle}){
  const isSubagentRoot = node.type==='session' && node.agent!==data.session_id;
  const open = !collapsed.has(node.id);
  const name = node.type==='session'
    ? (isSubagentRoot ? 'Subagent '+node.agent.slice(0,8) : node.label)
    : meta(node.type).name;
  const badges = node.type==='session' ? [] : treeBadges(node);
  // A `delegate` tool_call (and any other node whose dispatch nests children,
  // e.g. read_tool_artifact's artifact_query) still contributes to the
  // context window itself — show the same "+N role" summary TreeLeaf shows,
  // so grouped nodes don't silently drop their own ctx-window contribution.
  const summary = node.type!=='session' ? addedParts(node.ctx_state) : {added:'',removed:''};
  const onRowClick = ()=>{ onSel(node.id); toggle(node.id); };
  return html`<div class=${'agroup'+(isSubagentRoot?' sub':'')}>
    <div class=${'agent-head'+(node.id===sel?' sel':'')} onClick=${onRowClick}>
      <span class="row-dot sm" style=${{background:meta(node.type).dot}}></span>
      <span class="agent-name">${name}</span>
      ${badges.length ? html`<span class="tleaf-badges">
        ${badges.map((x,i)=>html`<span key=${i} class=${'nbadge sm '+x.c}>${x.t}</span>`)}
      </span>` : null}
      ${isSubagentRoot ? html`<span class="sub-tag">subagent</span>` : null}
      ${node.refusal_flag ? html`<span class="refusal-tag">refused</span>` : null}
      ${(summary.added || summary.removed) ? html`<span class="tleaf-sub">
        ${summary.added ? html`<span class="ctx-delta-add">${summary.added}</span>` : null}
        ${summary.added && summary.removed ? '  ' : null}
        ${summary.removed ? html`<span class="ctx-delta-rem">${summary.removed}</span>` : null}
      </span>` : null}
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
  const summary = addedParts(node.ctx_state);
  const badges = treeBadges(node);
  return html`<div ref=${ref} class=${'tleaf'+(selected?' sel':'')} onClick=${()=>onSel(node.id)}>
    <span class="row-dot sm" style=${{background:m.dot}}></span>
    <span class="tleaf-name">${m.name}</span>
    ${badges.length ? html`<span class="tleaf-badges">
      ${badges.map((x,i)=>html`<span key=${i} class=${'nbadge sm '+x.c}>${x.t}</span>`)}
    </span>` : null}
    ${node.refusal_flag ? html`<span class="refusal-tag">refused</span>` : null}
    <span class="tleaf-sub">
      ${summary.added ? html`<span class="ctx-delta-add">${summary.added}</span>` : null}
      ${summary.added && summary.removed ? '  ' : null}
      ${summary.removed ? html`<span class="ctx-delta-rem">${summary.removed}</span>` : null}
      ${(!summary.added && !summary.removed) ? '—' : null}
    </span>
  </div>`;
}

// A display copy of attributes with cryptic codes mapped to friendly labels.
function displayAttrs(node){
  const a = Object.assign({}, node.attributes||{});
  if(a.phase) a.phase = phaseLabel(a.phase);
  return a;
}

function Detail({node,mainAgent}){
  const m = meta(node.type);
  const subAgent = node.agent && node.agent!==mainAgent ? node.agent : null;
  const attrs = displayAttrs(node);
  const attrsBody = Object.keys(attrs).length ? html`<${CodeBox} value=${attrs}/>` : null;
  const badges = nodeBadges(node);
  return html`<div class="dwrap">
    <div class="dhead">
      <div class="dhead-top">
        <span class="dbadge" style=${{background:m.dot}}>${nodeTitle(node)}</span>
        ${subAgent ? html`<span class="sub-tag">subagent ${subAgent.slice(0,8)}</span>` : null}
        ${node.loop_flag ? html`<span class="loop-tag">loop</span>` : null}
        ${node.anomaly_flag ? html`<span class="anomaly-tag">anomaly</span>` : null}
        ${node.refusal_flag ? html`<span class="refusal-tag">refused</span>` : null}
      </div>
      ${badges.length ? html`<div class="badge-row">
        ${badges.map((x,i)=>html`<span key=${i} class=${'nbadge '+x.c}>${x.t}</span>`)}
      </div>` : null}
    </div>
    <${CtxCard} cs=${node.ctx_state} agent=${subAgent}/>
    <${Section} title="Outputs" data=${node.outputs} body=${outputsBody(node)} open=${true}/>
    <${Section} title="Inputs" data=${node.inputs} open=${true}/>
    <${Section} title="Attributes" data=${attrs} body=${attrsBody} open=${false}/>
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
      <${ContentBox} value=${content}/></div>`
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

// Dedicated, readable rendering of a refusal's full explanation (what/why/
// how-to-fix) so it never requires opening raw_envelope: reason, matched
// rule id, standard reference(s) as clickable links, and the safer
// alternative. Absent entirely on a non-refused tool_call (r.metadata has no
// `refusal` key), including every pre-change trace.
function RefusalDetail({refusal}){
  if(!refusal) return null;
  const refs = refusal.references || [];
  return html`<div class="tr-block">
    <div class="tr-label err">refused — ${refusal.rule_id}</div>
    <div>${refusal.reason}</div>
    ${refs.length ? html`<ul class="refusal-refs">
      ${refs.map((r,i)=>html`<li key=${i}>
        <a href=${r.url} target="_blank" rel="noopener noreferrer">${r.standard_id}</a>
      </li>`)}
    </ul>` : null}
    ${refusal.safer_alternative ? html`<div><b>Safer alternative:</b> ${refusal.safer_alternative}</div>` : null}
  </div>`;
}

function ToolResult({node}){
  const raw = node.outputs && node.outputs.result;
  const r = parseToolResult(raw) || {};
  const cmd = cmdText(node.inputs && node.inputs.args);
  const lang = (r.metadata || {}).lang || {};
  const hasResult = raw!=null && raw!=='';
  const refusal = (r.metadata || {}).refusal;
  // Split the envelope into labelled, language-highlighted boxes (command /
  // output / error) and always keep raw_envelope so nothing is hidden — an empty
  // `output` with the real signal in `error` would otherwise look like "no
  // output". Each box's language comes from the backend `metadata.lang` hint.
  // The tool/status/latency/artifact badges live in the node header (nodeBadges).
  return html`<div class="toolres">
    ${cmd ? html`<div class="tr-block"><div class="tr-label">command</div>
      <${ContentBox} value=${cmd} lang=${lang.command}/></div>` : null}
    ${r.output ? html`<div class="tr-block"><div class="tr-label">output</div>
      <${ContentBox} value=${r.output} lang=${lang.output}/></div>` : null}
    ${r.error ? html`<div class="tr-block"><div class="tr-label err">error</div>
      <${ContentBox} value=${r.error} lang=${lang.error}/></div>` : null}
    <${RefusalDetail} refusal=${refusal}/>
    ${hasResult ? html`<div class="tr-block"><div class="tr-label">raw_envelope</div>
      <${CodeBox} value=${r}/></div>`
      : html`<div class="tr-block muted">No result recorded.</div>`}
  </div>`;
}

function LlmOutputs({node}){
  const o = node.outputs || {};
  const calls = o.tool_calls || [];
  return html`<div class="toolres">
    ${o.content ? html`<div class="tr-block"><div class="tr-label">response</div>
      <${ContentBox} value=${o.content}/></div>` : null}
    ${o.reasoning ? html`<div class="tr-block"><div class="tr-label">reasoning</div>
      <${ContentBox} value=${o.reasoning}/></div>` : null}
    ${calls.length ? html`<div class="tr-block"><div class="tr-label">tool calls · ${calls.length}</div>
      <${CodeBox} value=${calls}/></div>` : null}
    ${(!o.content && !o.reasoning && !calls.length) ? html`<div class="tr-block muted">No output recorded.</div>` : null}
  </div>`;
}

// Mini read-only CodeMirror viewer: pretty JSON/text with syntax highlighting,
// a clickable JSON breadcrumb, fold/expand all, copy all, and ⌘F find (with
// Enter/Shift-Enter next/prev via CodeMirror's search keymap).
function CodeBox({value, lang:hint}){
  const {text,lang} = useMemo(()=>toDoc(value, hint), [value, hint]);
  const host = useRef(null);
  const viewRef = useRef(null);
  const [crumb,setCrumb] = useState([]);
  const [copied,setCopied] = useState(false);

  useEffect(()=>{
    if(!host.current || !CM) return;
    const exts = [
      CM.lineNumbers(),
      CM.EditorState.readOnly.of(true),
      CM.drawSelection(),
      CM.highlightActiveLine(),
      CM.highlightActiveLineGutter(),
      CM.syntaxHighlighting(CM.defaultHighlightStyle,{fallback:true}),
      CM.codeFolding(),
      CM.foldGutter(),
      CM.highlightSelectionMatches(),
      CM.search({top:true}),
      CM.keymap.of([...CM.searchKeymap, ...CM.foldKeymap]),
      CM.EditorView.lineWrapping,
      CB_THEME,
    ];
    if(lang==='json'){
      exts.push(CM.json());
      exts.push(CM.EditorView.updateListener.of(u=>{
        if(u.selectionSet || u.docChanged) setCrumb(jsonPathAt(u.state));
      }));
    } else if(lang==='python' && CM.python){
      exts.push(CM.python());
    } else if(lang==='shell' && CM.shell){
      exts.push(CM.shell());
    }
    const view = new CM.EditorView({
      state: CM.EditorState.create({doc:text, extensions:exts}),
      parent: host.current,
    });
    viewRef.current = view;
    return ()=>{ view.destroy(); viewRef.current=null; };
  }, [text, lang]);

  const run = (fn)=>()=>{ const v=viewRef.current; if(v){ fn(v); v.focus(); } };
  const jump = (from)=>()=>{ const v=viewRef.current; if(!v) return;
    v.dispatch({selection:{anchor:from}, scrollIntoView:true}); v.focus(); };
  const copy = ()=>{ if(navigator.clipboard) navigator.clipboard.writeText(text)
    .then(()=>{ setCopied(true); setTimeout(()=>setCopied(false),1200); }); };

  return html`<div class="cb">
    <div class="cb-bar">
      ${lang==='json' ? html`<div class="cb-crumbs">
        <span class="cb-crumb" onClick=${jump(0)}>root</span>
        ${crumb.map((c,i)=>html`<span key=${i}><span class="cb-sep">›</span>
          <span class="cb-crumb" onClick=${jump(c.from)}>${c.label}</span></span>`)}
      </div>` : html`<div class="cb-crumbs muted">${lang}</div>`}
      <div class="cb-actions">
        <button class="cb-btn" title="Find (⌘F)" onClick=${run(CM.openSearchPanel)}>find</button>
        ${lang==='json' ? html`
          <button class="cb-btn" title="Collapse all" onClick=${run(CM.foldAll)}>collapse</button>
          <button class="cb-btn" title="Expand all" onClick=${run(CM.unfoldAll)}>expand</button>` : null}
        <button class="cb-btn" onClick=${copy}>${copied?'✓ copied':'copy'}</button>
      </div>
    </div>
    <div class="cb-editor" ref=${host}></div>
  </div>`;
}

// Popup showing exactly what a supersession retirement replaced: the full
// original tool-result text next to the short stub that now stands in for
// it in the conversation sent to the model.
function RetirementModal({retirements,onClose}){
  return html`<div class="retire-overlay" onClick=${onClose}>
    <div class="retire-modal" onClick=${e=>e.stopPropagation()}>
      <div class="retire-modal-top">
        <span class="retire-modal-title">Retired tool results</span>
        <button class="retire-close" onClick=${onClose}>close</button>
      </div>
      ${retirements.map((r,i)=>html`<div key=${i} class="retire-item">
        <div class="retire-item-head">${r.tool_call_id ? 'tool_call_id: '+r.tool_call_id : 'message #'+r.index}</div>
        <div class="retire-label">Before (retired)</div>
        <pre class="retire-before">${r.before}</pre>
        <div class="retire-label">After (replaced with)</div>
        <pre class="retire-after">${r.after}</pre>
      </div>`)}
    </div>
  </div>`;
}

function CtxCard({cs,agent}){
  const [showRetired,setShowRetired] = useState(false);
  if(!cs || cs.tokens==null) return null;
  const comp = cs.composition || {};
  const total = cs.tokens || 0;
  const segs = ROLE_ORDER.filter(r=>comp[r]);
  const added = cs.added || {};
  const addedRoles = ROLE_ORDER.filter(r=>added[r]);
  const removedByRole = cs.removed_by_role || {};
  const removedRoles = ROLE_ORDER.filter(r=>removedByRole[r]);
  const retirements = cs.retirements || [];
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
    ${(addedRoles.length || removedRoles.length) ? html`<div class="ctx-delta">
      ${addedRoles.length ? html`<span class="ctx-delta-add">
        +${fmtNum(cs.added_total)} added
        <span class="muted">(${addedRoles.map(r=>ROLE_META[r].label+' '+fmtNum(added[r])).join(', ')})</span>
      </span>` : null}
      ${removedRoles.length ? html`<span class=${'ctx-delta-rem'+(retirements.length?' clickable':'')}
          title=${retirements.length?'Click to see what was retired':null}
          onClick=${retirements.length?()=>setShowRetired(true):null}>
        −${fmtNum(cs.removed)} retired
        <span class="muted">(${removedRoles.map(r=>ROLE_META[r].label+' '+fmtNum(removedByRole[r])).join(', ')})</span>
      </span>` : null}
    </div>` : null}
    ${showRetired ? html`<${RetirementModal} retirements=${retirements} onClose=${()=>setShowRetired(false)}/>` : null}
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
  if(typeof data==='string') return html`<${ContentBox} value=${data}/>`;
  if(Array.isArray(data)) return html`<${CodeBox} value=${data}/>`;
  if(typeof data==='object'){
    const keys = Object.keys(data);
    return html`${keys.map(k=>html`<div key=${k} class="tr-block">
      <div class="tr-label">${k}</div><${Value} v=${data[k]}/></div>`)}`;
  }
  return html`<span>${String(data)}</span>`;
}

function Value({v}){
  if(v==null) return html`<span class="muted">null</span>`;
  if(typeof v==='object') return html`<${CodeBox} value=${v}/>`;
  if(typeof v==='string' && (v.length>60 || v.includes('\\n'))) return html`<${ContentBox} value=${v}/>`;
  return html`<span>${String(v)}</span>`;
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
    """Concatenate the vendored UI library sources, load order preserved.

    Returns:
        The Preact/hooks/htm UMD bundles plus the CodeMirror IIFE bundle joined
        with newlines, ready to inline into a ``<script>`` element.
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
            ``/evals``                   → embedded eval dashboard UI
            ``/api/evals/...``           → eval dashboard JSON (see evals_server.py)
        """
        path = self.path.split("?")[0]
        eval_html = eval_dashboard_html(path)
        if path == "/":
            self._send_html()
        elif eval_html is not None:
            self._send_html(eval_html)
        elif path == "/api/sessions":
            self._send_json(list_sessions(self.base_dir))
        elif path.startswith("/api/evals/"):
            if not handle_eval_api_route(self, path, self.base_dir.resolve() / "evals"):
                self._send_json({"error": "not found"}, status=404)
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

    def _send_html(self, html: str | None = None) -> None:
        """Write the embedded HTML viewer (with vendored libs) as the response.

        Args:
            html: Page body to serve; defaults to the Trace Explorer page.
        """
        body = (html or _full_html()).encode()
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
