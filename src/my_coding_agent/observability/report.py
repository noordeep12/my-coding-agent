"""Generate a self-contained, offline HTML viewer for agent sessions.

``write_report`` renders every instrumented session under ``.my_coding_agent/``
into a single ``viewer.html`` — inline CSS + vanilla JS, with the trace-tree data
embedded as JSON. No server, no dependencies, no network: open the file in any
browser (works from ``file://``). The page is a two-pane viewer — a collapsible,
searchable pipeline trace tree on the left and the selected object's decision
panel (status, context-window bar, input/output, metadata) on the right — built
from the same :func:`build_trace_tree` model as everything else.

CLI::

    uv run my-coding-agent-viewer        # write + open .my_coding_agent/viewer.html
    uv run my-coding-agent-viewer --no-open
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Any

from . import reader
from .tree import build_trace_tree

DEFAULT_ROOT = reader.DEFAULT_ROOT
_DATA_TOKEN = "__OBSERVABILITY_DATA__"


def _session_analytics(s: reader.Session) -> dict[str, Any]:
    """Bundle the derived views the viewer renders (header chips + Overview node).

    Reuses the existing ``reader`` helpers verbatim — nothing is re-computed here.
    """
    bottlenecks = reader.bottlenecks(s)
    failures = sum(1 for t in s.tool_calls if t.ok is False or t.status == "error")
    return {
        "summary": {
            "total_tokens": sum(r["tokens"] for r in bottlenecks),
            "est_cost_usd": round(sum(r["cost_usd"] for r in bottlenecks), 6),
            "failures": failures,
        },
        "analytics": {
            "context_series": reader.context_series(s),
            "bottlenecks": bottlenecks,
            "loops": reader.detect_loops(s),
        },
    }


def build_payload(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Collect top-level sessions and their trace trees into a JSON payload."""
    by_id = reader.load_sessions_by_id(root)
    roots = [
        s
        for s in by_id.values()
        if not s.parent_session_id or s.parent_session_id not in by_id
    ]
    roots.sort(key=lambda s: s.started_at, reverse=True)
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "label": s.label,
                "ok": s.ok,
                "stop_reason": s.stop_reason,
                "started_at": s.started_at,
                "tree": build_trace_tree(s, by_id).to_dict(),
                **_session_analytics(s),
            }
            for s in roots
        ]
    }


def render_html(payload: dict[str, Any]) -> str:
    """Embed ``payload`` into the HTML template, safely escaped for ``<script>``."""
    # Escape ``</`` so embedded strings can never close the <script> early.
    data = json.dumps(payload, default=str).replace("</", "<\\/")
    return _TEMPLATE.replace(_DATA_TOKEN, data)


def write_report(
    root: str | Path = DEFAULT_ROOT, out: str | Path | None = None
) -> Path:
    """Render the viewer and write it to ``out`` (default ``<root>/viewer.html``)."""
    html = render_html(build_payload(root))
    out_path = Path(out) if out else Path(root) / "viewer.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point: generate the viewer and open it in a browser."""
    parser = argparse.ArgumentParser(description="Generate the agent session viewer.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="session directory root")
    parser.add_argument("--out", default=None, help="output HTML path")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)
    out_path = write_report(args.root, args.out)
    print(f"Wrote {out_path}")
    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Session Viewer</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
--accent:#58a6ff;--sel:#1f6feb2b;--blue:#388bfd;--green:#3fb950;--red:#f85149;
--amber:#d29922}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;align-items:center;gap:12px;padding:11px 18px;
border-bottom:1px solid var(--border);background:var(--panel);position:sticky;top:0;z-index:5}
header h1{font-size:15px;margin:0;font-weight:600;white-space:nowrap}
.sp{flex:1}
select,input,button{background:#0d1117;color:var(--text);border:1px solid var(--border);
border-radius:6px;padding:6px 10px;font:inherit}
button{cursor:pointer}
button:hover,select:hover,input:focus{border-color:var(--accent);outline:none}
main{display:grid;grid-template-columns:minmax(330px,2fr) 3fr;height:calc(100vh - 57px)}
#left,#right{overflow:auto;padding:14px 16px}
#left{border-right:1px solid var(--border)}
svg{flex:none;vertical-align:middle}
.row{display:flex;align-items:center;gap:7px;padding:3px 6px;border-radius:6px;
cursor:pointer;white-space:nowrap;color:var(--text)}
.row:hover{background:#ffffff0a}
.row.sel{background:var(--sel);box-shadow:inset 0 0 0 1px var(--accent)}
.tog{width:13px;text-align:center;color:var(--muted);user-select:none;flex:none;
font-size:11px}
.ic{width:17px;height:17px;color:var(--muted);display:flex;align-items:center}
.row.sel .ic{color:var(--accent)}
.ttl{overflow:hidden;text-overflow:ellipsis}
.tok{margin-left:auto;padding-left:8px;color:var(--green);font-size:11px;
font-variant-numeric:tabular-nums;flex:none}
.tok.rem{color:var(--red)}
.kids{margin-left:13px;border-left:1px solid var(--border);padding-left:6px}
.hidden{display:none}
#meta h2{font-size:16px;margin:0 0 5px;display:flex;align-items:center;gap:8px}
.meta-sub{color:var(--muted);font-size:12px;margin-bottom:14px;display:flex;
align-items:center;gap:7px}
.meta-sub code{background:#ffffff10;padding:1px 6px;border-radius:5px}
.rowst{width:15px;height:15px;flex:none;display:flex;align-items:center}
.ist-success{color:var(--green)}
.ist-failure{color:var(--red)}
.ist-warning{color:var(--amber)}
.kvk{color:var(--muted);font-size:12px;margin:8px 0 2px}
.ctx{margin:6px 0 18px}
.ctx .cap{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);
margin-bottom:5px}
.bar{height:12px;border-radius:6px;background:#ffffff10;border:1px solid var(--border);
display:flex;overflow:hidden}
.bar .seg{height:100%}
.seg.blue{background:var(--blue)}
.seg.green{background:var(--green)}
.seg.red{background:var(--red)}
.delta.add{color:var(--green)}
.delta.rem{color:var(--red)}
.mlabel{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;
margin:14px 0 3px}
.kv{display:grid;grid-template-columns:170px 1fr;gap:10px;padding:7px 0;
border-bottom:1px solid var(--border)}
.kv .k{color:var(--muted)}
pre{background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:11px;
overflow:auto;max-height:440px;white-space:pre-wrap;word-break:break-word;margin:4px 0;
font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.empty{color:var(--muted);padding:48px 12px;text-align:center}
.codebox{position:relative}
.copybtn{position:absolute;top:8px;right:8px;z-index:1;padding:2px 9px;font-size:11px;
opacity:.55;background:var(--panel)}
.codebox:hover .copybtn{opacity:1}
.chips{display:flex;gap:8px;font-size:12px}
.chip{background:#ffffff10;border:1px solid var(--border);border-radius:12px;padding:2px 9px}
.chip.bad{border-color:var(--red);color:var(--red)}
.gchart{display:flex;align-items:flex-end;gap:4px;height:120px;padding:8px 2px;margin:4px 0 18px;
border:1px solid var(--border);border-radius:8px;background:#0d1117}
.gcol{flex:1;min-width:8px;display:flex;flex-direction:column;justify-content:flex-end;height:100%}
.gcol .seg{width:100%;border-radius:3px 3px 0 0;background:var(--blue);min-height:2px}
.gcol .gx{text-align:center;font-size:10px;color:var(--muted);margin-top:3px}
table{width:100%;border-collapse:collapse;margin:4px 0 18px;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.loops .lp{display:flex;gap:8px;align-items:baseline;padding:6px 0;border-bottom:1px solid var(--border)}
.loops .tag{color:var(--muted);font-size:11px;text-transform:uppercase}
.loops .sig{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
font:12.5px ui-monospace,Menlo,monospace}
.loops .cnt{color:var(--amber)}
.ctxnote{font-size:12px;color:var(--muted);margin-top:7px}
</style>
</head>
<body>
<header>
  <h1>Agent Session Viewer</h1>
  <select id="sessions" title="session"></select>
  <input id="search" placeholder="filter the tree&#8230;" size="16">
  <button id="expand">Expand all</button>
  <button id="collapse">Collapse all</button>
  <span id="chips" class="chips"></span>
  <span class="sp"></span>
  <span class="meta-sub" style="margin:0">
    context bar: <span style="color:var(--blue)">&#9632; history</span>
    <span style="color:var(--green)">&#9632; added</span>
    <span style="color:var(--red)">&#9632; evicted</span></span>
</header>
<main>
  <div id="left"></div>
  <div id="right"><div id="meta"><div class="empty">Select an object in the tree.</div></div></div>
</main>
<script>window.__OBS__ = __OBSERVABILITY_DATA__;</script>
<script>
const S=(p)=>`<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
const ICONS={
 agent:S('<rect x="3" y="3" width="10" height="10" rx="2"/><path d="M6 6h4v4H6z"/>'),
 step:S('<path d="M6 4l5 4-5 4"/>'),
 context_manager:S('<rect x="2.5" y="3" width="11" height="10" rx="1.5"/><path d="M2.5 6h11"/>'),
 tool_router:S('<path d="M3 12V8a2 2 0 0 1 2-2h7"/><path d="M9 4l3 2-3 2"/>'),
 llm_call:S('<circle cx="8" cy="8" r="5"/>'),
 user_message:S('<rect x="2.5" y="3.5" width="11" height="9" rx="1.5"/><path d="M5 6.5h6M5 9h4"/>'),
 system_message:S('<circle cx="8" cy="8" r="2.3"/><path d="M8 2v2M8 12v2M2 8h2M12 8h2"/>'),
 tool_executor:S('<path d="M4 5l3 3-3 3"/><path d="M9 11h4"/>'),
 tool_output_validation:S('<path d="M3.5 8.5l2.6 2.6L12.5 5"/>'),
 overview:S('<path d="M2.5 13V9M6 13V4M9.5 13V7M13 13V2"/>')};
const STIC={success:S('<path d="M3.5 8.5l2.6 2.6L12.5 5"/>'),
 failure:S('<path d="M5 5l6 6M11 5l-6 6"/>'),
 warning:S('<path d="M8 3l5.5 9.5h-11z"/><path d="M8 7v2.6"/><circle cx="8" cy="11.4" r=".5" fill="currentColor"/>')};
function statusIcon(st,msg){const s=icon(st,STIC);s.classList.remove('ic');
  s.classList.add('rowst','ist-'+st);if(msg)s.title=msg;return s;}
const HANDLED=new Set(['ctx','status','message','input','output','reasoning','name','label',
 'context_window']);
const D=window.__OBS__||{sessions:[]};
let root=null,index={},collapsed=new Set(),selectedId=null,cur=null;

function el(tag,cls,txt){const e=document.createElement(tag);if(cls)e.className=cls;
if(txt!=null)e.textContent=txt;return e;}
function icon(name,map){const s=el('span','ic');s.innerHTML=(map||ICONS)[name]||'';return s;}
function hasVal(v){return v!==undefined&&v!==null&&v!==''&&!(Array.isArray(v)&&!v.length);}
function fmt(n){return (n||0).toLocaleString();}

function buildNode(node,container){
  const row=el('div','row');row.dataset.id=node.node_id;
  const hasKids=node.children&&node.children.length>0;
  const tog=el('span','tog',hasKids?'\\u25BE':'');
  row.append(tog,icon(node.type),el('span','ttl',node.title));
  const md=node.metadata||{};
  // Only flag problems inline; success is implicit (shown in the panel).
  if(md.status==='warning'||md.status==='failure')
    row.appendChild(statusIcon(md.status,md.message));
  // Right-aligned badge on any row that updates the context window.
  if(md.ctx&&md.ctx.removed>0)
    row.appendChild(el('span','tok rem','\\u2212'+fmt(md.ctx.removed)+' tok'));
  else if(md.ctx&&md.ctx.added>0)
    row.appendChild(el('span','tok','+'+(md.ctx.estimated?'~':'')+fmt(md.ctx.added)+' tok'));
  container.appendChild(row);
  let kids=null;
  if(hasKids){kids=el('div','kids');container.appendChild(kids);
    node.children.forEach(c=>buildNode(c,kids));}
  index[node.node_id]={node,row,kids,tog};
  tog.addEventListener('click',e=>{e.stopPropagation();toggle(node.node_id);});
  row.addEventListener('click',()=>select(node.node_id));
}
function toggle(id){const r=index[id];if(!r||!r.kids)return;
  if(collapsed.has(id)){collapsed.delete(id);r.kids.classList.remove('hidden');r.tog.textContent='\\u25BE';}
  else{collapsed.add(id);r.kids.classList.add('hidden');r.tog.textContent='\\u25B8';}}
function select(id){const r=index[id];if(!r)return;
  if(selectedId&&index[selectedId])index[selectedId].row.classList.remove('sel');
  selectedId=id;r.row.classList.add('sel');
  if(id==='overview')renderOverview();else renderMeta(r.node);}

function label(m,t){m.appendChild(el('div','mlabel',t));}

function ctxAddedBy(t){return t==='llm_call'?'this LLM output'
  :t==='tool_executor'?'this tool call':'this step';}
function renderCtx(m,c,node){if(!c||!c.window)return;
  label(m,'CONTEXT WINDOW');
  const wrap=el('div','ctx');const cap=el('div','cap');
  cap.appendChild(el('span',null,fmt(c.history)+' / '+fmt(c.window)+' tokens'+
    (c.agent_label?'  \\u00B7  '+c.agent_label:'')));
  if(c.removed>0){cap.appendChild(el('span','delta rem','\\u2212'+fmt(c.removed)+' evicted'));}
  else if(c.added>0){cap.appendChild(el('span','delta add','+'+(c.estimated?'~':'')+fmt(c.added)+' added'));}
  wrap.appendChild(cap);
  const bar=el('div','bar');
  const pct=v=>Math.max(0,Math.min(100,v/c.window*100));
  const hist=el('div','seg blue');hist.style.width=pct(c.history)+'%';bar.appendChild(hist);
  if(c.removed>0){const r=el('div','seg red');r.style.width=pct(c.removed)+'%';bar.appendChild(r);}
  else if(c.added>0){const g=el('div','seg green');g.style.width=pct(c.added)+'%';bar.appendChild(g);}
  wrap.appendChild(bar);
  // Plain-language line: how many tokens this step added to / evicted from the window.
  if(c.removed>0)
    wrap.appendChild(el('div','ctxnote','\\u2212'+fmt(c.removed)+' tokens evicted from the context window'));
  else if(c.added>0)
    wrap.appendChild(el('div','ctxnote','+'+(c.estimated?'~':'')+fmt(c.added)+
      ' tokens added to the context window by '+ctxAddedBy(node&&node.type)));
  m.appendChild(wrap);}

function copyText(text,btn){const ok=()=>{const o=btn.textContent;btn.textContent='Copied';
  setTimeout(()=>{btn.textContent=o;},1200);};
  const fallback=()=>{const ta=document.createElement('textarea');ta.value=text;
    document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}catch(e){}
    document.body.removeChild(ta);ok();};
  if(navigator.clipboard&&navigator.clipboard.writeText)
    navigator.clipboard.writeText(text).then(ok,fallback);
  else fallback();}
function renderValue(m,val){const box=el('div','codebox');
  const pre=el('pre');
  pre.textContent=(typeof val==='object')?JSON.stringify(val,null,2):String(val);
  const btn=el('button','copybtn','Copy');
  btn.addEventListener('click',e=>{e.stopPropagation();copyText(pre.textContent,btn);});
  box.append(btn,pre);m.appendChild(box);}

function renderMetadata(m,md){
  const rest=Object.keys(md).filter(k=>!HANDLED.has(k)&&hasVal(md[k]));
  if(!rest.length)return;label(m,'METADATA');
  rest.forEach(k=>{const v=md[k];
    if(typeof v==='object'){m.appendChild(el('div','kvk',k));renderValue(m,v);}
    else{const kv=el('div','kv');kv.appendChild(el('div','k',k));
      kv.appendChild(el('div','v',String(v)));m.appendChild(kv);}});}

function renderMeta(node){const m=document.getElementById('meta');m.innerHTML='';
  const md=node.metadata||{};
  const h=el('h2');h.appendChild(icon(node.type));h.appendChild(el('span',null,node.title));
  m.appendChild(h);
  // Job status (logo only) at the top, just before the timestamp.
  const sub=el('div','meta-sub');
  if(md.status&&STIC[md.status])sub.appendChild(statusIcon(md.status,md.message));
  if(node.timestamp)sub.appendChild(el('span',null,node.timestamp));
  m.appendChild(sub);
  renderCtx(m,md.ctx,node);
  if(hasVal(md.output)){label(m,'OUTPUT');renderValue(m,md.output);}
  if(hasVal(md.input)){label(m,'INPUT');renderValue(m,md.input);}
  renderMetadata(m,md);}

function renderOverview(){const m=document.getElementById('meta');m.innerHTML='';
  const h=el('h2');h.appendChild(icon('overview'));
  h.appendChild(el('span',null,'Session Overview'));m.appendChild(h);
  m.appendChild(el('div','meta-sub',cur.label+'  \\u00B7  '+cur.session_id));
  const a=cur.analytics||{};
  // Context growth — one bar per LLM call, height = % of window used.
  label(m,'CONTEXT GROWTH');
  const cs=a.context_series||{call:[]};
  if(cs.call&&cs.call.length){const ch=el('div','gchart');
    cs.call.forEach((c,i)=>{const col=el('div','gcol');const seg=el('div','seg');
      seg.style.height=Math.max(2,cs.pct[i])+'%';
      seg.title=fmt(cs.prompt_tokens[i])+' / '+fmt(cs.context_window[i])+' tokens ('+cs.pct[i]+'%)';
      col.appendChild(seg);col.appendChild(el('div','gx','#'+c));ch.appendChild(col);});
    m.appendChild(ch);}
  else m.appendChild(el('div','empty','No LLM calls with a context snapshot.'));
  // Bottlenecks — per-step tokens / latency / $ (sorted by tokens desc).
  label(m,'BOTTLENECKS');
  const bn=a.bottlenecks||[];
  if(bn.length){const t=el('table');
    t.innerHTML='<thead><tr><th>step</th><th class=num>calls</th>'+
      '<th class=num>tokens</th><th class=num>latency</th><th class=num>$</th></tr></thead>';
    const tb=el('tbody');
    bn.forEach(r=>{const tr=el('tr');
      tr.innerHTML='<td>'+r.step+'</td><td class=num>'+fmt(r.calls)+'</td><td class=num>'+
        fmt(r.tokens)+'</td><td class=num>'+r.latency_s+'s</td><td class=num>$'+
        r.cost_usd+'</td>';tb.appendChild(tr);});
    t.appendChild(tb);m.appendChild(t);}
  else m.appendChild(el('div','empty','No steps recorded.'));
  // Loops & redundancy — repeated tool calls / decisions (count > 1).
  label(m,'LOOPS / REDUNDANCY');
  const lp=a.loops||[];
  if(lp.length){const box=el('div','loops');
    lp.forEach(f=>{const row=el('div','lp');row.appendChild(el('span','tag',f.kind));
      row.appendChild(el('span','sig',f.signature));
      row.appendChild(el('span','cnt','\\u00D7'+f.count));box.appendChild(row);});
    m.appendChild(box);}
  else m.appendChild(el('div','empty','No repeated calls or decisions detected.'));}

function renderChips(){const c=document.getElementById('chips');c.innerHTML='';
  const s=(cur&&cur.summary)||{total_tokens:0,est_cost_usd:0,failures:0};
  c.appendChild(el('span','chip','$'+s.est_cost_usd));
  c.appendChild(el('span','chip',fmt(s.total_tokens)+' tok'));
  c.appendChild(el('span','chip'+(s.failures?' bad':''),s.failures+' fail'));}

function expandAll(){collapsed.clear();Object.values(index).forEach(r=>{
  if(r.kids){r.kids.classList.remove('hidden');r.tog.textContent='\\u25BE';}});}
function collapseAll(){Object.values(index).forEach(r=>{if(r.kids){
  collapsed.add(r.node.node_id);r.kids.classList.add('hidden');r.tog.textContent='\\u25B8';}});}
function filter(term){term=term.trim().toLowerCase();
  function visit(node){const r=index[node.node_id];
    const self=!term||node.title.toLowerCase().includes(term);
    let child=false;(node.children||[]).forEach(c=>{if(visit(c))child=true;});
    const vis=self||child;r.row.classList.toggle('hidden',!vis);
    if(r.kids){const show=term?child:!collapsed.has(node.node_id);
      r.kids.classList.toggle('hidden',!show);
      r.tog.textContent=show?'\\u25BE':'\\u25B8';}
    return vis;}
  if(root)visit(root);}

function buildOverviewRow(container){const row=el('div','row');row.dataset.id='overview';
  row.append(el('span','tog',''),icon('overview'),el('span','ttl','Session Overview'));
  container.appendChild(row);
  index['overview']={node:{node_id:'overview',type:'overview',title:'Session Overview',children:[]},
    row,kids:null,tog:null};
  row.addEventListener('click',()=>select('overview'));}

function renderSession(i){const s=D.sessions[i];if(!s)return;
  cur=s;root=s.tree;index={};collapsed=new Set();selectedId=null;
  const left=document.getElementById('left');left.innerHTML='';
  buildOverviewRow(left);buildNode(root,left);
  renderChips();select('overview');
  document.getElementById('search').value='';}

function init(){const sel=document.getElementById('sessions');
  if(!D.sessions.length){document.getElementById('left').innerHTML=
    '<div class="empty">No instrumented sessions found.<br>Run <code>uv run my-coding-agent</code>.</div>';
    return;}
  D.sessions.forEach((s,i)=>{const o=el('option',null,
    (s.ok?'\\u2713 ':'\\u2717 ')+s.label+'  \\u00B7  '+s.session_id+'  \\u00B7  '+s.stop_reason);
    o.value=i;sel.appendChild(o);});
  sel.addEventListener('change',()=>renderSession(+sel.value));
  document.getElementById('search').addEventListener('input',e=>filter(e.target.value));
  document.getElementById('expand').addEventListener('click',expandAll);
  document.getElementById('collapse').addEventListener('click',collapseAll);
  renderSession(0);}
init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
