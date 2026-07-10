"""Builder tab routes and embedded UI, served from the webui shell server.

A graph canvas over the framework's registered node types (`pipeline/registry.py`):
place nodes, connect edges, edit a node's options, designate start/end, save
the pipeline (via `store.py`'s generic `items` table, `table_name="pipelines"`),
reload it, and launch/monitor/stop a run (`builder_runs.py`). Reuses the same
offline-vendored Preact + htm bundle as the Trace Explorer / Eval Dashboard —
no CDN, no build step.
"""

from __future__ import annotations

import re
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from ..pipeline.registry import list_node_types, validate_runnable
from .builder_runs import RunRegistry
from .store import Store

_TABLE = "pipelines"
_PIPELINE_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")


class _JSONSender(Protocol):
    def _send_json(self, data: Any, status: int = 200) -> None: ...


# ── Embedded single-page HTML ────────────────────────────────────────────────
# ruff: noqa: E501
BUILDER_EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline Builder</title>
<style>
:root{
  --bg:#ffffff; --bg2:#f5f5f7; --panel:#fbfbfd; --line:#e5e5ea;
  --text:#1d1d1f; --muted:#86868b; --accent:#0071e3; --accent-soft:#e8f1fd;
  --pos:#1a7f37; --pos-bg:#e7f6ec; --neg:#d70015; --neg-bg:#fdeaec;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Monaco,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:var(--font);background:var(--bg2);color:var(--text);font-size:13px;-webkit-font-smoothing:antialiased}
#app{min-height:100vh;display:flex;flex-direction:column}
.topbar{display:flex;align-items:center;gap:16px;height:52px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.brand{font-weight:600;font-size:14px}
.spacer{flex:1}
button.btn{font-family:var(--font);font-size:12px;font-weight:500;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button.btn:hover{border-color:var(--accent)}
button.btn.primary{color:#fff;background:var(--accent);border-color:var(--accent)}
button.btn:disabled{opacity:0.5;cursor:default}
select,input[type=text]{font-family:var(--font);font-size:12px;border:1px solid var(--line);border-radius:6px;padding:6px 8px;background:var(--bg)}

.main{flex:1;display:flex;min-height:0}
.sidebar{width:220px;flex:none;border-right:1px solid var(--line);background:var(--panel);padding:12px;overflow:auto}
.sidebar h3{font-size:11px;color:var(--muted);text-transform:uppercase;margin:12px 0 6px}
.node-chip{display:block;width:100%;text-align:left;margin-bottom:6px;padding:8px 10px;border:1px dashed var(--line);border-radius:8px;background:var(--bg);cursor:pointer;font-size:12px}
.node-chip:hover{border-color:var(--accent)}
.pipeline-row{padding:6px 8px;border-radius:6px;cursor:pointer;font-size:12px;margin-bottom:2px}
.pipeline-row:hover{background:var(--bg2)}
.pipeline-row.on{background:var(--accent-soft);color:var(--accent)}

.canvas-wrap{flex:1;position:relative;overflow:auto;background:
  linear-gradient(var(--line) 1px, transparent 1px) 0 0/24px 24px,
  linear-gradient(90deg, var(--line) 1px, transparent 1px) 0 0/24px 24px,
  var(--bg)}
svg.edges{position:absolute;top:0;left:0;pointer-events:none;overflow:visible}
.node-box{position:absolute;width:150px;border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:8px 10px;cursor:grab;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.node-box.selected{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft)}
.node-box.start{border-left:4px solid var(--pos)}
.node-box.end{border-right:4px solid var(--neg)}
.node-box .type{font-weight:600;font-size:12px}
.node-box .role{font-size:10px;color:var(--muted)}
.node-box .connect-handle{position:absolute;right:-8px;top:50%;transform:translateY(-50%);width:14px;height:14px;border-radius:50%;background:var(--accent);cursor:crosshair}
.node-box .del{position:absolute;top:2px;right:4px;font-size:11px;color:var(--muted);cursor:pointer}

.panel{width:280px;flex:none;border-left:1px solid var(--line);background:var(--panel);padding:14px;overflow:auto}
.panel h3{font-size:12px;margin-bottom:10px}
.field{margin-bottom:10px}
.field label{display:block;font-size:11px;color:var(--muted);margin-bottom:4px}
.muted{color:var(--muted)}
.empty{padding:32px;text-align:center;color:var(--muted)}
.progress-badge{font-size:11px;padding:2px 8px;border-radius:6px;background:var(--accent-soft);color:var(--accent)}
.progress-badge.finished{background:var(--pos-bg);color:var(--pos)}
.progress-badge.failed,.progress-badge.stopped{background:var(--neg-bg);color:var(--neg)}
.trace-link{font-size:12px;color:var(--accent);cursor:pointer;text-decoration:underline}
</style>
</head>
<body>
<div id="app"></div>
<script>/*__VENDOR__*/</script>
<script>
const {h, render} = window.preact;
const {useState, useEffect, useRef, useCallback} = window.preactHooks;
const html = window.htm.bind(h);

function getJSON(url){ return fetch(url).then(r => r.json()); }
function postJSON(url, body){ return fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})}).then(r => r.json()); }
function putJSON(url, body){ return fetch(url, {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})}).then(r => r.json()); }
function delReq(url){ return fetch(url, {method:"DELETE"}).then(r => r.json()); }

function uid(){ return Math.random().toString(16).slice(2, 10); }

function emptyGraph(){ return {nodes: [], edges: [], start: null, end: null}; }

function App(){
  const [nodeTypes, setNodeTypes] = useState([]);
  const [pipelines, setPipelines] = useState([]);
  const [pipelineId, setPipelineId] = useState(null);
  const [name, setName] = useState("untitled");
  const [graph, setGraph] = useState(emptyGraph());
  const [selectedId, setSelectedId] = useState(null);
  const [linking, setLinking] = useState(null);
  const [taskPrompt, setTaskPrompt] = useState("");
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);
  const [validation, setValidation] = useState(null);
  const canvasRef = useRef(null);
  const dragRef = useRef(null);

  const refreshPipelines = useCallback(() => { getJSON("/api/builder/pipelines").then(setPipelines); }, []);

  useEffect(() => {
    getJSON("/api/builder/node-types").then(setNodeTypes);
    refreshPipelines();
  }, [refreshPipelines]);

  // Live-progress polling while a run is active.
  useEffect(() => {
    if(!runId) return;
    const timer = setInterval(() => {
      getJSON("/api/builder/runs/" + runId).then(s => {
        setRunStatus(s);
        if(s.phase === "finished" || s.phase === "stopped" || s.phase === "failed"){
          clearInterval(timer);
        }
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [runId]);

  const addNode = (type) => {
    const id = uid();
    const n = {id, type, x: 40 + graph.nodes.length * 40, y: 40 + (graph.nodes.length % 4) * 90, options: {}};
    setGraph(g => ({...g, nodes: [...g.nodes, n]}));
  };

  const deleteNode = (id) => {
    setGraph(g => ({
      nodes: g.nodes.filter(n => n.id !== id),
      edges: g.edges.filter(e => e.from !== id && e.to !== id),
      start: g.start === id ? null : g.start,
      end: g.end === id ? null : g.end,
    }));
    if(selectedId === id) setSelectedId(null);
  };

  const startLink = (id, e) => { e.stopPropagation(); setLinking(id); };
  const swallowClick = (e) => { e.stopPropagation(); };
  const finishLink = (id) => {
    if(linking && linking !== id){
      setGraph(g => ({...g, edges: [...g.edges.filter(e => e.from !== linking), {from: linking, to: id}]}));
    }
    setLinking(null);
  };

  const onNodeMouseDown = (id, e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const node = graph.nodes.find(n => n.id === id);
    dragRef.current = {id, offX: e.clientX - rect.left - node.x, offY: e.clientY - rect.top - node.y};
    setSelectedId(id);
  };
  const onCanvasMouseMove = (e) => {
    if(!dragRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const {id, offX, offY} = dragRef.current;
    const x = Math.max(0, e.clientX - rect.left - offX);
    const y = Math.max(0, e.clientY - rect.top - offY);
    setGraph(g => ({...g, nodes: g.nodes.map(n => n.id === id ? {...n, x, y} : n)}));
  };
  const onCanvasMouseUp = () => { dragRef.current = null; };

  const selected = graph.nodes.find(n => n.id === selectedId) || null;

  const savePipeline = () => {
    const payload = {name, graph};
    const req = pipelineId ? putJSON("/api/builder/pipelines/" + pipelineId, payload) : postJSON("/api/builder/pipelines", payload);
    req.then(res => {
      if(res.id) setPipelineId(res.id);
      refreshPipelines();
    });
  };

  const loadPipeline = (id) => {
    getJSON("/api/builder/pipelines/" + id).then(p => {
      setPipelineId(p.id);
      setName(p.name);
      setGraph(p.graph);
      setSelectedId(null);
      setRunId(null);
      setRunStatus(null);
      setValidation(null);
    });
  };

  const deletePipeline = (id, e) => {
    e.stopPropagation();
    delReq("/api/builder/pipelines/" + id).then(() => {
      if(pipelineId === id){ setPipelineId(null); setName("untitled"); setGraph(emptyGraph()); }
      refreshPipelines();
    });
  };

  const runPipeline = () => {
    if(!pipelineId){ setValidation("save the pipeline before running"); return; }
    postJSON("/api/builder/pipelines/" + pipelineId + "/run", {task_prompt: taskPrompt}).then(res => {
      if(res.error){ setValidation(res.error); return; }
      setValidation(null);
      setRunId(res.run_id);
      setRunStatus({phase: "starting"});
    });
  };

  const stopRun = () => { if(runId) postJSON("/api/builder/runs/" + runId + "/stop", {}); };

  const openTrace = () => {
    if(runStatus && runStatus.session_id){
      try{ window.parent && window.parent.postMessage({type:"mca:selection", tab:"traces", session: runStatus.session_id}, "*"); }catch(e){}
      window.parent && window.parent.postMessage({type:"mca:navigate", tab:"traces"}, "*");
    }
  };

  return html`
    <div class="topbar">
      <div class="brand">Pipeline Builder</div>
      <input type="text" value=${name} onInput=${e => setName(e.target.value)} />
      <button class="btn" onClick=${savePipeline}>Save</button>
      <div class="spacer"></div>
      <input type="text" placeholder="task prompt for run" style=${{width:"260px"}} value=${taskPrompt} onInput=${e => setTaskPrompt(e.target.value)} />
      <button class="btn primary" disabled=${!!runId && runStatus && runStatus.phase === "running"} onClick=${runPipeline}>Run</button>
      <button class="btn" disabled=${!runId || !runStatus || runStatus.phase !== "running"} onClick=${stopRun}>Stop</button>
      ${runStatus && html`<span class="progress-badge ${runStatus.phase}">${runStatus.phase}${runStatus.step_num ? " · step " + runStatus.step_num : ""}</span>`}
      ${runStatus && runStatus.phase === "finished" && runStatus.session_id && html`<span class="trace-link" onClick=${openTrace}>open trace →</span>`}
    </div>
    <div class="main">
      <div class="sidebar">
        <h3>Node types</h3>
        ${nodeTypes.map(t => html`<button class="node-chip" onClick=${() => addNode(t.name)}>+ ${t.name}</button>`)}
        <h3>Saved pipelines</h3>
        ${pipelines.map(p => html`
          <div class="pipeline-row ${p.id === pipelineId ? "on" : ""}" onClick=${() => loadPipeline(p.id)}>
            ${p.name} <span class="muted" onClick=${e => deletePipeline(p.id, e)}> ✕</span>
          </div>
        `)}
        ${!pipelines.length && html`<div class="muted">No saved pipelines yet.</div>`}
      </div>
      <div class="canvas-wrap" ref=${canvasRef} onMouseMove=${onCanvasMouseMove} onMouseUp=${onCanvasMouseUp} onMouseLeave=${onCanvasMouseUp}>
        <svg class="edges" width="100%" height="100%">
          ${graph.edges.map(e => {
            const from = graph.nodes.find(n => n.id === e.from);
            const to = graph.nodes.find(n => n.id === e.to);
            if(!from || !to) return null;
            const x1 = from.x + 150, y1 = from.y + 24, x2 = to.x, y2 = to.y + 24;
            return html`<line x1=${x1} y1=${y1} x2=${x2} y2=${y2} stroke="#0071e3" stroke-width="2" marker-end="url(#arrow)" />`;
          })}
          <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#0071e3" /></marker></defs>
        </svg>
        ${graph.nodes.map(n => html`
          <div class="node-box ${n.id === selectedId ? "selected" : ""} ${n.id === graph.start ? "start" : ""} ${n.id === graph.end ? "end" : ""}"
               style=${{left: n.x + "px", top: n.y + "px"}}
               onMouseDown=${e => onNodeMouseDown(n.id, e)}
               onClick=${() => finishLink(n.id)}>
            <span class="del" onClick=${e => { e.stopPropagation(); deleteNode(n.id); }}>✕</span>
            <div class="type">${n.type}</div>
            <div class="role">${n.id === graph.start ? "start" : ""} ${n.id === graph.end ? "end" : ""}</div>
            <div class="connect-handle" onMouseDown=${e => startLink(n.id, e)} onClick=${swallowClick} title="drag to connect"></div>
          </div>
        `)}
      </div>
      <div class="panel">
        ${selected ? html`
          <h3>${selected.type}</h3>
          <div class="field"><label>Node id</label><span class="mono">${selected.id}</span></div>
          <div class="field">
            <label>Role</label>
            <button class="btn" onClick=${() => setGraph(g => ({...g, start: selected.id}))}>Mark start</button>
            <button class="btn" onClick=${() => setGraph(g => ({...g, end: selected.id}))}>Mark end</button>
          </div>
          <div class="field"><label>Options</label><div class="muted">This node type has no editable options.</div></div>
        ` : html`<div class="empty">Select a node to edit it.</div>`}
        ${validation && html`<div class="field" style=${{color:"#d70015"}}>${validation}</div>`}
      </div>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById("app"));
</script>
</body>
</html>
"""

_BUILDER_VENDOR_TOKEN = "/*__VENDOR__*/"
_BUILDER_VENDOR_FILES = ("preact.min.js", "hooks.umd.js", "htm.umd.js")


@lru_cache(maxsize=1)
def _builder_vendor_js(vendor_dir: Path) -> str:
    return "\n".join(
        (vendor_dir / name).read_text(encoding="utf-8")
        for name in _BUILDER_VENDOR_FILES
    )


@lru_cache(maxsize=1)
def _full_builder_html(vendor_dir: Path) -> str:
    return BUILDER_EMBEDDED_HTML.replace(
        _BUILDER_VENDOR_TOKEN, _builder_vendor_js(vendor_dir)
    )


def builder_html(path: str) -> str | None:
    """Return the Builder tab's HTML for ``/builder``, else ``None``."""
    if path != "/builder":
        return None
    vendor_dir = Path(__file__).parent.parent / "viewer" / "_vendor"
    return _full_builder_html(vendor_dir)


def _pipeline_to_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("id", ""),
        "name": payload.get("name", ""),
        "graph": payload.get("graph"),
    }


def handle_builder_api_get(
    handler: _JSONSender, path: str, store: Store, run_registry: RunRegistry
) -> bool:
    """Dispatch a `/api/builder/...` GET request; return True if handled."""
    if path == "/api/builder/node-types":
        handler._send_json(list_node_types())
        return True

    if path == "/api/builder/pipelines":
        items = store.list_items(_TABLE)
        handler._send_json([_pipeline_to_response(item) for item in items])
        return True

    match = re.fullmatch(r"/api/builder/pipelines/([^/]+)", path)
    if match:
        item_id = match.group(1)
        payload = store.get_item(_TABLE, item_id)
        if payload is None:
            handler._send_json({"error": "pipeline not found"}, status=404)
        else:
            handler._send_json(_pipeline_to_response(payload))
        return True

    match = re.fullmatch(r"/api/builder/runs/([^/]+)", path)
    if match:
        run_id = match.group(1)
        if not _RUN_ID_RE.match(run_id):
            handler._send_json({"error": "invalid run id"}, status=400)
            return True
        run = run_registry.get(run_id)
        if run is None:
            handler._send_json({"error": "run not found"}, status=404)
        else:
            handler._send_json(run.status())
        return True

    return False


def _create_pipeline(
    handler: _JSONSender, store: Store, payload: dict[str, Any]
) -> None:
    item_id = uuid.uuid4().hex[:12]
    store.create_item(
        _TABLE,
        item_id,
        {"id": item_id, "name": payload.get("name", ""), "graph": payload.get("graph")},
    )
    handler._send_json({"id": item_id}, status=201)


def _write_existing_pipeline(
    handler: _JSONSender,
    method: str,
    store: Store,
    item_id: str,
    payload: dict[str, Any],
) -> None:
    if method == "PUT":
        if store.get_item(_TABLE, item_id) is None:
            handler._send_json({"error": "pipeline not found"}, status=404)
            return
        store.update_item(
            _TABLE,
            item_id,
            {
                "id": item_id,
                "name": payload.get("name", ""),
                "graph": payload.get("graph"),
            },
        )
        handler._send_json({"id": item_id})
        return
    store.delete_item(_TABLE, item_id)
    handler._send_json({"ok": True})


def _launch_run(
    handler: _JSONSender,
    store: Store,
    run_registry: RunRegistry,
    base_dir: Path,
    item_id: str,
    payload: dict[str, Any],
) -> None:
    pipeline = store.get_item(_TABLE, item_id)
    if pipeline is None:
        handler._send_json({"error": "pipeline not found"}, status=404)
        return
    graph = pipeline.get("graph") or {}
    error = validate_runnable(
        graph.get("nodes", []),
        graph.get("edges", []),
        graph.get("start"),
        graph.get("end"),
    )
    if error is not None:
        handler._send_json({"error": error}, status=400)
        return
    task_prompt = str(payload.get("task_prompt") or "").strip()
    if not task_prompt:
        handler._send_json({"error": "a task prompt is required to run"}, status=400)
        return
    run_id = run_registry.launch(
        task_prompt, int(payload.get("max_steps") or 40), base_dir
    )
    handler._send_json({"run_id": run_id}, status=201)


def _stop_run(handler: _JSONSender, run_registry: RunRegistry, run_id: str) -> None:
    run = run_registry.get(run_id)
    if run is None:
        handler._send_json({"error": "run not found"}, status=404)
        return
    run.stop()
    handler._send_json({"ok": True})


def handle_builder_api_write(
    handler: _JSONSender,
    method: str,
    path: str,
    store: Store,
    run_registry: RunRegistry,
    base_dir: Path,
    payload: dict[str, Any],
) -> bool:
    """Dispatch a `/api/builder/...` POST/PUT/DELETE request; return True if handled."""
    if path == "/api/builder/pipelines" and method == "POST":
        _create_pipeline(handler, store, payload)
        return True

    match = re.fullmatch(r"/api/builder/pipelines/([^/]+)", path)
    if match and method in ("PUT", "DELETE"):
        _write_existing_pipeline(handler, method, store, match.group(1), payload)
        return True

    match = re.fullmatch(r"/api/builder/pipelines/([^/]+)/run", path)
    if match and method == "POST":
        _launch_run(handler, store, run_registry, base_dir, match.group(1), payload)
        return True

    match = re.fullmatch(r"/api/builder/runs/([^/]+)/stop", path)
    if match and method == "POST":
        _stop_run(handler, run_registry, match.group(1))
        return True

    return False


__all__ = [
    "RunRegistry",
    "builder_html",
    "handle_builder_api_get",
    "handle_builder_api_write",
]
