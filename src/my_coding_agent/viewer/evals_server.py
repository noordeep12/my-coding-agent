"""Eval Dashboard routes and embedded UI, served from the Trace Explorer server.

Read-only over the persisted eval result store (`evals_reader.py`): renders
runs, datasets, and comparisons that already exist on disk. Never runs or
mutates an eval. Reuses the same offline-vendored Preact + htm bundle as the
Trace Explorer (no CDN, no build step) — see `server.py`'s `_vendor_js()`.
"""

from __future__ import annotations

import dataclasses
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from . import evals_reader

# Eval run ids are `uuid.uuid4().hex[:12]` (evals/results.py) — 12 lowercase
# hex chars; allow some slack for future id shapes without loosening past hex.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,64}$")


class _JSONSender(Protocol):
    def _send_json(self, data: Any, status: int = 200) -> None: ...


# ── Embedded single-page HTML (Apple-minimalist Preact UI, matches Trace Explorer) ──
# ruff: noqa: E501
EVAL_EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Dashboard</title>
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
.empty{padding:48px;text-align:center;color:var(--muted)}
.muted{color:var(--muted)}

.topbar{display:flex;align-items:center;gap:16px;height:52px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line)}
.brand{font-weight:600;font-size:14px}
.nav{display:flex;gap:4px}
.nav button{font-family:var(--font);font-size:12px;font-weight:500;color:var(--muted);background:transparent;border:none;border-radius:8px;padding:6px 12px;cursor:pointer}
.nav button:hover{color:var(--text);background:var(--bg2)}
.nav button.on{color:var(--accent);background:var(--accent-soft)}

.main{flex:1;padding:24px;max-width:1100px;width:100%;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.card h2{font-size:14px;margin-bottom:12px}
.metrics{display:flex;gap:24px;flex-wrap:wrap}
.metric{display:flex;flex-direction:column;gap:2px}
.metric .v{font-size:24px;font-weight:600}
.metric .l{font-size:11px;color:var(--muted)}

table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:var(--muted);font-weight:500;padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid var(--line)}
tr.clickable{cursor:pointer}
tr.clickable:hover{background:var(--bg2)}
.pill{display:inline-block;font-size:10px;font-weight:600;border-radius:5px;padding:2px 8px}
.pill.pass{color:var(--pos);background:var(--pos-bg)}
.pill.fail{color:var(--neg);background:var(--neg-bg)}
.mono{font-family:var(--mono);font-size:11px}

.trend{display:flex;align-items:flex-end;gap:4px;height:80px}
.trend .bar{flex:1;background:var(--accent-soft);border-radius:3px 3px 0 0;min-height:2px;position:relative}
.trend .bar.fail{background:var(--neg-bg)}
.trend .bar span{position:absolute;bottom:100%;left:50%;transform:translateX(-50%);font-size:9px;color:var(--muted);white-space:nowrap}

.case-row{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:8px;cursor:pointer}
.case-row:hover{border-color:var(--accent)}
.case-row .head{display:flex;justify-content:space-between;align-items:center}
.detail-block{margin-top:10px;padding-top:10px;border-top:1px solid var(--line);font-size:12px}
.detail-block .row{display:flex;gap:8px;margin-bottom:6px}
.detail-block .k{color:var(--muted);min-width:80px;flex:none}
pre{white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:11px;background:var(--bg2);border-radius:6px;padding:8px}

.back{font-size:12px;color:var(--accent);cursor:pointer;margin-bottom:12px;display:inline-block}
.stub{padding:32px;text-align:center;color:var(--muted)}
</style>
</head>
<body>
<div id="app"></div>
<script>/*__VENDOR__*/</script>
<script>
const {h, render} = window.preact;
const {useState, useEffect} = window.preactHooks;
const html = window.htm.bind(h);

function getJSON(url){ return fetch(url).then(r => r.json()); }

function Nav({view, setView}){
  const tabs = [
    ["overview","Overview"],
    ["runs","Run History"],
    ["datasets","Datasets"],
    ["compare","Compare"],
  ];
  return html`
    <div class="topbar">
      <div class="brand">Eval Dashboard</div>
      <div class="nav">
        ${tabs.map(([id,label]) => html`
          <button class=${view===id ? "on" : ""} onClick=${() => setView(id)}>${label}</button>
        `)}
      </div>
    </div>
  `;
}

function pct(v){ return v == null ? "—" : Math.round(v*100) + "%"; }

function Trend({runs}){
  if(!runs.length) return html`<div class="muted">No runs yet.</div>`;
  const ordered = [...runs].reverse();
  return html`
    <div class="trend">
      ${ordered.map(r => html`
        <div class="bar ${r.verdict}" style=${{height: Math.max(4, (r.headline_score||0)*76)+"px"}}>
          <span>${pct(r.headline_score)}</span>
        </div>
      `)}
    </div>
  `;
}

function Overview({runs}){
  if(!runs.length){
    return html`<div class="card"><div class="empty">No eval runs recorded yet.</div></div>`;
  }
  const latest = runs[0];
  return html`
    <div class="card">
      <h2>Latest run</h2>
      <div class="metrics">
        <div class="metric"><div class="v">${pct(latest.headline_score)}</div><div class="l">pass rate</div></div>
        <div class="metric"><div class="v">${latest.case_count}</div><div class="l">cases</div></div>
        <div class="metric"><div class="v">${latest.verdict}</div><div class="l">verdict</div></div>
        <div class="metric"><div class="v mono">${latest.model}</div><div class="l">model</div></div>
      </div>
    </div>
    <div class="card">
      <h2>Trend across runs</h2>
      <${Trend} runs=${runs} />
    </div>
  `;
}

function RunHistory({runs, openRun}){
  if(!runs.length) return html`<div class="card"><div class="empty">No eval runs recorded yet.</div></div>`;
  return html`
    <div class="card">
      <h2>Run history</h2>
      <table>
        <thead><tr><th>Run</th><th>Dataset</th><th>Model</th><th>Verdict</th><th>Score</th></tr></thead>
        <tbody>
          ${runs.map(r => html`
            <tr class="clickable" onClick=${() => openRun(r.run_id)}>
              <td class="mono">${r.run_id}</td>
              <td>${r.dataset}</td>
              <td class="mono">${r.model}</td>
              <td><span class="pill ${r.verdict}">${r.verdict}</span></td>
              <td>${pct(r.headline_score)}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function CaseDetail({c}){
  return html`
    <div class="detail-block">
      ${c.task != null && html`<div class="row"><div class="k">Task</div><div>${c.task}</div></div>`}
      ${c.expected != null && html`<div class="row"><div class="k">Expected</div><pre>${JSON.stringify(c.expected, null, 2)}</pre></div>`}
      <div class="row"><div class="k">Metrics</div><pre>${JSON.stringify(c.metrics, null, 2)}</pre></div>
      <div class="row"><div class="k">Detail</div><pre>${JSON.stringify(c.detail, null, 2)}</pre></div>
    </div>
  `;
}

function RunBreakdown({runId, back}){
  const [view, setViewData] = useState(null);
  const [openCase, setOpenCase] = useState(null);
  useEffect(() => { getJSON("/api/evals/runs/" + runId).then(setViewData); }, [runId]);
  if(!view) return html`<div class="card"><div class="empty">Loading…</div></div>`;
  if(view.error) return html`<div class="card"><div class="empty">Run not found.</div></div>`;
  const s = view.summary;
  return html`
    <span class="back" onClick=${back}>← Run history</span>
    <div class="card">
      <h2>Run ${s.run_id}</h2>
      <div class="metrics">
        <div class="metric"><div class="v">${pct(s.headline_score)}</div><div class="l">pass rate</div></div>
        <div class="metric"><div class="v">${s.case_count}</div><div class="l">cases</div></div>
        <div class="metric"><div class="v">${s.verdict}</div><div class="l">verdict</div></div>
        <div class="metric"><div class="v mono">${s.dataset}</div><div class="l">dataset</div></div>
      </div>
    </div>
    <div class="card">
      <h2>Cases</h2>
      ${view.cases.map(c => html`
        <div class="case-row" onClick=${() => setOpenCase(openCase === c.case_id ? null : c.case_id)}>
          <div class="head">
            <span class="mono">${c.case_id}</span>
            <span class="pill ${c.passed ? "pass" : "fail"}">${c.passed ? "pass" : "fail"}</span>
          </div>
          ${openCase === c.case_id && html`<${CaseDetail} c=${c} />`}
        </div>
      `)}
    </div>
  `;
}

function Datasets(){
  const [datasets, setDatasets] = useState(null);
  useEffect(() => { getJSON("/api/evals/datasets").then(setDatasets); }, []);
  if(datasets === null) return html`<div class="card"><div class="empty">Loading…</div></div>`;
  if(!datasets.length) return html`<div class="card"><div class="empty">No datasets recorded yet.</div></div>`;
  return html`
    <div class="card">
      <h2>Datasets</h2>
      <table>
        <thead><tr><th>Dataset</th><th>Version</th><th>Cases</th></tr></thead>
        <tbody>
          ${datasets.map(d => html`
            <tr>
              <td class="mono">${d.id}</td>
              <td>v${d.version}</td>
              <td>${d.case_ids.length}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

function Compare(){
  return html`
    <div class="card">
      <div class="stub">
        Two-run comparison is not yet available — it renders once the
        eval-run-comparison module lands.
      </div>
    </div>
  `;
}

function App(){
  const [view, setView] = useState("overview");
  const [runs, setRuns] = useState([]);
  const [openRunId, setOpenRunId] = useState(null);
  useEffect(() => { getJSON("/api/evals/runs").then(setRuns); }, []);

  const openRun = (id) => { setOpenRunId(id); setView("run"); };
  const backToHistory = () => { setOpenRunId(null); setView("runs"); };

  return html`
    <${Nav} view=${view === "run" ? "runs" : view} setView=${setView} />
    <div class="main">
      ${view === "overview" && html`<${Overview} runs=${runs} />`}
      ${view === "runs" && html`<${RunHistory} runs=${runs} openRun=${openRun} />`}
      ${view === "run" && html`<${RunBreakdown} runId=${openRunId} back=${backToHistory} />`}
      ${view === "datasets" && html`<${Datasets} />`}
      ${view === "compare" && html`<${Compare} />`}
    </div>
  `;
}

render(html`<${App} />`, document.getElementById("app"));
</script>
</body>
</html>
"""

_EVAL_VENDOR_TOKEN = "/*__VENDOR__*/"
_EVAL_VENDOR_FILES = ("preact.min.js", "hooks.umd.js", "htm.umd.js")


@lru_cache(maxsize=1)
def _eval_vendor_js(vendor_dir: Path) -> str:
    """Concatenate the Preact/hooks/htm bundles the eval dashboard needs.

    A smaller subset than the Trace Explorer's (no CodeMirror/markdown-it —
    the eval views render plain JSON/text, not code or LLM markdown).
    """
    return "\n".join(
        (vendor_dir / name).read_text(encoding="utf-8") for name in _EVAL_VENDOR_FILES
    )


@lru_cache(maxsize=1)
def _full_eval_html(vendor_dir: Path) -> str:
    """Return the eval dashboard page with the vendored libraries inlined."""
    return EVAL_EMBEDDED_HTML.replace(_EVAL_VENDOR_TOKEN, _eval_vendor_js(vendor_dir))


def eval_dashboard_html(path: str) -> str | None:
    """Return the eval dashboard's HTML for ``/evals``, else ``None``.

    Args:
        path: The request path.

    Returns:
        The HTML document to serve, or `None` if `path` isn't `/evals`.
    """
    if path != "/evals":
        return None
    vendor_dir = Path(__file__).parent / "_vendor"
    return _full_eval_html(vendor_dir)


def handle_eval_api_route(handler: _JSONSender, path: str, evals_root: Path) -> bool:
    """Dispatch a `/api/evals/...` GET request; return True if handled.

    Routes:
        ``/api/evals/runs``              → run history (newest first)
        ``/api/evals/runs/{run_id}``     → one run's full breakdown
        ``/api/evals/datasets``          → available datasets + versions
        ``/api/evals/compare``           → comparison stub (pending #142)

    Args:
        handler: The request handler; used to send the JSON response.
        path: The request path.
        evals_root: Directory holding eval results/datasets/cases
            (``<base_dir>/evals``).

    Returns:
        True if `path` matched an eval API route (response already sent).
    """
    if path == "/api/evals/runs":
        summaries = evals_reader.list_runs(root=evals_root)
        handler._send_json([dataclasses.asdict(s) for s in summaries])
        return True

    if path == "/api/evals/datasets":
        datasets = evals_reader.list_available_datasets(
            base_dir=evals_root / "datasets"
        )
        handler._send_json([dataclasses.asdict(d) for d in datasets])
        return True

    if path == "/api/evals/compare":
        handler._send_json(
            {
                "available": False,
                "message": (
                    "Two-run comparison is not yet available — pending the "
                    "eval-run-comparison module."
                ),
            }
        )
        return True

    match = re.fullmatch(r"/api/evals/runs/([^/]+)", path)
    if match:
        run_id = match.group(1)
        if not _RUN_ID_RE.match(run_id):
            handler._send_json({"error": "invalid run id"}, status=400)
            return True
        view = evals_reader.load_run(
            run_id, root=evals_root, cases_dir=evals_root / "cases"
        )
        if view is None:
            handler._send_json({"error": "run not found"}, status=404)
        else:
            handler._send_json(dataclasses.asdict(view))
        return True

    return False
