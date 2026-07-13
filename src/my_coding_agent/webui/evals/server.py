"""Eval Dashboard routes and embedded UI, served from the web UI shell.

Serves the single two-pane Evaluation management page against the
Evaluation/RunConfig/EvalConfig CRUD + run API (`webui/evals/api.py`).
Reuses the same offline-vendored Preact + htm bundle as the Trace Explorer
(no CDN, no build step) — see this module's `_eval_vendor_js()`.
"""

from __future__ import annotations

import dataclasses
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from . import reader as evals_reader

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
<title>Evaluations</title>
<style>
:root{
  --bg:#ffffff; --bg2:#f5f5f7; --panel:#fbfbfd; --line:#e5e5ea;
  --text:#1d1d1f; --muted:#86868b; --accent:#0071e3; --accent-soft:#e8f1fd;
  --pos:#1a7f37; --pos-bg:#e7f6ec; --neg:#d70015; --neg-bg:#fdeaec;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Monaco,monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#1c1c1e; --bg2:#000000; --panel:#232326; --line:#3a3a3c;
    --text:#f5f5f7; --muted:#98989d; --accent:#0a84ff; --accent-soft:#0a3d66;
    --pos:#32d74b; --pos-bg:#0e2a13; --neg:#ff453a; --neg-bg:#330a08;
  }
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:var(--font);background:var(--bg2);color:var(--text);font-size:13px;-webkit-font-smoothing:antialiased}
#app{height:100vh;display:flex;flex-direction:column}
.empty{padding:48px;text-align:center;color:var(--muted)}
.muted{color:var(--muted)}

.topbar{display:flex;align-items:center;gap:16px;height:52px;padding:0 20px;background:var(--bg);border-bottom:1px solid var(--line);flex:none}
.brand{font-weight:600;font-size:14px}

.layout{flex:1;display:flex;min-height:0}
.pane-left{flex:1;min-width:0;overflow:auto;padding:20px}
.pane-right{width:520px;flex:none;overflow:auto;padding:20px;border-left:1px solid var(--line);background:var(--panel)}

.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.card h2{font-size:14px;margin-bottom:12px}
.section{margin-bottom:18px}
.section h3{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.03em;margin-bottom:8px}

table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;color:var(--muted);font-weight:500;padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr.clickable{cursor:pointer}
tr.clickable:hover{background:var(--bg2)}
.pill{display:inline-block;font-size:10px;font-weight:600;border-radius:5px;padding:2px 8px}
.pill.pass{color:var(--pos);background:var(--pos-bg)}
.pill.fail{color:var(--neg);background:var(--neg-bg)}
.pill.no_checks,.pill.pending{color:var(--muted);background:var(--bg2)}
.mono{font-family:var(--mono);font-size:11px}
.link{color:var(--accent);cursor:pointer}
.link:hover{text-decoration:underline}
.actions{display:flex;gap:6px;flex-wrap:wrap}
.menu-wrap{position:relative;display:inline-block}
.menu-dropdown{position:absolute;right:0;top:calc(100% + 4px);background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.15);min-width:120px;z-index:10;overflow:hidden}
.menu-item{padding:8px 12px;font-size:12px;cursor:pointer;white-space:nowrap}
.menu-item:hover{background:var(--bg2)}
.menu-item.danger{color:var(--neg)}

pre{white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:11px;background:var(--bg2);border-radius:6px;padding:8px}

.form-row{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.form-row label{font-size:11px;color:var(--muted);min-width:110px}
.form-row.col{flex-direction:column;align-items:stretch}
.form-row.col label{margin-bottom:4px}
input[type=text],input[type=number],textarea,select{font-family:var(--font);font-size:12px;border:1px solid var(--line);border-radius:6px;padding:6px 8px;background:var(--bg);color:var(--text)}
input[type=text],input[type=number],select{flex:1;min-width:0}
textarea{font-family:var(--mono);width:100%;min-height:60px;flex:1}
label.check{display:flex;align-items:center;gap:6px;min-width:0}
button{font-family:var(--font);cursor:pointer}
button.primary{font-size:12px;font-weight:500;color:#fff;background:var(--accent);border:none;border-radius:6px;padding:7px 14px}
button.primary:hover{opacity:0.9}
button.primary:disabled{opacity:0.5;cursor:not-allowed}
button.secondary{font-size:12px;font-weight:500;color:var(--text);background:var(--bg2);border:1px solid var(--line);border-radius:6px;padding:7px 14px}
button.danger{font-size:11px;color:var(--neg);background:transparent;border:1px solid var(--neg);border-radius:6px;padding:4px 10px}
button.link-btn{font-size:11px;color:var(--accent);background:transparent;border:none;padding:2px 4px}
.error-msg{color:var(--neg);font-size:12px;margin:6px 0}
.ok-msg{color:var(--pos);font-size:12px;margin:6px 0}

.rule-block{border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:10px}
.rule-block .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.check-block{border:1px solid var(--line);border-radius:6px;padding:10px;margin:8px 0 8px 12px;background:var(--bg2)}
.check-block .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}

.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;z-index:100}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:24px;max-width:480px;width:90%;max-height:80vh;overflow:auto}
.modal h2{font-size:14px;margin-bottom:14px}
.modal .actions{justify-content:flex-end;margin-top:16px}
.stage-list{display:flex;flex-direction:column;gap:10px;margin:16px 0}
.stage{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted)}
.stage.active{color:var(--text);font-weight:500}
.stage.done{color:var(--pos)}
.stage .dot{width:8px;height:8px;border-radius:50%;background:var(--line);flex:none}
.stage.active .dot{background:var(--accent)}
.stage.done .dot{background:var(--pos)}
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
function postJSON(url, body){
  return fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})})
    .then(async r => { const data = await r.json(); if(!r.ok) throw new Error(data.error||"request failed"); return data; });
}
function putJSON(url, body){
  return fetch(url, {method:"PUT", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body||{})})
    .then(async r => { const data = await r.json(); if(!r.ok) throw new Error(data.error||"request failed"); return data; });
}
function del(url){
  return fetch(url, {method:"DELETE"})
    .then(async r => { if(r.status === 204) return null; const data = await r.json(); if(!r.ok) throw new Error(data.error||"request failed"); return data; });
}

function pct(v){ return v == null ? "—" : Math.round(v*100) + "%"; }

function newCheck(){
  return {name:"", description:"", method:"", input:"", expected:"", evaluator:"exact_match", threshold:1};
}
function newRule(){
  return {name:"", description:"", checks:[newCheck()]};
}
function emptyRunConfigDraft(){
  return {
    name:"", description:"", agent:"", model:"", provider:"",
    system_prompt:"", user_prompt_template:"", context_template:"",
    temperature:"", max_tokens:"", top_p:"", extra_params:"{}",
    tools_enabled:false, tool_config:"{}", memory_config:"{}",
    retrieval_config:"{}", env_vars:"{}",
  };
}
function emptyEvalConfigDraft(){
  return {name:"", rules:[newRule()]};
}

// ── Small building blocks ───────────────────────────────────────────────

function EmptyState({onCreate}){
  return html`
    <div class="card">
      <div class="empty">
        <div style=${{marginBottom:"12px"}}>No evaluations created</div>
        <button class="primary" onClick=${onCreate}>Create Eval</button>
      </div>
    </div>
  `;
}

function ThresholdInput({value, onChange, readOnly}){
  return html`
    <input type="number" step="0.01" min="0" max="1" value=${value} disabled=${readOnly}
      onInput=${e => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      style=${{maxWidth:"90px"}} />
  `;
}

function JsonEditor({label, value, onChange, readOnly}){
  const [err, setErr] = useState(null);
  return html`
    <div class="form-row col">
      <label>${label}</label>
      <textarea class="mono" disabled=${readOnly} value=${value} onInput=${e => {
        onChange(e.target.value);
        try{ JSON.parse(e.target.value || "{}"); setErr(null); }
        catch(ex){ setErr("invalid JSON"); }
      }}></textarea>
      ${err && html`<div class="error-msg">${err}</div>`}
    </div>
  `;
}

function ConfirmationModal({title, message, onConfirm, onCancel}){
  return html`
    <div class="modal-overlay">
      <div class="modal">
        <h2>${title}</h2>
        <div>${message}</div>
        <div class="actions">
          <button class="secondary" onClick=${onCancel}>Cancel</button>
          <button class="danger" onClick=${onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  `;
}

function ActionsMenu({items}){
  const [open, setOpen] = useState(false);
  return html`
    <div class="menu-wrap">
      <button class="secondary" onClick=${() => setOpen(o => !o)}>⋯</button>
      ${open && html`
        <div class="menu-dropdown" onMouseLeave=${() => setOpen(false)}>
          ${items.map(it => html`
            <div class="menu-item ${it.danger ? "danger" : ""}"
              onClick=${() => { setOpen(false); it.onClick(); }}>${it.label}</div>
          `)}
        </div>
      `}
    </div>
  `;
}

const RUN_STAGES = ["Preparing run", "Executing pipeline", "Running checks", "Calculating results"];

function ExecutionProgressModal({stageIndex, error, result, onClose}){
  return html`
    <div class="modal-overlay">
      <div class="modal">
        <h2>Running evaluation</h2>
        <div class="stage-list">
          ${RUN_STAGES.map((label, i) => html`
            <div class="stage ${i < stageIndex ? "done" : i === stageIndex ? "active" : ""}">
              <span class="dot"></span><span>${label}</span>
            </div>
          `)}
        </div>
        ${error && html`<div class="error-msg">${error}</div>`}
        ${result && html`<div class="ok-msg">Run complete: ${result.verdict || "done"}</div>`}
        ${(error || result) && html`
          <div class="actions"><button class="primary" onClick=${onClose}>Close</button></div>
        `}
      </div>
    </div>
  `;
}

// ── Run Configuration ────────────────────────────────────────────────────

function RunConfigSelector({runConfigs, mode, setMode, selectedId, setSelectedId, readOnly}){
  return html`
    <div class="section">
      <h3>Run Configuration</h3>
      <div class="form-row">
        <label>Source</label>
        <select value=${mode} disabled=${readOnly} onChange=${e => setMode(e.target.value)}>
          <option value="select">Select existing</option>
          <option value="new">Create New</option>
        </select>
      </div>
      ${mode === "select" && html`
        <div class="form-row">
          <label>Run Config</label>
          <select value=${selectedId || ""} disabled=${readOnly} onChange=${e => setSelectedId(e.target.value)}>
            <option value="">Select…</option>
            ${runConfigs.map(rc => html`<option value=${rc.id}>${rc.name}</option>`)}
          </select>
        </div>
      `}
    </div>
  `;
}

function RunConfigEditor({draft, setDraft, readOnly}){
  const setField = (k, v) => setDraft(prev => ({...prev, [k]: v}));
  return html`
    <div class="section">
      <div class="form-row"><label>Name</label><input type="text" disabled=${readOnly} value=${draft.name} onInput=${e => setField("name", e.target.value)} /></div>
      <div class="form-row"><label>Description</label><input type="text" disabled=${readOnly} value=${draft.description} onInput=${e => setField("description", e.target.value)} /></div>
      <div class="form-row"><label>Agent</label><input type="text" disabled=${readOnly} value=${draft.agent} onInput=${e => setField("agent", e.target.value)} /></div>
      <div class="form-row"><label>Model</label><input type="text" disabled=${readOnly} value=${draft.model} onInput=${e => setField("model", e.target.value)} /></div>
      <div class="form-row"><label>Provider</label><input type="text" disabled=${readOnly} value=${draft.provider} onInput=${e => setField("provider", e.target.value)} /></div>
      <div class="form-row col"><label>System prompt</label><textarea disabled=${readOnly} value=${draft.system_prompt} onInput=${e => setField("system_prompt", e.target.value)}></textarea></div>
      <div class="form-row col"><label>User prompt template</label><textarea disabled=${readOnly} value=${draft.user_prompt_template} onInput=${e => setField("user_prompt_template", e.target.value)}></textarea></div>
      <div class="form-row col"><label>Context template</label><textarea disabled=${readOnly} value=${draft.context_template} onInput=${e => setField("context_template", e.target.value)}></textarea></div>
      <div class="form-row"><label>Temperature</label><input type="number" step="0.1" disabled=${readOnly} value=${draft.temperature} onInput=${e => setField("temperature", e.target.value)} /></div>
      <div class="form-row"><label>Max tokens</label><input type="number" disabled=${readOnly} value=${draft.max_tokens} onInput=${e => setField("max_tokens", e.target.value)} /></div>
      <div class="form-row"><label>Top p</label><input type="number" step="0.1" disabled=${readOnly} value=${draft.top_p} onInput=${e => setField("top_p", e.target.value)} /></div>
      <${JsonEditor} label="Extra params" value=${draft.extra_params} onChange=${v => setField("extra_params", v)} readOnly=${readOnly} />
      <div class="form-row"><label class="check"><input type="checkbox" disabled=${readOnly} checked=${draft.tools_enabled} onChange=${e => setField("tools_enabled", e.target.checked)} /> Tools enabled</label></div>
      <${JsonEditor} label="Tool config" value=${draft.tool_config} onChange=${v => setField("tool_config", v)} readOnly=${readOnly} />
      <${JsonEditor} label="Memory config" value=${draft.memory_config} onChange=${v => setField("memory_config", v)} readOnly=${readOnly} />
      <${JsonEditor} label="Retrieval config" value=${draft.retrieval_config} onChange=${v => setField("retrieval_config", v)} readOnly=${readOnly} />
      <${JsonEditor} label="Env vars" value=${draft.env_vars} onChange=${v => setField("env_vars", v)} readOnly=${readOnly} />
    </div>
  `;
}

// ── Eval Configuration ────────────────────────────────────────────────────

function EvalConfigSelector({evalConfigs, mode, setMode, selectedId, setSelectedId, readOnly}){
  return html`
    <div class="section">
      <h3>Eval Configuration</h3>
      <div class="form-row">
        <label>Source</label>
        <select value=${mode} disabled=${readOnly} onChange=${e => setMode(e.target.value)}>
          <option value="select">Select existing</option>
          <option value="new">Create New</option>
        </select>
      </div>
      ${mode === "select" && html`
        <div class="form-row">
          <label>Eval Config</label>
          <select value=${selectedId || ""} disabled=${readOnly} onChange=${e => setSelectedId(e.target.value)}>
            <option value="">Select…</option>
            ${evalConfigs.map(ec => html`<option value=${ec.id}>${ec.name}</option>`)}
          </select>
        </div>
      `}
    </div>
  `;
}

function EvalCheckEditor({check, onChange, onRemove, readOnly}){
  const setField = (k, v) => onChange({...check, [k]: v});
  return html`
    <div class="check-block">
      <div class="head">
        <span class="mono">Check</span>
        ${!readOnly && html`<button class="danger" onClick=${onRemove}>Remove</button>`}
      </div>
      <div class="form-row"><label>Name</label><input type="text" disabled=${readOnly} value=${check.name} onInput=${e => setField("name", e.target.value)} /></div>
      <div class="form-row"><label>Description</label><input type="text" disabled=${readOnly} value=${check.description} onInput=${e => setField("description", e.target.value)} /></div>
      <div class="form-row"><label>Method</label><input type="text" disabled=${readOnly} value=${check.method} onInput=${e => setField("method", e.target.value)} /></div>
      <div class="form-row"><label>Input</label><input type="text" disabled=${readOnly} value=${check.input} onInput=${e => setField("input", e.target.value)} /></div>
      <div class="form-row"><label>Expected</label><input type="text" disabled=${readOnly} value=${check.expected} onInput=${e => setField("expected", e.target.value)} /></div>
      <div class="form-row">
        <label>Evaluator</label>
        <select value=${check.evaluator} disabled=${readOnly} onChange=${e => setField("evaluator", e.target.value)}>
          <option value="exact_match">exact_match</option>
          <option value="trajectory">trajectory</option>
          <option value="judge">judge</option>
        </select>
      </div>
      <div class="form-row"><label>Threshold</label><${ThresholdInput} value=${check.threshold} onChange=${v => setField("threshold", v)} readOnly=${readOnly} /></div>
    </div>
  `;
}

function EvalRuleEditor({rules, setRules, readOnly}){
  const setRule = (i, rule) => setRules(prev => prev.map((r, idx) => idx === i ? rule : r));
  const removeRule = (i) => setRules(prev => prev.filter((_, idx) => idx !== i));
  const addRule = () => setRules(prev => [...prev, newRule()]);
  const setCheck = (ri, ci, check) => setRule(ri, {...rules[ri], checks: rules[ri].checks.map((c, idx) => idx === ci ? check : c)});
  const removeCheck = (ri, ci) => setRule(ri, {...rules[ri], checks: rules[ri].checks.filter((_, idx) => idx !== ci)});
  const addCheck = (ri) => setRule(ri, {...rules[ri], checks: [...rules[ri].checks, newCheck()]});

  return html`
    <div class="section">
      <h3>Rules</h3>
      ${rules.map((rule, ri) => html`
        <div class="rule-block">
          <div class="head">
            <span class="mono">Rule ${ri + 1}</span>
            ${!readOnly && html`<button class="danger" onClick=${() => removeRule(ri)}>Remove Rule</button>`}
          </div>
          <div class="form-row"><label>Name</label><input type="text" disabled=${readOnly} value=${rule.name} onInput=${e => setRule(ri, {...rule, name: e.target.value})} /></div>
          <div class="form-row"><label>Description</label><input type="text" disabled=${readOnly} value=${rule.description} onInput=${e => setRule(ri, {...rule, description: e.target.value})} /></div>
          ${rule.checks.map((check, ci) => html`
            <${EvalCheckEditor} check=${check} onChange=${c => setCheck(ri, ci, c)} onRemove=${() => removeCheck(ri, ci)} readOnly=${readOnly} />
          `)}
          ${!readOnly && html`<button class="secondary" onClick=${() => addCheck(ri)}>Add Check</button>`}
        </div>
      `)}
      ${!readOnly && html`<button class="secondary" onClick=${addRule}>Add Rule</button>`}
    </div>
  `;
}

// ── Evaluation list ──────────────────────────────────────────────────────

function EvaluationRow({evaluation, runConfigs, evalConfigs, onDetails, onConfigure, onRun, onDelete, onOpenLastRun}){
  const rc = runConfigs.find(r => r.id === evaluation.run_config_id);
  const ec = evalConfigs.find(e => e.id === evaluation.eval_config_id);
  const lastRun = evaluation.last_run;
  const items = [
    {label: "Details", onClick: () => onDetails(evaluation)},
    {label: "Configure", onClick: () => onConfigure(evaluation)},
    {label: "Run", onClick: () => onRun(evaluation)},
    {label: "Delete", danger: true, onClick: () => onDelete(evaluation)},
  ];
  return html`
    <tr>
      <td>${evaluation.name}</td>
      <td>${evaluation.summary || html`<span class="muted">—</span>`}</td>
      <td class="mono">${rc ? rc.name : evaluation.run_config_id}</td>
      <td class="mono">${ec ? ec.name : evaluation.eval_config_id}</td>
      <td>${lastRun ? html`<span class="link" onClick=${() => onOpenLastRun(evaluation)}>${lastRun.run_id}</span>` : html`<span class="muted">—</span>`}</td>
      <td>${lastRun ? html`<span class="pill ${lastRun.verdict}">${lastRun.verdict}</span>` : html`<span class="pill pending">no runs</span>`}</td>
      <td><${ActionsMenu} items=${items} /></td>
    </tr>
  `;
}

function EvaluationTable({evaluations, runConfigs, evalConfigs, onDetails, onConfigure, onRun, onDelete, onOpenLastRun}){
  return html`
    <table>
      <thead>
        <tr><th>Name</th><th>Summary</th><th>Run Config</th><th>Eval Config</th><th>Last Run</th><th>Status</th><th>Actions</th></tr>
      </thead>
      <tbody>
        ${evaluations.map(ev => html`
          <${EvaluationRow} evaluation=${ev} runConfigs=${runConfigs} evalConfigs=${evalConfigs}
            onDetails=${onDetails} onConfigure=${onConfigure} onRun=${onRun} onDelete=${onDelete} onOpenLastRun=${onOpenLastRun} />
        `)}
      </tbody>
    </table>
  `;
}

// ── Configuration panel ──────────────────────────────────────────────────

function parseJsonField(value, label){
  try{ return JSON.parse(value || "{}"); }
  catch(e){ throw new Error(`${label} must be valid JSON`); }
}

function buildRunConfigPayload(draft){
  return {
    name: draft.name,
    description: draft.description,
    agent: draft.agent,
    model: draft.model,
    provider: draft.provider,
    system_prompt: draft.system_prompt,
    user_prompt_template: draft.user_prompt_template,
    context_template: draft.context_template,
    temperature: draft.temperature === "" ? null : Number(draft.temperature),
    max_tokens: draft.max_tokens === "" ? null : Number(draft.max_tokens),
    top_p: draft.top_p === "" ? null : Number(draft.top_p),
    extra_params: parseJsonField(draft.extra_params, "Extra params"),
    tools_enabled: draft.tools_enabled,
    tool_config: parseJsonField(draft.tool_config, "Tool config"),
    memory_config: parseJsonField(draft.memory_config, "Memory config"),
    retrieval_config: parseJsonField(draft.retrieval_config, "Retrieval config"),
    env_vars: parseJsonField(draft.env_vars, "Env vars"),
  };
}

function buildEvalConfigPayload(draft){
  return {
    name: draft.name,
    rules: draft.rules.map(rule => ({
      name: rule.name,
      description: rule.description,
      checks: rule.checks.map(check => ({
        name: check.name,
        description: check.description,
        method: check.method,
        input: check.input,
        expected: check.expected,
        evaluator: check.evaluator,
        threshold: check.threshold === "" ? 1 : Number(check.threshold),
      })),
    })),
  };
}

function draftFromRunConfig(rc){
  return {
    name: rc.name || "", description: rc.description || "", agent: rc.agent || "",
    model: rc.model || "", provider: rc.provider || "",
    system_prompt: rc.system_prompt || "", user_prompt_template: rc.user_prompt_template || "",
    context_template: rc.context_template || "",
    temperature: rc.temperature ?? "", max_tokens: rc.max_tokens ?? "", top_p: rc.top_p ?? "",
    extra_params: JSON.stringify(rc.extra_params || {}, null, 2),
    tools_enabled: !!rc.tools_enabled,
    tool_config: JSON.stringify(rc.tool_config || {}, null, 2),
    memory_config: JSON.stringify(rc.memory_config || {}, null, 2),
    retrieval_config: JSON.stringify(rc.retrieval_config || {}, null, 2),
    env_vars: JSON.stringify(rc.env_vars || {}, null, 2),
  };
}

function draftFromEvalConfig(ec){
  return {
    name: ec.name || "",
    rules: (ec.rules && ec.rules.length ? ec.rules : [newRule()]).map(rule => ({
      name: rule.name || "", description: rule.description || "",
      checks: (rule.checks && rule.checks.length ? rule.checks : [newCheck()]).map(check => ({
        name: check.name || "", description: check.description || "", method: check.method || "",
        input: check.input ?? "", expected: check.expected ?? "",
        evaluator: check.evaluator || "exact_match", threshold: check.threshold ?? 1,
      })),
    })),
  };
}

function EvaluationConfigPanel({editing, runConfigs, evalConfigs, readOnly, onSaved, onCancel}){
  const isEdit = !!(editing && editing.id);
  const [name, setName] = useState(editing ? editing.name : "");
  const [summary, setSummary] = useState(editing ? (editing.summary || "") : "");
  const [runMode, setRunMode] = useState("select");
  const [runConfigId, setRunConfigId] = useState(editing ? editing.run_config_id : "");
  const [runConfigDraft, setRunConfigDraft] = useState(emptyRunConfigDraft());
  const [evalMode, setEvalMode] = useState("select");
  const [evalConfigId, setEvalConfigId] = useState(editing ? editing.eval_config_id : "");
  const [evalConfigDraft, setEvalConfigDraft] = useState(emptyEvalConfigDraft());
  const [msg, setMsg] = useState(null);
  const [saving, setSaving] = useState(false);

  // Selecting an existing Run/Eval Config loads its full record into the
  // same editor used for "Create New" — read-only for Details, editable for
  // Configure — instead of the id being the only visible trace of it.
  useEffect(() => {
    if(runMode !== "select") return;
    const found = runConfigs.find(rc => rc.id === runConfigId);
    if(found) setRunConfigDraft(draftFromRunConfig(found));
  }, [runMode, runConfigId, runConfigs]);
  useEffect(() => {
    if(evalMode !== "select") return;
    const found = evalConfigs.find(ec => ec.id === evalConfigId);
    if(found) setEvalConfigDraft(draftFromEvalConfig(found));
  }, [evalMode, evalConfigId, evalConfigs]);

  const save = async () => {
    setMsg(null);
    if(!name){ setMsg({err:"Name is required"}); return; }
    setSaving(true);
    try{
      let resolvedRunConfigId = runConfigId;
      if(runMode === "new"){
        const payload = buildRunConfigPayload(runConfigDraft);
        if(!payload.name){ throw new Error("Run Configuration name is required"); }
        const created = await postJSON("/api/evals/run-configs", payload);
        resolvedRunConfigId = created.id;
      } else {
        if(!runConfigId){ throw new Error("Select or create a Run Configuration"); }
        const payload = buildRunConfigPayload(runConfigDraft);
        if(!payload.name){ throw new Error("Run Configuration name is required"); }
        await putJSON(`/api/evals/run-configs/${runConfigId}`, payload);
      }
      if(!resolvedRunConfigId){ throw new Error("Select or create a Run Configuration"); }

      let resolvedEvalConfigId = evalConfigId;
      if(evalMode === "new"){
        const payload = buildEvalConfigPayload(evalConfigDraft);
        if(!payload.name){ throw new Error("Eval Configuration name is required"); }
        const created = await postJSON("/api/evals/eval-configs", payload);
        resolvedEvalConfigId = created.id;
      } else {
        if(!evalConfigId){ throw new Error("Select or create an Eval Configuration"); }
        const payload = buildEvalConfigPayload(evalConfigDraft);
        if(!payload.name){ throw new Error("Eval Configuration name is required"); }
        await putJSON(`/api/evals/eval-configs/${evalConfigId}`, payload);
      }
      if(!resolvedEvalConfigId){ throw new Error("Select or create an Eval Configuration"); }

      const body = {name, summary, run_config_id: resolvedRunConfigId, eval_config_id: resolvedEvalConfigId};
      if(isEdit){
        await putJSON(`/api/evals/evaluations/${editing.id}`, body);
      } else {
        await postJSON("/api/evals/evaluations", body);
      }
      setSaving(false);
      onSaved();
    } catch(e){
      setSaving(false);
      setMsg({err: String(e.message || e)});
    }
  };

  return html`
    <div class="card">
      <span class="link" onClick=${onCancel}>← Close</span>
      <h2>${readOnly ? "Evaluation Details" : isEdit ? "Configure Evaluation" : "Create Evaluation"}</h2>
      <div class="section">
        <h3>Basic info</h3>
        <div class="form-row"><label>Name</label><input type="text" disabled=${readOnly} value=${name} onInput=${e => setName(e.target.value)} /></div>
        <div class="form-row"><label>Summary</label><input type="text" disabled=${readOnly} value=${summary} onInput=${e => setSummary(e.target.value)} /></div>
      </div>

      <${RunConfigSelector} runConfigs=${runConfigs} mode=${runMode} setMode=${setRunMode}
        selectedId=${runConfigId} setSelectedId=${setRunConfigId} readOnly=${readOnly} />
      ${(runMode === "new" || runConfigId) && html`<${RunConfigEditor} draft=${runConfigDraft} setDraft=${setRunConfigDraft} readOnly=${readOnly} />`}

      <${EvalConfigSelector} evalConfigs=${evalConfigs} mode=${evalMode} setMode=${setEvalMode}
        selectedId=${evalConfigId} setSelectedId=${setEvalConfigId} readOnly=${readOnly} />
      ${(evalMode === "new" || evalConfigId) && html`
        <div class="form-row"><label>Eval Config name</label><input type="text" disabled=${readOnly} value=${evalConfigDraft.name}
          onInput=${e => setEvalConfigDraft(prev => ({...prev, name: e.target.value}))} /></div>
        <${EvalRuleEditor} rules=${evalConfigDraft.rules} readOnly=${readOnly}
          setRules=${fn => setEvalConfigDraft(prev => ({...prev, rules: typeof fn === "function" ? fn(prev.rules) : fn}))} />
      `}

      ${msg && msg.err && html`<div class="error-msg">${msg.err}</div>`}
      <div class="actions">
        <button class="secondary" onClick=${onCancel}>${readOnly ? "Close" : "Cancel"}</button>
        ${!readOnly && html`<button class="primary" disabled=${saving} onClick=${save}>Save Evaluation</button>`}
      </div>
    </div>
  `;
}

// ── Last-run drill-down ──────────────────────────────────────────────────

function LastRunView({evaluation, evalConfig, back}){
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    setResult(null); setErr(null);
    getJSON(`/api/evals/evaluations/${evaluation.id}/runs/${evaluation.last_run.run_id}`)
      .then(r => { if(r && r.error) setErr(r.error); else setResult(r); })
      .catch(e => setErr(String(e)));
  }, [evaluation.id, evaluation.last_run.run_id]);

  const checksById = {};
  (evalConfig ? evalConfig.rules : []).forEach(rule => {
    rule.checks.forEach(check => { checksById[check.id] = {rule, check}; });
  });

  return html`
    <div class="card">
      <span class="link" onClick=${back}>← Back</span>
      <h2>Run ${evaluation.last_run.run_id}</h2>
      ${err && html`<div class="error-msg">${err}</div>`}
      ${!result && !err && html`<div class="empty">Loading…</div>`}
      ${result && html`
        <div class="section">
          <div><span class="muted">Verdict:</span> <span class="pill ${evaluation.last_run.verdict}">${evaluation.last_run.verdict}</span></div>
          <div><span class="muted">Timestamp:</span> ${result.timestamp}</div>
        </div>
        <div class="section">
          <h3>Checks</h3>
          ${!result.scores.length && html`<div class="muted">No checks were scored for this run.</div>`}
          ${result.scores.map(score => {
            const ctx = checksById[score.case_id];
            return html`
              <div class="check-block">
                <div class="head">
                  <span class="mono">${ctx ? ctx.check.name : score.case_id}</span>
                  <span class="pill ${score.passed ? "pass" : "fail"}">${score.passed ? "pass" : "fail"}</span>
                </div>
                <pre>${JSON.stringify(score.detail, null, 2)}</pre>
              </div>
            `;
          })}
        </div>
      `}
    </div>
  `;
}

// ── App ───────────────────────────────────────────────────────────────────

function App(){
  const [evaluations, setEvaluations] = useState([]);
  const [runConfigs, setRunConfigs] = useState([]);
  const [evalConfigs, setEvalConfigs] = useState([]);
  const [panel, setPanel] = useState(null); // null | {} (new) | evaluation (edit/details)
  const [panelReadOnly, setPanelReadOnly] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [runModal, setRunModal] = useState(null); // {stageIndex, error, result}
  const [lastRunTarget, setLastRunTarget] = useState(null);
  const [listError, setListError] = useState(null);

  const reloadAll = () => {
    getJSON("/api/evals/evaluations").then(d => setEvaluations(d.evaluations || [])).catch(e => setListError(String(e)));
    getJSON("/api/evals/run-configs").then(d => setRunConfigs(d.run_configs || []));
    getJSON("/api/evals/eval-configs").then(d => setEvalConfigs(d.eval_configs || []));
  };
  useEffect(reloadAll, []);

  const onSaved = () => { setPanel(null); setPanelReadOnly(false); reloadAll(); };
  const openCreate = () => { setPanelReadOnly(false); setPanel({}); };
  const openConfigure = (evaluation) => { setPanelReadOnly(false); setPanel(evaluation); };
  const openDetails = (evaluation) => { setPanelReadOnly(true); setPanel(evaluation); };
  const closePanel = () => { setPanel(null); setPanelReadOnly(false); };

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  // The run happens on a server-side background thread (not this request) —
  // poll .../runs/{run_id} until it stops answering "running" (202), rather
  // than blocking on one long request. The UI (and this tab's own polling)
  // stays responsive to other actions the whole time.
  const runEvaluation = async (evaluation) => {
    setRunModal({stageIndex: 0, error: null, result: null});
    const advance = (i) => setRunModal(prev => prev && {...prev, stageIndex: i});
    const timers = [setTimeout(() => advance(1), 350)];
    try{
      const {run_id} = await postJSON(`/api/evals/evaluations/${evaluation.id}/run`);
      advance(2);
      let result = await getJSON(`/api/evals/evaluations/${evaluation.id}/runs/${run_id}`);
      while(result && result.status === "running"){
        await sleep(800);
        result = await getJSON(`/api/evals/evaluations/${evaluation.id}/runs/${run_id}`);
      }
      advance(3);
      setRunModal({stageIndex: 4, error: null, result: {run_id, verdict: (result.aggregate_metrics && result.aggregate_metrics.nothing_to_score) ? "no_checks" : (result.scores.every(s => s.passed) ? "pass" : "fail")}});
      reloadAll();
    } catch(e){
      timers.forEach(clearTimeout);
      setRunModal({stageIndex: 0, error: String(e.message || e), result: null});
    }
  };

  const confirmDelete = async () => {
    await del(`/api/evals/evaluations/${deleteTarget.id}`);
    setDeleteTarget(null);
    reloadAll();
  };

  const openLastRun = async (evaluation) => {
    const evalConfig = evalConfigs.find(ec => ec.id === evaluation.eval_config_id)
      || await getJSON(`/api/evals/eval-configs/${evaluation.eval_config_id}`);
    setLastRunTarget({evaluation, evalConfig});
  };

  return html`
    <div class="topbar"><div class="brand">Evaluations</div></div>
    <div class="layout">
      <div class="pane-left">
        ${listError && html`<div class="error-msg">${listError}</div>`}
        ${!evaluations.length
          ? html`<${EmptyState} onCreate=${openCreate} />`
          : html`
            <div class="card">
              <div class="form-row" style=${{justifyContent:"flex-end"}}>
                <button class="primary" onClick=${openCreate}>Create Eval</button>
              </div>
              <${EvaluationTable} evaluations=${evaluations} runConfigs=${runConfigs} evalConfigs=${evalConfigs}
                onDetails=${openDetails} onConfigure=${openConfigure} onRun=${runEvaluation} onDelete=${setDeleteTarget} onOpenLastRun=${openLastRun} />
            </div>
          `}
      </div>
      <div class="pane-right">
        ${lastRunTarget
          ? html`<${LastRunView} evaluation=${lastRunTarget.evaluation} evalConfig=${lastRunTarget.evalConfig} back=${() => setLastRunTarget(null)} />`
          : panel !== null
            ? html`<${EvaluationConfigPanel} editing=${panel} runConfigs=${runConfigs} evalConfigs=${evalConfigs} readOnly=${panelReadOnly}
                onSaved=${onSaved} onCancel=${closePanel} />`
            : html`<div class="card"><div class="empty">Use "⋯" → Details/Configure on a row, or Create Eval, to open this panel.</div></div>`}
      </div>
    </div>
    ${deleteTarget && html`
      <${ConfirmationModal} title="Delete evaluation"
        message="Delete this evaluation? This action cannot be undone."
        onConfirm=${confirmDelete} onCancel=${() => setDeleteTarget(null)} />
    `}
    ${runModal && html`
      <${ExecutionProgressModal} stageIndex=${runModal.stageIndex} error=${runModal.error} result=${runModal.result}
        onClose=${() => setRunModal(null)} />
    `}
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
    vendor_dir = Path(__file__).parent.parent.parent / "viewer" / "_vendor"
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
