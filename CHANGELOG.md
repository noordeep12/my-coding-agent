# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-07-13

### Added

- Node-based agentic pipeline: an explicit DAG (`ContextGuard → ToolRouting → LLMCall → ToolDispatch → AnomalyDetect → FinalizeStep`) with a typed `RunContext` data contract between nodes.
- Dangerous-command refusal gate: deterministic, local checks on every `bash` call (recursive deletes, remote-content-piped-to-shell, raw-device writes, fork bombs, permission blasts, credential exfiltration, destructive force-pushes), with a structured refusal and logged/recorded event on a match.
- Exfiltration guard blocking outbound secrets (`.env`, SSH keys, cloud credentials, `.netrc`, PEM/key material) before an outbound tool call sends its payload.
- Network egress filter checking `fetch_web` destinations against a maintained malicious-domain blocklist.
- OS-level Seatbelt sandbox (opt-in, macOS, `--sandbox`) confining `bash` subprocess filesystem writes and network egress at the OS level.
- Untrusted-content confinement: provenance marking and explicit demarcation of content pulled from outside the task, plus refusal of build/install scripts run against agent-cloned repos once untrusted content is in play.
- Runtime anomaly detection flagging same-class tool-failure streaks as they happen, in main agents and subagents.
- Skills: user-authored `SKILL.md` files loaded on demand via `use_skill`, including three bundled example skills.
- Run resilience and resume: transient LLM failures are classified and absorbed with a patient bounded retry; completed steps are checkpointed so a dead run can continue with `--resume`/`--resume-last`.
- Context handoff: structured progress summary and fresh continuation when the context window fills up.
- Eval harness: case runner, result store, versioned dataset model, deterministic trajectory scoring, rubric-based LLM judge scorer with calibration, run comparison and CI regression gate, declarative YAML run config, and terminal/Trace-Explorer verdict surfacing.
- Session observability: Trace Explorer web viewer with tool-def capture, CodeMirror content viewer, sanitized markdown rendering, added/retired token color-coding, and per-session persistence under `.my_coding_agent/<session_id>/`.
- Interactive-default CLI: paste-mode prompt by default, plus `--config` for running declarative YAML run configs end to end.
- Lifecycle hook seam for deterministic pre/post extension points.
- Overlap of independent read-only tool calls within a single turn.
- `expected_report` channel for shaped agent-to-caller hand-back.

### Changed

- Refactored the monolithic tool executor into distinct engine/pipeline/observability/utils modules, and the repository into a src-layout package.
- Retired provably-superseded tool results at step start and skip clean-finish re-summarization to reduce wasted context.
- Offer the full toolset every step; removed the separate tool-routing selection step.
- Unified web UI shell consolidated into the Trace Explorer viewer directly (webui shell and separate Evals/Admin tabs removed).

### Removed

- Standalone web UI pipeline-builder tab, separate eval-config HTTP surface, and unwired tool-routing selector/node.
- Unused `msgpack` dependency.

### Fixed

- Eval exception hierarchy aligned with `MyCodingAgentError`.
- Internal API surface contracts tightened; dead eval-config HTTP surface removed.
- Admin LLM connection settings wired into interface-launched eval runs.

## [0.1.1] - 2026-05-25

### Added

- Initial tagged release of the from-scratch Python agent framework: local LLM-driven agentic loop, decorator-based tools, and baseline CLI.

[Unreleased]: https://github.com/noordeep12/my-coding-agent/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/noordeep12/my-coding-agent/compare/v0.1.1...v2.0.0
[0.1.1]: https://github.com/noordeep12/my-coding-agent/releases/tag/v0.1.1
