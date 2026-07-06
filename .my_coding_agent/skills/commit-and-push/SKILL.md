---
name: commit-and-push
description: Stage, write a Conventional Commit that passes this repo's commit-msg hooks, and push the current branch.
---

This repo enforces commit shape with `commit-msg` pre-commit hooks (see
`.pre-commit-config.yaml` and `.gitmessage`). A commit missing any required
element is rejected locally before it lands. Drive it like this:

1. **Review what you are committing.** Run `git status` and `git diff --staged`.
   Stage only files that trace to the task — never `.env`, keys, `*.jsonl`, or
   anything under `projects/`/`sessions/`/`cache/`.

2. **Write the subject** as `type(scope): description`:
   - `type` ∈ `feat|fix|refactor|docs|test|chore|perf|ci` (imperative, present tense).
   - Subject ≤ 72 chars (`commit-subject-length` hook).
   - Format checked by `commit-subject-format`.

3. **Write a body** explaining *why*, not what — a non-empty paragraph
   (`commit-body-required`). Wrap at ~72 chars.

4. **Add a `Refs:` footer** referencing an existing GitHub issue:
   `Refs: #<issue-number>` (`commit-refs-footer`). If no issue exists for the
   change, create one first with `gh-axi issue create` before committing.

5. **Commit and push.** `git commit`, then `git push -u origin <branch>`.
   Never push to `main`/`master`; never force-push a shared branch without
   explicit confirmation.

If a `commit-msg` hook rejects the message, read its output — it names the exact
rule that failed — fix the subject/body/footer, and retry. Do not use
`--no-verify` to bypass the hooks.
