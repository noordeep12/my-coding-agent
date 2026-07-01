Vendored third-party JS for the Trace Explorer UI (offline, no CDN).

- preact.min.js       — Preact 10.24.3 (UMD, exposes window.preact)
- hooks.umd.js        — Preact Hooks 10.24.3 (UMD, exposes window.preactHooks)
- htm.umd.js          — htm 3.1.1 (UMD, exposes window.htm)
- codemirror.bundle.js — CodeMirror 6.0.2 (prebuilt IIFE, exposes window.CM6):
  a curated read-only editor surface (state/view, lang-json, search, folding,
  syntax highlighting) used by the detail-panel content boxes. No runtime build.

Rebuild codemirror.bundle.js (only when its deps change): create an entry.js
that imports the needed @codemirror packages and assigns them to `window.CM6`
(state, view, language, search, lang-json), then:

    npm install codemirror @codemirror/state @codemirror/view \
      @codemirror/language @codemirror/search @codemirror/lang-json esbuild
    npx esbuild entry.js --bundle --format=iife --minify \
      --legal-comments=none --outfile=codemirror.bundle.js

Do not edit by hand. Excluded from coverage per CONTRIBUTE.md.
