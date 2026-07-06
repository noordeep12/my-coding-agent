Vendored third-party JS for the Trace Explorer UI (offline, no CDN).

- preact.min.js       — Preact 10.24.3 (UMD, exposes window.preact)
- hooks.umd.js        — Preact Hooks 10.24.3 (UMD, exposes window.preactHooks)
- htm.umd.js          — htm 3.1.1 (UMD, exposes window.htm)
- codemirror.bundle.js — CodeMirror 6 (prebuilt IIFE, exposes window.CM6):
  a curated read-only editor surface (state/view, lang-json, lang-python, shell
  via legacy-modes, search, folding, syntax highlighting) used by the
  detail-panel content boxes. No runtime build.
- markdown-it.bundle.js — markdown-it 14.3.0 (prebuilt IIFE, exposes
  window.markdownit): CommonMark + tables parser used to render free-text
  content boxes as formatted markdown.
- dompurify.bundle.js — DOMPurify 3.4.11 (prebuilt IIFE, exposes
  window.DOMPurify): sanitizes markdown-it's HTML output before it reaches
  the DOM, since session content is untrusted LLM output.

Rebuild codemirror.bundle.js (only when its deps change): create an entry.js
that imports the needed @codemirror packages and assigns them to `window.CM6`
(state, view, language, search, lang-json, lang-python, and shell wrapped from
`@codemirror/legacy-modes/mode/shell` via `StreamLanguage` — there is no
`@codemirror/lang-shell`), then:

    npm install codemirror @codemirror/state @codemirror/view \
      @codemirror/language @codemirror/search @codemirror/lang-json \
      @codemirror/lang-python @codemirror/legacy-modes esbuild
    npx esbuild entry.js --bundle --format=iife --minify \
      --legal-comments=none --outfile=codemirror.bundle.js

Rebuild markdown-it.bundle.js / dompurify.bundle.js (only when their deps
change): create an entry.js for each that imports the package and assigns it
to the exposed global (`window.markdownit = require('markdown-it')`;
`window.DOMPurify = require('dompurify')`), then:

    npm install markdown-it dompurify esbuild
    npx esbuild entry-markdownit.js --bundle --format=iife --minify \
      --legal-comments=none --outfile=markdown-it.bundle.js
    npx esbuild entry-dompurify.js --bundle --format=iife --minify \
      --legal-comments=none --outfile=dompurify.bundle.js

Do not edit by hand. Excluded from coverage per CONTRIBUTE.md.
