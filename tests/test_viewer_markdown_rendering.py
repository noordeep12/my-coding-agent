"""Tests for the rendered-markdown view (issue #112).

The embedded UI is plain JS inside a Python string (no JS test runtime in
this repo's CI — see `test_viewer_server.py` for the established pattern),
so the render-pipeline behaviour is verified at the source level: the exact
`markdown-it` → `DOMPurify` → DOM ordering, the `MarkdownBox`/`ContentBox`
scope boundary, and the Rendered/Raw toggle. Vendor manifest and fail-fast
startup behaviour are verified as real Python execution.
"""

from __future__ import annotations

import pytest

from my_coding_agent.utils.exceptions import MyCodingAgentError
from my_coding_agent.viewer.server import (
    _VENDOR_DIR,
    _VENDOR_FILES,
    EMBEDDED_HTML,
    _check_vendor_assets,
)

# ── Vendoring (offline, fail-fast) ──────────────────────────────────────────


class TestVendorAssets:
    def test_manifest_includes_markdown_it_and_dompurify(self):
        assert "markdown-it.bundle.js" in _VENDOR_FILES
        assert "dompurify.bundle.js" in _VENDOR_FILES

    def test_manifest_files_exist_on_disk(self):
        # Real vendored files, not mocks — a missing bundle here is a build bug.
        _check_vendor_assets()
        for name in _VENDOR_FILES:
            assert (_VENDOR_DIR / name).is_file()

    def test_check_vendor_assets_fails_fast_when_markdown_bundle_missing(
        self, tmp_path, monkeypatch
    ):
        for name in _VENDOR_FILES:
            if name == "markdown-it.bundle.js":
                continue
            (tmp_path / name).write_text("// stub", encoding="utf-8")
        monkeypatch.setattr("my_coding_agent.viewer.server._VENDOR_DIR", tmp_path)
        with pytest.raises(MyCodingAgentError, match="markdown-it.bundle.js"):
            _check_vendor_assets()

    def test_check_vendor_assets_fails_fast_when_dompurify_bundle_missing(
        self, tmp_path, monkeypatch
    ):
        for name in _VENDOR_FILES:
            if name == "dompurify.bundle.js":
                continue
            (tmp_path / name).write_text("// stub", encoding="utf-8")
        monkeypatch.setattr("my_coding_agent.viewer.server._VENDOR_DIR", tmp_path)
        with pytest.raises(MyCodingAgentError, match="dompurify.bundle.js"):
            _check_vendor_assets()

    def test_no_network_calls_in_render_pipeline_source(self):
        # Offline guarantee: the render helper must only reference the two
        # vendored globals, never fetch()/XHR/a CDN URL.
        start = EMBEDDED_HTML.index("function renderMarkdownHTML")
        end = EMBEDDED_HTML.index("function highlightFences")
        src = EMBEDDED_HTML[start:end]
        assert "window.markdownit" not in src  # instantiated once, at module scope
        assert "window.DOMPurify.sanitize" in src
        assert "fetch(" not in src
        assert "http://" not in src
        assert "https://" not in src


# ── Render pipeline: parse (html:false) → sanitize → single insertion path ──


class TestRenderPipelineSource:
    def test_markdownit_configured_html_false(self):
        assert "new window.markdownit({html:false" in EMBEDDED_HTML

    def test_sanitize_wraps_markdownit_output_before_return(self):
        start = EMBEDDED_HTML.index("function renderMarkdownHTML")
        end = EMBEDDED_HTML.index("function highlightFences")
        src = EMBEDDED_HTML[start:end]
        # DOMPurify.sanitize must be the value returned, and _md.render must
        # feed into it — sanitize is the last thing that happens to the HTML.
        assert "_md.render(" in src
        assert "return window.DOMPurify.sanitize(raw)" in src

    def test_single_innerhtml_insertion_uses_sanitized_output(self):
        start = EMBEDDED_HTML.index("function MarkdownBox")
        end = EMBEDDED_HTML.index("function ContentBox")
        src = EMBEDDED_HTML[start:end]
        assert "host.current.innerHTML = out;" in src
        assert "renderMarkdownHTML(text)" in src

    def test_rendered_output_memoized_on_content(self):
        start = EMBEDDED_HTML.index("function MarkdownBox")
        end = EMBEDDED_HTML.index("function ContentBox")
        src = EMBEDDED_HTML[start:end]
        assert "useMemo(()=>renderMarkdownHTML(text), [text])" in src


# ── Fenced-code highlighting via vendored CodeMirror ─────────────────────────


class TestFencedCodeHighlighting:
    def test_highlight_fences_uses_cm6_for_known_languages(self):
        start = EMBEDDED_HTML.index("function highlightFences")
        end = EMBEDDED_HTML.index("function MarkdownBox")
        src = EMBEDDED_HTML[start:end]
        assert "lang==='json'" in src
        assert "lang==='python'" in src
        assert "CM.shell" in src

    def test_unsupported_language_falls_back_to_plain_pre(self):
        start = EMBEDDED_HTML.index("function highlightFences")
        end = EMBEDDED_HTML.index("function MarkdownBox")
        src = EMBEDDED_HTML[start:end]
        # The early `return` (no CM extension pushed) leaves the original
        # <pre><code> markup DOMPurify already let through — still visually
        # distinct monospace code, not replaced by a CM instance.
        assert "else return;" in src
        assert "pre.replaceWith(host)" in src


# ── MarkdownBox: Rendered default, Raw reachable via toggle ─────────────────


class TestRenderedRawToggle:
    def test_rendered_is_default_view(self):
        start = EMBEDDED_HTML.index("function MarkdownBox")
        end = EMBEDDED_HTML.index("function ContentBox")
        src = EMBEDDED_HTML[start:end]
        assert "useState(false)" in src  # raw=false by default

    def test_raw_reuses_codebox_with_original_value(self):
        start = EMBEDDED_HTML.index("function MarkdownBox")
        end = EMBEDDED_HTML.index("function ContentBox")
        src = EMBEDDED_HTML[start:end]
        # Raw must pass the *original* value/hint straight to CodeBox, so the
        # exact source (whitespace/escapes) is what CodeBox already renders
        # byte-exact today — no transformation applied on the raw path.
        assert "html`<${CodeBox} value=${value} lang=${hint}/>`" in src


# ── ContentBox: scope boundary (free-text vs JSON/hinted) ───────────────────


class TestContentBoxScope:
    def test_content_box_delegates_to_todoc_for_the_scope_decision(self):
        start = EMBEDDED_HTML.index("function ContentBox")
        end = EMBEDDED_HTML.index("// Walk the JSON syntax tree")
        src = EMBEDDED_HTML[start:end]
        assert "toDoc(value, hint)" in src
        assert "lang==='text'" in src
        assert "MarkdownBox" in src
        assert "CodeBox" in src

    def test_free_text_dispatch_points_route_through_content_box(self):
        # LLM response/reasoning and report content are the primary free-text
        # surfaces issue #112 called out; command/output/error and the
        # generic DataView/Value string branches join the same dispatcher.
        for needle in (
            "<${ContentBox} value=${o.content}/>",
            "<${ContentBox} value=${o.reasoning}/>",
            "<${ContentBox} value=${content}/>",
        ):
            assert needle in EMBEDDED_HTML

    def test_json_and_array_branches_still_use_codebox_directly(self):
        # DataView's array branch and Value's object branch are unaffected —
        # objects/arrays always resolve to lang:'json' in toDoc, so routing
        # them through CodeBox directly (not ContentBox) is behaviourally
        # identical but keeps the diff minimal at those call sites.
        start = EMBEDDED_HTML.index("function DataView")
        end = EMBEDDED_HTML.index("function Value")
        src = EMBEDDED_HTML[start:end]
        assert "Array.isArray(data)) return html`<${CodeBox} value=${data}/>`;" in src

    def test_tool_calls_array_unaffected(self):
        assert "<${CodeBox} value=${calls}/>" in EMBEDDED_HTML
