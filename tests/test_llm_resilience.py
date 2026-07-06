"""Tests for LLM failure classification and two-phase outage recovery (D1/D2).

Network-free (CONTRIBUTE.md §30): ``_request_with_retry`` is mocked so no request
reaches a real server, and ``time.sleep`` is patched so patient-phase waits are
instant. These assert the *behavior* the spec names — a failed response is never
turned into a successful completion or an empty assistant turn, retryable classes
are absorbed within tolerance, and non-retryable classes fail fast.
"""

import json

import httpx
import pytest

from my_coding_agent.engine.llm.errors import (
    LLMHTTPStatusError,
    LLMMalformedBodyError,
    LLMTransportError,
)
from my_coding_agent.engine.llm.schema import (
    CLASSIFICATION_HTTP_STATUS,
    CLASSIFICATION_MALFORMED_BODY,
    CLASSIFICATION_TRANSPORT,
)


class _Resp:
    """Minimal httpx.Response stand-in with a fixed status/body."""

    def __init__(self, payload, content=b"{}", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code
        self.text = content.decode() if isinstance(content, bytes) else str(content)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok(content="hi"):
    return _Resp(
        {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


@pytest.fixture(autouse=True)
def _no_sleep(mocker):
    """Make every patient-phase wait instant so tests never actually block."""
    mocker.patch("my_coding_agent.engine.llm.time.sleep", return_value=None)


# ── classification (D1) ───────────────────────────────────────────────────────


def test_http_500_json_body_classified_http_status_no_message(bare_llm, mocker):
    # HTTP 500 with a JSON error body → http-status, and no assistant message /
    # usage row is produced from it. Tolerance 0 so it fails immediately.
    mocker.patch.dict("os.environ", {"MCA_LLM_OUTAGE_TOLERANCE_S": "0"})
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp(
            {"error": "boom"}, content=b'{"error":"boom"}', status_code=500
        ),
    )
    with pytest.raises(LLMHTTPStatusError) as exc:
        bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert exc.value.classification == CLASSIFICATION_HTTP_STATUS
    assert exc.value.status_code == 500
    assert bare_llm.llm_calls == []  # never became a successful completion


def test_non_json_body_classified_malformed_with_status_and_prefix(bare_llm, mocker):
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp(None, content=b"<html>oops</html>", status_code=200),
    )
    with pytest.raises(LLMMalformedBodyError) as exc:
        bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert exc.value.classification == CLASSIFICATION_MALFORMED_BODY
    assert "non-JSON response" in str(exc.value)
    assert "html" in str(exc.value)  # body prefix named
    assert bare_llm.llm_calls == []


def test_http_200_empty_choices_classified_malformed_not_empty_turn(bare_llm, mocker):
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp({"choices": [], "usage": {}}, status_code=200),
    )
    with pytest.raises(LLMMalformedBodyError) as exc:
        bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert exc.value.classification == CLASSIFICATION_MALFORMED_BODY
    assert "choices" in str(exc.value)
    assert bare_llm.llm_calls == []  # not an empty assistant turn


def test_http_400_fails_fast_no_patient_phase(bare_llm, mocker):
    # 400 is a protocol bug: fail at once as http-status, never sleep/probe.
    recorder = mocker.Mock()
    bare_llm._recorder = recorder
    sleep = mocker.patch("my_coding_agent.engine.llm.time.sleep")
    req = mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        return_value=_Resp(
            {"error": "bad"}, content=b'{"error":"bad"}', status_code=400
        ),
    )
    with pytest.raises(LLMHTTPStatusError) as exc:
        bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert exc.value.classification == CLASSIFICATION_HTTP_STATUS
    assert exc.value.retryable is False
    assert req.call_count == 1  # one attempt, no retry
    sleep.assert_not_called()
    # The fail-fast case still emits the failure event (documented "non-retryable"
    # purpose), exactly like the tolerance-exceeded branch.
    recorder.record_llm_failure.assert_called_once()
    _, kwargs = recorder.record_llm_failure.call_args
    assert kwargs["classification"] == CLASSIFICATION_HTTP_STATUS
    assert bare_llm.llm_calls == []


# ── outage absorption (D2) ────────────────────────────────────────────────────


def test_outage_within_tolerance_recovers_with_state_intact(bare_llm, mocker):
    # Server unreachable for a few probes, then answers — the run continues and
    # the recovered call records its usage normally.
    recorder = mocker.Mock()
    bare_llm._recorder = recorder
    bare_llm._context_window = 8192  # avoid the lazy network probe
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        side_effect=[
            httpx.ConnectError("down"),
            httpx.ConnectError("down"),
            _ok("recovered"),
        ],
    )
    resp = bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert resp.json()["choices"][0]["message"]["content"] == "recovered"
    assert bare_llm.llm_calls[-1]["total"] == 2  # usage recorded after recovery
    assert recorder.record_llm_wait.call_count == 2
    recorder.record_llm_recovery.assert_called_once()
    recorder.record_llm_failure.assert_not_called()


def test_outage_beyond_tolerance_raises_classified_and_records_failure(
    bare_llm, mocker
):
    recorder = mocker.Mock()
    bare_llm._recorder = recorder
    mocker.patch.dict("os.environ", {"MCA_LLM_OUTAGE_TOLERANCE_S": "0"})
    mocker.patch.object(
        bare_llm, "_request_with_retry", side_effect=httpx.ConnectError("down")
    )
    with pytest.raises(LLMTransportError) as exc:
        bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert exc.value.classification == CLASSIFICATION_TRANSPORT
    recorder.record_llm_failure.assert_called_once()
    assert bare_llm.llm_calls == []


def test_custom_env_tolerance_is_respected(bare_llm, mocker):
    mocker.patch.dict("os.environ", {"MCA_LLM_OUTAGE_TOLERANCE_S": "42"})
    assert bare_llm._outage_tolerance_s() == 42.0


def test_bad_env_tolerance_falls_back_to_default(bare_llm, mocker):
    from my_coding_agent.engine.llm.schema import DEFAULT_OUTAGE_TOLERANCE_S

    mocker.patch.dict("os.environ", {"MCA_LLM_OUTAGE_TOLERANCE_S": "not-a-number"})
    assert bare_llm._outage_tolerance_s() == DEFAULT_OUTAGE_TOLERANCE_S


def test_429_is_retryable(bare_llm, mocker):
    recorder = mocker.Mock()
    bare_llm._recorder = recorder
    bare_llm._context_window = 8192
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        side_effect=[
            _Resp({"error": "rate"}, content=b"{}", status_code=429),
            _ok("after-429"),
        ],
    )
    resp = bare_llm.chat_completion([{"role": "user", "content": "q"}])
    assert resp.json()["choices"][0]["message"]["content"] == "after-429"
    recorder.record_llm_wait.assert_called_once()


# ── observability: real Recorder writes the events to events.jsonl ────────────


def test_recovery_events_land_in_events_jsonl(bare_llm, mocker, tmp_path):
    from my_coding_agent.observability.recorder import Recorder

    sdir = tmp_path / "sid"
    sdir.mkdir()
    bare_llm._recorder = Recorder(session_id="sid", session_dir=sdir)
    bare_llm._context_window = 8192
    mocker.patch.object(
        bare_llm,
        "_request_with_retry",
        side_effect=[httpx.ConnectError("down"), _ok("ok")],
    )
    bare_llm.chat_completion([{"role": "user", "content": "q"}])
    events = [
        json.loads(line)
        for line in (sdir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    types = [e["type"] for e in events]
    assert "llm_wait" in types
    assert "llm_recovery" in types
    wait = next(e for e in events if e["type"] == "llm_wait")
    assert wait["classification"] == CLASSIFICATION_TRANSPORT
    assert "attempt" in wait and "delay_s" in wait
