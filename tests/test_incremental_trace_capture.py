"""Fidelity and size-growth verification for incremental trace capture.

Round-trips synthetic sessions through ``Recorder`` (write) and the trace
reader (read/reconstruct), per incremental-trace-capture tasks 4.1/4.2.
"""

from __future__ import annotations

from my_coding_agent.observability.recorder import Recorder
from my_coding_agent.viewer.reader import _read_events

_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_RESPONSE = {"choices": [{"message": {"content": "ok"}}]}


def _make_recorder(tmp_path, name="sess"):
    sdir = tmp_path / name
    sdir.mkdir()
    return Recorder(session_id=name, session_dir=sdir), sdir / "events.jsonl"


def _record_main_call(rec, call, messages):
    rec.record_llm_call(
        kind="main",
        call=call,
        latency_s=0.1,
        usage=_USAGE,
        messages=messages,
        context_window=8192,
        response_data=_RESPONSE,
    )


class TestRoundTripFidelity:
    def test_appends_and_handoff_replacement_reconstruct_exactly(self, tmp_path):
        rec, path = _make_recorder(tmp_path)
        recorded_inputs: list[list[dict]] = []

        m1 = [{"role": "user", "content": "hi"}]
        _record_main_call(rec, 1, m1)
        recorded_inputs.append(m1)

        m2 = [*m1, {"role": "assistant", "content": "a1"}]
        _record_main_call(rec, 2, m2)
        recorded_inputs.append(m2)

        m3 = [*m2, {"role": "user", "content": "u2"}]
        _record_main_call(rec, 3, m3)
        recorded_inputs.append(m3)

        # Handoff: brand-new list of brand-new message objects.
        m4 = [{"role": "system", "content": "reset"}, {"role": "user", "content": "u3"}]
        _record_main_call(rec, 4, m4)
        recorded_inputs.append(m4)

        m5 = [*m4, {"role": "assistant", "content": "a4"}]
        _record_main_call(rec, 5, m5)
        recorded_inputs.append(m5)

        events = [e for e in _read_events(path) if e["type"] == "llm_call"]
        assert len(events) == len(recorded_inputs)
        for ev, expected in zip(events, recorded_inputs, strict=True):
            assert ev["messages"] == expected

    def test_in_place_appends_to_the_live_list_reconstruct_exactly(self, tmp_path):
        """Matches the real pipeline's actual mutation pattern: ``ctx.messages``
        is one list object that gets ``.append()``-ed to after each call
        returns (llm_call.py appends the assistant reply, tool_dispatch.py
        appends tool results) — it is never rebound to a new list except on
        handoff. Round-tripping this exact pattern is what caught the
        reference-aliasing bug where the recorder held a live reference
        instead of a snapshot.
        """
        rec, path = _make_recorder(tmp_path)
        recorded_inputs: list[list[dict]] = []

        live = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}]
        _record_main_call(rec, 1, live)
        recorded_inputs.append(list(live))

        live.append({"role": "assistant", "content": "a1"})
        live.append({"role": "tool", "content": "t1"})
        _record_main_call(rec, 2, live)
        recorded_inputs.append(list(live))

        live.append({"role": "assistant", "content": "a2"})
        _record_main_call(rec, 3, live)
        recorded_inputs.append(list(live))

        events = [e for e in _read_events(path) if e["type"] == "llm_call"]
        assert len(events) == len(recorded_inputs)
        for ev, expected in zip(events, recorded_inputs, strict=True):
            assert ev["messages"] == expected


class TestSizeGrowthBound:
    def test_growth_is_sub_quadratic(self, tmp_path):
        def _record_n_calls(n, name):
            rec, path = _make_recorder(tmp_path, name)
            messages: list[dict] = []
            for call in range(1, n + 1):
                messages = [
                    *messages,
                    {
                        "role": "user" if call % 2 else "assistant",
                        "content": f"m{call}",
                    },
                ]
                _record_main_call(rec, call, messages)
            return path.stat().st_size

        size_n = _record_n_calls(50, "n")
        size_2n = _record_n_calls(100, "two_n")

        assert size_2n < 2.5 * size_n
