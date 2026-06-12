"""Tests for the response-parsing helpers in utils.py.

These helpers normalize raw chat-completion payloads; the contract is that they
always return a sane type (dict/str) even when the payload is empty or malformed.
"""

import json

import pytest

from my_coding_agent.utils import (
    extract_finish_reason,
    extract_message,
    extract_usage,
    parse_tool_args,
)


def _resp(payload, factory):
    return factory(payload)


# --- extract_message ---------------------------------------------------------


def test_extract_message_returns_message_dict(fake_response):
    resp = fake_response({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
    assert extract_message(resp) == {"role": "assistant", "content": "hi"}


def test_extract_message_empty_choices_returns_empty_dict(fake_response):
    assert extract_message(fake_response({"choices": []})) == {}


def test_extract_message_missing_choices_returns_empty_dict(fake_response):
    assert extract_message(fake_response({})) == {}


def test_extract_message_null_message_returns_empty_dict(fake_response):
    assert extract_message(fake_response({"choices": [{"message": None}]})) == {}


# --- extract_finish_reason ---------------------------------------------------


@pytest.mark.parametrize(
    "reason,expected",
    [("STOP", "stop"), ("Tool_Calls", "tool_calls"), ("", "")],
)
def test_extract_finish_reason_lowercases(reason, expected, fake_response):
    resp = fake_response({"choices": [{"finish_reason": reason}]})
    assert extract_finish_reason(resp) == expected


def test_extract_finish_reason_missing_returns_empty(fake_response):
    assert extract_finish_reason(fake_response({"choices": [{}]})) == ""


def test_extract_finish_reason_no_choices_returns_empty(fake_response):
    assert extract_finish_reason(fake_response({"choices": []})) == ""


# --- extract_usage -----------------------------------------------------------


def test_extract_usage_returns_usage_dict(fake_response):
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    assert extract_usage(fake_response({"usage": usage})) == usage


def test_extract_usage_missing_returns_empty_dict(fake_response):
    assert extract_usage(fake_response({})) == {}


def test_extract_usage_null_returns_empty_dict(fake_response):
    assert extract_usage(fake_response({"usage": None})) == {}


# --- parse_tool_args ---------------------------------------------------------


def test_parse_tool_args_passes_dict_through():
    args = {"command": "ls"}
    assert parse_tool_args(args) is args


def test_parse_tool_args_parses_json_string():
    assert parse_tool_args('{"command": "ls -la"}') == {"command": "ls -la"}


def test_parse_tool_args_empty_json_object():
    assert parse_tool_args("{}") == {}


def test_parse_tool_args_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_tool_args("{not json}")
