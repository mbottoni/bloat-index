"""Tests for the recording mock.

The mock's job is to be boringly predictable. When it is not, adapters loop
or crash and the resulting numbers look like framework problems -- which has
already happened once, and is the reason these tests exist.
"""

import json
import urllib.request
from pathlib import Path

import pytest

from bloatindex.server import FINAL_ANSWER, TOOL_NAME, serve

WEATHER_TOOL = {
    "type": "function",
    "function": {"name": TOOL_NAME, "parameters": {"type": "object", "properties": {}}},
}
FINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}},
    },
}


@pytest.fixture
def server(tmp_path):
    httpd, recorder, port = serve(tmp_path / "requests.jsonl")
    yield f"http://127.0.0.1:{port}/v1", recorder
    httpd.shutdown()


def post(base, path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def test_first_turn_returns_a_tool_call(server):
    base, _ = server
    r = post(base, "/chat/completions", {"messages": [], "tools": [WEATHER_TOOL]})
    calls = r["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == TOOL_NAME


def test_second_turn_ends_the_episode(server):
    base, _ = server
    post(base, "/chat/completions", {"messages": [], "tools": [WEATHER_TOOL]})
    r = post(base, "/chat/completions", {"messages": [], "tools": [WEATHER_TOOL]})
    assert r["choices"][0]["message"]["content"] == FINAL_ANSWER


def test_no_tool_call_when_none_were_declared(server):
    """A framework that forwards no schemas must not get a call it cannot route."""
    base, _ = server
    r = post(base, "/chat/completions", {"messages": []})
    assert r["choices"][0]["message"]["content"] == FINAL_ANSWER


def test_final_answer_convention_is_honoured(server):
    """Frameworks that terminate via a tool must be terminated via that tool.

    Without this, smolagents loops to max_steps and appears to burn tens of
    thousands of tokens on a two-step task.
    """
    base, _ = server
    tools = [WEATHER_TOOL, FINAL_TOOL]
    post(base, "/chat/completions", {"messages": [], "tools": tools})
    r = post(base, "/chat/completions", {"messages": [], "tools": tools})
    call = r["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "final_answer"
    assert FINAL_ANSWER in json.loads(call["arguments"])["answer"]


def test_usage_is_always_zero(server):
    """We count tokens ourselves; a framework must not be able to affect that."""
    base, _ = server
    r = post(base, "/chat/completions", {"messages": [], "tools": [WEATHER_TOOL]})
    assert r["usage"]["total_tokens"] == 0


def test_anthropic_dialect(server):
    base, _ = server
    r = post(base, "/messages", {"messages": [], "tools": [{"name": TOOL_NAME}]})
    assert r["content"][0]["type"] == "tool_use"
    assert r["content"][0]["name"] == TOOL_NAME


def test_requests_are_recorded_in_order(server):
    base, recorder = server
    post(base, "/chat/completions", {"messages": [], "tools": [WEATHER_TOOL], "marker": 1})
    post(base, "/chat/completions", {"messages": [], "marker": 2})
    assert [r["body"]["marker"] for r in recorder.requests] == [1, 2]


def test_flush_writes_jsonl(server, tmp_path):
    base, recorder = server
    post(base, "/chat/completions", {"messages": [], "marker": 1})
    recorder.flush()
    lines = recorder.out_path.read_text().strip().splitlines()
    assert json.loads(lines[0])["body"]["marker"] == 1
