"""Tests for the part that decides what counts as overhead.

Everything this project claims rests on the attribution in tokens.py, so it
is the part most worth pinning down.
"""

import json

import pytest

from bloatindex import tokens

TASK = "What is the weather in Sao Paulo?"


def test_buckets_sum_to_total():
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Current Task: {TASK}"},
            {"role": "assistant", "content": "thinking"},
        ],
        "tools": [{"type": "function", "function": {"name": "t", "parameters": {}}}],
    }
    b = tokens.analyse(body, TASK)
    assert b.total == b.tool_schemas + b.system + b.history + b.scaffold + b.task


def test_task_is_not_counted_as_overhead():
    body = {"messages": [{"role": "user", "content": TASK}]}
    b = tokens.analyse(body, TASK)
    assert b.task == tokens.count(TASK)
    assert b.overhead == 0


def test_prompt_wrapping_lands_in_scaffold():
    """A framework that rewrites your prompt should be charged for the wrapper."""
    wrapped = f"Current Task: {TASK}\n\nExpected criteria: The answer."
    body = {"messages": [{"role": "user", "content": wrapped}]}
    b = tokens.analyse(body, TASK)
    assert b.task == tokens.count(TASK)
    assert b.scaffold > 0
    assert b.scaffold == tokens.count(wrapped) - tokens.count(TASK)


def test_task_credited_only_once():
    """Re-sent history must not keep earning free task credit."""
    body = {
        "messages": [
            {"role": "user", "content": TASK},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": TASK},
        ]
    }
    b = tokens.analyse(body, TASK)
    assert b.task == tokens.count(TASK)
    assert b.scaffold == tokens.count(TASK)  # the second copy is overhead


def test_anthropic_system_field_is_counted():
    body = {
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": TASK}],
    }
    b = tokens.analyse(body, TASK, dialect="anthropic")
    assert b.system == tokens.count("You are helpful.")


def test_tool_calls_outside_content_are_counted():
    """OpenAI puts assistant tool calls beside content, not inside it."""
    body = {
        "messages": [
            {"role": "user", "content": TASK},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
        ]
    }
    b = tokens.analyse(body, TASK)
    assert b.history > 0


def test_content_blocks_are_flattened():
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": TASK}]}]
    }
    b = tokens.analyse(body, TASK)
    assert b.task == tokens.count(TASK)


def test_summarise_reports_first_and_episode_separately():
    reqs = [
        {"body": {"messages": [{"role": "user", "content": TASK}], "tools": []}, "dialect": "openai"},
        {
            "body": {
                "messages": [
                    {"role": "user", "content": TASK},
                    {"role": "assistant", "content": "a" * 100},
                ]
            },
            "dialect": "openai",
        },
    ]
    s = tokens.summarise(reqs, TASK)
    assert s["n_requests"] == 2
    assert s["episode"]["total"] > s["first_request"]["total"]
    # The task string is sent twice, so the episode credits it twice.
    assert s["episode"]["task"] == 2 * tokens.count(TASK)


def test_empty_episode_reports_error_rather_than_zero():
    """Silence must not be reported as a perfect score."""
    assert "error" in tokens.summarise([], TASK)
