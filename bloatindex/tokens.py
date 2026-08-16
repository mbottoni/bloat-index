"""Splits a recorded request into the parts a user asked for and the rest.

The headline number this project reports -- overhead tokens -- is defined
here, so the definition is worth stating plainly:

    overhead = every token in the request that is not the task string
               the caller actually passed in

Tool schemas, system prompts, format instructions, scratchpad boilerplate
and re-sent history are all overhead. That is not a pejorative; some of it
buys real capability. It is simply the price, and the price is measurable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import tiktoken

# o200k_base backs the GPT-4o family. The absolute counts shift a little
# under other tokenisers, but every framework is measured with the same one,
# so the comparison between them holds.
_ENC = tiktoken.get_encoding("o200k_base")


def count(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text, disallowed_special=()))


def _text_of(content) -> str:
    """Flatten a message body to text.

    Content is a bare string in the simple case, and a list of typed blocks
    in the multimodal / tool-result case that both providers now use.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") in ("tool_use", "tool_result"):
                    # Serialised whole: the JSON really is sent over the wire.
                    parts.append(json.dumps(block))
                else:
                    parts.append(json.dumps(block))
        return "\n".join(parts)
    return json.dumps(content)


@dataclass
class Breakdown:
    """Token counts for one request, partitioned so the parts sum to total."""

    total: int = 0
    tool_schemas: int = 0
    system: int = 0
    history: int = 0
    scaffold: int = 0  # framework text injected into user-role turns
    task: int = 0      # the caller's actual question

    @property
    def overhead(self) -> int:
        return self.total - self.task

    @property
    def overhead_pct(self) -> float:
        return 100.0 * self.overhead / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["overhead"] = self.overhead
        d["overhead_pct"] = round(self.overhead_pct, 1)
        return d


def analyse(body: dict, task_prompt: str, dialect: str = "openai") -> Breakdown:
    """Partition one recorded request body.

    `task_prompt` is the exact string the adapter was asked to send. Any user
    turn containing it is credited with the task's tokens and charged scaffold
    for the remainder, which is how prompt wrappers become visible.
    """
    b = Breakdown()
    task_tokens = count(task_prompt)

    tools = body.get("tools") or body.get("functions") or []
    if tools:
        b.tool_schemas = count(json.dumps(tools, separators=(",", ":")))

    # Anthropic carries the system prompt beside the messages, not inside them.
    if dialect == "anthropic" and body.get("system"):
        b.system += count(_text_of(body["system"]))

    task_credited = False
    for msg in body.get("messages", []):
        role = msg.get("role", "")
        text = _text_of(msg.get("content"))

        # Assistant tool calls live outside `content` in the OpenAI dialect.
        if msg.get("tool_calls"):
            text += "\n" + json.dumps(msg["tool_calls"], separators=(",", ":"))

        n = count(text)
        if role == "system":
            b.system += n
        elif role == "user":
            if not task_credited and task_prompt and task_prompt in text:
                b.task += task_tokens
                b.scaffold += max(0, n - task_tokens)
                task_credited = True
            else:
                b.scaffold += n
        else:  # assistant, tool, function -- prior turns being re-sent
            b.history += n

    b.total = b.tool_schemas + b.system + b.history + b.scaffold + b.task
    return b


def summarise(requests: list[dict], task_prompt: str) -> dict:
    """Aggregate a whole episode.

    Two numbers matter and they are different. `first_request` is what the
    framework costs to say hello -- the fixed toll paid before any work. The
    episode total is what the same task costs once every re-sent turn is
    counted, and it is the one that scales with agent loop length.
    """
    per_request = [
        analyse(r["body"], task_prompt, r.get("dialect", "openai")) for r in requests
    ]
    if not per_request:
        return {"error": "no requests recorded"}

    first = per_request[0]
    return {
        "n_requests": len(per_request),
        "first_request": first.as_dict(),
        "episode": {
            "total": sum(p.total for p in per_request),
            "overhead": sum(p.overhead for p in per_request),
            "task": sum(p.task for p in per_request),
            "tool_schemas": sum(p.tool_schemas for p in per_request),
            "system": sum(p.system for p in per_request),
            "history": sum(p.history for p in per_request),
            "scaffold": sum(p.scaffold for p in per_request),
        },
        "per_request": [p.as_dict() for p in per_request],
    }
