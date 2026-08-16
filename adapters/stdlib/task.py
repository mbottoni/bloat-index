"""Baseline: a tool-calling agent loop in the Python standard library.

This adapter exists to answer one question -- what does the task cost when
nothing is added to it? Every other row in the table is read against this one.

It is not a recommendation. There is no retry logic, no streaming, no
provider fallback, no tracing. Those things cost tokens and dependencies for
a reason, and the point of the index is to price them, not to pretend they
are worthless.
"""

import json
import os
import urllib.request

BASE = os.environ["BLOATINDEX_BASE_URL"]
SPEC = json.load(open(os.environ["BLOATINDEX_SPEC"]))
N_TOOLS = int(os.environ.get("BLOATINDEX_N_TOOLS", "2"))

TOOLS = [
    {"type": "function", "function": t} for t in SPEC["tools"][:N_TOOLS]
]

IMPLS = {
    "get_weather": lambda city: f"{city}: 22C, clear",
    "add": lambda a, b: str(a + b),
}


def call(messages):
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps({"model": "gpt-4o", "messages": messages, "tools": TOOLS}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer sk-bloatindex"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["choices"][0]["message"]


def run():
    messages = [{"role": "user", "content": SPEC["task_prompt"]}]
    for _ in range(6):  # loop guard, not a retry policy
        msg = call(messages)
        if not msg.get("tool_calls"):
            return msg["content"]
        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            result = IMPLS[fn["name"]](**json.loads(fn["arguments"]))
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )
    raise RuntimeError("agent did not terminate")


if __name__ == "__main__":
    print(run())
