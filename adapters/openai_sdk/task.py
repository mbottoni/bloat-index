"""Baseline: the official OpenAI SDK, no agent framework.

The realistic floor. Most production agents that skip a framework look
roughly like this.
"""

import json
import os

from openai import OpenAI

SPEC = json.load(open(os.environ["BLOATINDEX_SPEC"]))
N_TOOLS = int(os.environ.get("BLOATINDEX_N_TOOLS", "2"))

client = OpenAI(base_url=os.environ["BLOATINDEX_BASE_URL"], api_key="sk-bloatindex")
TOOLS = [{"type": "function", "function": t} for t in SPEC["tools"][:N_TOOLS]]

IMPLS = {
    "get_weather": lambda city: f"{city}: 22C, clear",
    "add": lambda a, b: str(a + b),
}


def run():
    messages = [{"role": "user", "content": SPEC["task_prompt"]}]
    for _ in range(6):
        msg = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=TOOLS
        ).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            result = IMPLS[tc.function.name](**json.loads(tc.function.arguments))
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
    raise RuntimeError("agent did not terminate")


if __name__ == "__main__":
    print(run())
