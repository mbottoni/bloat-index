"""Pydantic-AI agent.

Tool functions are generated from the shared spec so their signatures,
type hints and docstrings match what every other adapter declares.
Pydantic-AI derives its JSON Schema from the signature, so building the
functions this way is what keeps the comparison honest.
"""

import json
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import Tool

SPEC = json.load(open(os.environ["BLOATINDEX_SPEC"]))
N_TOOLS = int(os.environ.get("BLOATINDEX_N_TOOLS", "2"))

_ANN = {"string": "str", "number": "float", "integer": "int", "boolean": "bool"}

RESULTS = {
    "get_weather": lambda kw: f"{kw['city']}: 22C, clear",
    "add": lambda kw: str(kw["a"] + kw["b"]),
}


def _impl(name, kwargs):
    fn = RESULTS.get(name)
    return fn(kwargs) if fn else f"{name} not implemented"


def make_fn(tool):
    """Build a real function whose signature mirrors the spec's schema."""
    params = tool["parameters"]
    required = params.get("required", [])
    args = []
    for pname, pschema in params["properties"].items():
        ann = _ANN.get(pschema.get("type"), "str")
        args.append(f"{pname}: {ann}" if pname in required else f"{pname}: {ann} | None = None")
    src = (
        f"def {tool['name']}({', '.join(args)}) -> str:\n"
        f"    \"\"\"{tool['description']}\"\"\"\n"
        f"    return _impl({tool['name']!r}, locals())\n"
    )
    ns = {"_impl": _impl}
    exec(src, ns)  # noqa: S102 -- generating from our own spec file
    return ns[tool["name"]]


def run():
    model = OpenAIChatModel(
        "gpt-4o",
        provider=OpenAIProvider(
            base_url=os.environ["BLOATINDEX_BASE_URL"], api_key="sk-bloatindex"
        ),
    )
    tools = [
        Tool(make_fn(t), name=t["name"], description=t["description"])
        for t in SPEC["tools"][:N_TOOLS]
    ]
    agent = Agent(model, tools=tools)
    return agent.run_sync(SPEC["task_prompt"]).output


if __name__ == "__main__":
    print(run())
