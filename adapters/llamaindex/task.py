"""LlamaIndex FunctionAgent.

Tool functions are generated from the shared spec; LlamaIndex reads their
signature and docstring to build the schema it sends.
"""

import asyncio
import json
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

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
    exec(src, ns)  # noqa: S102
    return ns[tool["name"]]


async def _run():
    llm = OpenAI(
        model="gpt-4o",
        api_base=os.environ["BLOATINDEX_BASE_URL"],
        api_key="sk-bloatindex",
    )
    tools = [
        FunctionTool.from_defaults(
            fn=make_fn(t), name=t["name"], description=t["description"]
        )
        for t in SPEC["tools"][:N_TOOLS]
    ]
    agent = FunctionAgent(tools=tools, llm=llm)
    return str(await agent.run(SPEC["task_prompt"]))


if __name__ == "__main__":
    print(asyncio.run(_run()))
