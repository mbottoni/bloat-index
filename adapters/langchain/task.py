"""LangChain / LangGraph ReAct agent.

Tools are built from the shared spec with the same names, descriptions and
JSON Schema as every other adapter, so any difference in the numbers comes
from the framework rather than from how the task was written.
"""

import json
import os

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import Field, create_model

SPEC = json.load(open(os.environ["BLOATINDEX_SPEC"]))
N_TOOLS = int(os.environ.get("BLOATINDEX_N_TOOLS", "2"))

_PY = {"string": str, "number": float, "integer": int, "boolean": bool}

IMPLS = {
    "get_weather": lambda city: f"{city}: 22C, clear",
    "add": lambda a, b: str(a + b),
}


def _args_model(tool):
    """Turn the spec's JSON Schema into the pydantic model LangChain wants."""
    params = tool["parameters"]
    required = set(params.get("required", []))
    fields = {}
    for pname, pschema in params["properties"].items():
        typ = _PY.get(pschema.get("type"), str)
        default = ... if pname in required else None
        if pname not in required:
            typ = typ | None
        fields[pname] = (typ, Field(default, description=pschema.get("description", "")))
    return create_model(f"{tool['name']}_args", **fields)


def _impl(name):
    fn = IMPLS.get(name)
    if fn:
        return lambda **kw: fn(**kw)
    return lambda **kw: f"{name} not implemented"


def build_tools():
    return [
        StructuredTool.from_function(
            func=_impl(t["name"]),
            name=t["name"],
            description=t["description"],
            args_schema=_args_model(t),
        )
        for t in SPEC["tools"][:N_TOOLS]
    ]


def run():
    llm = ChatOpenAI(
        model="gpt-4o",
        base_url=os.environ["BLOATINDEX_BASE_URL"],
        api_key="sk-bloatindex",
        temperature=0,
    )
    agent = create_react_agent(llm, build_tools())
    result = agent.invoke({"messages": [("user", SPEC["task_prompt"])]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(run())
