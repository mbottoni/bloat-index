"""CrewAI single-agent crew.

CrewAI is role-based: an agent needs a role, a goal and a backstory, and a
task needs a description and an expected output. Those fields are required
by the API, so they are part of what the framework costs -- they are written
here as tersely as the framework permits, to avoid inflating its numbers
with prose of our own invention.
"""

import json
import os

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, create_model

SPEC = json.load(open(os.environ["BLOATINDEX_SPEC"]))
N_TOOLS = int(os.environ.get("BLOATINDEX_N_TOOLS", "2"))

_PY = {"string": str, "number": float, "integer": int, "boolean": bool}

RESULTS = {
    "get_weather": lambda kw: f"{kw.get('city')}: 22C, clear",
    "add": lambda kw: str(kw.get("a", 0) + kw.get("b", 0)),
}


def _args_model(tool) -> type[BaseModel]:
    params = tool["parameters"]
    required = set(params.get("required", []))
    fields = {}
    for pname, pschema in params["properties"].items():
        typ = _PY.get(pschema.get("type"), str)
        if pname in required:
            fields[pname] = (typ, Field(..., description=pschema.get("description", "")))
        else:
            fields[pname] = (typ | None, Field(None, description=pschema.get("description", "")))
    return create_model(f"{tool['name']}_args", **fields)


def make_tool(tool) -> BaseTool:
    schema = _args_model(tool)
    name, desc = tool["name"], tool["description"]

    class _Tool(BaseTool):
        model_config = {"arbitrary_types_allowed": True}

        def _run(self, **kwargs) -> str:
            fn = RESULTS.get(self.name)
            return fn(kwargs) if fn else f"{self.name} not implemented"

    return _Tool(name=name, description=desc, args_schema=schema)


def run():
    os.environ.setdefault("OPENAI_API_KEY", "sk-bloatindex")
    agent = Agent(
        role="Assistant",
        goal="Answer the user's question.",
        backstory="You answer questions using the tools you are given.",
        tools=[make_tool(t) for t in SPEC["tools"][:N_TOOLS]],
        llm="gpt-4o",
        verbose=False,
    )
    task = Task(
        description=SPEC["task_prompt"],
        expected_output="The answer.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff())


if __name__ == "__main__":
    print(run())
