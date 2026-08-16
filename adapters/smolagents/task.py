"""smolagents ToolCallingAgent.

smolagents declares tool inputs directly as a dict, so the spec's schema is
passed through with very little translation.
"""

import json
import os

from smolagents import OpenAIServerModel, Tool, ToolCallingAgent

SPEC = json.load(open(os.environ["BLOATINDEX_SPEC"]))
N_TOOLS = int(os.environ.get("BLOATINDEX_N_TOOLS", "2"))

RESULTS = {
    "get_weather": lambda kw: f"{kw['city']}: 22C, clear",
    "add": lambda kw: str(kw["a"] + kw["b"]),
}


def _dispatch(name, kwargs):
    fn = RESULTS.get(name)
    return fn(kwargs) if fn else f"{name} not implemented"


def make_tool(spec_tool):
    params = spec_tool["parameters"]
    required = set(params.get("required", []))
    inputs = {}
    for pname, pschema in params["properties"].items():
        entry = {
            "type": pschema.get("type", "string"),
            "description": pschema.get("description", ""),
        }
        if pname not in required:
            entry["nullable"] = True
        inputs[pname] = entry

    # smolagents validates that `forward` names each declared input
    # explicitly, so the method is generated to match the spec exactly
    # rather than accepting **kwargs.
    argnames = list(params["properties"])
    sig = ", ".join(a if a in required else f"{a}=None" for a in argnames)
    passthrough = ", ".join(f"{a}={a}" for a in argnames)
    src = (
        f"def forward(self, {sig}):\n"
        f"    return _dispatch(self.name, dict({passthrough}))\n"
    )
    ns = {"_dispatch": _dispatch}
    exec(src, ns)  # noqa: S102 -- generated from our own spec file

    return type(
        f"{spec_tool['name']}_tool",
        (Tool,),
        {
            "name": spec_tool["name"],
            "description": spec_tool["description"],
            "inputs": inputs,
            "output_type": "string",
            "forward": ns["forward"],
        },
    )()


def run():
    model = OpenAIServerModel(
        model_id="gpt-4o",
        api_base=os.environ["BLOATINDEX_BASE_URL"],
        api_key="sk-bloatindex",
    )
    agent = ToolCallingAgent(
        tools=[make_tool(t) for t in SPEC["tools"][:N_TOOLS]],
        model=model,
        add_base_tools=False,
        verbosity_level=0,
    )
    return str(agent.run(SPEC["task_prompt"]))


if __name__ == "__main__":
    print(run())
