# Methodology

Every definition this benchmark relies on, written out so disagreements can be about the choices rather than about what was measured.

## The task

One question — *"What is the weather in Sao Paulo?"* — answered with one tool call, then a final response. Two model round-trips.

It is deliberately the smallest task that still exercises tool calling. Fixed cost is what we are isolating, and a larger task would bury it under work that has nothing to do with the framework.

## The tools

All adapters read [`adapters/spec.json`](adapters/spec.json), which holds twelve tool definitions in JSON Schema. Runs use the first *N*.

Adapters translate that schema into their framework's native tool type — a `StructuredTool`, a `FunctionTool`, a `BaseTool` subclass — but **must not** rename a tool, reword a description, drop a field, or change a type. Several adapters generate their tool functions with `exec` precisely so the signature is derived from the spec rather than hand-written and quietly divergent.

Where a framework derives its schema from a Python signature and docstring (pydantic-ai, llama-index), the generated function carries the spec's description as its docstring, so the same text reaches the wire.

## Token counting

- Tokeniser: **`o200k_base`**, the encoding behind the GPT-4o family.
- Counts come from **the request we recorded**, never from provider-reported usage. The mock always reports zero usage, so a framework cannot influence its score by how it handles that field.
- Absolute counts shift somewhat under a different tokeniser. Every framework is measured with the same one, so comparisons between rows hold.

### Attribution

Each recorded request is partitioned into five buckets that sum to the total:

| bucket | what it is |
|---|---|
| `tool_schemas` | `json.dumps` of the `tools`/`functions` array |
| `system` | system-role messages, plus Anthropic's top-level `system` field |
| `history` | assistant, tool and function messages — prior turns being re-sent |
| `scaffold` | user-role text that is **not** your task string |
| `task` | your task string, credited once, to the first user turn containing it |

**Overhead** is defined as `total - task`: every token in the request that is not the string the caller passed in.

The scaffold bucket is what catches prompt rewriting. If a framework sends `Current Task: What is the weather in Sao Paulo?`, the task string is credited 8 tokens and the wrapper is charged as scaffold.

Two totals are reported. `first_request` is the fixed toll paid before any work happens. `episode` sums every request in the run, and is the figure that scales with loop length.

## Dependencies and disk

- One `uv` virtualenv per adapter, created with `--python 3.12`. Nothing is shared.
- `deps` is the package count from `uv pip list`. A bare uv venv seeds nothing, so every entry was pulled in by the adapter's own requirements.
- `install` is the on-disk size of `site-packages`, **excluding `__pycache__`**. Bytecode caches are written lazily on first import, so counting them would make the number depend on how many times the adapter had been run — an early version of this harness reported langchain at 38 MB clean and 53 MB after a few runs.

## Import time

Median of five cold subprocesses. The timer runs *inside* the child and brackets the import statement only, so interpreter startup is excluded. Each sample is a fresh process, so nothing is served from a warm module cache.

`stdlib` reports no import time — there is nothing to import.

## Lines of code

Non-blank, non-comment lines of `task.py`, with docstrings excluded via an AST pass. The adapters' module docstrings are commentary for readers of this repository, not part of the program, and counting them would penalise the adapters that explain themselves best.

This is the weakest metric here. It measures one person's rendering of the task, and reasonable people would write these differently. Treat it as indicative.

## Repeats

Each adapter runs three times. Token counts are usually byte-identical run to run, but some frameworks stamp a timestamp or UUID into the prompt, and a single sample would hide that. Tables report the median; the JSON keeps min and max, and the README states plainly whether anything varied.

## The mock

A standard-library HTTP server implementing the OpenAI and Anthropic chat endpoints. It plays a fixed script: turn 1 returns a `get_weather` tool call, every later turn ends the episode.

Two details matter for fairness:

1. **It only returns a tool call if the framework actually declared that tool.** A framework that never forwarded its schemas would otherwise receive a call for a tool it does not know, and crash in a way that looks like the framework's bug.

2. **It answers in the shape each framework waits for.** Some frameworks do not treat a plain assistant message as terminal — smolagents expects a `final_answer` tool call and will otherwise loop to `max_steps`. Before this was handled, smolagents appeared to make 21 requests and burn 33,448 tokens on a two-step task. That was our bug, and reporting it would have been a false accusation. If your framework has a similar terminal convention, the mock needs to learn it; please open an issue.

### What the mock does not emulate

Rate limits, latency, streaming backpressure, refusals, malformed tool arguments, and context-length errors. Frameworks whose value shows up in handling those conditions are not being credited for it here — see the limits section of the README.

## Known limitations

Beyond the mock: single task shape, single provider dialect per adapter, default configuration only, and no measurement of answer quality. The README lists these prominently rather than in a footnote, because a benchmark that hides its limits earns the pushback it gets.
