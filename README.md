# Bloat Index

**What does an agent framework cost you before it does any work?**

Every framework here runs the *same* task — answer one question, using tools, with one tool call — against a recording server that captures the exact bytes each one puts on the wire. Then we count.

No API key. No spend. Every number below is reproduced from scratch by CI and committed back into this file.

---

## The headline

> Your question is 8 tokens. With 12 tools declared, it ships inside a request of **804 tokens**.
>
> **99% of that first request is overhead — and switching frameworks barely moves it.**

The overhead is tool schemas, and every framework pays it about equally, including the one that uses no framework at all. The industry conversation is about which framework is lighter. The measurement says that is close to the wrong question.

Where frameworks *do* differ, enormously, is what they cost your machine: **0 to 135 dependencies, 4 kB to 581 MB on disk, and a cold import from 265 ms to 1.4 s.**

## Results

One task. Two tools. `overhead tokens` is every token in the first request that is not the user's question.

<!-- BEGIN:TABLE_MAIN -->
| framework | deps | install | import | LOC | overhead tokens | overhead % | episode tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | 26 | 33 MB | 439 ms | 46 | 114 | 93% | 287 |
| [langchain + langgraph](https://github.com/langchain-ai/langchain) | 44 | 65 MB | 611 ms | 51 | 117 | 94% | 293 |
| [openai (SDK only)](https://github.com/openai/openai-python) | 14 | 16 MB | 659 ms | 28 | 117 | 94% | 294 |
| [stdlib (urllib)](https://docs.python.org/3/library/urllib.request.html) | 0 | 5 kB | -- | 37 | 117 | 94% | 293 |
| [llama-index](https://github.com/run-llama/llama_index) | 66 | 175 MB | 1.14 s | 47 | 124 | 94% | 308 |
| [crewai](https://github.com/crewAIInc/crewAI) | 135 | 684 MB | 2.52 s | 51 | 192 | 96% | 443 |
| [smolagents](https://github.com/huggingface/smolagents) | 37 | 67 MB | 369 ms | 59 | 1,058 | 99% | 2,184 |
<!-- END:TABLE_MAIN -->

`episode tokens` is the whole run — the first request plus every turn re-sent afterwards.

<!-- BEGIN:STABILITY -->
Every framework produced byte-identical requests across all 3 runs, so these figures carry no run-to-run variance.
<!-- END:STABILITY -->

## Where the overhead comes from

The same run with <!-- BEGIN:NTOOLS_LARGE -->12<!-- END:NTOOLS_LARGE --> tools declared, broken into parts. They sum to the total.

<!-- BEGIN:TABLE_COMPOSITION -->
| framework | tool schemas | system prompt | scaffold | your task | total |
|---|---:|---:|---:|---:|---:|
| pydantic-ai | 786 | 0 | 0 | 8 | 794 |
| openai (SDK only) | 796 | 0 | 0 | 8 | 804 |
| stdlib (urllib) | 796 | 0 | 0 | 8 | 804 |
| langchain + langgraph | 848 | 0 | 0 | 8 | 856 |
| llama-index | 893 | 0 | 0 | 8 | 901 |
| crewai | 952 | 24 | 33 | 8 | 1,017 |
| smolagents | 864 | 1,550 | 3 | 8 | 2,425 |
<!-- END:TABLE_COMPOSITION -->

- **tool schemas** — the JSON description of the tools you offered
- **system prompt** — instructions the framework adds on your behalf
- **scaffold** — framework text wrapped around your prompt (roles, goals, output-format demands)
- **your task** — the string you actually passed in

Two things stand out.

`smolagents` ships a **6,282-character system prompt** — a full Action/Observation protocol specification with worked examples, re-sent on every turn. That is a deliberate design decision, not an oversight: it is teaching the model a reasoning loop rather than relying on native tool-calling. But it is a decision you pay for per request, and it is the single largest framework-attributable cost in this table.

`crewai` is the only one that rewrites your prompt. Your question arrives at the model as `Current Task: <your question>`, followed by `This is the expected criteria for your final answer: ...` and an instruction to return complete content rather than a summary — plus a system prompt assembled from the role and goal its API requires. Modest in tokens, but worth knowing that the string you passed in is not the string the model sees.

Everyone else is within noise of the no-framework baseline. **On tokens, most agent frameworks are close to free.**

## How overhead scales with your toolbox

Overhead tokens in the first request, as the number of declared tools grows:

<!-- BEGIN:TABLE_SCALING -->
| framework | 2 tools | 12 tools | growth |
|---|---:|---:|---:|
| pydantic-ai | 114 | 786 | +672 |
| langchain + langgraph | 117 | 848 | +731 |
| openai (SDK only) | 117 | 796 | +679 |
| stdlib (urllib) | 117 | 796 | +679 |
| llama-index | 124 | 893 | +769 |
| crewai | 192 | 1,009 | +817 |
| smolagents | 1,058 | 2,417 | +1,359 |
<!-- END:TABLE_SCALING -->

This is the finding that matters. Going from 2 tools to 12 costs roughly 680 tokens **regardless of framework**, because it is your schemas, not their code. Paid on every request, for the whole life of the conversation.

If you want to cut agent token cost, the lever is **how many tools you declare**, not which framework declares them. Tool schemas are the one part of the context nobody is optimising and everybody is paying for.

## How it works

The hard part of measuring this is that frameworks are opaque about what they send. So we do not ask them — we intercept.

```
adapter (isolated venv)  ──HTTP──>  recording server  ──>  requests.jsonl
      │                                    │
      │                          plays a fixed 2-turn script
      └── identical task + identical       │
          tool schemas from spec.json      └──> tokenised with o200k_base
                                                and split into parts
```

- **The server is standard library only.** A measurement harness has no business adding dependencies to the thing it measures.
- **It reports zero usage.** We count tokens ourselves from the recorded request, so no framework can influence its own score by trusting or ignoring provider-reported usage.
- **Tool schemas come from one shared [`spec.json`](adapters/spec.json).** Adapters translate that into their framework's own tool type but may not rename, reword, or drop a field.
- **One virtualenv per adapter**, built by `uv`, so nothing shares dependencies.
- **Each adapter runs 3 times.** Frameworks that stamp a timestamp or UUID into the prompt would otherwise be reported from a single lucky sample.

## Reproduce it

```bash
git clone https://github.com/mbottoni/bloat-index
cd bloat-index
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python tiktoken

.venv/bin/python -m bloatindex.measure                  # all adapters, 2 tools
.venv/bin/python -m bloatindex.measure --tools 12       # the scaling run
.venv/bin/python -m bloatindex.report                   # regenerate this README
```

Runs offline apart from installing the frameworks. Takes a few minutes, most of it `uv` downloading CrewAI.

Raw captured requests land in `results/raw/` — every claim here is auditable down to the byte each framework sent.

## What this does not measure

Stated up front, because a benchmark that hides its limits deserves the pushback it gets:

- **Capability.** A framework charging more tokens may be buying retries, structured output, streaming, observability, or multi-agent routing. This measures the price, not whether it is worth paying.
- **Quality.** The model is mocked, so nothing here says whether a framework produces *better* answers. It cannot.
- **One task shape.** Single agent, one tool call, no memory, no RAG, no handoffs. Frameworks built for orchestration are being measured on the task they are least suited to. That is deliberate — it isolates fixed cost — but do not read it as a verdict on complex workloads.
- **Default configuration.** Each adapter follows the path its own docs steer you to. A tuned deployment can do better, and system prompts in particular are usually overridable.
- **Install size counts what is installed**, excluding `__pycache__`, which grows as you run and would make the figure depend on run count.

If an adapter misrepresents your framework, that is a bug and a PR fixing it is welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Adding a framework

Create `adapters/<name>/` with three files — `meta.json`, `requirements.txt`, and a `task.py` that reads `BLOATINDEX_BASE_URL`, `BLOATINDEX_SPEC` and `BLOATINDEX_N_TOOLS`, runs the task, and prints the answer. It is picked up automatically. [`adapters/stdlib/`](adapters/stdlib/) is the reference implementation, at 37 lines.

## Methodology

Full definitions — how tokens are attributed, why `o200k_base`, how LOC is counted, what the mock does and does not emulate — are in [METHODOLOGY.md](METHODOLOGY.md).

---

MIT licensed. Measurements refresh weekly via [GitHub Actions](.github/workflows/measure.yml).
