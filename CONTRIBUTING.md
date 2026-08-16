# Contributing

Two kinds of contribution matter most here: **adding a framework**, and **telling us an adapter is unfair to yours**.

## "Your adapter misrepresents my framework"

This is the most valuable issue you can open, and it will be treated as a bug rather than an argument.

The benchmark is only worth anything if each adapter is what a competent user of that framework would actually write, following its own documentation. If ours is not, we want the patch.

Helpful issues include: what the adapter does, what it should do instead, and ideally a PR. If the fix changes the numbers, that is fine — the numbers are outputs, not commitments.

What will *not* be accepted is a change that makes a framework look better by giving it an easier task: fewer tools, a shorter prompt, a trimmed system message that the default configuration would still send. The comparison rests on every framework receiving identical input.

If your framework's default system prompt is large and you would rather it were not counted, the honest fix is to change the default, not the benchmark.

## Adding a framework

Create `adapters/<name>/` with three files.

**`meta.json`**

```json
{
  "name": "yourframework",
  "display": "your-framework",
  "homepage": "https://github.com/you/yourframework",
  "import_module": "yourframework",
  "note": "Anything a reader should know about how it was configured."
}
```

`import_module` is what gets timed for cold import — use the module a real user imports to build an agent, not the cheapest one available.

**`requirements.txt`** — the install as your docs recommend it. Prefer the slim extra if you publish one, and say so in `note`.

**`task.py`** — reads three environment variables:

| variable | meaning |
|---|---|
| `BLOATINDEX_BASE_URL` | OpenAI-compatible base URL to point the client at |
| `BLOATINDEX_SPEC` | path to `adapters/spec.json` |
| `BLOATINDEX_N_TOOLS` | how many tools from the spec to declare |

It must build an agent with the first `N_TOOLS` tools from the spec, run `spec["task_prompt"]`, and **print the final answer to stdout**. The harness checks that the answer contains `22C`, which is how a silently broken adapter gets caught.

[`adapters/stdlib/task.py`](adapters/stdlib/task.py) is the reference implementation at 37 lines. [`adapters/pydantic_ai/task.py`](adapters/pydantic_ai/task.py) shows the pattern for frameworks that derive schemas from function signatures.

Then:

```bash
.venv/bin/python -m bloatindex.measure yourframework --tools 2 --repeats 3
.venv/bin/python -m bloatindex.measure yourframework --tools 12 --repeats 3
.venv/bin/python -m bloatindex.report
```

Commit the adapter, the refreshed `results/*.json`, and the regenerated `README.md`.

### If your framework will not terminate

Some frameworks have their own terminal convention — a `final_answer` tool, a structured-output contract — and will loop against a mock that just returns text. `bloatindex/server.py` handles the `final_answer` case; if yours differs, open an issue rather than working around it in the adapter. Getting this wrong produces spectacular and completely fake numbers, which is a problem we have already had once.

## Ground rules for the numbers

- Never hand-edit a table in `README.md`. Run `python -m bloatindex.report`.
- Never commit results measured on a machine other than the one that ran every adapter in that file — cross-machine rows are not comparable.
- If you change how something is measured, say so in `METHODOLOGY.md` in the same PR.
