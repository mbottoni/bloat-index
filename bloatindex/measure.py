"""Runs every adapter in its own environment and records what it cost.

One virtualenv per adapter, built by uv. Nothing is shared between them, so
a heavy framework cannot make a light one look worse by leaving transitive
dependencies lying around.

Each adapter is measured `repeats` times. Token counts are usually identical
run to run, but some frameworks stamp a timestamp or a UUID into the prompt,
and a benchmark that reported a single sample would hide that. We publish the
median and the observed range instead.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import tokens
from .server import serve

ROOT = Path(__file__).resolve().parent.parent
ADAPTERS = ROOT / "adapters"
VENVS = ROOT / ".venvs"
SPEC_PATH = ADAPTERS / "spec.json"
PYTHON = "3.12"

# Import time is measured as the median of this many cold subprocesses.
IMPORT_SAMPLES = 5
RUN_TIMEOUT = 300


@dataclass
class AdapterResult:
    name: str
    display: str
    homepage: str = ""
    note: str = ""
    ok: bool = False
    error: str = ""
    n_deps: int = 0
    install_bytes: int = 0
    import_ms: float | None = None
    import_ms_range: list[float] = field(default_factory=list)
    loc: int = 0
    install_seconds: float = 0.0
    runs: list[dict] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    answer_ok: bool = False

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("runs", None)
        d["n_runs"] = len(self.runs)
        return d


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _venv_python(name: str) -> Path:
    return VENVS / name / "bin" / "python"


def _dir_bytes(path: Path) -> int:
    """On-disk size of site-packages, excluding bytecode caches.

    __pycache__ is skipped deliberately. It is written lazily the first time
    a module is imported, so counting it would make the figure depend on how
    many times the adapter had been run -- an early version of this harness
    reported langchain at 38 MB on a clean venv and 53 MB after a few runs.
    What we report is what the install puts on disk.
    """
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts:
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _site_packages(name: str) -> Path | None:
    for p in (VENVS / name / "lib").glob("python*/site-packages"):
        return p
    return None


def _loc(path: Path) -> int:
    """Non-blank, non-comment, non-docstring lines of the adapter task.

    Counts what a developer must actually write to express the task. Module
    docstrings in our adapters are commentary for readers of this repo, not
    part of the program, so they are excluded.
    """
    import ast

    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None

    skip: set[int] = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc and node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0]
                    skip.update(range(expr.lineno, (expr.end_lineno or expr.lineno) + 1))

    n = 0
    for i, line in enumerate(src.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or i in skip:
            continue
        n += 1
    return n


def install(name: str, force: bool = False) -> tuple[bool, str, float]:
    """Build the adapter's virtualenv. Returns (ok, message, seconds)."""
    venv = VENVS / name
    req = ADAPTERS / name / "requirements.txt"
    if force and venv.exists():
        shutil.rmtree(venv)

    started = time.perf_counter()
    if not venv.exists():
        r = _run(["uv", "venv", str(venv), "--python", PYTHON])
        if r.returncode != 0:
            return False, f"uv venv failed: {r.stderr.strip()[:400]}", 0.0

    if req.exists() and req.read_text().strip():
        r = _run(
            [
                "uv", "pip", "install",
                "--python", str(_venv_python(name)),
                "-r", str(req),
            ]
        )
        if r.returncode != 0:
            return False, f"install failed: {r.stderr.strip()[:400]}", 0.0

    return True, "", time.perf_counter() - started


def count_deps(name: str) -> int:
    r = _run(["uv", "pip", "list", "--python", str(_venv_python(name)), "--format", "json"])
    if r.returncode != 0:
        return 0
    try:
        pkgs = json.loads(r.stdout)
    except json.JSONDecodeError:
        return 0
    # uv seeds nothing into a bare venv, so every entry here was pulled in by
    # the adapter's own requirements.
    return len(pkgs)


def measure_import(name: str, module: str | None) -> tuple[float | None, list[float]]:
    """Median wall time to import the framework's entry point, cold.

    Interpreter startup is excluded: the timer runs inside the child process
    and brackets the import statement only. Each sample is a fresh process so
    nothing is served from an already-warm module cache.
    """
    if not module:
        return None, []
    py = _venv_python(name)
    code = (
        "import time,importlib;"
        f"t=time.perf_counter();importlib.import_module({module!r});"
        "print((time.perf_counter()-t)*1000)"
    )
    samples = []
    for _ in range(IMPORT_SAMPLES):
        r = _run([str(py), "-c", code])
        if r.returncode == 0:
            try:
                samples.append(float(r.stdout.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                pass
    if not samples:
        return None, []
    return round(statistics.median(samples), 1), [round(min(samples), 1), round(max(samples), 1)]


def run_task(name: str, n_tools: int, run_idx: int) -> dict:
    """Execute the adapter once against a fresh recording server."""
    out = ROOT / "results" / "raw" / f"{name}.tools{n_tools}.run{run_idx}.jsonl"
    httpd, recorder, port = serve(out)
    try:
        env = {
            **os.environ,
            "BLOATINDEX_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "BLOATINDEX_SPEC": str(SPEC_PATH),
            "BLOATINDEX_N_TOOLS": str(n_tools),
            # Keep frameworks from phoning home; telemetry would add requests
            # that have nothing to do with the task.
            "OPENAI_API_KEY": "sk-bloatindex",
            "ANTHROPIC_API_KEY": "sk-bloatindex",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
            "CREWAI_DISABLE_TELEMETRY": "true",
            "OTEL_SDK_DISABLED": "true",
            "ANONYMIZED_TELEMETRY": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
        }
        started = time.perf_counter()
        proc = _run(
            [str(_venv_python(name)), str(ADAPTERS / name / "task.py")],
            env=env,
            cwd=str(ADAPTERS / name),
            timeout=RUN_TIMEOUT,
        )
        elapsed = time.perf_counter() - started
    finally:
        httpd.shutdown()
        recorder.flush()

    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "seconds": round(elapsed, 2),
        "requests": recorder.requests,
    }


def measure(name: str, n_tools: int = 2, repeats: int = 3, force: bool = False) -> AdapterResult:
    meta = json.loads((ADAPTERS / name / "meta.json").read_text())
    res = AdapterResult(
        name=name,
        display=meta.get("display", name),
        homepage=meta.get("homepage", ""),
        note=meta.get("note", ""),
    )

    ok, err, secs = install(name, force=force)
    if not ok:
        res.error = err
        return res
    res.install_seconds = round(secs, 1)

    res.n_deps = count_deps(name)
    sp = _site_packages(name)
    res.install_bytes = _dir_bytes(sp) if sp else 0
    res.import_ms, res.import_ms_range = measure_import(name, meta.get("import_module"))
    res.loc = _loc(ADAPTERS / name / "task.py")

    spec = json.loads(SPEC_PATH.read_text())
    prompt = spec["task_prompt"]

    for i in range(repeats):
        run = run_task(name, n_tools, i)
        res.runs.append(run)
        if not run["ok"]:
            res.error = run["stderr"][-600:] or "task exited non-zero"
            return res

    # Token numbers come from the recorded traffic, summarised per run and
    # then reduced across runs so the report can show a spread.
    summaries = [tokens.summarise(r["requests"], prompt) for r in res.runs]
    good = [s for s in summaries if "error" not in s]
    if not good:
        res.error = "no requests were recorded -- did the adapter reach the server?"
        return res

    def med(path):
        vals = []
        for s in good:
            cur = s
            for key in path:
                cur = cur[key]
            vals.append(cur)
        return {
            "median": round(statistics.median(vals), 1),
            "min": min(vals),
            "max": max(vals),
        }

    res.tokens = {
        "n_requests": med(["n_requests"]),
        "first_request": {
            k: med(["first_request", k])
            for k in ("total", "overhead", "task", "tool_schemas", "system", "scaffold", "history")
        },
        "episode": {
            k: med(["episode", k])
            for k in ("total", "overhead", "task", "tool_schemas", "system", "scaffold", "history")
        },
        "sample": good[0],
    }

    last = res.runs[-1]["stdout"]
    res.answer_ok = spec["expected_answer_contains"].lower() in last.lower()
    res.ok = True
    return res


def discover() -> list[str]:
    return sorted(
        p.name for p in ADAPTERS.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="bloatindex", description="Measure agent framework overhead.")
    ap.add_argument("adapters", nargs="*", help="adapter names (default: all)")
    ap.add_argument("--tools", type=int, default=2, help="how many tools to expose")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--force-reinstall", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    names = args.adapters or discover()
    results = []
    for name in names:
        print(f"[{name}] measuring ...", flush=True)
        r = measure(name, n_tools=args.tools, repeats=args.repeats, force=args.force_reinstall)
        status = "ok" if r.ok else f"FAILED: {r.error.splitlines()[0][:120] if r.error else '?'}"
        extra = ""
        if r.ok:
            fr = r.tokens["first_request"]
            extra = f" | {r.n_deps} deps, {r.install_bytes/1e6:.0f} MB, {fr['overhead']['median']:.0f} overhead tokens"
        print(f"[{name}] {status}{extra}", flush=True)
        results.append(r)

    payload = {
        "spec": json.loads(SPEC_PATH.read_text()),
        "config": {
            "n_tools": args.tools,
            "repeats": args.repeats,
            "python": PYTHON,
            "tokenizer": "o200k_base",
            "import_samples": IMPORT_SAMPLES,
        },
        "results": [r.as_dict() for r in results],
    }
    out = Path(args.out) if args.out else ROOT / "results" / f"tools{args.tools}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
