"""Renders measurement JSON into the tables embedded in README.md.

The README is generated, not hand-maintained. Numbers in a README that are
typed by a human go stale silently; numbers written by this module go stale
loudly, because the workflow that refreshes them fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fmt_bytes(n: int) -> str:
    if n == 0:
        return "0"
    if n < 1_000_000:
        return f"{n/1e3:.0f} kB"
    return f"{n/1e6:.0f} MB"


def _fmt_ms(v) -> str:
    if v is None:
        return "--"
    return f"{v/1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def _order(results: list[dict]) -> list[dict]:
    """Cheapest first, by the number the project is actually about."""
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    ok.sort(key=lambda r: r["tokens"]["first_request"]["overhead"]["median"])
    return ok + bad


def main_table(data: dict) -> str:
    """The headline table: what one task costs in each framework."""
    rows = [
        "| framework | deps | install | import | LOC | overhead tokens | overhead % | episode tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in _order(data["results"]):
        if not r.get("ok"):
            rows.append(f"| {r['display']} | ⚠️ failed | | | | | | |")
            continue
        fr = r["tokens"]["first_request"]
        total = fr["total"]["median"]
        overhead = fr["overhead"]["median"]
        pct = 100.0 * overhead / total if total else 0.0
        link = f"[{r['display']}]({r['homepage']})" if r.get("homepage") else r["display"]
        rows.append(
            f"| {link} | {r['n_deps']} | {_fmt_bytes(r['install_bytes'])} "
            f"| {_fmt_ms(r['import_ms'])} | {r['loc']} | {overhead:,.0f} "
            f"| {pct:.0f}% | {r['tokens']['episode']['total']['median']:,.0f} |"
        )
    return "\n".join(rows)


def composition_table(data: dict) -> str:
    """Where the overhead actually comes from, per framework."""
    rows = [
        "| framework | tool schemas | system prompt | scaffold | your task | total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in _order(data["results"]):
        if not r.get("ok"):
            continue
        fr = r["tokens"]["first_request"]
        rows.append(
            f"| {r['display']} | {fr['tool_schemas']['median']:,.0f} "
            f"| {fr['system']['median']:,.0f} | {fr['scaffold']['median']:,.0f} "
            f"| {fr['task']['median']:,.0f} | {fr['total']['median']:,.0f} |"
        )
    return "\n".join(rows)


def scaling_table(small: dict, large: dict) -> str:
    """How overhead moves when the toolbox grows."""
    n_small = small["config"]["n_tools"]
    n_large = large["config"]["n_tools"]
    by_name = {r["name"]: r for r in large["results"] if r.get("ok")}

    rows = [
        f"| framework | {n_small} tools | {n_large} tools | growth |",
        "|---|---:|---:|---:|",
    ]
    for r in _order(small["results"]):
        if not r.get("ok") or r["name"] not in by_name:
            continue
        a = r["tokens"]["first_request"]["overhead"]["median"]
        b = by_name[r["name"]]["tokens"]["first_request"]["overhead"]["median"]
        rows.append(f"| {r['display']} | {a:,.0f} | {b:,.0f} | +{b - a:,.0f} |")
    return "\n".join(rows)


def stability_note(data: dict) -> str:
    """State plainly whether repeats disagreed, rather than hiding it."""
    unstable = []
    for r in data["results"]:
        if not r.get("ok"):
            continue
        fr = r["tokens"]["first_request"]["overhead"]
        if fr["min"] != fr["max"]:
            unstable.append(f"{r['display']} ({fr['min']:,}-{fr['max']:,})")
    reps = data["config"]["repeats"]
    if not unstable:
        return (
            f"Every framework produced byte-identical requests across all {reps} runs, "
            "so these figures carry no run-to-run variance."
        )
    return (
        f"Across {reps} runs, these frameworks varied their request: "
        + ", ".join(unstable)
        + ". The tables report the median."
    )


def render(readme: Path, small: dict, large: dict) -> str:
    text = readme.read_text()
    blocks = {
        "TABLE_MAIN": main_table(small),
        "TABLE_COMPOSITION": composition_table(large),
        "TABLE_SCALING": scaling_table(small, large),
        "STABILITY": stability_note(small),
    }
    # Inline scalars are written without surrounding newlines so they can sit
    # mid-sentence; the tables get their own lines.
    inline = {"NTOOLS_LARGE": str(large["config"]["n_tools"])}

    for key, value in blocks.items():
        pattern = re.compile(
            rf"(<!-- BEGIN:{key} -->)(.*?)(<!-- END:{key} -->)", re.DOTALL
        )
        if not pattern.search(text):
            raise SystemExit(f"README is missing the {key} markers")
        text = pattern.sub(rf"\1\n{value}\n\3", text)

    for key, value in inline.items():
        pattern = re.compile(
            rf"(<!-- BEGIN:{key} -->)(.*?)(<!-- END:{key} -->)", re.DOTALL
        )
        text = pattern.sub(rf"\g<1>{value}\g<3>", text)
    return text


def main() -> int:
    small = json.loads((ROOT / "results" / "tools2.json").read_text())
    large = json.loads((ROOT / "results" / "tools12.json").read_text())
    readme = ROOT / "README.md"
    readme.write_text(render(readme, small, large))
    print("README.md updated from results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
