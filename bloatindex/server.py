"""A recording stand-in for the OpenAI and Anthropic chat APIs.

Every framework under test is pointed at this server instead of a real
provider. It plays a fixed two-turn script -- one tool call, then a final
answer -- and writes each incoming request body to a JSONL file. Those
recorded bodies are the raw material for every token number we report.

Implemented on the standard library alone. Anything we imported here would
have to be installed next to the framework under test, and a measurement
harness has no business adding to the thing it is measuring.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The scripted episode. Turn 1 asks for a tool, turn 2 answers. Frameworks
# that re-plan or retry will make more calls than this; they simply keep
# receiving the final answer, and the extra calls show up in their totals.
TOOL_NAME = "get_weather"
TOOL_ARGS = {"city": "Sao Paulo"}
FINAL_ANSWER = "It is 22C and clear in Sao Paulo."


class _Handler(BaseHTTPRequestHandler):
    # Set by the server factory below.
    recorder: "Recorder" = None  # type: ignore[assignment]

    # Silence per-request logging; it would interleave with framework output.
    def log_message(self, fmt, *args):  # noqa: A002
        pass

    # -- plumbing ---------------------------------------------------------

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks: list[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):  # noqa: N802
        # Some clients probe /v1/models on construction to validate the key.
        if self.path.rstrip("/").endswith("/models"):
            self._send(
                {
                    "object": "list",
                    "data": [
                        {"id": m, "object": "model", "owned_by": "bloat-index"}
                        for m in ("gpt-4o", "gpt-4o-mini", "claude-sonnet-4")
                    ],
                }
            )
        else:
            self._send({"status": "ok"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"_unparseable": raw.decode("utf-8", "replace")}

        anthropic = "/messages" in self.path
        turn = self.recorder.record(self.path, body, dialect="anthropic" if anthropic else "openai")

        declared = self._declared_tools(body, anthropic)

        # The first request gets a tool call, every later one ends the episode.
        # "Later" rather than "second" so a framework that re-plans still
        # terminates instead of hanging the harness.
        want_tool = turn == 0 and TOOL_NAME in declared

        # Some frameworks do not treat a plain assistant message as the end of
        # a run: smolagents, for one, expects the model to call a `final_answer`
        # tool and will otherwise loop until it hits max_steps. Answering in the
        # shape each framework actually waits for is the difference between
        # measuring its overhead and measuring our own bug.
        finisher = self._finisher(declared)
        if turn > 0 and finisher:
            name, arg = finisher
            if anthropic:
                self._send(self._anthropic_response(False, tool=(name, {arg: FINAL_ANSWER})))
            elif body.get("stream"):
                self._send_sse(self._openai_stream(False, tool=(name, {arg: FINAL_ANSWER})))
            else:
                self._send(self._openai_response(False, tool=(name, {arg: FINAL_ANSWER})))
            return

        if anthropic:
            self._send(self._anthropic_response(want_tool))
        elif body.get("stream"):
            self._send_sse(self._openai_stream(want_tool))
        else:
            self._send(self._openai_response(want_tool))

    @staticmethod
    def _declared_tools(body: dict, anthropic: bool) -> dict:
        """Map tool name -> schema for whatever the caller offered.

        A framework that never forwards its tool definitions would otherwise
        receive a call for a tool it does not know about, and crash in a way
        that looks like our bug rather than its behaviour.
        """
        tools = body.get("tools") or body.get("functions") or []
        out = {}
        for t in tools:
            if not isinstance(t, dict):
                continue
            if not anthropic and "function" in t:
                t = t["function"]
            name = t.get("name")
            if name:
                out[name] = t
        return out

    @staticmethod
    def _finisher(declared: dict) -> tuple[str, str] | None:
        """Find a terminal tool the framework expects us to call, if any.

        Returns (tool_name, first_argument_name), since the answer has to be
        passed as that tool's argument rather than as message content.
        """
        for name in ("final_answer", "FinalAnswer", "submit_answer"):
            schema = declared.get(name)
            if schema is None:
                continue
            params = schema.get("parameters") or schema.get("input_schema") or {}
            props = list((params.get("properties") or {}).keys())
            return name, (props[0] if props else "answer")
        return None

    # -- response shapes --------------------------------------------------

    def _openai_response(self, want_tool: bool, tool: tuple[str, dict] | None = None) -> dict:
        if want_tool or tool:
            name, args = tool if tool else (TOOL_NAME, TOOL_ARGS)
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_bloatindex",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": FINAL_ANSWER}
            finish = "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            # Usage is deliberately zeroed. We count tokens ourselves from the
            # recorded request, so a framework cannot influence our numbers by
            # trusting or ignoring what the provider reports.
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _openai_stream(self, want_tool: bool, tool: tuple[str, dict] | None = None) -> list[dict]:
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4o",
        }
        if want_tool or tool:
            name, args = tool if tool else (TOOL_NAME, TOOL_ARGS)
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_bloatindex",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": FINAL_ANSWER}
            finish = "stop"
        return [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        ]

    def _anthropic_response(self, want_tool: bool, tool: tuple[str, dict] | None = None) -> dict:
        if want_tool or tool:
            name, args = tool if tool else (TOOL_NAME, TOOL_ARGS)
            content = [{"type": "tool_use", "id": "toolu_bloatindex", "name": name, "input": args}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": FINAL_ANSWER}]
            stop = "end_turn"
        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": content,
            "stop_reason": stop,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


class Recorder:
    """Collects request bodies in arrival order, safe across worker threads."""

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self._lock = threading.Lock()
        self._turn = 0
        self.requests: list[dict] = []

    def record(self, path: str, body: dict, dialect: str) -> int:
        with self._lock:
            turn = self._turn
            self._turn += 1
            self.requests.append(
                {"turn": turn, "path": path, "dialect": dialect, "body": body}
            )
        return turn

    def flush(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w") as fh:
            for req in self.requests:
                fh.write(json.dumps(req) + "\n")


def serve(out_path: Path, port: int = 0) -> tuple[ThreadingHTTPServer, Recorder, int]:
    """Start the recorder on a background thread. Port 0 picks a free one."""
    recorder = Recorder(out_path)
    handler = type("BoundHandler", (_Handler,), {"recorder": recorder})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, recorder, httpd.server_address[1]


if __name__ == "__main__":  # manual smoke test: python -m bloatindex.server
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "requests.jsonl")
    httpd, rec, port = serve(out)
    print(f"recording to {out}, listening on http://127.0.0.1:{port}/v1")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rec.flush()
        print(f"\nwrote {len(rec.requests)} requests")
