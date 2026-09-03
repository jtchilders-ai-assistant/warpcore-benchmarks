"""Fake OpenAI endpoint that reproduces the ISSUES #15 defect over real HTTP.

Fixtures prove classify() is right; this proves the whole HTTP path is right --
request shape, response parsing, exit code -- without needing a mis-configured
GPU. Serves `content: null` with the answer in `reasoning`, exactly as vLLM's
poolside_v1 parser did on Laguna.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "fake/broken-parser-model"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # keep test output clean

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"object": "list", "data": [{"id": MODEL}]})
        else:
            self.send_error(404)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        # The defect: stopped normally, billed tokens, no content, answer hidden.
        self._send({
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "2 cm/hour x 4 hours = 8 cm\n\nThe answer is 8",
                },
            }],
            "usage": {"completion_tokens": 129},
        })


if __name__ == "__main__":
    srv = HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"serving on {srv.server_address[1]}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
