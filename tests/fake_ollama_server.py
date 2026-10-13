"""A tiny local HTTP server that speaks just enough of the Ollama API for our tests.

It lets the real langchain-ollama and ollama client code run, so the tests check what we
really send (context length, temperature, JSON schema) without needing Ollama installed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllamaServer:
    def __init__(
        self,
        installed_model_names: list[str],
        chat_reply_text: str = "hello",
        chat_status_code: int = 200,
    ) -> None:
        self.installed_model_names = installed_model_names
        self.chat_reply_text = chat_reply_text
        self.chat_status_code = chat_status_code
        self.chat_requests: list[dict] = []
        self.http_server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self.server_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.http_server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        self.server_thread.start()

    def stop(self) -> None:
        self.http_server.shutdown()
        self.http_server.server_close()

    def _build_handler(self):
        fake_server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def _send_json(self, status_code: int, body: dict) -> None:
                body_bytes = (json.dumps(body) + "\n").encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

            def do_GET(self):
                if self.path == "/api/tags":
                    models = [{"model": name} for name in fake_server.installed_model_names]
                    self._send_json(200, {"models": models})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                request_body = json.loads(self.rfile.read(content_length))
                fake_server.chat_requests.append(request_body)
                if fake_server.chat_status_code != 200:
                    self._send_json(fake_server.chat_status_code, {"error": "model not found"})
                    return
                self._send_json(
                    200,
                    {
                        "model": request_body.get("model", ""),
                        "created_at": "2026-01-01T00:00:00Z",
                        "message": {"role": "assistant", "content": fake_server.chat_reply_text},
                        "done": True,
                        "done_reason": "stop",
                    },
                )

        return Handler
