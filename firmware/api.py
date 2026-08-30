import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import storage


def make_handler(db):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, payload):
            body = json.dumps(payload, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/events"):
                self.respond(storage.recent(db))
            elif self.path.startswith("/stats"):
                self.respond({
                    "totals": storage.totals(db),
                    "diverted_fraction": storage.diverted_fraction(db),
                })
            elif self.path == "/health":
                self.respond({"ok": True})
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return Handler


def serve(db, port):
    server = HTTPServer(("0.0.0.0", port), make_handler(db))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
