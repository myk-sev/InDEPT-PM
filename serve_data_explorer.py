"""Serve the PurpleAir data explorer with a local backup endpoint."""

from __future__ import annotations

import argparse
import json
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from backup_core_data import create_backup


ROOT = Path(__file__).resolve().parent


class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def backup_status(root: Path) -> dict[str, object]:
    manifest_path = root / "backups/manifest.json"
    if not manifest_path.is_file():
        return {"last_run_utc": None, "file_count": 0, "total_bytes": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "last_run_utc": manifest["created_at_utc"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def explorer_server(root: Path = ROOT, port: int = 8766) -> ThreadingHTTPServer:
    explorer = root / "purpleair_pair_exclusions/results"
    backup_lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if urlsplit(self.path).path == "/api/backup-status":
                try:
                    self.send_json(200, backup_status(root))
                except (KeyError, OSError, json.JSONDecodeError) as problem:
                    self.send_json(500, {"error": f"Backup status unavailable: {problem}"})
                return
            super().do_GET()

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/backup":
                self.send_json(404, {"error": "Not found"})
                return
            request_host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if urlsplit(f"//{request_host}").hostname not in {"127.0.0.1", "localhost"} or (
                origin and urlsplit(origin).netloc != request_host
            ):
                self.send_json(403, {"error": "Cross-origin backup request rejected"})
                return
            if not backup_lock.acquire(blocking=False):
                self.send_json(409, {"error": "A backup is already running"})
                return
            try:
                create_backup(root)
                self.send_json(200, backup_status(root))
            except (OSError, ValueError) as problem:
                self.send_json(500, {"error": f"Backup failed: {problem}"})
            finally:
                backup_lock.release()

        def log_message(self, message: str, *args: object) -> None:
            print(f"{self.address_string()} - {message % args}")

    server = LocalHTTPServer(
        ("127.0.0.1", port), partial(Handler, directory=str(explorer))
    )
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = explorer_server(port=args.port)
    url = f"http://127.0.0.1:{server.server_port}/location_history_explorer.html"
    print(f"Data explorer: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
