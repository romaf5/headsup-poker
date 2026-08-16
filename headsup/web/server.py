"""Dependency-free HTTP server: JSON API around :class:`GameSession` + the static web UI.

    python -m headsup.web                     # opens http://127.0.0.1:8000/
    python -m headsup.web --opponent onnx --port 8080 --no-browser
"""

import argparse
import glob
import json
import mimetypes
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from headsup.paths import DEFAULT_ONNX_PATH, DEFAULT_POLICY_PATH, MODELS_DIR, ROOT
from headsup.web.session import GameSession

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SIMPLE_OPPONENTS = [
    ("cfr", "DeepCFR policy (models/deepcfr_policy.pth)"),
    ("onnx", "PPO exploiter (models/rl_games_exploiter.onnx)"),
    ("random", "Random bot"),
    ("call", "Calling station"),
    ("allin", "Maniac (always all-in)"),
    ("raise", "Always raise"),
]


def available_players():
    """Player specs offered in the settings dialog: built-ins + policies found on disk."""
    out = [{"spec": s, "label": l} for s, l in SIMPLE_OPPONENTS]
    seen = {DEFAULT_POLICY_PATH, DEFAULT_ONNX_PATH}
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")) + glob.glob(os.path.join(ROOT, "runs", "*", "policy.pth"))):
        if path in seen:
            continue
        seen.add(path)
        rel = os.path.relpath(path, ROOT)
        out.append({"spec": f"cfr:{rel}", "label": f"DeepCFR policy ({rel})"})
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx"))):
        if path in seen:
            continue
        rel = os.path.relpath(path, ROOT)
        out.append({"spec": f"onnx:{rel}", "label": f"ONNX exploiter ({rel})"})
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*iterates*.pt")) + glob.glob(os.path.join(ROOT, "runs", "*", "iterates.pt"))):
        rel = os.path.relpath(path, ROOT)
        out.append({"spec": f"sdcfr:{rel}", "label": f"SD-CFR average strategy ({rel})"})
    return out


class SessionHolder:
    """Mutable reference so /api/new can swap the session under the handler."""

    def __init__(self, session):
        self.session = session
        self.lock = threading.RLock()


def make_handler(holder: SessionHolder):
    class Handler(BaseHTTPRequestHandler):
        server_version = "headsup-poker/0.2"

        def log_message(self, fmt, *args):
            if os.environ.get("HEADSUP_WEB_LOG"):
                super().log_message(fmt, *args)

        def _send(self, body, ctype, status=200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=200):
            self._send(json.dumps(payload).encode(), "application/json", status)

        def _static(self, name):
            path = os.path.normpath(os.path.join(STATIC_DIR, name))
            if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
                return self.send_error(404)
            with open(path, "rb") as f:
                body = f.read()
            self._send(body, mimetypes.guess_type(path)[0] or "application/octet-stream")

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            url = urlparse(self.path)
            path = url.path
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            q = parse_qs(url.query)
            try:
                with holder.lock:
                    s = holder.session
                    if path == "/api/state":
                        return self._json(s.state(reveal_bot=q.get("reveal", ["0"])[0] == "1"))
                    if path == "/api/advice":
                        return self._json(s.advice())
                    if path == "/api/players":
                        return self._json({"players": available_players()})
            except Exception as exc:
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._body()
                with holder.lock:
                    s = holder.session
                    if path == "/api/action":
                        return self._json(s.act(body["action"]))
                    if path == "/api/next":
                        return self._json(s.next_hand())
                    if path == "/api/bot_step":
                        return self._json(s.bot_step())
                    if path == "/api/agent_step":
                        return self._json(s.agent_step(bool(body.get("deterministic", False))))
                    if path == "/api/new":
                        cfg = dict(holder.session.settings)
                        for k in ("opponent", "advisor", "stack_size", "small_blind", "big_blind", "raise_cap", "seed", "deterministic"):
                            if k in body:
                                cfg[k] = body[k]
                        seed = cfg.get("seed")
                        cfg["seed"] = int(seed) if seed not in (None, "") else None
                        cfg["advisor"] = cfg.get("advisor") or None
                        holder.session = GameSession(device=holder.session.device, **cfg)
                        return self._json(holder.session.state())
                self.send_error(404)
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(session, host="127.0.0.1", port=8000, open_browser=True):
    holder = SessionHolder(session)
    httpd = ThreadingHTTPServer((host, port), make_handler(holder))
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"Heads-up poker table open at {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opponent", default="cfr", help="bot: cfr[:path.pth] | onnx[:path.onnx] | random | call | allin | raise")
    p.add_argument("--advisor", default=DEFAULT_POLICY_PATH, help="policy used for advice/autoplay ('' to disable)")
    p.add_argument("--stack", type=int, default=100)
    p.add_argument("--small-blind", type=int, default=1)
    p.add_argument("--big-blind", type=int, default=2)
    p.add_argument("--raise-cap", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true", help="bot plays argmax instead of sampling")
    p.add_argument("--device", default=None, help="torch device for the bot/advisor (default: auto)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args(argv)
    session = GameSession(
        opponent=a.opponent,
        advisor=a.advisor or None,
        stack_size=a.stack,
        small_blind=a.small_blind,
        big_blind=a.big_blind,
        raise_cap=a.raise_cap,
        seed=a.seed,
        deterministic=a.deterministic,
        device=a.device,
    )
    if session.advisor_error:
        print(f"(advisor disabled: {session.advisor_error})")
    serve(session, a.host, a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
