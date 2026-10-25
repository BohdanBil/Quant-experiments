"""Local web dashboard for the risk report.

    python serve.py                     # http://127.0.0.1:8000, opens your browser
    python serve.py --port 8050 --no-browser

Shows the results of the last `python main.py` run (results/dashboard.json, created by running
main.py if it is missing) and runs new what-if simulations for other weights and horizons.
The server only listens on this computer (127.0.0.1).
"""
import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import main as config
from portfolio_risk import dashboard, data
from portfolio_risk.models import FilteredHistoricalModel, NormalModel
from portfolio_risk.simulation import simulate
from portfolio_risk.volatility import HAR

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
FILES = {"/api/results": ROOT / "results" / "dashboard.json",
         "/report.txt": ROOT / "results" / "report.txt"}
CONTENT_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
                 ".json": "application/json", ".txt": "text/plain", ".svg": "image/svg+xml"}
MAX_PATHS, MAX_HORIZON = 50_000, 250      # bigger runs take tens of seconds and ~1 GB of memory


class WhatIf:
    """Fits the main model once, then simulates any weights and horizon on request."""

    def __init__(self):
        ohlc = data.load_ohlc(config.TICKERS, config.START, config.END)
        self.log_returns = data.log_returns(ohlc["Close"])
        self.main = FilteredHistoricalModel(self.log_returns, window=config.FHS_WINDOW, vol=HAR(),
                                            fit_window=config.VOL_WINDOW, rv=data.range_variance(ohlc))
        self.normal = NormalModel(self.log_returns.iloc[-config.WINDOW:])
        self.lock = threading.Lock()        # one simulation at a time keeps memory use bounded

    def run(self, request):
        given = request.get("weights") or {}
        weights = pd.Series({t: float(given.get(t, 0)) for t in config.TICKERS})
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Weights must be zero or positive, and not all zero.")
        weights /= weights.sum()
        horizon, paths = int(request.get("horizon", 100)), int(request.get("paths", 25_000))
        value = float(request.get("value", config.VALUE))
        if not (1 <= horizon <= MAX_HORIZON and 1_000 <= paths <= MAX_PATHS and value > 0):
            raise ValueError(f"Horizon must be 1-{MAX_HORIZON} days, paths 1,000-{MAX_PATHS:,}, "
                             "and the value positive.")
        days = sorted({d for d in (1, 10, horizon) if d <= horizon})
        with self.lock:
            sims = [simulate(m, weights, value, horizon, paths, np.random.default_rng(config.SEED), days)
                    for m in (self.normal, self.main)]
            return dashboard.portfolio_summary(sims, sims[1], weights, value, days, self.log_returns,
                                               config.STRESS_PERIODS, config.STOP_LOSS, config.PUT_STRIKE)


class Handler(BaseHTTPRequestHandler):
    what_if = None

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        file = FILES.get(path) or (WEB / path.lstrip("/")).resolve()
        if file not in FILES.values() and WEB not in file.parents:
            return self.send_error(HTTPStatus.NOT_FOUND)     # nothing outside web/ is served
        if not file.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        self._send(file.read_bytes(), CONTENT_TYPES.get(file.suffix, "application/octet-stream"))

    def do_POST(self):
        if self.path != "/api/whatif":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            result = self.what_if.run(json.loads(body or b"{}"))
            self._send(json.dumps(dashboard.clean(result)).encode(), "application/json")
        except (ValueError, TypeError) as error:
            self._send(json.dumps({"error": str(error)}).encode(), "application/json",
                       HTTPStatus.BAD_REQUEST)

    def _send(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args = parser.parse_args()

    if not FILES["/api/results"].exists():
        print("No results yet, running main.py first (about 20 seconds)...")
        subprocess.run([sys.executable, str(ROOT / "main.py")], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    print("Fitting the model for what-if simulations...")
    Handler.what_if = WhatIf()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
