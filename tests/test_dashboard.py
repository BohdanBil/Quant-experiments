import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd

import serve
from portfolio_risk import dashboard


class CleanTest(unittest.TestCase):
    def test_makes_results_json_friendly(self):
        raw = {"a": np.float64(1.23456789), "b": np.array([1.0, np.nan]), "c": np.int64(3),
               "d": pd.Timestamp("2026-10-23"), 1: [np.bool_(True), None]}
        cleaned = dashboard.clean(raw)
        self.assertEqual(cleaned, {"a": 1.23457, "b": [1.0, None], "c": 3, "d": "2026-10-23",
                                   "1": [True, None]})
        json.dumps(cleaned, allow_nan=False)        # strict JSON: no NaN or numpy types left


class ServerTest(unittest.TestCase):
    """The server must serve the page and results, and nothing else from the project."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def status(self, path):
        try:
            with urllib.request.urlopen(self.base + path) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_serves_the_page(self):
        self.assertEqual(self.status("/"), 200)
        self.assertEqual(self.status("/app.js"), 200)

    def test_refuses_files_outside_web(self):
        for path in ("/../serve.py", "/%2e%2e/serve.py", "/../main.py", "/web/../main.py", "/missing.js"):
            self.assertEqual(self.status(path), 404, path)


if __name__ == "__main__":
    unittest.main()
