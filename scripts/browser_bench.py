"""Time the Whisper front end in a browser: transformers.js against onnxprep on WebGPU and WASM.

usage: uv run python scripts/browser_bench.py [--json docs/browser.json]

Serves bench/ with cross-origin isolation (so WASM gets all threads) and drives it in headless Chromium.
Set CHROMIUM_PATH to use a specific browser; otherwise run `uv run playwright install chromium` once.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import pathlib
import tempfile
import threading

import onnx
from playwright.sync_api import sync_playwright

from onnxprep.audio import whisper_log_mel

ROOT = pathlib.Path(__file__).resolve().parent.parent


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="also write the results to this file")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        site = pathlib.Path(tmp)
        (site / "index.html").write_text((ROOT / "bench" / "index.html").read_text())
        for name, model in [("single", whisper_log_mel()), ("batch", whisper_log_mel(batch=True))]:
            model.ir_version = min(model.ir_version, 9)  # onnxruntime-web 1.30 refuses newer IR versions
            onnx.save(model, site / f"{name}.onnx")
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(IsolatedHandler, directory=tmp))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=os.environ.get("CHROMIUM_PATH") or None,
                args=["--enable-unsafe-webgpu", "--disable-background-timer-throttling"],
            )
            page = browser.new_page()
            page.on("console", lambda m: m.text.startswith("[bench]") and print(" ", m.text[8:], flush=True))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html?batch={args.batch}")
            page.wait_for_function("window.benchResults !== undefined", timeout=600_000, polling=1000)
            report = page.evaluate("window.benchResults")
            browser.close()
        server.shutdown()
    if "error" in report:
        raise SystemExit(report["error"])
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
