"""Which nodes of each recipe stay on each execution provider, and how fast is it?

usage: uv run python scripts/provider_report.py
"""

import json
import re
import subprocess
import sys
import tempfile

import onnxruntime as ort

from onnxprep.audio import whisper_log_mel

PROBE = r"""
import sys, time, numpy as np, onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 0
import json
prov, opts = sys.argv[2], json.loads(sys.argv[3])
s = ort.InferenceSession(sys.argv[1], so, providers=[(prov, opts), "CPUExecutionProvider"])
x = (np.random.default_rng(0).standard_normal(480000) * 0.1).astype(np.float32)
for _ in range(3): s.run(None, {"audio": x})
t = time.perf_counter(); n = 20
for _ in range(n): s.run(None, {"audio": x})
print(f"TIME_MS {(time.perf_counter() - t) / n * 1000:.2f}", flush=True)
"""

PROVIDER_OPTIONS = {"CoreMLExecutionProvider": {"ModelFormat": "MLProgram"}}
VARIANTS = {
    "conv": dict(stft="conv"),
    "conv static_input": dict(stft="conv", static_input=True),
    "native": dict(stft="native"),
}

providers = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
print(f"{'whisper_log_mel variant':24s} {'provider':26s} {'nodes on EP':>12s} {'ms / 30s clip':>14s}")
for label, kw in VARIANTS.items():
    m = whisper_log_mel(80, **kw)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(m.SerializeToString())
    for prov in providers:
        opts = json.dumps(PROVIDER_OPTIONS.get(prov, {}))
        r = subprocess.run([sys.executable, "-c", PROBE, f.name, prov, opts], capture_output=True, text=True)
        log = r.stdout + r.stderr
        t = re.search(r"TIME_MS ([\d.]+)", log)
        cap = re.findall(r"number of nodes in the graph: (\d+) number of nodes supported by \w+: (\d+)", log)
        nodes = f"{cap[-1][1]}/{cap[-1][0]}" if cap else ("all" if prov == "CPUExecutionProvider" else "?")
        print(f"{label:24s} {prov:26s} {nodes:>12s} {t.group(1) if t else 'ERR':>14s}")
