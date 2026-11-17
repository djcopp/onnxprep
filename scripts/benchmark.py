"""Time the Whisper front end against Hugging Face's feature extractor, and show where each node runs.

usage: uv run python scripts/benchmark.py [--json docs/benchmark.json]

Each ONNX configuration runs in a subprocess with verbose logging, which is the only place onnxruntime
reports how many nodes an execution provider took.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort

from onnxprep.audio import N_SAMPLES, SAMPLE_RATE, whisper_log_mel

PROBE = r"""
import json, sys, time
import numpy as np, onnxruntime as ort
path, provider, options = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
so = ort.SessionOptions()
so.log_severity_level = 0
providers = [(provider, options)] + (["CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [])
s = ort.InferenceSession(path, so, providers=providers)
x = (np.random.default_rng(0).standard_normal(480_000) * 0.1).astype(np.float32)
for _ in range(3):
    s.run(None, {"audio": x})
times = []
for _ in range(30):
    t = time.perf_counter()
    s.run(None, {"audio": x})
    times.append((time.perf_counter() - t) * 1000)
times.sort()
print(f"MEDIAN_MS {times[len(times) // 2]:.3f}", flush=True)
"""

PROVIDER_OPTIONS = {"CoreMLExecutionProvider": {"ModelFormat": "MLProgram"}}
VARIANTS = {
    "conv": {"stft": "conv"},
    "conv, static input": {"stft": "conv", "static_input": True},
    "native STFT": {"stft": "native"},
}


def median_ms(fn, warmup: int = 3, reps: int = 30) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def baselines(audio: np.ndarray) -> list[dict]:
    """Hugging Face's two implementations, called directly. Its __call__ silently picks the torch one
    whenever torch is installed, so timing __call__ alone says nothing about the numpy path."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=80)
    batch = audio[None]
    rows = [{"name": "Hugging Face, numpy", "ms": median_ms(lambda: fe._np_extract_fbank_features(batch, "cpu"))}]
    try:
        import torch

        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        devices += ["cuda"] if torch.cuda.is_available() else []
        for device in devices:
            ms = median_ms(lambda d=device: fe._torch_extract_fbank_features(batch, d))
            rows.append({"name": f"Hugging Face, torch {device}", "ms": ms})
    except ImportError:
        pass
    return rows


def onnx_rows() -> list[dict]:
    providers = [p for p in ort.get_available_providers() if p != "AzureExecutionProvider"]
    rows = []
    for label, kwargs in VARIANTS.items():
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(whisper_log_mel(80, **kwargs).SerializeToString())
        for provider in providers:
            options = json.dumps(PROVIDER_OPTIONS.get(provider, {}))
            run = subprocess.run(
                [sys.executable, "-c", PROBE, f.name, provider, options], capture_output=True, text=True
            )
            log = run.stdout + run.stderr
            ms = re.search(r"MEDIAN_MS ([\d.]+)", log)
            placed = re.findall(r"number of nodes in the graph: (\d+) number of nodes supported by \w+: (\d+)", log)
            rows.append(
                {
                    "variant": label,
                    "provider": provider.removesuffix("ExecutionProvider"),
                    "nodes": f"{placed[-1][1]}/{placed[-1][0]}" if placed else "all",
                    "ms": float(ms.group(1)) if ms else None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()

    audio = (np.random.default_rng(0).standard_normal(N_SAMPLES) * 0.1).astype(np.float32)
    report = {
        "machine": f"{platform.system()} {platform.machine()}",
        "onnxruntime": ort.__version__,
        "clip_seconds": N_SAMPLES / SAMPLE_RATE,
        "baselines": baselines(audio),
        "onnx": onnx_rows(),
    }
    print(f"One 30 s clip, median of 30 runs ({report['machine']}, onnxruntime {report['onnxruntime']})\n")
    for b in report["baselines"]:
        print(f"  {b['name']:34s} {b['ms']:8.2f} ms")
    print()
    print(f"  {'onnxprep variant':22s} {'provider':10s} {'nodes on provider':>18s} {'ms':>8s}")
    for r in report["onnx"]:
        ms = f"{r['ms']:.2f}" if r["ms"] is not None else "failed"
        print(f"  {r['variant']:22s} {r['provider']:10s} {r['nodes']:>18s} {ms:>8s}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
