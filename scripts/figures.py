"""Draw the README figures and print the accuracy numbers quoted in the README.

Timings come from docs/browser.json (scripts/browser_bench.py).

usage: uv run python scripts/figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import svg  # noqa: E402

from onnxprep.audio import whisper_log_mel  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
SR = 16_000
CLIPS = [0.37, 7.0, 30.0, 41.5]


def signal(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * (200 + 900 * t) * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def accuracy() -> dict[str, list[float]]:
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(feature_size=80)
    errors: dict[str, list[float]] = {"conv": [], "native": []}
    for stft in errors:
        session = ort.InferenceSession(whisper_log_mel(80, stft=stft).SerializeToString())
        for seconds in CLIPS:
            audio = signal(seconds)
            ref = fe(audio, sampling_rate=SR, return_tensors="np").input_features
            errors[stft].append(float(np.abs(session.run(None, {"audio": audio})[0] - ref).max()))
    return errors


def browser_chart(theme: svg.Theme) -> svg.Canvas:
    report = json.loads((ROOT / "docs" / "browser.json").read_text())
    labels = {
        ("transformers.js", 1): ("transformers.js", None),
        ("onnxprep, webgpu", 1): ("onnxprep, WebGPU", 0),
        ("onnxprep, wasm", 1): ("onnxprep, WASM", 0),
        ("onnxprep, webgpu", report["env"]["batch"]): (f"onnxprep, WebGPU, {report['env']['batch']} per call", 0),
    }
    rows = []
    for r in report["results"]:
        key = (r["name"], r["batch"])
        if key in labels and "ms_per_clip" in r:
            rows.append((labels[key][0], r["ms_per_clip"], labels[key][1]))
    rows.sort(key=lambda row: row[1])
    return svg.time_chart(rows, theme, "time per 30 s clip in Chromium, log scale (shorter is better)", log=True)


def pipeline_diagram(theme: svg.Theme) -> svg.Canvas:
    t = theme
    c = svg.Canvas(880, 200, t)
    steps = [
        ("audio", ["16 kHz mono", "any length"], False),
        ("Pad, Slice", ["to 30 s"], False),
        ("Conv", ["DFT kernels", "hop 160"], True),
        ("Mul, Add", ["power"], False),
        ("MatMul", ["mel filters"], False),
        ("Log, Max, Div", ["log scaling"], False),
        ("features", ["1 × 80 × 3000"], False),
    ]
    w, gap, y = 114, 13, 30
    for i, (name, lines, accent) in enumerate(steps):
        x = i * (w + gap)
        c.box(x, y, w, 82, name, lines, accent=accent, mono_title=0 < i < len(steps) - 1, title_size=13)
        if i:
            c.arrow([(x - gap, y + 41), (x - 2, y + 41)])
    conv_x = 2 * (w + gap) + w / 2
    c.line(conv_x, y + 82, conv_x, 150, t.series[0], width=1.2, dash="3 3")
    c.text(
        conv_x,
        166,
        "the STFT as a strided convolution, which every GPU backend implements",
        size=12,
        color=t.muted,
        anchor="middle",
    )
    return c


def main() -> None:
    errors = accuracy()
    print("max abs error:", {k: [f"{e:.1e}" for e in v] for k, v in errors.items()})
    svg.write(pipeline_diagram, "pipeline", "The Whisper front end as standard ONNX ops", OUT)
    svg.write(browser_chart, "browser", "Time per 30 second clip in the browser", OUT)
    print(f"wrote {len(list(OUT.glob('*.svg')))} files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
