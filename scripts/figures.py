"""Draw the README figures. Accuracy is measured here; timings come from docs/benchmark.json.

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


def accuracy_chart(errors, theme: svg.Theme) -> svg.Canvas:
    """Grouped vertical bars on a log axis: max abs error per clip length, conv vs native STFT."""
    t = theme
    width, height, left, top, bottom = 880, 300, 70, 40, 250
    c = svg.Canvas(width, height, t)
    lo, hi = 1e-6, 1e-1

    def sy(v):
        return bottom - (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * (bottom - top)

    for k in range(-6, 0):
        v = 10.0**k
        c.line(left, sy(v), width, sy(v), t.grid)
        c.text(left - 10, sy(v), f"1e{k}", size=11, color=t.muted, anchor="end")
    series = [("conv STFT (default)", "conv"), ("native STFT op", "native")]
    lx = left
    for i, (name, _) in enumerate(series):
        c.rect(lx, 8, 12, 12, t.series[i], rx=3)
        c.text(lx + 18, 14.5, name, size=12, color=t.muted)
        lx += 18 + 7.2 * len(name) + 24
    group_w = (width - left) / len(CLIPS)
    bar_w = 28
    for j, seconds in enumerate(CLIPS):
        cx = left + group_w * (j + 0.5)
        for i, (_, key) in enumerate(series):
            v = errors[key][j]
            x = cx - bar_w - 2 + i * (bar_w + 4)
            y = sy(v)
            c.add(
                f'<path d="M{x:.1f},{bottom} V{y + 4:.1f} Q{x:.1f},{y:.1f} {x + 4:.1f},{y:.1f} H{x + bar_w - 4:.1f} '
                f'Q{x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + 4:.1f} V{bottom} Z" fill="{t.series[i]}"/>'
            )
            c.text(x + bar_w / 2, y - 10, f"{v:.0e}".replace("e-0", "e-"), size=11, anchor="middle")
        c.text(cx, bottom + 18, f"{seconds:g} s clip", size=12, color=t.muted, anchor="middle")
    c.text(
        left,
        bottom + 40,
        "largest absolute error against Hugging Face WhisperFeatureExtractor, log scale",
        size=11,
        color=t.muted,
    )
    return c


def speed_chart(theme: svg.Theme) -> svg.Canvas:
    report = json.loads((ROOT / "docs" / "benchmark.json").read_text())
    rows = [(b["name"], b["ms"], None) for b in report["baselines"]]
    labels = {
        ("conv", "CPU"): ("onnxprep on CPU", 0),
        ("conv, static input", "CoreML"): ("onnxprep on CoreML", 0),
        ("native STFT", "CPU"): ("native STFT op on CPU", 1),
    }
    for r in report["onnx"]:
        key = (r["variant"], r["provider"])
        if key in labels and r["ms"]:
            rows.append((labels[key][0], r["ms"], labels[key][1]))
    rows.sort(key=lambda row: row[1])
    return svg.time_chart(rows, theme, "time per 30 s clip, log scale (shorter is better)", log=True)


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
    svg.write(lambda th: accuracy_chart(errors, th), "accuracy", "Error against the reference", OUT)
    svg.write(speed_chart, "speed", "Time per 30 second clip", OUT)
    print(f"wrote {len(list(OUT.glob('*.svg')))} files to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
