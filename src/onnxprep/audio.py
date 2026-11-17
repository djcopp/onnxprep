"""Audio front ends as standard-op ONNX graphs.

``whisper_log_mel`` reproduces OpenAI Whisper's ``log_mel_spectrogram(pad_or_trim(audio))``:
raw 16 kHz mono float32 PCM in, ``[1, n_mels, 3000]`` log-mel features out. A browser, mobile or server
app can pass microphone samples to it directly, without porting the DSP code.

Two STFT strategies:

* ``stft="conv"`` (default): the STFT is a strided ``Conv`` whose kernels are the windowed DFT basis.
  Every GPU execution provider implements ``Conv`` (CUDA, TensorRT, CoreML, DirectML, WebGPU).
* ``stft="native"``: the opset 17 ``STFT`` operator. A smaller file, but most GPU execution providers
  don't implement it, so it runs on the CPU.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import onnx

from ._graph import Graph

# Whisper constants (openai/whisper/audio.py)
SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480_000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3_000


# Filterbanks and windows. These run in numpy when the graph is built, not at inference time.


def _hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-12) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Slaney-scale, Slaney-normalised triangular filters, the ``librosa.filters.mel`` defaults (and what
    Whisper ships in ``mel_filters.npz``). Returns ``[n_mels, n_fft // 2 + 1]`` float32."""
    fmax = sr / 2 if fmax is None else fmax
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = _mel_to_hz_slaney(np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def hann_window(n: int, periodic: bool = True) -> np.ndarray:
    m = n if periodic else n - 1
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / m)).astype(np.float64)


def _dft_conv_kernels(n_fft: int, window: np.ndarray) -> np.ndarray:
    """``[2 * (n_fft//2+1), 1, n_fft]`` Conv weights: windowed cos (real) then windowed sin (imag) bases."""
    k = np.arange(n_fft // 2 + 1)[:, None]
    n = np.arange(n_fft)[None, :]
    ang = 2 * np.pi * k * n / n_fft
    real = np.cos(ang) * window[None, :]
    imag = -np.sin(ang) * window[None, :]
    return np.concatenate([real, imag], axis=0)[:, None, :].astype(np.float32)


# Graph pieces


def _pad_or_trim(g: Graph, x: str, length: int) -> str:
    """[B, samples]: zero-pad at the end or truncate to exactly ``length`` samples (branch-free)."""
    n = g.op("Shape", x, start=-1)
    need = g.op("Max", g.op("Sub", g.const([length], dtype=np.int64), n), g.const([0], dtype=np.int64))
    pads = g.op("Concat", g.const([0, 0, 0], dtype=np.int64), need, axis=0)  # [begin B, begin L, end B, end L]
    padded = g.op("Pad", x, pads, g.const(0.0, dtype=np.float32), mode="constant")
    return g.op(
        "Slice",
        padded,
        g.const([0], dtype=np.int64),
        g.const([length], dtype=np.int64),
        g.const([1], dtype=np.int64),
    )


def _power_spectrogram(g: Graph, x: str, n_fft: int, hop: int, stft: str) -> str:
    """``[B, samples]`` signals to a ``[B, n_fft//2+1, frames]`` power spectrum. Frames are centred with
    reflect padding, as in ``torch.stft(center=True, pad_mode="reflect")``."""
    bins = n_fft // 2 + 1
    window = hann_window(n_fft)
    x3 = g.op("Unsqueeze", x, g.const([1], dtype=np.int64))  # [B, 1, samples]
    x3 = g.op("Pad", x3, g.const([0, 0, n_fft // 2, 0, 0, n_fft // 2], dtype=np.int64), mode="reflect")
    if stft == "conv":
        spec = g.op(
            "Conv",
            x3,
            g.const(_dft_conv_kernels(n_fft, window), name="dft_kernels"),
            strides=[hop],
            kernel_shape=[n_fft],
            pads=[0, 0],  # explicit: CoreML's ML Program converter rejects Conv without `pads`
            dilations=[1],
            group=1,
        )  # [1, 2*bins, F]
        sq = g.op("Mul", spec, spec)
        re = g.op(
            "Slice",
            sq,
            g.const([0], dtype=np.int64),
            g.const([bins], dtype=np.int64),
            g.const([1], dtype=np.int64),
        )
        im = g.op(
            "Slice",
            sq,
            g.const([bins], dtype=np.int64),
            g.const([2 * bins], dtype=np.int64),
            g.const([1], dtype=np.int64),
        )
        return g.op("Add", re, im)  # [1, bins, F]
    if stft == "native":
        sig = g.op("Transpose", x3, perm=[0, 2, 1])  # [1, L, 1]
        spec = g.op(
            "STFT",
            sig,
            g.const(hop, dtype=np.int64),
            g.const(window.astype(np.float32)),
            g.const(n_fft, dtype=np.int64),
            onesided=1,
        )  # [1, F, bins, 2]
        sq = g.op("Mul", spec, spec)
        p = g.op("ReduceSum", sq, g.const([3], dtype=np.int64), keepdims=0)  # [1, F, bins]
        return g.op("Transpose", p, perm=[0, 2, 1])
    raise ValueError(f"stft must be 'conv' or 'native', got {stft!r}")


def whisper_log_mel(
    n_mels: int = 80,
    *,
    stft: Literal["conv", "native"] = "conv",
    pad_to_30s: bool = True,
    static_input: bool = False,
    batch: bool = False,
) -> onnx.ModelProto:
    """Whisper's log-mel front end as a standalone ONNX model.

    Input ``audio``: float32 ``[num_samples]``, mono, 16 kHz, range [-1, 1].
    Output ``input_features``: float32 ``[1, n_mels, 3000]`` (or ``[1, n_mels, num_samples // 160]`` when
    ``pad_to_30s=False``).

    Use ``n_mels=80`` for Whisper tiny to large-v2 and ``n_mels=128`` for large-v3 and turbo.

    ``static_input=True`` fixes the input at exactly 480,000 samples, and the caller pads or trims. With
    static shapes CoreML runs 15 of 17 nodes instead of 15 of 23, and the clip takes about half as long.

    ``batch=True`` takes ``[batch, num_samples]`` and returns ``[batch, n_mels, 3000]``, one clip per row.
    Every row shares ``num_samples``, so pad shorter clips with zeros (Whisper pads with zeros anyway).
    """
    if static_input and not pad_to_30s:
        raise ValueError("static_input=True implies a fixed 30 s window; use pad_to_30s=True")
    g = Graph("whisper_log_mel")
    samples = N_SAMPLES if static_input else "num_samples"
    if batch:
        x = g.input("audio", np.float32, ["batch", samples])
    else:
        x = g.op("Unsqueeze", g.input("audio", np.float32, [samples]), g.const([0], dtype=np.int64))
    if pad_to_30s and not static_input:
        x = _pad_or_trim(g, x, N_SAMPLES)
    power = _power_spectrogram(g, x, N_FFT, HOP_LENGTH, stft)
    # Whisper drops the last STFT frame: stft[..., :-1]
    power = g.op(
        "Slice",
        power,
        g.const([0], dtype=np.int64),
        g.const([-1], dtype=np.int64),
        g.const([2], dtype=np.int64),
    )
    mel = g.op("MatMul", g.const(mel_filterbank(SAMPLE_RATE, N_FFT, n_mels), name="mel_filters"), power)
    log = g.op(
        "Div",
        g.op("Log", g.op("Max", mel, g.const(1e-10, dtype=np.float32))),
        g.const(np.log(10.0), dtype=np.float32),
    )  # log10
    peak = g.op("ReduceMax", log, axes=[1, 2], keepdims=1)  # per clip
    floor = g.op("Sub", peak, g.const(8.0, dtype=np.float32))
    log = g.op("Max", log, floor)
    out = g.op("Div", g.op("Add", log, g.const(4.0, dtype=np.float32)), g.const(4.0, dtype=np.float32))
    frames = N_FRAMES if pad_to_30s else "num_frames"
    g.output(out, "input_features", np.float32, ["batch" if batch else 1, n_mels, frames])
    return g.model(
        doc=f"Whisper log-mel front end ({n_mels} mels, stft={stft}). Built by onnxprep from standard ONNX ops.",
        metadata={
            "onnxprep.recipe": "whisper_log_mel",
            "n_mels": n_mels,
            "stft": stft,
            "sample_rate": SAMPLE_RATE,
            "pad_to_30s": pad_to_30s,
            "static_input": static_input,
            "batch": batch,
        },
    )
