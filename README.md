# onnxprep

**Portable pre/post-processing as pure standard-op ONNX graphs — raw input in, final answer out, on any runtime and any GPU.**

Every team that ships a model to the browser, a phone or an edge box re-implements its preprocessing by hand —
Whisper's log-mel front end alone has been ported to JavaScript, Rust, Mojo and more, and tiny mismatches in
padding or normalisation silently cost accuracy. `onnxprep` builds those steps as ONNX graphs using **only
standard `ai.onnx` operators** (no custom-op libraries), verified against the reference implementations, so the
exact same file runs in onnxruntime on CUDA / TensorRT / CoreML / DirectML / CPU, in the browser via
onnxruntime-web (WebGPU / WASM), and in any other ONNX runtime.

> Status: early (v0.1). First recipe: Whisper's log-mel spectrogram. More below.

## Quick start

```python
import numpy as np, onnx, onnxruntime as ort
from onnxprep.audio import whisper_log_mel

model = whisper_log_mel(n_mels=80)  # 128 for large-v3 / turbo
onnx.save(model, "whisper_frontend.onnx")  # ~0.7 MB, no custom ops

session = ort.InferenceSession("whisper_frontend.onnx")
audio = np.zeros(16_000 * 5, np.float32)  # mono 16 kHz PCM, any length
features = session.run(None, {"audio": audio})[0]  # [1, 80, 3000] — feed straight to the Whisper encoder
```

In the browser it's the same file:

```js
import * as ort from "onnxruntime-web/webgpu";
const session = await ort.InferenceSession.create("whisper_frontend.onnx", { executionProviders: ["webgpu", "wasm"] });
const { input_features } = await session.run({ audio: new ort.Tensor("float32", pcm, [pcm.length]) });
```

## Accuracy

Parity-tested against Hugging Face's `WhisperFeatureExtractor` (which matches OpenAI's implementation) across
clip lengths (0.37 s, 7 s, 30 s, 41.5 s), silence, and 80/128 mel bins:

| STFT strategy | max abs error vs reference |
|---|---|
| `stft="conv"` (default) | **3e-5** (80 mels), 5e-5 (128 mels) |
| `stft="native"` (ONNX `STFT` op) | 4.5e-4 – 3.1e-3 |

The mel filterbank matches `librosa.filters.mel` to 4e-9.

## Why the STFT is a convolution

ONNX has a native `STFT` operator, but most GPU execution providers don't implement it, so it silently falls
back to CPU — and in onnxruntime it is also slower and less precise. `onnxprep` instead expresses the STFT as a
strided `Conv` whose kernels are the windowed DFT basis. `Conv` is implemented by every accelerator backend.

Measured on an Apple Silicon Mac, one 30 s clip (`scripts/provider_report.py`):

| variant | provider | nodes on provider | ms / clip |
|---|---|---|---|
| conv | CPU | all | **2.9** |
| conv, `static_input=True` | CoreML (ML Program) | 15 / 17 | 4.7 |
| conv | CoreML (ML Program) | 15 / 23 | 8.6 |
| native `STFT` | CPU | all | 73 |
| native `STFT` | CoreML (ML Program) | 4 / 23 | 74 |

Findings so far: the conv formulation is ~25× faster than the native op on CPU; static input shapes let GPU
providers take most of the graph; and CoreML's ML Program converter rejects `Conv` nodes without an explicit
`pads` attribute, so recipes always set it. For a single small clip, Apple's CPU path is still faster than
dispatching to CoreML — GPU wins are expected for batched / longer workloads and are on the benchmark roadmap.

## Roadmap

- [x] Whisper log-mel (80 / 128 mels), conv + native STFT, parity tests
- [ ] Browser parity + benchmarks on onnxruntime-web (WebGPU and WASM)
- [ ] Image preprocessing from raw RGBA bytes: resize, letterbox, crop, normalise (YOLO, CLIP, ViT presets)
- [ ] Post-processing: softmax / top-k, box decoding + NMS, CTC greedy decode
- [ ] `onnxprep fuse`: attach pre/post graphs to an existing model → one self-contained `.onnx`
- [ ] `onnxprep doctor`: report which nodes fall off the GPU on each execution provider, and rewrite them

## Development

```bash
uv sync
uv run pytest                                  # parity tests
uv run ruff check . && uv run ruff format --check .
uv run python scripts/provider_report.py       # per-provider placement + timing
```

MIT licensed.
