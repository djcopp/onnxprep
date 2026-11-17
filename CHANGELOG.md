# Changelog

## 0.2.0

- `whisper_log_mel(batch=True)`: many clips per call, `[batch, samples]` in, `[batch, n_mels, 3000]` out.
- `onnxprep` command line tool for writing models without Python code.
- Models now load in onnxruntime-web and older onnxruntime releases (IR version 8).
- Prebuilt models attached to each GitHub release.
- In-browser demo and a browser benchmark against transformers.js.

## 0.1.0

- Whisper's log-mel front end (80 or 128 mel bins) as a standard-op ONNX graph, matching Hugging Face's
  feature extractor to within 2e-5.
