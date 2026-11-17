"""Write the prebuilt models that ship with each GitHub release.

usage: uv run python scripts/build_models.py [out_dir]
"""

import pathlib
import sys

import onnx

from onnxprep.audio import whisper_log_mel

VARIANTS = {
    "whisper_frontend_80.onnx": {"n_mels": 80},
    "whisper_frontend_128.onnx": {"n_mels": 128},
    "whisper_frontend_80_batch.onnx": {"n_mels": 80, "batch": True},
    "whisper_frontend_128_batch.onnx": {"n_mels": 128, "batch": True},
}

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "dist/models")
    out.mkdir(parents=True, exist_ok=True)
    for name, kwargs in VARIANTS.items():
        onnx.save(whisper_log_mel(**kwargs), out / name)
        print(f"{out / name}: {(out / name).stat().st_size / 1e6:.1f} MB")
