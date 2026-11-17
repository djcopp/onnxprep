"""Command line: write a recipe to an .onnx file.

onnxprep whisper -o whisper_frontend.onnx
onnxprep whisper --mels 128 --batch -o whisper_frontend_128_batch.onnx
"""

from __future__ import annotations

import argparse
import pathlib

import onnx

from onnxprep.audio import whisper_log_mel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="onnxprep", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="recipe", required=True)
    w = sub.add_parser("whisper", help="Whisper's log-mel front end: 16 kHz audio in, encoder features out")
    w.add_argument("-o", "--output", type=pathlib.Path, required=True, help="where to write the .onnx file")
    w.add_argument("--mels", type=int, choices=[80, 128], default=80, help="80 up to large-v2, 128 for large-v3/turbo")
    w.add_argument("--batch", action="store_true", help="take [batch, samples] instead of [samples]")
    w.add_argument("--static", action="store_true", help="fix the input at exactly 30 s (480,000 samples)")
    w.add_argument("--no-pad", action="store_true", help="don't pad or trim to 30 s; output samples // 160 frames")
    w.add_argument("--native-stft", action="store_true", help="use the STFT operator instead of a convolution")
    args = parser.parse_args(argv)

    model = whisper_log_mel(
        args.mels,
        stft="native" if args.native_stft else "conv",
        pad_to_30s=not args.no_pad,
        static_input=args.static,
        batch=args.batch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
