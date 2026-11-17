import numpy as np
import onnx
import onnxruntime as ort
import pytest

from onnxprep.audio import N_FRAMES, mel_filterbank, whisper_log_mel

librosa = pytest.importorskip("librosa")
transformers = pytest.importorskip("transformers")

SR = 16_000
# conv STFT is float32-exact to ~1e-4; the native STFT op is noticeably less precise in onnxruntime
TOL = {"conv": 2e-4, "native": 5e-3}


def _session(model: onnx.ModelProto):
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def _reference(audio: np.ndarray, n_mels: int) -> np.ndarray:
    fe = transformers.WhisperFeatureExtractor(feature_size=n_mels)
    return fe(audio, sampling_rate=SR, return_tensors="np").input_features


def _signal(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    chirp = np.sin(2 * np.pi * (200 + 900 * t) * t)
    return (0.3 * chirp + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.mark.parametrize("n_mels", [80, 128])
def test_filterbank_matches_librosa(n_mels):
    ours = mel_filterbank(SR, 400, n_mels)
    ref = librosa.filters.mel(sr=SR, n_fft=400, n_mels=n_mels)
    np.testing.assert_allclose(ours, ref, atol=1e-7)


@pytest.mark.parametrize("stft", ["conv", "native"])
@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("seconds", [0.37, 7.0, 30.0, 41.5])
def test_matches_huggingface(stft, n_mels, seconds):
    audio = _signal(seconds)
    out = _session(whisper_log_mel(n_mels, stft=stft)).run(None, {"audio": audio})[0]
    ref = _reference(audio, n_mels)
    assert out.shape == (1, n_mels, N_FRAMES)
    np.testing.assert_allclose(out, ref, atol=TOL[stft])


def test_silence():
    audio = np.zeros(SR * 3, np.float32)
    out = _session(whisper_log_mel()).run(None, {"audio": audio})[0]
    np.testing.assert_allclose(out, _reference(audio, 80), atol=TOL["conv"])


def test_no_padding_mode_frame_count():
    audio = _signal(4.0)
    out = _session(whisper_log_mel(pad_to_30s=False)).run(None, {"audio": audio})[0]
    assert out.shape == (1, 80, audio.size // 160)
    # same frames as the padded version, except edge effects near the end of the signal
    padded = _session(whisper_log_mel()).run(None, {"audio": audio})[0]
    assert np.abs(out[..., 5:-5] - padded[..., 5 : out.shape[-1] - 5]).max() < 1.0


@pytest.mark.parametrize("stft", ["conv", "native"])
def test_only_standard_ops(stft):
    m = whisper_log_mel(stft=stft)
    assert {imp.domain for imp in m.opset_import} == {""}
    assert all(n.domain == "" for n in m.graph.node)
    onnx.checker.check_model(m, full_check=True)


def test_static_input_matches_dynamic():
    audio = _signal(30.0)
    static = _session(whisper_log_mel(static_input=True)).run(None, {"audio": audio})[0]
    np.testing.assert_allclose(static, _reference(audio, 80), atol=TOL["conv"])
    m = whisper_log_mel(static_input=True)
    assert m.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 480_000


def test_static_input_requires_padding():
    with pytest.raises(ValueError):
        whisper_log_mel(static_input=True, pad_to_30s=False)
