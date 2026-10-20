import subprocess

import librosa
import numpy as np
import pytest

from inference.audio import AudioError, decode_audio, validate_duration
from tests.conftest import make_wav_bytes


def test_decode_matches_librosa_load(tmp_path):
    data = make_wav_bytes(seconds=3, sr=44100, channels=2)
    p = tmp_path / "a.wav"
    p.write_bytes(data)
    ref, _ = librosa.load(p, sr=16000, mono=True)
    got = decode_audio(data)
    assert got.dtype == np.float32 and got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_decode_mp3_and_webm(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(make_wav_bytes(seconds=2))
    for ext, codec in (("mp3", "libmp3lame"), ("webm", "libopus")):
        out = tmp_path / f"a.{ext}"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", codec, str(out)], check=True)
        y = decode_audio(out.read_bytes())
        assert abs(len(y) / 16000 - 2.0) < 0.2


def test_decode_garbage():
    with pytest.raises(AudioError):
        decode_audio(b"not audio at all")


def test_validate_duration():
    sr = 16000
    ok = np.ones(sr * 6, dtype=np.float32) * 0.1
    assert validate_duration(ok, sr, 5, 120) == pytest.approx(6.0)
    with pytest.raises(AudioError, match="corto"):
        validate_duration(ok[: sr * 2], sr, 5, 120)
    with pytest.raises(AudioError, match="largo"):
        validate_duration(ok, sr, 1, 5)
    with pytest.raises(AudioError, match="silencio"):
        validate_duration(np.zeros(sr * 6, dtype=np.float32), sr, 5, 120)
