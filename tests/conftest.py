import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.config import ModelSpec
from inference.encoder import XlsrEncoder
from inference.predictor import Predictor


def make_wav_bytes(seconds=6.0, sr=44100, channels=2, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))
    y = np.stack([y, 0.5 * y], axis=1) if channels == 2 else y
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_spec():
    return ModelSpec(name="tiny", n_layers=3, hidden=32, head_dim=8, min_duration_s=1.0, max_duration_s=30.0)


@pytest.fixture(scope="session")
def tiny_predictor(tiny_spec):
    return Predictor.load(tiny_spec, encoder=XlsrEncoder.tiny_random(hidden=32, layers=2))
