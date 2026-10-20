import pytest

from inference.audio import AudioError
from tests.conftest import make_wav_bytes


def test_predict_bytes_stub(tiny_predictor):
    pred = tiny_predictor.predict_bytes(make_wav_bytes(seconds=3, channels=2))
    assert tiny_predictor.stub and pred.stub
    assert pred.label in ("HC", "AD")
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert pred.duration_s == pytest.approx(3.0, abs=0.01)


def test_deterministic(tiny_predictor):
    d = make_wav_bytes(seconds=2)
    assert tiny_predictor.predict_bytes(d).probabilities == tiny_predictor.predict_bytes(d).probabilities


def test_too_short(tiny_predictor):
    with pytest.raises(AudioError):
        tiny_predictor.predict_bytes(make_wav_bytes(seconds=0.3))
