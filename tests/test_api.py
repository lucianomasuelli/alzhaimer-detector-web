import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_wav_bytes


@pytest.fixture
def client(tiny_predictor):
    with TestClient(create_app(lambda: tiny_predictor)) as c:
        yield c


def test_health(client):
    j = client.get("/health").json()
    assert j == {"status": "ok", "model": "tiny", "stub": True, "encoder_layers": None}


def test_predict_ok(client):
    r = client.post("/predict", files={"file": ("a.wav", make_wav_bytes(3), "audio/wav")})
    assert r.status_code == 200
    assert set(r.json()["probabilities"]) == {"HC", "AD"}


def test_bad_extension(client):
    assert client.post("/predict", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 415


def test_too_short_is_422(client):
    r = client.post("/predict", files={"file": ("a.wav", make_wav_bytes(0.3), "audio/wav")})
    assert r.status_code == 422


def test_garbage_is_422(client):
    assert client.post("/predict", files={"file": ("a.wav", b"garbage", "audio/wav")}).status_code == 422


def test_index(client):
    assert "prototipo de investigación" in client.get("/").text.lower()
