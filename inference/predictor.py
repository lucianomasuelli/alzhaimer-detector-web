"""Predictor: bytes/ruta de audio -> prediccion. Sin dependencias de FastAPI."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol

import torch

from .audio import decode_audio, validate_duration
from .config import ModelSpec
from .encoder import XlsrEncoder
from .head import FrozenFeatureHead


@dataclass
class Prediction:
    label: str
    probabilities: dict[str, float]
    threshold: float
    duration_s: float
    model: str
    stub: bool

    def to_dict(self):
        return asdict(self)


class BasePredictor(Protocol):
    spec: ModelSpec

    def predict_bytes(self, data: bytes) -> Prediction: ...


class Predictor:
    def __init__(self, spec: ModelSpec, encoder: XlsrEncoder, head: FrozenFeatureHead, stub: bool = False):
        self.spec, self.encoder, self.stub = spec, encoder, stub
        self.head = head.eval()

    @classmethod
    def load(cls, spec: ModelSpec, head_path: str | Path | None = None, encoder: XlsrEncoder | None = None):
        """head_path=None -> cabeza con pesos aleatorios (stub)."""
        encoder = encoder or XlsrEncoder.from_pretrained(spec.encoder_id, spec.encoder_layers(), spec.n_layers)
        if head_path is None:
            torch.manual_seed(0)            # stub reproducible
        head = FrozenFeatureHead(spec.n_layers, spec.hidden, spec.head_dim, spec.dropout,
                                 tuple(spec.layer_range) if spec.layer_range else None, spec.pool_stats)
        if head_path is not None:
            head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        return cls(spec, encoder, head, stub=head_path is None)

    def predict_wave(self, wave) -> Prediction:
        dur = validate_duration(wave, self.spec.sample_rate, self.spec.min_duration_s, self.spec.max_duration_s)
        feats = self.encoder.embed(wave, self.spec.sample_rate)
        with torch.inference_mode():
            p_ad = float(torch.sigmoid(self.head(feats.unsqueeze(0))).item())
        label = self.spec.labels[1] if p_ad >= self.spec.threshold else self.spec.labels[0]
        return Prediction(label, {self.spec.labels[0]: 1 - p_ad, self.spec.labels[1]: p_ad},
                          self.spec.threshold, dur, self.spec.name, self.stub)

    def predict_bytes(self, data: bytes) -> Prediction:
        return self.predict_wave(decode_audio(data, self.spec.sample_rate))

    def predict_path(self, path: str | Path) -> Prediction:
        return self.predict_bytes(Path(path).read_bytes())
