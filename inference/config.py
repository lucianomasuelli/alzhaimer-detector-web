"""ModelSpec: todo lo que define un modelo servible. Se carga de config.json (HF Hub o local)."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENCODER_ID = "facebook/wav2vec2-xls-r-1b"


@dataclass
class ModelSpec:
    name: str = "e1-audio"
    encoder_id: str = ENCODER_ID
    n_layers: int = 49
    hidden: int = 1280
    head_dim: int = 128
    dropout: float = 0.4
    layer_range: list[int] | None = None       # [lo, hi) o None = todas
    pool_stats: str = "both"
    threshold: float = 0.5                      # P(AD) >= threshold -> AD
    labels: list[str] = field(default_factory=lambda: ["HC", "AD"])  # indice 1 = AD
    sample_rate: int = 16000
    min_duration_s: float = 5.0
    max_duration_s: float = 220.0               # cubre el maximo de entrenamiento (AD: 217.9 s)
    versions: dict = field(default_factory=dict)  # torch/transformers/librosa usados al entrenar

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))
