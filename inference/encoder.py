"""XLS-R congelado + estadisticas por capa (identico a extract_pooled_features.py)."""

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .head import layer_stats


class XlsrEncoder:
    def __init__(self, model, processor):
        self.model = model.eval()
        self.processor = processor
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, encoder_id: str):
        return cls(Wav2Vec2Model.from_pretrained(encoder_id),
                   Wav2Vec2FeatureExtractor.from_pretrained(encoder_id))

    @classmethod
    def tiny_random(cls, hidden=32, layers=2):
        """Encoder aleatorio diminuto para tests/CI (sin red)."""
        cfg = Wav2Vec2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=2,
                             intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 4),
                             conv_kernel=(10, 4), num_feat_extract_layers=2,
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
        return cls(Wav2Vec2Model(cfg), Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True))

    @torch.inference_mode()
    def embed(self, wave: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """float32 mono 16 kHz -> (L, 2, H)."""
        inputs = self.processor(wave, sampling_rate=sr, return_tensors="pt")
        out = self.model(**inputs, output_hidden_states=True)
        return layer_stats(out.hidden_states)
