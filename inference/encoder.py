"""XLS-R congelado + estadisticas por capa (identico a extract_pooled_features.py).

Recorte opcional (num_layers=N): se cargan solo las N primeras capas transformer. hidden_states[i]
para i < N es la entrada de la capa i, identica a la del modelo completo; el ultimo hidden state
del modelo recortado lleva el layer_norm final (en la variante stable-layer-norm de XLS-R) y se
descarta. Se devuelven N capas y el resto se rellena con ceros hasta pad_to: la cabeza las ignora
porque layer_range las excluye.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .head import layer_stats


class XlsrEncoder:
    def __init__(self, model, processor, num_layers=None, pad_to=None):
        self.model = model.eval()
        self.processor = processor
        self.num_layers, self.pad_to = num_layers, pad_to
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, encoder_id: str, num_layers: int | None = None, pad_to: int | None = None):
        kw = {} if num_layers is None else {"num_hidden_layers": num_layers}
        return cls(Wav2Vec2Model.from_pretrained(encoder_id, **kw),
                   Wav2Vec2FeatureExtractor.from_pretrained(encoder_id), num_layers, pad_to)

    @classmethod
    def tiny_random(cls, hidden=32, layers=2, stable_layer_norm=True):
        """Encoder aleatorio diminuto para tests/CI (sin red). XLS-R usa stable layer norm."""
        cfg = Wav2Vec2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=2,
                             intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 4),
                             conv_kernel=(10, 4), num_feat_extract_layers=2,
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                             do_stable_layer_norm=stable_layer_norm,
                             feat_extract_norm="layer" if stable_layer_norm else "group")
        return cls(Wav2Vec2Model(cfg), Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True))

    @torch.inference_mode()
    def embed(self, wave: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """float32 mono 16 kHz -> (L, 2, H)."""
        inputs = self.processor(wave, sampling_rate=sr, return_tensors="pt")
        out = self.model(**inputs, output_hidden_states=True)
        if self.num_layers is None:
            return layer_stats(out.hidden_states)
        stats = layer_stats(out.hidden_states[: self.num_layers])
        if self.pad_to and self.pad_to > stats.shape[0]:
            pad = stats.new_zeros(self.pad_to - stats.shape[0], *stats.shape[1:])
            stats = torch.cat([stats, pad], dim=0)
        return stats
