"""Cabeza E1 (audio-only) vendoreada de lucianomasuelli/alzhaimer-detector @ e1413c7.

Origen: modules/model_frozen.py (FrozenFeatureHead, LayerWeightedPool) y
modules/pooled_features.py (layer_stats). Se copia el minimo para no depender de
mamba_ssm ni del resto del repo de investigacion. Mantener los nombres de parametros
y buffers (layer_logits, mu, sd, ...) para que el state_dict entrenado cargue sin cambios.
"""

import torch
import torch.nn as nn


def layer_stats(hidden_states):
    """hidden_states: tupla/lista de L tensores (1, T, H) o un tensor (L, T, H).
    Devuelve (L, 2, H) float32 en CPU: [:, 0] = media, [:, 1] = desvio (poblacional) sobre T."""
    if isinstance(hidden_states, (tuple, list)):
        hs = torch.stack([h.squeeze(0) if h.dim() == 3 else h for h in hidden_states], dim=0)
    else:
        hs = hidden_states
    hs = hs.float()
    mean = hs.mean(dim=1)
    std = hs.std(dim=1, unbiased=False) if hs.shape[1] > 1 else torch.zeros_like(mean)
    out = torch.stack([mean, std], dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).cpu()


class LayerWeightedPool(nn.Module):
    """(B, L, 2, H) -> (B, 2H): estandariza cada feature y hace suma ponderada softmax sobre capas."""

    def __init__(self, n_layers, hidden, layer_range=None, stats="both"):
        super().__init__()
        if stats not in ("both", "mean"):
            raise ValueError(f"pool_stats invalido: {stats} (both|mean)")
        self.stats = stats
        self.out_dim = (2 if stats == "both" else 1) * hidden
        lo, hi = (0, n_layers) if layer_range is None else layer_range
        if not (0 <= lo < hi <= n_layers):
            raise ValueError(f"layer_range {layer_range} invalido para {n_layers} capas")
        self.lo, self.hi = lo, hi
        self.layer_logits = nn.Parameter(torch.zeros(hi - lo))
        self.hidden = hidden
        self.register_buffer("mu", torch.zeros(n_layers, 2, hidden))
        self.register_buffer("sd", torch.ones(n_layers, 2, hidden))

    def forward(self, x):
        x = (x - self.mu) / self.sd
        x = x[:, self.lo:self.hi]
        if self.stats == "mean":
            x = x[:, :, :1]
        w = torch.softmax(self.layer_logits, dim=0).view(1, -1, 1, 1)
        return (w * x).sum(dim=1).flatten(1)


class FrozenFeatureHead(nn.Module):
    """Variante audio-only de FrozenFeatureHead. Las claves del state_dict coinciden con
    las del original con modalities='audio' (audio.pool.*, audio.proj.*, classifier.*)."""

    def __init__(self, n_layers=49, hidden=1280, head_dim=128, dropout=0.4,
                 layer_range=None, pool_stats="both", num_classes=1):
        super().__init__()
        pool = LayerWeightedPool(n_layers, hidden, layer_range, pool_stats)
        if head_dim == 0:
            proj, n_in = nn.Identity(), pool.out_dim
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, num_classes))
        else:
            proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(pool.out_dim, head_dim),
                                 nn.GELU(), nn.Dropout(dropout))
            n_in = head_dim
            self.classifier = nn.Sequential(nn.LayerNorm(n_in), nn.Dropout(dropout),
                                            nn.Linear(n_in, num_classes))
        self.audio = nn.ModuleDict({"pool": pool, "proj": proj})

    def forward(self, audio):
        return self.classifier(self.audio["proj"](self.audio["pool"](audio)))
