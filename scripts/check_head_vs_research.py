"""Verifica que inference/head.py (vendoreada) da las mismas salidas que modules/model_frozen.py del
repo de investigacion, con pesos aleatorios y el mismo state_dict.

Uso: python scripts/check_head_vs_research.py /ruta/a/alzhaimer-detector
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.head import FrozenFeatureHead


def main(repo):
    spec = importlib.util.spec_from_file_location("model_frozen", Path(repo) / "modules" / "model_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for head_dim, layers in ((128, None), (0, None), (128, [0, 16])):
        cfg = SimpleNamespace(modalities="audio", head_dim=head_dim, dropout=0.4, num_classes=1,
                              audio_n_layers=49, audio_input_size=1280, audio_layers=layers)
        ref = mod.FrozenFeatureHead(cfg).eval()
        with torch.no_grad():
            ref.audio["pool"].mu.normal_(); ref.audio["pool"].sd.uniform_(0.5, 2)
            ref.audio["pool"].layer_logits.normal_()
        new = FrozenFeatureHead(49, 1280, head_dim, 0.4, tuple(layers) if layers else None).eval()
        new.load_state_dict(ref.state_dict())                       # strict: mismas claves
        x = torch.randn(3, 49, 2, 1280)
        torch.testing.assert_close(new(x), ref((x, None)))
        print(f"OK head_dim={head_dim} layers={layers}")


if __name__ == "__main__":
    main(sys.argv[1])
