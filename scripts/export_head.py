"""Convierte un checkpoint de entrenamiento (logs/.../model_fold_K.pth) en head.pth + config.json,
listos para subir al HF Hub (huggingface-cli upload TU_USUARIO/alzheimer-head .).

Uso: python scripts/export_head.py model_fold_0.pth OUT_DIR --head-dim 128 --dropout 0.4 \
         [--layer-range 0 16] [--threshold 0.5] [--max-duration 120]
"""
import argparse
import sys
from pathlib import Path

import torch
import transformers
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.config import ModelSpec
from inference.head import FrozenFeatureHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--layer-range", type=int, nargs=2)
    ap.add_argument("--pool-stats", default="both")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=120.0)
    ap.add_argument("--name", default="e1-audio")
    a = ap.parse_args()

    sd = torch.load(a.checkpoint, map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    spec = ModelSpec(name=a.name, head_dim=a.head_dim, dropout=a.dropout, layer_range=a.layer_range,
                     pool_stats=a.pool_stats, threshold=a.threshold, max_duration_s=a.max_duration,
                     versions={"torch": torch.__version__, "transformers": transformers.__version__,
                               "librosa": librosa.__version__})
    head = FrozenFeatureHead(spec.n_layers, spec.hidden, spec.head_dim, spec.dropout,
                             tuple(spec.layer_range) if spec.layer_range else None, spec.pool_stats)
    head.load_state_dict(sd)                       # strict: falla si la arquitectura no coincide
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pth")
    spec.to_json(out / "config.json")
    print("exportado a", out)


if __name__ == "__main__":
    main()
