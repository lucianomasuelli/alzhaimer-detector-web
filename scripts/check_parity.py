"""Paridad numerica: embedding del pipeline de inferencia vs el .pt precomputado en entrenamiento.

Uso: python scripts/check_parity.py AUDIO.wav REF_pooledaudio_audioxlsr1b.pt [--atol 1e-3]
El .pt de referencia es el (49, 2, 1280) generado por modules/preprocess/extract_pooled_features.py.
Correr en el cluster (donde estan dataset/ y los embeddings) con las mismas versiones de librerias.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.audio import decode_audio
from inference.config import ENCODER_ID
from inference.encoder import XlsrEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("reference")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--encoder", default=ENCODER_ID)
    a = ap.parse_args()
    ref = torch.load(a.reference, map_location="cpu")
    got = XlsrEncoder.from_pretrained(a.encoder).embed(decode_audio(Path(a.audio).read_bytes()))
    assert got.shape == ref.shape, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    diff = (got - ref).abs()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    print(f"max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} cosine={cos:.8f}")
    for l in (0, 8, 16, 32, 48):
        print(f"  capa {l:2d}: max|diff|={diff[l].max():.3e}")
    ok = diff.max().item() <= a.atol
    print("PARIDAD OK" if ok else "PARIDAD FALLA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
