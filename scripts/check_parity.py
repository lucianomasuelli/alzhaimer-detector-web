"""Paridad numerica: embedding del pipeline de inferencia vs el .pt precomputado en entrenamiento.

Uso: python scripts/check_parity.py AUDIO.wav REF_pooledaudio_audioxlsr1b.pt [--atol 1e-3] [--truncate 16]
--truncate N: encoder recortado a N capas; compara solo las capas [0, N) (las demas quedan en cero).
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
    ap.add_argument("--truncate", type=int, help="recortar el encoder a N capas")
    a = ap.parse_args()
    ref = torch.load(a.reference, map_location="cpu")
    enc = XlsrEncoder.from_pretrained(a.encoder, a.truncate, ref.shape[0] if a.truncate else None)
    got = enc.embed(decode_audio(Path(a.audio).read_bytes()))
    assert got.shape == ref.shape, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    if a.truncate:
        got, ref = got[: a.truncate], ref[: a.truncate]
    diff = (got - ref).abs()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    print(f"max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} cosine={cos:.8f}")
    for l in (x for x in (0, 8, 15, 16, 32, 48) if x < got.shape[0]):
        print(f"  capa {l:2d}: max|diff|={diff[l].max():.3e}")
    ok = diff.max().item() <= a.atol
    print("PARIDAD OK" if ok else "PARIDAD FALLA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
