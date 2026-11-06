"""Latencia y memoria en CPU del pipeline completo (decodificacion + XLS-R + cabeza).

Uso: python scripts/benchmark.py [--durations 10 30 60 120] [--repeats 3] [--tiny] [--config config.json] [--truncate]
Reporta tiempo por duracion de audio y RSS pico del proceso (MB). Usa el stub si no hay checkpoint.
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.config import ModelSpec
from inference.encoder import XlsrEncoder
from inference.predictor import Predictor
from tests.conftest import make_wav_bytes


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", type=float, nargs="+", default=[10, 30, 60, 120])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tiny", action="store_true", help="encoder aleatorio diminuto (solo smoke test)")
    ap.add_argument("--head", help="head.pth (opcional)")
    ap.add_argument("--config", help="config.json (opcional; p. ej. out_e1_early/config.json)")
    ap.add_argument("--truncate", action="store_true", help="encoder recortado a layer_range[1] capas")
    a = ap.parse_args()

    print(f"torch threads={torch.get_num_threads()}")
    t = time.perf_counter()
    spec = ModelSpec.from_json(a.config) if a.config else ModelSpec()
    spec.max_duration_s = max(a.durations) + 1
    spec.truncate_encoder = a.truncate
    if a.truncate and not spec.layer_range:
        spec.layer_range = [0, 16]          # E1-audio-early
    enc = None
    if a.tiny:
        spec = ModelSpec(name="tiny", n_layers=3, hidden=32, head_dim=8, max_duration_s=spec.max_duration_s)
        enc = XlsrEncoder.tiny_random(hidden=32, layers=2)
    pred = Predictor.load(spec, a.head, encoder=enc)
    print(f"carga: {time.perf_counter() - t:.1f}s | RSS tras cargar: {rss_mb():.0f} MB")
    for d in a.durations:
        data = make_wav_bytes(d, sr=16000, channels=1)
        times = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            pred.predict_bytes(data)
            times.append(time.perf_counter() - t)
        print(f"{d:6.0f}s audio: mediana {sorted(times)[len(times)//2]:6.2f}s  min {min(times):6.2f}s  | RSS pico {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
