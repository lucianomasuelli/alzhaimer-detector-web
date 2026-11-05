"""Igual que scripts/check_parity.py pero con el encoder en el dispositivo pedido (el de inference/
corre siempre en CPU). Mismo processor/modelo/layer_stats; solo agrega .to(device).
Guarda el embedding en --save para comparar GPU vs CPU."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / "alzhaimer-detector-web"))
from inference.audio import decode_audio
from inference.config import ENCODER_ID
from inference.encoder import XlsrEncoder
from inference.head import layer_stats

ap = argparse.ArgumentParser()
ap.add_argument("audio"); ap.add_argument("reference")
ap.add_argument("--device", default="cuda"); ap.add_argument("--atol", type=float, default=1e-3)
ap.add_argument("--save")
a = ap.parse_args()
enc = XlsrEncoder.from_pretrained(ENCODER_ID)
enc.model.to(a.device)
wave = decode_audio(Path(a.audio).read_bytes())
with torch.inference_mode():
    inputs = enc.processor(wave, sampling_rate=16000, return_tensors="pt")
    out = enc.model(**{k: v.to(a.device) for k, v in inputs.items()}, output_hidden_states=True)
    got = layer_stats(out.hidden_states)
ref = torch.load(a.reference, map_location="cpu")
assert got.shape == ref.shape, (got.shape, ref.shape)
diff = (got - ref).abs()
cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
print(f"device={a.device} samples={len(wave)} max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} cosine={cos:.8f}")
print("  por capa max|diff|:", " ".join(f"{l}:{diff[l].max():.2e}" for l in range(diff.shape[0])))
print("PARIDAD OK" if diff.max().item() <= a.atol else "PARIDAD FALLA")
if a.save: torch.save(got, a.save)
