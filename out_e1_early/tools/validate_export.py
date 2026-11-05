"""Paso 5 (cabeza exportada vs prob de outer_predictions, sobre .pt precomputados) y paso 6 (punta a punta)."""
import json, sys
from pathlib import Path
import torch
WEB = Path.home() / "alzhaimer-detector-web"; sys.path.insert(0, str(WEB))
from inference.config import ModelSpec
from inference.head import FrozenFeatureHead
from inference.predictor import Predictor
RES = Path.home() / "dementiaDetection/Code/alzhaimer-detector"
D = RES / "dataset/Ivanova/train"
OUT = WEB / "out_e1_early"
pred = json.loads((RES / "logs/ablation/ivanova_e1_abl-early/seed_42/outer_predictions.json").read_text())
f0 = sorted([p for p in pred if p["fold"] == 0], key=lambda p: p["uid"])
spec = ModelSpec.from_json(OUT / "config.json"); print("spec:", spec)
head = FrozenFeatureHead(spec.n_layers, spec.hidden, spec.head_dim, spec.dropout,
                         tuple(spec.layer_range) if spec.layer_range else None, spec.pool_stats)
head.load_state_dict(torch.load(OUT / "head.pth", map_location="cpu", weights_only=True)); head.eval()
grp = lambda p: "AD" if p["y_true"] == 1 else "HC"
print("\n== Paso 5: n =", len(f0))
diffs = []
with torch.no_grad():
    for p in f0:
        x = torch.load(D / "text" / grp(p) / f"{p['uid']}_pooledaudio_audioxlsr1b.pt", map_location="cpu")
        pr = torch.sigmoid(head(x.unsqueeze(0))).item(); diffs.append(abs(pr - p["prob"]))
        print(f"  {p['uid']:14s} {grp(p)} ref={p['prob']:.7f} got={pr:.7f} |diff|={diffs[-1]:.2e}")
print(f"PASO 5 max|diff|={max(diffs):.3e} -> {'OK' if max(diffs) < 1e-5 else 'FALLA'}")
print("\n== Paso 6 (Predictor, CPU)")
pr6 = Predictor.load(spec, OUT / "head.pth")
for cls in ("AD", "HC"):
    p = next(q for q in f0 if grp(q) == cls)          # primer uid alfabetico de la clase en fold 0
    a = next((D / "audio" / cls / f"{p['uid']}{e}") for e in (".wav", ".mp3") if (D / "audio" / cls / f"{p['uid']}{e}").exists())
    try:
        r = pr6.predict_path(a)
        print(f"  {p['uid']} ({a.name}) ref={p['prob']:.7f} P(AD)={r.probabilities['AD']:.7f} "
              f"|diff|={abs(r.probabilities['AD']-p['prob']):.2e} label={r.label} dur={r.duration_s:.1f}s")
    except Exception as e:
        print(f"  {p['uid']} ({a.name}) ERROR: {type(e).__name__}: {e}")
