"""Costo de inferencia de E1-fusion, E1-audio y E1-audio-early (completo y recortado) sobre los
mismos audios: latencia por etapa, Real-Time Factor (RTF = tiempo de proceso / duracion del audio),
memoria pico, parametros y tiempo de carga.

Cada variante corre en un subproceso propio (memoria pico limpia). Las cabezas tienen pesos
aleatorios: el costo no depende de los pesos y la cabeza es despreciable frente a los encoders.
No reproduce la clasificacion: las metricas salen de las corridas de entrenamiento (ver README).

  e1-fusion             decode + XLS-R (49 capas) + Whisper turbo + DistilmBERT + cabeza audio+texto
  e1-audio              decode + XLS-R (49 capas) + cabeza
  e1-audio-early        decode + XLS-R (49 capas) + cabeza sobre capas [0,16)   (estado actual)
  e1-audio-early-trunc  decode + XLS-R recortado a 16 capas + cabeza [0,16)    (TRUNCATE_ENCODER=1)

Uso (cluster, entorno cogni; e1-fusion necesita openai-whisper y el repo de investigacion):
  python scripts/compare_inference.py --audio-dir DATASET/Ivanova/train/audio --n 30 \\
      --research-repo ~/dementiaDetection/Code/alzhaimer-detector --device cpu --threads 2 \\
      --out results_inference/cpu
Smoke test local, sin red: python scripts/compare_inference.py --tiny --out /tmp/x
Consolidar corridas separadas (p. ej. una variante por job):
  python scripts/compare_inference.py --merge out_a/e1-fusion.json out_b/e1-audio.json ... --out results_inference/cpu
"""
import argparse
import importlib.util
import json
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.audio import decode_audio  # noqa: E402
from inference.config import ENCODER_ID  # noqa: E402
from inference.encoder import XlsrEncoder  # noqa: E402
from inference.head import FrozenFeatureHead, layer_stats  # noqa: E402

VARIANTS = ["e1-fusion", "e1-audio", "e1-audio-early", "e1-audio-early-trunc"]
TEXT_ID = "distilbert-base-multilingual-cased"
EARLY_HI = 16


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def n_params(*models):
    return sum(p.numel() for m in models for p in m.parameters())


class Timer:
    def __init__(self, device):
        self.cuda = device.startswith("cuda")
        self.t = {}

    def __call__(self, name, fn, *a, **kw):
        if self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        if self.cuda:
            torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
        return out


def load_variant(variant, a):
    """Devuelve (run(wave, timer) -> P(AD), modelos para contar parametros)."""
    dev = a.device
    if a.tiny:
        full_layers, hi = 4, 2
        enc = XlsrEncoder.tiny_random(hidden=32, layers=full_layers)
        if variant.endswith("trunc"):
            enc.model.encoder.layers = enc.model.encoder.layers[:hi]
            enc.num_layers, enc.pad_to = hi, full_layers + 1
    else:
        full_layers, hi = 48, EARLY_HI
        trunc = variant.endswith("trunc")
        enc = XlsrEncoder.from_pretrained(a.encoder, hi if trunc else None, full_layers + 1 if trunc else None)
    enc.model.to(dev)
    L, H = full_layers + 1, enc.model.config.hidden_size
    models = [enc.model]

    def audio_feats(wave, timer):
        inputs = timer("xlsr", enc.processor, wave, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            out = timer("xlsr", enc.model, **{k: v.to(dev) for k, v in inputs.items()}, output_hidden_states=True)
            hs = out.hidden_states if enc.num_layers is None else out.hidden_states[: enc.num_layers]
            x = timer("pool", layer_stats, hs)
        if enc.pad_to and enc.pad_to > x.shape[0]:
            x = torch.cat([x, x.new_zeros(enc.pad_to - x.shape[0], *x.shape[1:])])
        return x.to(dev)

    if variant != "e1-fusion":
        rng = (0, hi) if "early" in variant else None
        head = FrozenFeatureHead(L, H, 128, 0.4, rng).eval().to(dev)
        models.append(head)

        def run(wave, timer):
            x = audio_feats(wave, timer)
            with torch.inference_mode():
                return float(torch.sigmoid(timer("head", head, x[None])).item())
        return run, models

    if a.tiny:
        raise SystemExit("e1-fusion no tiene modo --tiny (necesita Whisper y DistilmBERT)")
    import unicodedata

    import whisper
    from transformers import AutoTokenizer, DistilBertModel
    spec = importlib.util.spec_from_file_location("model_frozen", Path(a.research_repo) / "modules" / "model_frozen.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)

    asr = whisper.load_model("turbo", device=dev)
    tok = AutoTokenizer.from_pretrained(TEXT_ID)
    tmodel = DistilBertModel.from_pretrained(TEXT_ID).eval().to(dev)
    cfg = SimpleNamespace(modalities="both", head_dim=128, dropout=0.4, num_classes=1,
                          audio_n_layers=L, audio_input_size=H,
                          text_n_layers=tmodel.config.n_layers + 1, text_input_size=tmodel.config.dim)
    head = mf.FrozenFeatureHead(cfg).eval().to(dev)
    models += [asr, tmodel, head]

    def run(wave, timer):
        x = audio_feats(wave, timer)
        # Como en preprocesswhisper.py (word_timestamps=True). fp16 solo en GPU.
        res = timer("asr", asr.transcribe, wave, word_timestamps=True, fp16=dev.startswith("cuda"))
        text = unicodedata.normalize("NFC", res["text"].strip())
        with torch.inference_mode():
            inputs = timer("text", tok, text, return_tensors="pt", truncation=True, max_length=384)
            out = timer("text", tmodel, **{k: v.to(dev) for k, v in inputs.items()}, output_hidden_states=True)
            t = timer("text", layer_stats, out.hidden_states).to(dev)
            return float(torch.sigmoid(timer("head", head, (x[None], t[None]))).item())
    return run, models


def run_variant(variant, a, files):
    torch.set_num_threads(a.threads)
    rss_base = rss_mb()
    t0 = time.perf_counter()
    run, models = load_variant(variant, a)
    load_s = time.perf_counter() - t0
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if a.tiny:      # bytes de wav sinteticos en memoria
        import io

        import soundfile as sf
        items = []
        for i, d in enumerate((5, 10, 20)):
            buf = io.BytesIO()
            sf.write(buf, (0.1 * np.random.default_rng(i).standard_normal(16000 * d)).astype(np.float32), 16000,
                     format="WAV")
            items.append((f"synthetic_{d}s.wav", buf.getvalue()))
    else:
        items = [(f, None) for f in files]

    def data_of(item):
        return item[1] if item[1] is not None else Path(item[0]).read_bytes()

    run(decode_audio(data_of(items[0])), Timer(a.device))      # warmup, no se mide
    rows = []
    for item in items:
        f, data = item[0], data_of(item)
        timer = Timer(a.device)
        t0 = time.perf_counter()
        wave = timer("decode", decode_audio, data)
        p = run(wave, timer)
        total = time.perf_counter() - t0
        dur = len(wave) / 16000
        rows.append({"file": str(f), "duration_s": dur, "latency_s": total, "rtf": total / dur,
                     "stages_s": timer.t, "p_ad": p})
        print(f"  [{variant}] {Path(str(f)).name}: {dur:6.1f}s audio -> {total:6.2f}s (RTF {total / dur:.3f})", flush=True)
    out = {"variant": variant, "device": a.device, "threads": a.threads, "load_s": load_s,
           "params_m": n_params(*models) / 1e6, "rss_base_mb": rss_base, "rss_peak_mb": rss_mb(),
           "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if a.device.startswith("cuda") else None,
           "torch": torch.__version__, "rows": rows}
    return out


def pick_files(a):
    if a.tiny:
        return []
    files = [p for d in ("AD", "HC") for p in sorted((Path(a.audio_dir) / d).glob("*"))
             if p.suffix.lower() in (".wav", ".mp3")]
    random.Random(a.seed).shuffle(files)
    return [str(p) for p in files[: a.n]]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summarize(results, out_dir):
    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    lines = ["**Tiempo de inferencia por audio**", "",
             "| variante | latencia media ± DE (s) | mediana (s) | p95 (s) | RTF media ± DE | RTF mediana | RTF p95 |",
             "|---|---|---|---|---|---|---|"]
    res = ["**Recursos y etapas**", "",
           "| variante | params (M) | carga (s) | RSS pico (MB) | GPU pico (MB) | etapas (media, s) |",
           "|---|---|---|---|---|---|"]
    for r in results:
        lat = [x["latency_s"] for x in r["rows"]]
        rtf = [x["rtf"] for x in r["rows"]]
        stages = {k: statistics.mean(x["stages_s"].get(k, 0.0) for x in r["rows"])
                  for k in r["rows"][0]["stages_s"]}
        st = ", ".join(f"{k} {v:.2f}" for k, v in stages.items())
        gpu = f"{r['gpu_peak_mb']:.0f}" if r["gpu_peak_mb"] is not None else "-"
        lines.append(f"| {r['variant']} | {statistics.mean(lat):.2f} ± {sd(lat):.2f} | {statistics.median(lat):.2f} | "
                     f"{q(lat, .95):.2f} | {statistics.mean(rtf):.3f} ± {sd(rtf):.3f} | {statistics.median(rtf):.3f} | "
                     f"{q(rtf, .95):.3f} |")
        res.append(f"| {r['variant']} | {r['params_m']:.1f} | {r['load_s']:.1f} | {r['rss_peak_mb']:.0f} | {gpu} | {st} |")
    lines += [""] + res
    r0 = results[0]
    durs = [x["duration_s"] for x in r0["rows"]]
    head = (f"Dispositivo: {r0['device']} | hilos: {r0['threads']} | torch {r0['torch']} | {len(durs)} audios "
            f"(duracion mediana {statistics.median(durs):.1f} s, rango {min(durs):.1f}-{max(durs):.1f} s). "
            "RTF = tiempo de proceso / duracion del audio (< 1: mas rapido que tiempo real); DE = desvio estandar "
            "entre audios. "
            "Latencia: desde los bytes del archivo hasta P(AD), una solicitud a la vez, sin la carga del modelo.")
    md = head + "\n\n" + "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(md)
    with open(out_dir / "per_file.csv", "w") as f:
        f.write("variant,file,duration_s,latency_s,rtf\n")
        for r in results:
            for x in r["rows"]:
                f.write(f"{r['variant']},{Path(x['file']).name},{x['duration_s']:.3f},{x['latency_s']:.4f},{x['rtf']:.5f}\n")
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--audio-dir")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2, help="hilos de torch (2 = vCPU tipico de un hosting chico)")
    ap.add_argument("--encoder", default=ENCODER_ID)
    ap.add_argument("--research-repo", help="necesario para e1-fusion (cabeza audio+texto)")
    ap.add_argument("--tiny", action="store_true", help="smoke test con un encoder aleatorio diminuto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", nargs="+", metavar="JSON",
                    help="consolidar JSON de corridas separadas (<variante>.json) en un solo summary.md")
    ap.add_argument("--_one", help=argparse.SUPPRESS)        # uso interno: correr una sola variante
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.merge:
        results = [json.loads(Path(f).read_text()) for f in a.merge]
        sets = {tuple(sorted(Path(x["file"]).name for x in r["rows"])) for r in results}
        if len(sets) > 1:   # mismas --n y --seed en todas las corridas
            raise SystemExit("las corridas no usan los mismos audios: no son comparables")
        summarize(results, out_dir)
        return

    if a._one:
        files = json.loads((out_dir / "files.json").read_text())
        (out_dir / f"{a._one}.json").write_text(json.dumps(run_variant(a._one, a, files), indent=1))
        return

    if not a.tiny and not a.audio_dir:
        ap.error("--audio-dir es obligatorio (salvo con --tiny)")
    if "e1-fusion" in a.variants and not a.tiny and not a.research_repo:
        ap.error("e1-fusion necesita --research-repo")
    variants = [v for v in a.variants if not (a.tiny and v == "e1-fusion")]
    (out_dir / "files.json").write_text(json.dumps(pick_files(a)))
    passthrough = [x for x in sys.argv[1:]]
    if "--variants" in passthrough:                 # cada subproceso corre solo su variante
        i = passthrough.index("--variants")
        j = i + 1
        while j < len(passthrough) and not passthrough[j].startswith("--"):
            j += 1
        del passthrough[i:j]
    results = []
    for v in variants:
        print(f"== {v}", flush=True)
        subprocess.run([sys.executable, __file__, *passthrough, "--_one", v], check=True)
        results.append(json.loads((out_dir / f"{v}.json").read_text()))
    summarize(results, out_dir)


if __name__ == "__main__":
    main()
