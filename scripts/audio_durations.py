"""Estadisticas de duracion/canales/sample rate de un directorio de audios (sin decodificar).

Uso: python scripts/audio_durations.py dataset/Ivanova/train/audio [--csv out.csv]
Espera subcarpetas AD/HC/MCI (el nombre de la carpeta padre se usa como grupo).
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def info(path):
    try:
        i = sf.info(path)
        return i.duration, i.samplerate, i.channels
    except Exception:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate,channels",
                            "-of", "json", str(path)], capture_output=True, text=True)
        j = json.loads(p.stdout)
        s = j["streams"][0]
        return float(j["format"]["duration"]), int(s["sample_rate"]), int(s["channels"])


def summarize(name, d):
    d = np.array(d)
    q = np.percentile(d, [5, 50, 95, 99])
    print(f"{name:6s} n={len(d):4d} mean={d.mean():6.1f}s median={q[1]:6.1f} P5={q[0]:6.1f} "
          f"P95={q[2]:6.1f} P99={q[3]:6.1f} max={d.max():6.1f} min={d.min():6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.root).rglob("*")):
        if p.suffix.lower() in EXT:
            dur, sr, ch = info(p)
            rows.append((p.parent.name, p.name, dur, sr, ch))
    if not rows:
        raise SystemExit("No se encontraron audios")
    summarize("TOTAL", [r[2] for r in rows])
    for g in sorted({r[0] for r in rows}):
        summarize(g, [r[2] for r in rows if r[0] == g])
    print("sample rates:", sorted({r[3] for r in rows}), "| canales (1/2):",
          sum(r[4] == 1 for r in rows), "/", sum(r[4] == 2 for r in rows))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "file", "duration_s", "sample_rate", "channels"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
