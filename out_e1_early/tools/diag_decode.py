"""Diagnostico (no cambia nada): decodificacion de entrenamiento (librosa.load) vs inference.audio.decode_audio."""
import sys
from pathlib import Path
import numpy as np, librosa, soundfile as sf
sys.path.insert(0, str(Path.home() / "alzhaimer-detector-web"))
from inference.audio import decode_audio
D = Path.home() / "dementiaDetection/Code/alzhaimer-detector/dataset/Ivanova/train/audio"
for rel in sys.argv[1:]:
    p = D / rel
    ref, _ = librosa.load(p, sr=16000, mono=True)
    got = decode_audio(p.read_bytes())
    i = sf.info(p)
    raw_sf, sr = sf.read(p, dtype="float32", always_2d=True)
    print(f"{rel}: sf.info frames={i.frames} sr={i.samplerate} ch={i.channels} fmt={i.format}/{i.subtype}")
    print(f"  librosa.load n={len(ref)}  decode_audio n={len(got)}  diff_n={len(got)-len(ref)}")
    n = min(len(ref), len(got))
    print(f"  max|diff| (alineado al inicio)={np.abs(ref[:n]-got[:n]).max():.3e}")
    # mejor desplazamiento en +-3000 muestras sobre los primeros 10 s
    seg = 160000; best = min(range(-3000, 3001, 1), key=lambda k: np.abs(ref[max(0,k):max(0,k)+seg] - got[max(0,-k):max(0,-k)+seg]).mean())
    a = ref[max(0,best):]; b = got[max(0,-best):]; m = min(len(a), len(b))
    print(f"  mejor offset={best} muestras@16k ({best/16:.2f} ms) -> max|diff|={np.abs(a[:m]-b[:m]).max():.3e}")
