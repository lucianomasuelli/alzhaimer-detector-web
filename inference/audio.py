"""Decodificacion y validacion de audio en memoria (nada se escribe a disco).

Replica librosa.load(sr=16000, mono=True), que usa el pipeline de entrenamiento: decodifica
al sample rate nativo conservando canales, promedia canales y remuestrea con librosa.
"""

import json
import subprocess

import librosa
import numpy as np

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".flac"}


class AudioError(ValueError):
    """Entrada invalida; el mensaje es apto para mostrar al usuario."""


def _probe(data: bytes) -> tuple[int, int]:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", "-i", "pipe:0"],
        input=data, capture_output=True, timeout=30)
    try:
        s = json.loads(p.stdout)["streams"][0]
        return int(s["sample_rate"]), int(s["channels"])
    except (KeyError, IndexError, ValueError):
        raise AudioError("No se pudo leer el audio: formato no reconocido o archivo corrupto.")


def decode_audio(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """bytes de un archivo de audio -> float32 mono a target_sr."""
    sr, ch = _probe(data)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-map", "0:a:0", "-f", "f32le",
         "-acodec", "pcm_f32le", "pipe:1"],
        input=data, capture_output=True, timeout=120)
    if p.returncode != 0 or not p.stdout:
        raise AudioError("No se pudo decodificar el audio.")
    y = np.frombuffer(p.stdout, dtype=np.float32)
    y = y[: len(y) // ch * ch].reshape(-1, ch).T          # (ch, n)
    y = y.mean(axis=0) if ch > 1 else y[0]
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def validate_duration(y: np.ndarray, sr: int, min_s: float, max_s: float) -> float:
    dur = len(y) / sr
    if dur < min_s:
        raise AudioError(f"Audio demasiado corto ({dur:.1f} s). Minimo: {min_s:.0f} s.")
    if dur > max_s:
        raise AudioError(f"Audio demasiado largo ({dur:.0f} s). Maximo: {max_s:.0f} s.")
    if not np.isfinite(y).all() or float(np.abs(y).max()) == 0.0:
        raise AudioError("El audio esta vacio o es silencio.")
    return dur
