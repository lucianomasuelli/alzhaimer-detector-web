---
title: Alzheimer voice demo
sdk: docker
app_port: 7860
---

# Demo: detección de Alzheimer por voz (prototipo de investigación)

Web app de demo de la tesina de Licenciatura en Ciencias de la Computación. Recibe un audio (archivo o micrófono) de
la lectura del primer párrafo del Quijote y devuelve la predicción AD vs HC del modelo E1 audio-only
(XLS-R-1B congelado + cabeza chica), definido en
[`alzhaimer-detector`](https://github.com/lucianomasuelli/alzhaimer-detector).

> **No es una herramienta diagnóstica.** Está entrenado con ~270 sujetos, una sola tarea y condiciones de grabación
> fijas. Ver "Limitaciones".

## Arquitectura

```
inference/   pipeline testeable sin FastAPI: audio.py (ffmpeg→mono 16 kHz), encoder.py (XLS-R + media/std por capa),
             head.py (cabeza vendoreada), predictor.py (Predictor), registry.py, config.py (ModelSpec)
app/         FastAPI: GET /health, POST /predict, frontend estático (HTML/JS plano)
scripts/     check_parity, check_head_vs_research, benchmark, audio_durations, export_head
tests/       pytest (sin red: encoder XLS-R aleatorio diminuto)
```

Pipeline (replica `modules/preprocess/extract_pooled_features.py`): bytes → soundfile (wav/mp3/flac/ogg) o ffmpeg (webm/m4a) → promedio de
canales → `librosa.resample` a 16 kHz → `Wav2Vec2FeatureExtractor` → XLS-R fp32, `output_hidden_states` →
media y desvío poblacional por capa `(49, 2, 1280)` → cabeza (z-score con `mu/sd` del checkpoint, suma ponderada
de capas, MLP) → `sigmoid` = P(AD). Audio completo en una pasada (sin chunking). Los audios nunca se escriben a disco.

## Uso local

```bash
pip install -r requirements-dev.txt     # requiere ffmpeg en el sistema
pytest                                  # sin red
uvicorn app.main:app --port 7860        # descarga XLS-R-1B (~4 GB) la primera vez
curl -F file=@lectura.wav localhost:7860/predict
```

Sin checkpoint, la cabeza tiene **pesos aleatorios** (`"stub": true` en la respuesta y banner en la UI).
Con checkpoint: `MODEL_REPO=usuario/alzheimer-head` (repo del HF Hub con `head.pth` + `config.json`,
`MODEL_REVISION` para fijar el commit) o `MODEL_CONFIG=config.json MODEL_HEAD=head.pth` locales.
Otras variables: `MAX_UPLOAD_MB` (25).

## Exportar el checkpoint

```bash
python scripts/export_head.py logs/.../model_fold_0.pth out/ --head-dim 128 --dropout 0.4 --threshold 0.5
huggingface-cli upload usuario/alzheimer-head out/ .
```

El `state_dict` incluye los buffers `mu/sd` (estandarización del train), así que no hace falta ningún otro artefacto.
`config.json` es un `ModelSpec`: debe coincidir con la config de entrenamiento (`head_dim`, `dropout`,
`layer_range`, `pool_stats`) y fija umbral y duración máxima.

## Paridad con el pipeline de entrenamiento

Correr en el cluster, donde están `dataset/` y los embeddings, con las mismas versiones de torch/transformers/librosa:

```bash
python scripts/check_parity.py dataset/Ivanova/train/audio/AD/<uid>.wav \
    dataset/Ivanova/train/text/AD/<uid>_pooledaudio_audioxlsr1b.pt
python scripts/check_head_vs_research.py /ruta/a/alzhaimer-detector   # cabeza vendoreada == original
python scripts/audio_durations.py dataset/Ivanova/train/audio --csv durations.csv   # fijar max_duration_s
```

Resultados en el cluster (`out_e1_early/REPORT.md`): WAV en GPU idéntico bit a bit; en CPU difiere ~5e-3 en el
embedding (la extracción usó la GPU, probablemente con TF32), con un efecto ~1e-6 en P(AD). Los MP3 se leen con
soundfile (libsndfile), igual que `librosa.load`; ffmpeg se usa solo para webm/m4a. `requirements.txt` fija las
versiones del entorno de entrenamiento.

## Benchmark

```bash
python scripts/benchmark.py --durations 10 30 60 --repeats 3     # modelo real; --tiny = smoke test
```

## Deploy en Hugging Face Spaces (Docker, CPU)

1. Crear un Space (SDK: Docker) y agregarlo como remoto: `git remote add space https://huggingface.co/spaces/USUARIO/NOMBRE`.
2. Settings → Variables: `MODEL_REPO`, `MODEL_REVISION` (y `HF_TOKEN` como secret si el repo del modelo es privado).
3. `git push space main`. El puerto es 7860 (ya configurado en el front-matter y el Dockerfile).
4. El primer arranque descarga XLS-R (~4 GB); para evitarlo en cada reinicio, descomentar el `RUN` de pre-descarga en el Dockerfile.

CPU free: 2 vCPU / 16 GB. XLS-R-1B fp32 ocupa ~4 GB; esperar decenas de segundos por audio (medir con el benchmark).

## Limitaciones

- Entrenado solo con la lectura del primer párrafo del Quijote (Ivanova, español europeo); el texto exacto
  mostrado en la UI (`app/static/index.html`) debe confirmarse con el del dataset.
- El modelo puede apoyarse en condiciones de grabación (canales, duración; las mejores capas son las tempranas): audios
  de micrófono de navegador (webm/opus) están fuera de distribución y la salida puede carecer de valor.
- Umbral calibrado con ~32 sujetos de validación; la probabilidad no está calibrada y no es un riesgo clínico.

## Extensibilidad (no implementada)

`ModelSpec` + `registry.get_predictor` permiten sumar modelos por config; solo se mantiene uno en memoria
(cada encoder grande ocupa varios GB). No hay selector en la UI.
