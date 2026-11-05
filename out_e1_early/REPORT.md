# Export E1-audio-early para la web app — reporte

Fecha: 2026-11-04. Cluster: Mendieta (`short`, NVIDIA A30). Repo de investigación en branch `mendieta`
(commit `0012d68`), sin modificar. Repo web clonado en `912ceb6` (`main`), sin commits ni push.

## Resumen

| Verificación | Resultado |
|---|---|
| 1. Umbral fold 0 (summary vs outer_predictions) | **OK** (coinciden) |
| 3a. `check_head_vs_research.py` | **OK** (3/3) |
| 3b. Paridad embedding, `.wav` en GPU | **OK** (max\|diff\| = 0, idéntico bit a bit) |
| 3b. Paridad embedding, `.mp3` en GPU | **FALLA** (6.6e-2 y 2.7e-3 > 1e-3) |
| 3b. Paridad embedding, CPU (todos) | **FALLA** (2.98e-3 … 6.29e-2 > 1e-3) |
| 3c. Duraciones | P99 total = 152.3 s → **MAX = 160 s** |
| 5. Cabeza exportada vs `prob` (55 sujetos fold 0) | **OK** (max\|diff\| = 1.19e-7 < 1e-5) |
| 6. Punta a punta (Predictor, CPU) | AD wav: \|diff\| = 1.3e-6; HC mp3: \|diff\| = 1.4e-5 (sin tolerancia definida) |

## 1. Checkpoint

- Seeds completas (con `subject_level_metrics.json`): las 20 (`42, 123, 456, 789, 1111, 1234, 2024, 2468, 5555,
  8080, 9999, 11223, 13579, 24680, 27182, 31415, 33445, 55667, 77889, 98765`).
- Seed elegida (menor número): **42**. Fold: **0**.
- Checkpoint: `~/dementiaDetection/Code/alzhaimer-detector/logs/ablation/ivanova_e1_abl-early/seed_42/model_fold_0.pth`
  (`logs` es symlink a `~/dementiaDetection/data/logs`), sha256 `542c661e…2e5a`.
- Umbral: `cross_fold_summary.txt`, bloque `Fold 0` → `Threshold: 0.6699999999999999`.
  `outer_predictions.json`: las 55 filas con `fold == 0` tienen `threshold = 0.6699999999999999` (un único valor). **Coinciden.**

## 2. Entorno (`cogni`, nodo de cómputo)

```
python 3.10.19 (main, Oct 21 2025, 16:43:05) [GCC 11.2.0]
torch 2.9.1+cu128      (CUDA 12.8, NVIDIA A30)
transformers 4.57.5
librosa 0.11.0
soundfile 0.13.1       (libsndfile 1.2.2)
numpy 2.2.6
soxr 1.1.0             (resampler que usa librosa por defecto)
audioread 3.0.1
ffmpeg / ffprobe 7.1   (/home/lmasuelli/miniconda3/envs/cogni/bin/)
```

`Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-1b")` → **`do_normalize = True`**:
```
{"do_normalize": true, "feature_extractor_type": "Wav2Vec2FeatureExtractor", "feature_size": 1,
 "padding_side": "right", "padding_value": 0, "return_attention_mask": true, "sampling_rate": 16000}
```

## 3a. Cabeza vendoreada vs investigación

```
OK head_dim=128 layers=None
OK head_dim=0 layers=None
OK head_dim=128 layers=[0, 16]
```

## 3b. Paridad del embedding (49, 2, 1280)

La extracción original (`e1_01_extract_1179719`) corrió en **GPU**, sin ningún fallback OOM→CPU (0 líneas `OOM` en el log).
`scripts/check_parity.py` siempre corre el encoder en CPU, así que:
- **CPU**: `scripts/check_parity.py AUDIO REF.pt` sin cambios.
- **GPU**: `out_e1_early/tools/parity_device.py --device cuda`. Es el mismo cálculo (`decode_audio`, processor,
  `XlsrEncoder`, `layer_stats` de `inference/`) con un `.to(device)` agregado. También se corrió con `--device cpu`
  y dio exactamente los mismos números que `check_parity.py`.

Sujetos: el primer uid alfabético de cada combinación clase/canales/formato dentro del fold 0. En este dataset
todos los estéreo son `.wav` y todos los `.mp3` son mono, así que se agregó un 5.º sujeto (mono `.wav`).
Canales y formato salen de `ffprobe`. Todos los audios están a 44.1 kHz.

| uid | clase | canales | formato | dispositivo | max\|diff\| | mean\|diff\| | coseno | resultado |
|---|---|---|---|---|---|---|---|---|
| AD-M-82-321 | AD | 1 | mp3 | GPU | 6.560e-02 | 3.100e-03 | 0.99999821 | **FALLA** |
| AD-M-82-321 | AD | 1 | mp3 | CPU | 6.288e-02 | 3.107e-03 | 0.99999815 | **FALLA** |
| AD-M-57-163 | AD | 2 | wav | GPU | 0.000e+00 | 0.000e+00 | 1.00000107 | OK |
| AD-M-57-163 | AD | 2 | wav | CPU | 7.874e-03 | 7.513e-05 | 1.00000119 | **FALLA** |
| HC-M-59-291 | HC | 1 | mp3 | GPU | 2.651e-03 | 4.280e-05 | 1.00000143 | **FALLA** |
| HC-M-59-291 | HC | 1 | mp3 | CPU | 2.983e-03 | 5.674e-05 | 1.00000119 | **FALLA** |
| HC-M-65-48 | HC | 2 | wav | GPU | 0.000e+00 | 0.000e+00 | 1.00000072 | OK |
| HC-M-65-48 | HC | 2 | wav | CPU | 4.894e-03 | 9.273e-05 | 1.00000072 | **FALLA** |
| HC-M-84-231 | HC | 1 | wav | GPU | 0.000e+00 | 0.000e+00 | 1.00000072 | OK |
| HC-M-84-231 | HC | 1 | wav | CPU | 4.581e-03 | 6.619e-05 | 1.00000036 | **FALLA** |

(Un coseno > 1 es redondeo de float32.)

GPU vs CPU (los dos embeddings del pipeline web, comparados entre sí):

| uid | max\|diff\| | coseno |
|---|---|---|
| AD-M-82-321 | 1.231e-02 | 1.00000095 |
| AD-M-57-163 | 7.874e-03 | 1.00000119 |
| HC-M-59-291 | 2.960e-03 | 1.00000179 |
| HC-M-65-48 | 4.894e-03 | 1.00000072 |
| HC-M-84-231 | 4.581e-03 | 1.00000036 |

Diferencias por capa (max\|diff\|; `check_parity.py` imprime las capas 0/8/16/32/48; las 49 capas están en
`logs/job2_1185663.out`):

| uid / disp. | capa 0 | capa 8 | capa 16 | capa 32 | capa 48 |
|---|---|---|---|---|---|
| AD-M-82-321 mp3 / CPU | 3.614e-03 | 3.204e-02 | 5.300e-02 | 4.668e-02 | 6.619e-03 |
| AD-M-57-163 wav / CPU | 3.105e-03 | 5.865e-04 | 6.189e-04 | 1.223e-03 | 1.720e-03 |
| HC-M-59-291 mp3 / CPU | 2.983e-03 | 6.462e-04 | 7.362e-04 | 9.775e-04 | 5.343e-04 |
| HC-M-65-48 wav / CPU | 3.101e-03 | 5.215e-04 | 2.557e-03 | 2.987e-03 | 3.161e-04 |
| HC-M-84-231 wav / CPU | 3.120e-03 | 6.495e-04 | 1.427e-03 | 2.037e-03 | 3.080e-04 |
| AD-M-82-321 mp3 / GPU | 3.60e-03 | 3.22e-02 | 5.39e-02 | 4.96e-02 | 7.06e-03 |
| HC-M-59-291 mp3 / GPU | 3.50e-05 | 9.97e-05 | 7.11e-04 | 1.75e-03 | 5.89e-04 |
| wav / GPU (los 3) | 0 | 0 | 0 | 0 | 0 |

Salida completa: `out_e1_early/logs/job2_1185663.out`.

### Diagnóstico de las fallas (sin cambiar código)

**a) MP3: los decodificadores difieren.** En entrenamiento se usó `librosa.load(sr=16000, mono=True)`, que lee el MP3 con
libsndfile 1.2.2 (mpg123). La web decodifica con ffmpeg 7.1. `out_e1_early/tools/diag_decode.py` (salida en
`logs/diag_decode.out`; srun job 1185671, que repitió la salida 5 veces) da:

```
AD/AD-M-82-321.mp3: librosa.load n=466061  decode_audio n=466460  diff_n=399
  max|diff| (alineado al inicio)=9.271e-04 ; mejor offset=0
HC/HC-M-59-291.mp3: librosa.load n=930780  decode_audio n=930813  diff_n=33
  max|diff| (alineado al inicio)=2.584e-05 ; mejor offset=0
HC/HC-M-84-231.wav: librosa.load n=622395  decode_audio n=622395  diff_n=0
  max|diff| (alineado al inicio)=0.000e+00
```

ffmpeg devuelve muestras de más al final: 399 en AD-M-82-321 (~25 ms a 16 kHz) y 33 en HC-M-59-291. Probablemente sea el
padding del encoder MP3, que libsndfile/mpg123 recorta y ffmpeg no. Además, las muestras en común difieren un poco
(hasta 9.3e-4). Esas muestras extra entran en la media y el desvío de todas las capas, y el error crece con la
profundidad (en AD-M-82-321 llega a ~5e-2 en las capas 15–20). En WAV la forma de onda es idéntica bit a bit.

**b) CPU vs GPU: ~3e-3 desde la capa 0, también en WAV.** Con la forma de onda idéntica, la GPU reproduce la referencia
exactamente y la CPU no. La diferencia ya está en la capa 0 (salida de la proyección de features del extractor
convolucional, ~3.1e-3 en todos los sujetos). Probable causa, **no verificada**: cuDNN usa TF32 por defecto en las
convoluciones (`torch.backends.cudnn.allow_tf32 = True`, A30 Ampere), y la referencia se extrajo así. La CPU calcula en
fp32 estricto. El orden de magnitud (~5e-3, hasta 1.2e-2 en mp3) es mayor que los ~1e-4 esperados.

**Impacto en la predicción:** chico en los 2 casos medidos (paso 6). Para HC-M-59-291 (mp3, CPU),
\|ΔP(AD)\| = 1.4e-5; para AD-M-57-163 (wav, CPU), 1.3e-6. AD-M-82-321, el mp3 con la mayor diferencia de
embedding, no se pasó por la cabeza: no estaba en el paso 6.

## 3c. Duraciones

`python scripts/audio_durations.py dataset/Ivanova/train/audio --csv durations.csv`:

```
TOTAL  n= 361 mean=  48.2s median=  41.2 P5=  27.0 P95=  90.5 P99= 152.3 max= 217.9 min=  19.5
AD     n=  74 mean=  62.7s median=  53.3 P5=  30.5 P95= 131.1 P99= 211.6 max= 217.9 min=  25.4
HC     n= 197 mean=  40.5s median=  37.5 P5=  25.3 P95=  67.6 P99=  93.5 max= 102.4 min=  19.5
MCI    n=  90 mean=  52.9s median=  45.4 P5=  28.6 P95=  95.9 P99= 163.9 max= 189.4 min=  24.7
sample rates: [44100] | canales (1/2): 130 / 231
```

MAX = ceil(152.3 / 10) · 10 = **160 s**. CSV: `out_e1_early/durations.csv` (copia de `~/alzhaimer-detector-web/durations.csv`).

## 4. Export

```
python scripts/export_head.py .../seed_42/model_fold_0.pth out_e1_early/ \
  --name e1-audio-early --head-dim 128 --dropout 0.4 --layer-range 0 16 \
  --threshold 0.6699999999999999 --max-duration 160
```
`load_state_dict` estricto sin errores. `head.pth` sha256 `3c543aef…2644`. `config.json`: `threshold 0.6699999999999999`,
`layer_range [0,16]`, `max_duration_s 160.0`, `versions {torch 2.9.1+cu128, transformers 4.57.5, librosa 0.11.0}`.

## 5. Validación del export (cabeza sola, `.pt` precomputados)

55/55 sujetos del fold 0: **max\|sigmoid(logit) − prob\| = 1.192e-07** → OK (< 1e-5). Detalle en `logs/job3_1185664.out`.

## 6. Punta a punta (`Predictor.load(spec, head.pth).predict_path`, CPU)

Sujetos: el primer uid alfabético de cada clase en el fold 0.

| uid | clase | archivo | prob (outer_predictions) | P(AD) web | \|diff\| | label (umbral 0.67) |
|---|---|---|---|---|---|---|
| AD-M-57-163 | AD | wav estéreo, 88.1 s | 0.9944475 | 0.9944488 | 1.31e-06 | AD |
| HC-M-59-291 | HC | mp3 mono, 58.2 s | 0.9781969 | 0.9781826 | 1.44e-05 | AD (*) |

(*) Es un falso positivo que ya tiene el modelo en entrenamiento (prob 0.978), no un problema del export.

## requirements.txt sugerido

```
torch==2.9.1
transformers==4.57.5
librosa==0.11.0
soundfile==0.13.1
numpy==2.2.6
soxr==1.1.0
```
Además, ffmpeg/ffprobe **7.1** a nivel sistema (Docker). Para la paridad en MP3 importa más el decodificador que la
versión de Python (ver 3b-a). En PyPI, `torch==2.9.1` es la build por defecto; la de entrenamiento fue `+cu128`.

## Anomalías / notas

1. **La paridad 3b falla** en MP3 (GPU y CPU) y en CPU para todos los formatos (ver diagnóstico). No se cambió código.
   Las dos causas que se ven: el decodificador MP3 (ffmpeg vs libsndfile/mpg123: largo y valores) y la diferencia
   numérica GPU vs CPU (¿TF32?). Decisión pendiente: aceptar la tolerancia, decodificar MP3 igual que en
   entrenamiento, o recortar las muestras extra.
2. **ffmpeg:** en el login no estaba en el PATH, así que se creó el entorno `ffmpeg` (ffmpeg 9.0.2) según la regla. Pero
   `cogni` ya trae ffmpeg/ffprobe 7.1 en su `bin`, que va primero en el PATH. Todo corrió con **7.1** y el entorno
   `ffmpeg` no se usó (se puede borrar con `conda env remove -n ffmpeg`).
3. La partición `short` exige pedir ≥1 GPU, así que todos los jobs pidieron `--gres=gpu:1`.
4. **MAX = 160 s** usa el P99 de los 361 audios, **MCI incluido** (90). Hay 4 audios > 160 s (AD-M-84-313, AD-W-71-47,
   MCI-M-81-80, MCI-W-50-205), y **AD-W-71-47 está en el fold 0**: la web lo rechazaría aunque el modelo lo vio entero.
   En AD, P99 = 211.6 s.
5. El repo de investigación pasó a la branch `mendieta` durante la tarea (lo cambió el usuario). Checkpoints y
   dataset están fuera del repo (symlinks a `~/dementiaDetection/data`), así que no les afecta.
6. Archivos auxiliares no versionados: `out_e1_early/tools/` (scripts de job, wrapper GPU, validación y diagnóstico) y
   `out_e1_early/logs/` (salidas de slurm; `logs/emb/` tiene los embeddings GPU/CPU de los 5 sujetos, ~5 MB).
   También quedó `~/alzhaimer-detector-web/durations.csv` en la raíz del repo web (sin trackear).

Jobs: 1185641 (entorno, 3a, 3c), 1185663 (3b), 1185664 (export, 5, 6), 1185671 (diagnóstico de decodificación).
