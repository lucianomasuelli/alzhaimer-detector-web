FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces ejecuta como UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY --chown=user inference ./inference
COPY --chown=user app ./app

# Opcional: hornear XLS-R en la imagen para evitar descargar ~4 GB en cada arranque en frio
# RUN python -c "from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor as F; \
#   Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-1b'); F.from_pretrained('facebook/wav2vec2-xls-r-1b')"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
