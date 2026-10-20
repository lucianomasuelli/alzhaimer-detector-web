import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference.audio import ALLOWED_EXTENSIONS, AudioError
from inference.config import ModelSpec
from inference.registry import get_predictor

from . import settings
from .schemas import HealthOut, PredictionOut

STATIC = Path(__file__).parent / "static"


def load_default_predictor():
    spec, head = ModelSpec(), None
    if settings.MODEL_REPO:
        from huggingface_hub import hf_hub_download
        cfg = hf_hub_download(settings.MODEL_REPO, "config.json", revision=settings.MODEL_REVISION)
        head = hf_hub_download(settings.MODEL_REPO, "head.pth", revision=settings.MODEL_REVISION)
        spec = ModelSpec.from_json(cfg)
    elif settings.MODEL_CONFIG:
        spec = ModelSpec.from_json(settings.MODEL_CONFIG)
        head = settings.MODEL_HEAD or None
    return get_predictor(spec, head)


def create_app(predictor_factory=load_default_predictor) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor_factory()      # una sola vez al iniciar
        app.state.lock = asyncio.Semaphore(1)          # 1 inferencia a la vez (CPU/RAM)
        yield

    app = FastAPI(title="Alzheimer voice demo", lifespan=lifespan)

    @app.get("/health", response_model=HealthOut)
    def health():
        p = app.state.predictor
        return HealthOut(status="ok", model=p.spec.name, stub=p.stub)

    @app.post("/predict", response_model=PredictionOut)
    async def predict(file: UploadFile = File(...)):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Formato no admitido ({ext}). Admitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        data = await file.read(limit + 1)             # solo en memoria, nunca a disco
        if len(data) > limit:
            raise HTTPException(413, f"Archivo demasiado grande (max {settings.MAX_UPLOAD_MB} MB).")
        async with app.state.lock:
            try:
                pred = await asyncio.to_thread(app.state.predictor.predict_bytes, data)
            except AudioError as e:
                raise HTTPException(422, str(e))
        return pred.to_dict()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
