import os

MODEL_REPO = os.getenv("MODEL_REPO", "")            # repo HF con head.pth + config.json; vacio = stub
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
MODEL_CONFIG = os.getenv("MODEL_CONFIG", "")        # config.json local (opcional)
MODEL_HEAD = os.getenv("MODEL_HEAD", "")            # head.pth local (opcional)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
