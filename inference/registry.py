"""Registry minimo de modelos. v1: una sola entrada. Cada encoder grande ocupa varios GB,
asi que solo se mantiene cargado un Predictor a la vez; el encoder se comparte por id."""

from .config import ModelSpec
from .predictor import Predictor

_loaded: dict[str, Predictor] = {}


def get_predictor(spec: ModelSpec, head_path=None) -> Predictor:
    if spec.name not in _loaded:
        _loaded.clear()                      # liberar el modelo anterior antes de cargar otro
        _loaded[spec.name] = Predictor.load(spec, head_path)
    return _loaded[spec.name]
