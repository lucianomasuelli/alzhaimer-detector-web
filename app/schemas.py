from pydantic import BaseModel


class PredictionOut(BaseModel):
    label: str
    probabilities: dict[str, float]
    threshold: float
    duration_s: float
    model: str
    stub: bool


class HealthOut(BaseModel):
    status: str
    model: str
    stub: bool
