from pydantic import BaseModel


class PredictionResponse(BaseModel):

    prediction: str

    confidence: float

    fresh_probability: float

    rotten_probability: float


class HealthResponse(BaseModel):

    status: str