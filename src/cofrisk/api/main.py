import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from cofrisk.data.validation import EXPECTED_FEATURES
from cofrisk.models.predictor import RiskPredictor


MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="CofRisk API",
    description="Corporate default risk prediction API",
    version="1.0.0",
)

predictor = RiskPredictor()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float]


class PredictionResponse(BaseModel):
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_class: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    received_features = set(request.features.keys())
    expected_features = set(EXPECTED_FEATURES)

    missing_features = expected_features - received_features
    unknown_features = received_features - expected_features

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing features",
                "features": sorted(missing_features),
            },
        )

    if unknown_features:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unknown features",
                "features": sorted(unknown_features),
            },
        )

    start = time.perf_counter()

    result = predictor.predict(request.features)

    latency_ms = (time.perf_counter() - start) * 1000

    print(
        f"prediction_completed "
        f"risk_class={result['risk_class']} "
        f"latency_ms={latency_ms:.2f} "
        f"model_version={MODEL_VERSION}"
    )

    return result