"""FastAPI application for heart disease prediction."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"

model: Any | None = None
preprocessor: Any | None = None


class HeartDiseaseRequest(BaseModel):
    """Validated predictor values for a single heart disease prediction."""

    model_config = ConfigDict(extra="forbid")

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


class HeartDiseasePrediction(BaseModel):
    """Prediction result returned by the prediction endpoint."""

    prediction: int
    prediction_label: str
    probability: float


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Load and validate serialized model artifacts at application startup."""
    global model, preprocessor

    missing_paths = [
        path for path in (MODEL_PATH, PREPROCESSOR_PATH) if not path.is_file()
    ]
    if missing_paths:
        missing_list = ", ".join(str(path) for path in missing_paths)
        LOGGER.error("Required model artifact(s) not found: %s", missing_list)
        raise RuntimeError(
            f"Required model artifact(s) not found: {missing_list}"
        )

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as error:
        LOGGER.exception("Failed to load model artifacts")
        raise RuntimeError("Failed to load model artifacts") from error

    if not callable(getattr(loaded_model, "predict", None)):
        LOGGER.error("Loaded model does not provide a predict method")
        raise RuntimeError("Loaded model artifact does not support prediction")
    if not callable(getattr(loaded_preprocessor, "transform", None)):
        LOGGER.error("Loaded preprocessor does not provide a transform method")
        raise RuntimeError(
            "Loaded preprocessor artifact does not support transform"
        )
    if not callable(getattr(loaded_model, "predict_proba", None)):
        LOGGER.error("Loaded model does not provide prediction probabilities")
        raise RuntimeError(
            "Loaded model artifact does not support prediction probabilities"
        )

    model = loaded_model
    preprocessor = loaded_preprocessor
    LOGGER.info(
        "Loaded model from %s and preprocessor from %s",
        MODEL_PATH,
        PREPROCESSOR_PATH,
    )
    yield
    LOGGER.info("Shutting down heart disease prediction API")


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease presence from clinical measurements.",
    version="1.0.0",
    lifespan=lifespan,
)

# Keep metrics available for monitoring without adding the endpoint to Swagger.
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.get("/")
def root() -> dict[str, str]:
    """Return API name and available endpoint hints."""
    return {
        "message": "Heart Disease Prediction API",
        "health": "/health",
        "prediction": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the model and preprocessor are loaded."""
    if model is None or preprocessor is None:
        LOGGER.warning(
            "Health check requested while model artifacts are unavailable"
        )
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded",
        )
    return {"status": "healthy"}


@app.post("/predict", response_model=HeartDiseasePrediction)
def predict(
    payload: HeartDiseaseRequest,
    request: Request,
) -> HeartDiseasePrediction:
    """Transform request features and return a binary prediction and score.

    Args:
        payload: Validated clinical feature values.
        request: FastAPI request context used to access application state.

    Returns:
        The predicted class, readable class label, and disease probability.

    Raises:
        HTTPException: If model artifacts are unavailable or inference fails.
    """
    loaded_model = model
    loaded_preprocessor = preprocessor
    if loaded_model is None or loaded_preprocessor is None:
        LOGGER.error(
            "Prediction requested while model artifacts are unavailable"
        )
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded",
        )

    try:
        features = pd.DataFrame([payload.model_dump()])
        expected_features = getattr(
            loaded_preprocessor,
            "feature_names_in_",
            None,
        )
        if expected_features is not None:
            features = features.reindex(columns=expected_features)

        transformed_features = loaded_preprocessor.transform(features)
        prediction = int(loaded_model.predict(transformed_features)[0])
        classes = list(loaded_model.classes_)
        if prediction not in (0, 1):
            raise ValueError(f"Unexpected prediction class: {prediction}")
        if 1 not in classes:
            raise ValueError("Loaded model has no positive class 1")

        positive_class_index = classes.index(1)
        probability = float(
            loaded_model.predict_proba(transformed_features)[0][
                positive_class_index
            ]
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Model returned invalid probability: {probability}"
            )

        prediction_label = (
            "Heart Disease" if prediction == 1 else "No Heart Disease"
        )
        LOGGER.info(
            "Prediction completed for %s: class=%d, probability=%.4f",
            request.client.host if request.client else "unknown client",
            prediction,
            probability,
        )
        return HeartDiseasePrediction(
            prediction=prediction,
            prediction_label=prediction_label,
            probability=probability,
        )
    except HTTPException:
        raise
    except Exception as error:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from error
