"""Tests for the saved heart disease classifier."""

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def model_artifacts(
    model_artifact_paths: tuple[Path, Path],
    model_and_preprocessor: tuple[Any, Any],
) -> tuple[Any, Any]:
    """Load the fitted classifier and its matching feature preprocessor."""
    model_path, preprocessor_path = model_artifact_paths
    assert model_path.is_file(), f"Expected trained model at {model_path}"
    assert preprocessor_path.is_file(), (
        f"Expected fitted preprocessor at {preprocessor_path}"
    )
    return model_and_preprocessor


def test_best_model_artifact_exists(
    model_artifact_paths: tuple[Path, Path],
) -> None:
    """Confirm the expected trained model file exists."""
    model_path, _ = model_artifact_paths
    assert model_path.is_file()
    assert model_path.stat().st_size > 0


def test_best_model_can_be_loaded(model_artifacts: tuple[Any, Any]) -> None:
    """Load the serialized classifier and check its inference interface."""
    model, _ = model_artifacts

    assert callable(getattr(model, "predict", None))
    assert callable(getattr(model, "predict_proba", None))


def test_prediction_returns_valid_class(
    model_artifacts: tuple[Any, Any],
    model_sample_features: Any,
) -> None:
    """Return one of the documented binary target classes."""
    model, _ = model_artifacts

    prediction = int(model.predict(model_sample_features)[0])

    assert prediction in {0, 1}


def test_prediction_probability_is_between_zero_and_one(
    model_artifacts: tuple[Any, Any],
    model_sample_features: Any,
) -> None:
    """Return a valid probability for the positive disease class."""
    model, _ = model_artifacts
    class_index = list(model.classes_).index(1)

    probability = float(
        model.predict_proba(model_sample_features)[0][class_index]
    )

    assert 0.0 <= probability <= 1.0
