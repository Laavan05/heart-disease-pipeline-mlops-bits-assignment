"""Shared fixtures for model and API tests."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocess import (
    clean_data,
    create_preprocessor,
    load_data,
    split_features_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"


@pytest.fixture(scope="session")
def model_artifact_paths(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Use repository artifacts or create temporary test-only equivalents.

    Trained model files are git-ignored in this project. Creating equivalent
    artifacts in pytest's temporary directory allows clean checkouts to run
    the model and API tests without polluting the repository.
    """
    model_path = PROJECT_ROOT / "models" / "best_model.pkl"
    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor.pkl"
    if model_path.is_file() and preprocessor_path.is_file():
        return model_path, preprocessor_path

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Prepared dataset required to create test artifacts: "
            f"{DATASET_PATH}"
        )

    output_dir = tmp_path_factory.mktemp("heart_model_artifacts")
    test_model_path = output_dir / "best_model.pkl"
    test_preprocessor_path = output_dir / "preprocessor.pkl"

    data = clean_data(load_data(DATASET_PATH))
    X, y = split_features_target(data)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    preprocessor = create_preprocessor(X_train)
    transformed_train = preprocessor.fit_transform(X_train)
    model: Any = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(transformed_train, y_train)

    joblib.dump(model, test_model_path)
    joblib.dump(preprocessor, test_preprocessor_path)
    return test_model_path, test_preprocessor_path


@pytest.fixture(scope="session")
def model_and_preprocessor(
    model_artifact_paths: tuple[Path, Path],
) -> tuple[Any, Any]:
    """Load a classifier and the preprocessor used with its feature data."""
    model_path, preprocessor_path = model_artifact_paths
    return joblib.load(model_path), joblib.load(preprocessor_path)


@pytest.fixture(scope="session")
def model_sample_features(model_and_preprocessor: tuple[Any, Any]) -> Any:
    """Transform one dataset record using the matching fitted preprocessor."""
    _, preprocessor = model_and_preprocessor
    data = pd.read_csv(DATASET_PATH)
    sample = data.drop(columns=["target"]).head(1)
    expected_features = getattr(preprocessor, "feature_names_in_", None)
    if expected_features is not None:
        sample = sample.reindex(columns=expected_features)

    return preprocessor.transform(sample)
