"""Tests for dataset loading, splitting, and feature preprocessing."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src import preprocess


@pytest.fixture
def project_root() -> Path:
    """Return the repository root containing the prepared dataset."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_dataset_path(tmp_path: Path) -> Path:
    """Write a small binary-classification dataset with missing features."""
    row_count = 30
    data = pd.DataFrame(
        {
            "age": np.arange(30, 30 + row_count, dtype=float),
            "cp": np.tile([1, 2, 3], row_count // 3),
            "chol": np.arange(170, 170 + row_count * 3, 3, dtype=float),
            "target": np.tile([0, 1], row_count // 2),
        }
    )
    data.loc[0, "age"] = np.nan
    data.loc[1, "chol"] = np.nan
    dataset_path = tmp_path / "heart_sample.csv"
    data.to_csv(dataset_path, index=False)
    return dataset_path


@pytest.fixture
def preprocessing_result(
    synthetic_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, object, pd.Series, pd.Series, object]:
    """Run the preprocessing workflow without writing a model artifact."""
    monkeypatch.setattr(preprocess, "dump", Mock())
    return preprocess.preprocess_data(synthetic_dataset_path)


def test_dataset_loading(project_root: Path) -> None:
    """Load the prepared heart disease CSV and verify its schema."""
    dataset_path = project_root / "data" / "raw" / "heart.csv"

    data = preprocess.load_data(dataset_path)

    assert not data.empty
    assert "target" in data.columns
    assert len(data) == 920


def test_target_column_must_exist() -> None:
    """Reject datasets that do not contain the supervised target."""
    data = pd.DataFrame({"age": [45.0, 55.0]})

    with pytest.raises(KeyError, match="target"):
        preprocess.clean_data(data)


def test_train_test_split_works(
    preprocessing_result: tuple[object, object, pd.Series, pd.Series, object],
) -> None:
    """Create non-empty, stratified training and test partitions."""
    X_train, X_test, y_train, y_test, _ = preprocessing_result

    assert X_train.shape[0] == 24
    assert X_test.shape[0] == 6
    assert len(y_train) == 24
    assert len(y_test) == 6
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_preprocessing_pipeline_is_created(
    synthetic_dataset_path: Path,
) -> None:
    """Build a ColumnTransformer from cleaned predictor features."""
    data = preprocess.clean_data(preprocess.load_data(synthetic_dataset_path))
    X, _ = preprocess.split_features_target(data)

    transformer = preprocess.create_preprocessor(X)

    assert isinstance(transformer, ColumnTransformer)
    assert transformer.transformers


def test_transformed_train_and_test_outputs_are_not_empty(
    preprocessing_result: tuple[object, object, pd.Series, pd.Series, object],
) -> None:
    """Verify that each transformed partition contains feature values."""
    X_train, X_test, y_train, y_test, _ = preprocessing_result

    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] > 0
    assert X_test.shape[1] > 0
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
