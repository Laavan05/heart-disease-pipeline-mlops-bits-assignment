"""Data loading and preprocessing utilities for heart disease modeling."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "target"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MAX_CATEGORICAL_CARDINALITY = 10


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from ``filepath``.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        Dataset contents as a pandas DataFrame.

    Raises:
        FileNotFoundError: If ``filepath`` does not identify a file.
        pandas.errors.ParserError: If pandas cannot parse the CSV structure.
    """
    path = Path(filepath)
    LOGGER.info("Loading dataset from %s", path)
    if not path.is_file():
        LOGGER.error("Dataset file does not exist: %s", path)
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    data = pd.read_csv(path)
    LOGGER.info("Loaded dataset with %d rows and %d columns", *data.shape)
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric, remove duplicates, and validate the target.

    Invalid numeric values are coerced to missing values so feature imputers
    can handle them consistently during train/test preprocessing. Rows with a
    missing target are removed because they cannot be used for supervised
    learning. Feature missing values are retained for pipeline imputation.

    Args:
        df: Raw dataset containing the target column.

    Returns:
        A cleaned copy of the dataset.

    Raises:
        KeyError: If the target column is missing.
        ValueError: If no usable records or predictor columns remain.
    """
    LOGGER.info("Cleaning dataset with %d rows and %d columns", *df.shape)
    if TARGET_COLUMN not in df.columns:
        LOGGER.error("Required target column %r is missing", TARGET_COLUMN)
        raise KeyError(f"Required target column {TARGET_COLUMN!r} is missing")

    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_target_count = int(cleaned[TARGET_COLUMN].isna().sum())
    if invalid_target_count:
        LOGGER.warning(
            "Dropping %d rows with missing or non-numeric targets",
            invalid_target_count,
        )
        cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        LOGGER.info("Removing %d duplicate rows", duplicate_count)
        cleaned = cleaned.drop_duplicates()

    if cleaned.empty:
        LOGGER.error("No rows remain after cleaning")
        raise ValueError("No rows remain after cleaning")
    if len(cleaned.columns) < 2:
        LOGGER.error("No predictor columns remain after cleaning")
        raise ValueError(
            "Dataset must contain at least one predictor and the target"
        )

    target_values = set(cleaned[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        LOGGER.error(
            "Target contains values outside the expected binary classes: %s",
            target_values,
        )
        raise ValueError(
            "Target must contain only binary values 0 and 1; "
            f"found {sorted(target_values)}"
        )

    feature_data = cleaned.drop(columns=TARGET_COLUMN)
    missing_features = int(feature_data.isna().sum().sum())
    LOGGER.info(
        "Cleaning complete: %d rows remain; %d feature values are missing",
        len(cleaned),
        missing_features,
    )
    return cleaned.reset_index(drop=True)


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from the binary target.

    Args:
        df: Cleaned dataset containing the target column.

    Returns:
        A tuple containing ``X`` (predictors) and ``y`` (target labels).

    Raises:
        KeyError: If the target column is missing.
        ValueError: If the dataset has no predictor columns.
    """
    if TARGET_COLUMN not in df.columns:
        LOGGER.error("Required target column %r is missing", TARGET_COLUMN)
        raise KeyError(f"Required target column {TARGET_COLUMN!r} is missing")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    if X.empty or X.shape[1] == 0:
        LOGGER.error("Dataset has no predictor columns")
        raise ValueError("Dataset must contain at least one predictor column")

    LOGGER.info(
        "Separated %d predictor columns and %d target labels",
        X.shape[1],
        len(y),
    )
    return X, y


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create imputing, scaling, and encoding transformers for predictors.

    Numeric columns are median-imputed and standardized. Categorical columns
    are identified from non-numeric dtypes and low-cardinality integer-valued
    numeric columns, then most-frequent-imputed and one-hot encoded. The
    cardinality rule supports categorical features stored as numbers in the
    source dataset.

    Args:
        X: Predictor dataframe used to identify the feature types.

    Returns:
        An unfitted sklearn ColumnTransformer.

    Raises:
        ValueError: If ``X`` contains no columns.
    """
    if X.shape[1] == 0:
        LOGGER.error("Cannot create a preprocessor for an empty feature set")
        raise ValueError("At least one predictor column is required")

    categorical_columns: list[Any] = []
    numeric_columns: list[Any] = []

    for column in X.columns:
        series = X[column]
        if not pd.api.types.is_numeric_dtype(series):
            categorical_columns.append(column)
            continue

        observed = series.dropna()
        is_integer_valued = observed.empty or np.isclose(
            observed.to_numpy(dtype=float),
            np.round(observed.to_numpy(dtype=float)),
        ).all()
        if (
            is_integer_valued
            and observed.nunique() <= MAX_CATEGORICAL_CARDINALITY
        ):
            categorical_columns.append(column)
        else:
            numeric_columns.append(column)

    LOGGER.info(
        "Identified %d numeric and %d categorical predictor columns",
        len(numeric_columns),
        len(categorical_columns),
    )

    transformers: list[tuple[str, Pipeline, list[Any]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_and_split_data(
    filepath: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, and stratify raw features for model selection.

    Keeping the raw feature frames allows cross-validation pipelines to fit
    imputers, scalers, and encoders independently inside each fold.

    Args:
        filepath: Path to the prepared CSV file.

    Returns:
        Raw training and test features, followed by their target labels.
    """
    data = clean_data(load_data(filepath))
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=y,
    )
    LOGGER.info(
        "Split data into %d training and %d test records",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def preprocess_data(
    filepath: str | Path,
) -> tuple[Any, Any, pd.Series, pd.Series, ColumnTransformer]:
    """Clean data, split it, fit preprocessing, and persist the fitted object.

    The split is stratified and performed before fitting any imputer, scaler,
    or encoder to avoid leaking test-set information into training. The
    fitted preprocessor is saved to ``models/preprocessor.pkl`` under the
    repository root that contains this module.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        A tuple of transformed training features, transformed test features,
        training labels, test labels, and the fitted preprocessor.

    Raises:
        FileNotFoundError: If the input file does not exist.
        KeyError: If the target column is missing.
        ValueError: If data cleaning or stratified splitting is not possible.
    """
    LOGGER.info("Starting preprocessing for %s", filepath)
    X_train, X_test, y_train, y_test = load_and_split_data(filepath)

    preprocessor = create_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    LOGGER.info(
        "Transformed features: training shape %s; test shape %s",
        X_train_processed.shape,
        X_test_processed.shape,
    )

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = model_dir / "preprocessor.pkl"
    dump(preprocessor, preprocessor_path)
    LOGGER.info("Saved fitted preprocessor to %s", preprocessor_path)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    repository_root = Path(__file__).resolve().parents[1]
    dataset_path = repository_root / "data" / "raw" / "heart.csv"
    X_train_processed, X_test_processed, y_train, y_test, _ = preprocess_data(
        dataset_path
    )
    LOGGER.info(
        "Preprocessing smoke test complete: X_train=%s, X_test=%s, "
        "y_train=%s, y_test=%s",
        X_train_processed.shape,
        X_test_processed.shape,
        y_train.shape,
        y_test.shape,
    )
