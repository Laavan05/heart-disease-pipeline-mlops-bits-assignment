"""Train heart disease classifiers and track experiments with MLflow."""

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

if __package__:
    from .preprocess import preprocess_data
    from .train import (
        _save_confusion_matrix,
        evaluate_model,
        train_model,
    )
else:
    from preprocess import preprocess_data
    from train import _save_confusion_matrix, evaluate_model, train_model

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
DEFAULT_EXPERIMENT_NAME = "Heart Disease Prediction"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"


def _model_configurations() -> list[tuple[str, Any, dict[str, Any]]]:
    """Build the two requested model configurations and their logged params.

    Returns:
        A list of model name, estimator, and MLflow parameter dictionary.
    """
    return [
        (
            "Logistic Regression",
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {
                "model_name": "Logistic Regression",
                "random_state": RANDOM_STATE,
                "max_iter": 1000,
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
            ),
            {
                "model_name": "Random Forest",
                "random_state": RANDOM_STATE,
                "n_estimators": 200,
            },
        ),
    ]


def _save_and_log_confusion_matrix(
    matrix: np.ndarray,
    model_name: str,
    figure_path: Path,
) -> None:
    """Save a confusion matrix locally and log it as a run artifact.

    Args:
        matrix: Confusion matrix in binary class order.
        model_name: Human-readable classifier name.
        figure_path: Local path at which to save the plot.
    """
    _save_confusion_matrix(matrix, model_name, figure_path)
    mlflow.log_artifact(
        str(figure_path),
        artifact_path="confusion_matrix",
    )


def run_mlflow_training(
    filepath: str | Path = DEFAULT_DATA_PATH,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> tuple[pd.DataFrame, str]:
    """Train, evaluate, and track each model as a separate MLflow run.

    Preprocessing is fit once on the training partition and shared by both
    candidate models. Each run logs its parameters, evaluation metrics,
    confusion-matrix image, and fitted scikit-learn model. After comparing
    ROC-AUC (with F1 as a tie-breaker), the best model is saved locally and
    the summary CSV is logged to both runs.

    The MLflow tracking URI can be configured through the
    ``MLFLOW_TRACKING_URI`` environment variable. If unset, MLflow's default
    local tracking store is used.

    Args:
        filepath: Path to the prepared CSV dataset.
        experiment_name: Name of the MLflow experiment.

    Returns:
        A tuple containing the comparison dataframe and selected model name.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        LOGGER.info("Using MLflow tracking URI %s", tracking_uri)

    LOGGER.info("Setting MLflow experiment to %s", experiment_name)
    mlflow.set_experiment(experiment_name)

    LOGGER.info("Preprocessing dataset from %s", filepath)
    X_train, X_test, y_train, y_test, _ = preprocess_data(filepath)

    metric_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    fitted_models: dict[str, Any] = {}

    for model_name, candidate, parameters in _model_configurations():
        run_name = model_name.lower().replace(" ", "_")
        LOGGER.info("Starting MLflow run for %s", model_name)
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.log_params(parameters)
            fitted_model = train_model(candidate, X_train, y_train)
            metrics, matrix, report = evaluate_model(
                fitted_model,
                X_test,
                y_test,
            )

            mlflow.log_metrics(metrics)
            figure_name = (
                "logistic_confusion_matrix.png"
                if model_name == "Logistic Regression"
                else "random_forest_confusion_matrix.png"
            )
            figure_path = SCREENSHOTS_DIR / figure_name
            _save_and_log_confusion_matrix(
                matrix,
                model_name,
                figure_path,
            )
            trusted_types = (
                ["sklearn.tree._tree.Tree"]
                if model_name == "Random Forest"
                else None
            )
            mlflow.sklearn.log_model(
                sk_model=fitted_model,
                name="model",
                skops_trusted_types=trusted_types,
            )

            metric_rows.append({"model": model_name, **metrics})
            run_ids.append(active_run.info.run_id)
            fitted_models[model_name] = fitted_model

        print(f"\n{model_name} classification report:\n{report}")

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=["roc_auc", "f1_score"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    print("\nModel comparison (sorted by ROC-AUC, then F1 score):")
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_model_name = str(metrics_df.iloc[0]["model"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(fitted_models[best_model_name], best_model_path)
    LOGGER.info(
        "Saved best model (%s) to %s",
        best_model_name,
        best_model_path,
    )

    metrics_path = MODELS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    LOGGER.info("Saved model comparison results to %s", metrics_path)

    for run_id in run_ids:
        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
            run_id=run_id,
        )

    LOGGER.info(
        "Selected %s with ROC-AUC=%.4f and F1=%.4f",
        best_model_name,
        metrics_df.iloc[0]["roc_auc"],
        metrics_df.iloc[0]["f1_score"],
    )
    return metrics_df, best_model_name


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    comparison, best_model = run_mlflow_training()
    LOGGER.info(
        "MLflow training workflow complete; best model: %s; "
        "evaluated models: %d",
        best_model,
        len(comparison),
    )
