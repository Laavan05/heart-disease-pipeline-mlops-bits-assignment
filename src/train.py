"""Train and evaluate heart disease classifiers."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__:
    from .preprocess import create_preprocessor, load_and_split_data
else:
    from preprocess import create_preprocessor, load_and_split_data

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "heart.csv"
)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SCREENSHOTS_DIR = Path(__file__).resolve().parents[1] / "screenshots"
CV_SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


def train_model(model: Any, X_train: Any, y_train: pd.Series) -> Any:
    """Fit a classifier on the provided preprocessed training data.

    Args:
        model: An unfitted scikit-learn-compatible classifier.
        X_train: Transformed training features.
        y_train: Training target labels.

    Returns:
        The fitted classifier.
    """
    model_name = type(model).__name__
    LOGGER.info("Training %s on %d records", model_name, len(y_train))
    model.fit(X_train, y_train)
    LOGGER.info("Finished training %s", model_name)
    return model


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray, str]:
    """Calculate classification metrics and diagnostic outputs.

    ROC-AUC is calculated from the model's positive-class probabilities where
    available, or its decision scores otherwise.

    Args:
        model: A fitted binary scikit-learn classifier.
        X_test: Transformed test features.
        y_test: Test target labels.

    Returns:
        A tuple containing a metric dictionary, confusion matrix, and
        human-readable classification report.

    Raises:
        ValueError: If the classifier cannot provide probabilities or decision
            scores, or if the test set does not contain both target classes.
    """
    model_name = type(model).__name__
    LOGGER.info("Evaluating %s on %d records", model_name, len(y_test))
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        positive_class_index = list(model.classes_).index(1)
        positive_scores = model.predict_proba(X_test)[:, positive_class_index]
    elif hasattr(model, "decision_function"):
        positive_scores = model.decision_function(X_test)
    else:
        LOGGER.error("%s provides no scores for ROC-AUC", model_name)
        raise ValueError(
            f"{model_name} must implement predict_proba or decision_function "
            "to calculate ROC-AUC"
        )

    if y_test.nunique() < 2:
        LOGGER.error(
            "ROC-AUC cannot be computed because the test set has one class"
        )
        raise ValueError(
            "ROC-AUC requires both target classes in the test set"
        )

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, positive_scores)),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Heart Disease", "Heart Disease"],
        zero_division=0,
    )

    LOGGER.info(
        "%s evaluation: accuracy=%.4f, precision=%.4f, recall=%.4f, "
        "f1=%.4f, roc_auc=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics, matrix, report


def tune_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, GridSearchCV]:
    """Tune candidate classifiers with leakage-safe stratified CV.

    Each search contains the feature preprocessor so that imputers, scaling,
    and encoding are fitted only on the training fold.
    """
    configurations: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_leaf": [1, 2],
            },
        ),
    }
    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    searches: dict[str, GridSearchCV] = {}

    for model_name, (classifier, parameter_grid) in configurations.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(X_train)),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=list(CV_SCORING),
            refit="roc_auc",
            cv=cross_validator,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        LOGGER.info(
            "Tuning %s with %d-fold stratified cross-validation",
            model_name,
            cross_validator.n_splits,
        )
        search.fit(X_train, y_train)
        searches[model_name] = search
        LOGGER.info(
            "%s cross-validation selected %s with ROC-AUC %.4f",
            model_name,
            search.best_params_,
            search.best_score_,
        )

    return searches


def cross_validation_metrics(search: GridSearchCV) -> dict[str, float]:
    """Return mean and standard deviation for the selected CV parameter set."""
    best_index = search.best_index_
    metrics: dict[str, float] = {}
    for metric_name in CV_SCORING:
        metrics[f"cv_{metric_name}_mean"] = float(
            search.cv_results_[f"mean_test_{metric_name}"][best_index]
        )
        metrics[f"cv_{metric_name}_std"] = float(
            search.cv_results_[f"std_test_{metric_name}"][best_index]
        )
    return metrics


def save_model(model: Any, filepath: str | Path) -> Path:
    """Persist a fitted model with joblib.

    Args:
        model: Fitted model to serialize.
        filepath: Destination path for the model artifact.

    Returns:
        The resolved path to the saved model.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    LOGGER.info("Saved fitted model to %s", path.resolve())
    return path


def _save_confusion_matrix(
    matrix: np.ndarray,
    model_name: str,
    filepath: Path,
) -> None:
    """Render a labeled confusion matrix and save it as a PNG image.

    Args:
        matrix: 2-by-2 confusion matrix in class order 0, 1.
        model_name: Human-readable classifier name for the chart title.
        filepath: Destination image path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=f"{model_name} Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        xticklabels=["No Disease (0)", "Disease (1)"],
        yticks=[0, 1],
        yticklabels=["No Disease (0)", "Disease (1)"],
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:,}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row_index, column_index] > threshold
                    else "black"
                ),
                fontsize=12,
            )
    figure.tight_layout()
    figure.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved %s confusion matrix plot to %s", model_name, filepath)


def run_training(
    filepath: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, str]:
    """Train, compare, and save two classifiers using the prepared dataset.

    Candidate models are tuned with five-fold stratified cross-validation.
    Preprocessing is fitted within each fold. The best model is selected by
    mean cross-validation ROC-AUC, with mean cross-validation F1 as a
    tie-breaker, and is evaluated once on the held-out test partition.

    Args:
        filepath: Path to the prepared input CSV.

    Returns:
        A tuple containing the model-comparison dataframe and selected model
        name.
    """
    LOGGER.info("Starting model training workflow for %s", filepath)
    X_train, X_test, y_train, y_test = load_and_split_data(filepath)
    searches = tune_candidate_models(X_train, y_train)

    metric_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for model_name, search in searches.items():
        fitted_model = search.best_estimator_
        metrics, matrix, report = evaluate_model(
            fitted_model,
            X_test,
            y_test,
        )
        fitted_models[model_name] = fitted_model
        metric_rows.append(
            {
                "model": model_name,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                **cross_validation_metrics(search),
                **metrics,
            }
        )

        print(f"\n{model_name} classification report:\n{report}")
        figure_name = (
            "logistic_confusion_matrix.png"
            if model_name == "Logistic Regression"
            else "random_forest_confusion_matrix.png"
        )
        _save_confusion_matrix(
            matrix,
            model_name,
            SCREENSHOTS_DIR / figure_name,
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = metrics_df.sort_values(
        by=["cv_roc_auc_mean", "cv_f1_mean"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    print(
        "\nModel comparison (selected by CV ROC-AUC, then CV F1; "
        "hold-out metrics shown separately):"
    )
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    save_model(
        best_pipeline.named_steps["classifier"],
        MODELS_DIR / "best_model.pkl",
    )
    save_model(
        best_pipeline.named_steps["preprocessor"],
        MODELS_DIR / "preprocessor.pkl",
    )

    metrics_path = MODELS_DIR / "model_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    LOGGER.info("Saved model evaluation results to %s", metrics_path)
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
    selected_metrics, selected_model = run_training()
    LOGGER.info(
        "Training workflow complete; selected model: %s; metrics rows: %d",
        selected_model,
        len(selected_metrics),
    )
