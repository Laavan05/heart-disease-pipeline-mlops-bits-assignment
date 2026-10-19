"""Train and evaluate heart disease classifiers."""

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
    from .preprocess import preprocess_data
else:
    from preprocess import preprocess_data

LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "heart.csv"
)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SCREENSHOTS_DIR = Path(__file__).resolve().parents[1] / "screenshots"


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

    The shared preprocessing function cleans the data, makes a stratified
    train/test split, fits preprocessing on training data only, transforms
    both partitions, and saves the fitted preprocessor. The best classifier
    is selected by ROC-AUC, with F1 score as the tie-breaker.

    Args:
        filepath: Path to the prepared input CSV.

    Returns:
        A tuple containing the model-comparison dataframe and selected model
        name.
    """
    LOGGER.info("Starting model training workflow for %s", filepath)
    X_train, X_test, y_train, y_test, _ = preprocess_data(filepath)

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }

    metric_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for model_name, candidate in models.items():
        fitted_model = train_model(candidate, X_train, y_train)
        metrics, matrix, report = evaluate_model(
            fitted_model,
            X_test,
            y_test,
        )
        fitted_models[model_name] = fitted_model
        metric_rows.append({"model": model_name, **metrics})

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
    save_model(fitted_models[best_model_name], MODELS_DIR / "best_model.pkl")

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
