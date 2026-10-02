"""Classification metrics and evaluation plots."""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    probabilities = np.asarray(probabilities).reshape(-1)
    predicted = (probabilities >= 0.5).astype(np.int32)
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def save_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, float],
    history: Any,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as result_file:
        json.dump(metrics, result_file, indent=2)

    predicted = (probabilities.reshape(-1) >= 0.5).astype(np.int32)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        ax=axes[0],
    )
    axes[0].set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    axes[1].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC AUC = {metrics['roc_auc']:.3f}",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(
        title="ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.05),
    )
    axes[1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, label in (("accuracy", "Accuracy"), ("val_accuracy", "Validation accuracy")):
        if key in history.history:
            axes[0].plot(history.history[key], label=label)
    for key, label in (("loss", "Loss"), ("val_loss", "Validation loss")):
        if key in history.history:
            axes[1].plot(history.history[key], label=label)
    axes[0].set(title="Training accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(figure)
