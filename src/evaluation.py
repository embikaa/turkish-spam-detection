"""
Evaluation utilities for model performance analysis.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray = None) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    # Class distribution in test set
    unique, counts = np.unique(y_true, return_counts=True)
    metrics["class_distribution"] = {
        ["Genuine", "Spam"][int(i)]: int(c)
        for i, c in zip(unique, counts)
    }

    # Full classification report
    report = classification_report(
        y_true, y_pred,
        target_names=["Genuine", "Spam"],
        output_dict=True
    )
    metrics["classification_report"] = report

    return metrics
