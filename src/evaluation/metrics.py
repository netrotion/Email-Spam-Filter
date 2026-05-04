from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true, y_pred, y_prob: list[float] | None = None
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    return metrics


def metrics_to_markdown(title: str, metrics: dict[str, Any]) -> str:
    confusion = metrics["confusion_matrix"]
    report = metrics["classification_report"]
    spam_report = report.get("1", {})
    ham_report = report.get("0", {})

    lines = [
        f"# {title}",
        "",
        "## Overall Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1-score: {metrics['f1_score']:.4f}",
    ]

    if "roc_auc" in metrics:
        lines.extend(
            [
                f"- ROC AUC: {metrics['roc_auc']:.4f}",
                f"- Brier Score: {metrics['brier_score']:.4f}",
                f"- Log Loss: {metrics['log_loss']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-Class Metrics",
            "",
            f"- Ham precision/recall/F1: {ham_report.get('precision', 0.0):.4f} / {ham_report.get('recall', 0.0):.4f} / {ham_report.get('f1-score', 0.0):.4f}",
            f"- Spam precision/recall/F1: {spam_report.get('precision', 0.0):.4f} / {spam_report.get('recall', 0.0):.4f} / {spam_report.get('f1-score', 0.0):.4f}",
            "",
            "## Confusion Matrix",
            "",
            "| Actual \\ Predicted | Ham | Spam |",
            "| --- | ---: | ---: |",
            f"| Ham | {confusion[0][0]} | {confusion[0][1]} |",
            f"| Spam | {confusion[1][0]} | {confusion[1][1]} |",
            "",
        ]
    )
    return "\n".join(lines)
