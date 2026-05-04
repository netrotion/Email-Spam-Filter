from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MODEL_ARTIFACT_PATH, PROCESSED_DATA_DIR, TRAINING_RESULTS_PATH
from src.evaluation.metrics import compute_binary_classification_metrics, metrics_to_markdown
from src.models.inference import load_model_metadata, load_trained_pipeline


def evaluate_saved_model() -> dict:
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed test split at {test_path}. Run `python -m src.models.train` first."
        )

    dataframe = pd.read_csv(test_path)
    pipeline = load_trained_pipeline(MODEL_ARTIFACT_PATH)
    predictions = pipeline.predict(dataframe["text"].tolist())
    probabilities = pipeline.predict_proba(dataframe["text"].tolist())[:, 1]
    metrics = compute_binary_classification_metrics(
        dataframe["label"].tolist(), predictions.tolist(), probabilities.tolist()
    )

    training_metadata = load_model_metadata()
    title = f"Evaluation Results ({training_metadata.get('best_model', 'unknown model')})"
    TRAINING_RESULTS_PATH.write_text(metrics_to_markdown(title, metrics), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = evaluate_saved_model()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"Detailed report written to: {Path(TRAINING_RESULTS_PATH)}")


if __name__ == "__main__":
    main()
