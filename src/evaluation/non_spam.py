from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_TRAINING_CONFIG, DOCS_DIR, PROCESSED_DATA_DIR
from src.data.dataset import extract_email_text_from_file
from src.data.download import ensure_spamassassin_corpus
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.inference import load_model_metadata, load_trained_pipeline


NON_SPAM_EVALUATION_PATH = DOCS_DIR / "NON_SPAM_EVALUATION.md"


def _configured_ham_sources(metadata: dict[str, Any]) -> tuple[str, ...]:
    configured_sources = (metadata.get("config") or {}).get("spamassassin_ham_sources")
    if isinstance(configured_sources, (list, tuple)):
        return tuple(str(source) for source in configured_sources)
    return DEFAULT_TRAINING_CONFIG.spamassassin_ham_sources


def _load_non_spam_corpora() -> dict[str, list[str]]:
    corpora = ensure_spamassassin_corpus(force=False)
    loaded: dict[str, list[str]] = {}

    for corpus_name, folder in corpora.items():
        if corpus_name.startswith("spam"):
            continue

        texts: list[str] = []
        for path in sorted(Path(folder).iterdir()):
            if not path.is_file():
                continue
            try:
                text = extract_email_text_from_file(path)
            except Exception:
                continue
            if text.strip():
                texts.append(text)
        loaded[corpus_name] = texts

    return loaded


def _spamassassin_sources_inside_training(metadata: dict[str, Any]) -> bool:
    source_counts = (metadata.get("dataset_summary") or {}).get("source_counts") or {}
    return any(
        f"spamassassin_{source_name}" in source_counts
        for source_name in _configured_ham_sources(metadata)
    )


def _load_heldout_non_spam_test_sources(metadata: dict[str, Any]) -> dict[str, list[str]]:
    test_frame = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    heldout: dict[str, list[str]] = {}

    for source_name in _configured_ham_sources(metadata):
        dataset_source = f"spamassassin_{source_name}"
        source_rows = test_frame[
            (test_frame["source"] == dataset_source) & (test_frame["label"] == 0)
        ]
        texts = source_rows["text"].fillna("").astype(str).tolist()
        if texts:
            heldout[source_name] = texts

    return heldout


def _top_false_positive_examples(
    texts: list[str],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    limit: int = 5,
) -> list[dict[str, Any]]:
    false_positive_indexes = [index for index, pred in enumerate(predictions) if int(pred) == 1]
    ranked = sorted(false_positive_indexes, key=lambda index: probabilities[index], reverse=True)

    examples = []
    for index in ranked[:limit]:
        examples.append(
            {
                "spam_probability": round(float(probabilities[index]), 4),
                "preview": texts[index][:180].replace("\n", " "),
            }
        )
    return examples


def _evaluate_non_spam_sources(pipeline) -> dict[str, Any]:
    loaded = _load_non_spam_corpora()
    results: dict[str, Any] = {}

    for source_name, texts in loaded.items():
        probabilities = pipeline.predict_proba(texts)[:, 1]
        predictions = pipeline.predict(texts)
        results[source_name] = {
            "samples": len(texts),
            "predicted_ham": int((predictions == 0).sum()),
            "predicted_spam": int((predictions == 1).sum()),
            "false_positive_rate": float((predictions == 1).mean()) if len(predictions) else 0.0,
            "avg_spam_probability": float(probabilities.mean()) if len(probabilities) else 0.0,
            "top_false_positives": _top_false_positive_examples(
                texts=texts,
                probabilities=probabilities,
                predictions=predictions,
            ),
        }

    return results


def _threshold_sweep(pipeline, *, non_spam_sources: dict[str, list[str]]) -> list[dict[str, Any]]:
    test_frame = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    test_probabilities = pipeline.predict_proba(test_frame["text"].tolist())[:, 1]
    y_true = test_frame["label"].to_numpy()

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    source_probabilities = {
        source_name: pipeline.predict_proba(texts)[:, 1]
        for source_name, texts in non_spam_sources.items()
    }
    combined_non_spam = (
        np.concatenate(list(source_probabilities.values()))
        if source_probabilities
        else np.asarray([], dtype=float)
    )

    rows = []
    for threshold in thresholds:
        test_predictions = (test_probabilities >= threshold).astype(int)
        metrics = compute_binary_classification_metrics(
            y_true.tolist(),
            test_predictions.tolist(),
            test_probabilities.tolist(),
        )
        row = {
            "threshold": threshold,
            "test_accuracy": float(metrics["accuracy"]),
            "test_precision": float(metrics["precision"]),
            "test_recall": float(metrics["recall"]),
            "test_f1": float(metrics["f1_score"]),
            "all_ham_fpr": float((combined_non_spam >= threshold).mean()) if len(combined_non_spam) else 0.0,
        }
        for source_name, probabilities in source_probabilities.items():
            row[f"{source_name}_fpr"] = float((probabilities >= threshold).mean())
        rows.append(row)

    return rows


def recommend_next_step(threshold_rows: list[dict[str, Any]]) -> dict[str, str]:
    baseline = next(row for row in threshold_rows if abs(row["threshold"] - 0.50) < 1e-9)
    strictest = next(row for row in threshold_rows if abs(row["threshold"] - 0.95) < 1e-9)

    if strictest["all_ham_fpr"] > 0.20:
        return {
            "decision": "augment_ham_data_and_retrain",
            "reason": (
                "Ngay ca khi nang threshold len 0.95, false-positive rate tren non-spam van vuot 20%, "
                "nen threshold tuning don thuan khong du."
            ),
            "next_step": (
                "Them easy_ham, easy_ham_2, hard_ham vao nhom ham, retrain model contextual, "
                "sau do benchmark lai tren ca test split hien tai va bo non-spam ngoai mien."
            ),
        }

    if baseline["all_ham_fpr"] > 0.10:
        return {
            "decision": "tune_threshold_then_retest",
            "reason": "Model van gap false positive tren non-spam, nhung threshold tuning con co the giam duoc.",
            "next_step": (
                "Thu calibration/thay doi threshold phuc vu production, sau do danh gia lai precision-recall va UX risk."
            ),
        }

    return {
        "decision": "keep_current_model",
        "reason": "False-positive rate tren non-spam dang o muc chap nhan duoc.",
        "next_step": "Giu nguyen model hien tai va tiep tuc theo doi drift tren du lieu that.",
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_report(
    *,
    model_name: str,
    non_spam_results: dict[str, Any],
    threshold_rows: list[dict[str, Any]],
    recommendation: dict[str, str],
    benchmark_mode: str,
) -> str:
    lines = [
        f"# Non-Spam Generalization Benchmark ({model_name})",
        "",
        "## Summary",
        "",
    ]
    if benchmark_mode == "heldout_test_sources":
        lines.extend(
            [
                "This benchmark evaluates the saved production model on the held-out test split for SpamAssassin ham sources that are now part of the training corpus.",
                "The goal is to measure false positives on legitimate emails without re-scoring rows that may already have appeared in training.",
            ]
        )
    else:
        lines.extend(
            [
                "This benchmark evaluates the saved production model on ham-only email corpora from the SpamAssassin Public Corpus.",
                "The goal is to measure false positives on legitimate emails that were not part of the Kaggle + Enron production training mix.",
            ]
        )
    lines.extend(
        [
            "",
            "## Ham-Only Source Results",
            "",
            "| Source | Samples | Predicted Ham | Predicted Spam | False-Positive Rate | Avg Spam Probability |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for source_name, result in non_spam_results.items():
        lines.append(
            "| "
            f"{source_name} | "
            f"{result['samples']} | "
            f"{result['predicted_ham']} | "
            f"{result['predicted_spam']} | "
            f"{_format_percent(result['false_positive_rate'])} | "
            f"{_format_percent(result['avg_spam_probability'])} |"
        )

    lines.extend(
        [
            "",
            "## Threshold Sweep",
            "",
            "| Threshold | Test Accuracy | Test Precision | Test Recall | Test F1 | easy_ham FPR | easy_ham_2 FPR | hard_ham FPR | All Ham FPR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in threshold_rows:
        lines.append(
            "| "
            f"{row['threshold']:.2f} | "
            f"{_format_percent(row['test_accuracy'])} | "
            f"{_format_percent(row['test_precision'])} | "
            f"{_format_percent(row['test_recall'])} | "
            f"{_format_percent(row['test_f1'])} | "
            f"{_format_percent(row.get('easy_ham_fpr', 0.0))} | "
            f"{_format_percent(row.get('easy_ham_2_fpr', 0.0))} | "
            f"{_format_percent(row.get('hard_ham_fpr', 0.0))} | "
            f"{_format_percent(row['all_ham_fpr'])} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Decision: `{recommendation['decision']}`",
            f"- Reason: {recommendation['reason']}",
            f"- Recommended next step: {recommendation['next_step']}",
            "",
            "## Representative False Positives",
            "",
        ]
    )

    for source_name, result in non_spam_results.items():
        lines.append(f"### {source_name}")
        lines.append("")
        if not result["top_false_positives"]:
            lines.append("- No false positives found.")
            lines.append("")
            continue
        for example in result["top_false_positives"]:
            lines.append(
                f"- Spam probability `{example['spam_probability']:.4f}`: {example['preview']}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def evaluate_non_spam_generalization() -> dict[str, Any]:
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing processed test split at {test_path}. Run `python -m src.models.train` first."
        )

    pipeline = load_trained_pipeline()
    metadata = load_model_metadata()
    model_name = metadata.get("best_model", "unknown-model")

    benchmark_mode = "heldout_test_sources" if _spamassassin_sources_inside_training(metadata) else "external_corpora"
    non_spam_texts = (
        _load_heldout_non_spam_test_sources(metadata)
        if benchmark_mode == "heldout_test_sources"
        else _load_non_spam_corpora()
    )

    non_spam_results = {}
    for source_name, texts in non_spam_texts.items():
        probabilities = pipeline.predict_proba(texts)[:, 1]
        predictions = pipeline.predict(texts)
        non_spam_results[source_name] = {
            "samples": len(texts),
            "predicted_ham": int((predictions == 0).sum()),
            "predicted_spam": int((predictions == 1).sum()),
            "false_positive_rate": float((predictions == 1).mean()) if len(predictions) else 0.0,
            "avg_spam_probability": float(probabilities.mean()) if len(probabilities) else 0.0,
            "top_false_positives": _top_false_positive_examples(
                texts=texts,
                probabilities=probabilities,
                predictions=predictions,
            ),
        }

    threshold_rows = _threshold_sweep(pipeline, non_spam_sources=non_spam_texts)
    recommendation = recommend_next_step(threshold_rows)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    NON_SPAM_EVALUATION_PATH.write_text(
        _markdown_report(
            model_name=model_name,
            non_spam_results=non_spam_results,
            threshold_rows=threshold_rows,
            recommendation=recommendation,
            benchmark_mode=benchmark_mode,
        ),
        encoding="utf-8",
    )

    return {
        "model_name": model_name,
        "non_spam_results": non_spam_results,
        "threshold_rows": threshold_rows,
        "recommendation": recommendation,
        "benchmark_mode": benchmark_mode,
        "output_path": str(NON_SPAM_EVALUATION_PATH),
    }


def main() -> None:
    payload = evaluate_non_spam_generalization()
    print(f"Model: {payload['model_name']}")
    print(f"Decision: {payload['recommendation']['decision']}")
    print(f"Detailed report written to: {payload['output_path']}")


if __name__ == "__main__":
    main()
