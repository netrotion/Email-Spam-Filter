from __future__ import annotations

import json
import os
import pathlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import CONTEXTUAL_ENCODER_DIR, MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH


def load_trained_pipeline(model_path: Path | None = None):
    path = model_path or MODEL_ARTIFACT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m src.models.train` first."
        )
    pipeline = _joblib_load_cross_platform(path)
    _normalize_loaded_pipeline_paths(pipeline)
    return pipeline


def _joblib_load_cross_platform(path: Path):
    if os.name != "nt":
        return joblib.load(path)

    original_posix_path = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return joblib.load(path)
    finally:
        pathlib.PosixPath = original_posix_path


def _normalize_loaded_pipeline_paths(pipeline) -> None:
    encoder_dir = getattr(pipeline, "encoder_dir", None)
    if encoder_dir is None:
        return

    encoder_path = Path(encoder_dir)
    if encoder_path.exists():
        pipeline.encoder_dir = encoder_path
        return

    if CONTEXTUAL_ENCODER_DIR.exists():
        pipeline.encoder_dir = CONTEXTUAL_ENCODER_DIR


def load_model_metadata(metadata_path: Path | None = None) -> dict[str, Any]:
    path = metadata_path or MODEL_METADATA_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_probability(pipeline, texts: list[str]) -> np.ndarray:
    probabilities = pipeline.predict_proba(texts)
    return probabilities[:, 1]


def _decision_scores(pipeline, texts: list[str]) -> np.ndarray:
    if hasattr(pipeline, "decision_function"):
        return np.asarray(pipeline.decision_function(texts))

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "decision_function"):
        return np.asarray(classifier.decision_function(pipeline.named_steps["features"].transform(texts)))
    probabilities = _positive_probability(pipeline, texts)
    return probabilities * 2.0 - 1.0


def _get_classifier_coefficients(classifier):
    if hasattr(classifier, "coef_"):
        return np.ravel(classifier.coef_)
    if hasattr(classifier, "calibrated_classifiers_") and classifier.calibrated_classifiers_:
        calibrated = classifier.calibrated_classifiers_[0]
        estimator = getattr(calibrated, "estimator", None)
        if estimator is not None and hasattr(estimator, "coef_"):
            return np.ravel(estimator.coef_)
    return None


def _clean_feature_name(name: str) -> str:
    cleaned = str(name)
    for prefix in ("word_tfidf__", "char_tfidf__", "email_stats__", "stats__"):
        cleaned = cleaned.replace(prefix, "")
    return cleaned


def explain_prediction(
    pipeline, text: str, predicted_label_id: int, top_n: int = 8
) -> list[dict[str, float | str]]:
    if not getattr(pipeline, "supports_explanations", True):
        return []
    if not hasattr(pipeline, "named_steps"):
        return []

    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    coefficients = _get_classifier_coefficients(classifier)
    if coefficients is None:
        return []

    feature_vector = features.transform([text])
    contributions = feature_vector.multiply(coefficients).toarray().ravel()
    feature_names = np.asarray(features.get_feature_names_out(), dtype=object)
    non_zero_indexes = np.flatnonzero(contributions)
    if len(non_zero_indexes) == 0:
        return []

    if predicted_label_id == 1:
        ordered = np.argsort(contributions[non_zero_indexes])[::-1]
        ranked_indexes = non_zero_indexes[ordered[:top_n]]
    else:
        ordered = np.argsort(contributions[non_zero_indexes])
        ranked_indexes = non_zero_indexes[ordered[:top_n]]

    explanations: list[dict[str, float | str]] = []
    for index in ranked_indexes:
        raw_score = float(contributions[index])
        if predicted_label_id == 1 and raw_score <= 0:
            continue
        if predicted_label_id == 0 and raw_score >= 0:
            continue
        explanations.append(
            {
                "feature": _clean_feature_name(feature_names[index]),
                "contribution": round(abs(raw_score), 6),
                "direction": "spam" if raw_score > 0 else "ham",
            }
        )
    return explanations


def predict_texts(
    texts: list[str], pipeline=None, metadata: dict[str, Any] | None = None, top_n: int = 8
) -> list[dict[str, Any]]:
    model = pipeline or load_trained_pipeline()
    model_metadata = metadata or load_model_metadata()
    probabilities = _positive_probability(model, texts)
    predictions = model.predict(texts)
    decision_scores = _decision_scores(model, texts)

    results: list[dict[str, Any]] = []
    for text, prediction, probability, decision_score in zip(
        texts, predictions, probabilities, decision_scores, strict=True
    ):
        predicted_label = "spam" if int(prediction) == 1 else "ham"
        results.append(
            {
                "label": predicted_label,
                "label_id": int(prediction),
                "spam_probability": float(probability),
                "ham_probability": float(1.0 - probability),
                "confidence": float(max(probability, 1.0 - probability)),
                "decision_score": float(decision_score),
                "top_signals": explain_prediction(
                    model, text, predicted_label_id=int(prediction), top_n=top_n
                ),
                "model_name": model_metadata.get("best_model"),
            }
        )
    return results


def predict_single_text(text: str, top_n: int = 8) -> dict[str, Any]:
    return predict_texts([text], top_n=top_n)[0]
