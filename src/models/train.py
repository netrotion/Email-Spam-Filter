from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import (
    CONTEXTUAL_ENCODER_DIR,
    DEFAULT_TRAINING_CONFIG,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    TRAINING_RESULTS_PATH,
    TrainingConfig,
    ensure_project_directories,
)
from src.data.dataset import DatasetBundle, prepare_dataset
from src.evaluation.metrics import compute_binary_classification_metrics, metrics_to_markdown
from src.models.contextual import ContextualTransformerSpamPipeline
from src.models.pipeline import build_candidate_pipelines


def _evaluate_pipeline(pipeline, features: list[str], labels: list[int]) -> tuple[list[int], list[float]]:
    predictions = list(pipeline.predict(features))
    probabilities = pipeline.predict_proba(features)[:, 1].tolist()
    return predictions, probabilities


def _contextual_candidate_name(model_name: str) -> str:
    short_name = model_name.rsplit("/", maxsplit=1)[-1]
    normalized = re.sub(r"[^a-z0-9]+", "_", short_name.casefold()).strip("_")
    return f"contextual_{normalized or 'transformer'}"


def _build_candidate_builders(mode: str, config: TrainingConfig | None = None) -> dict[str, Any]:
    training_config = config or DEFAULT_TRAINING_CONFIG
    builders: dict[str, Any] = {}
    if mode in {"baseline", "all"}:
        builders.update(build_candidate_pipelines(training_config))
    if mode in {"contextual", "all"}:
        builders[_contextual_candidate_name(training_config.contextual_model_name)] = lambda: ContextualTransformerSpamPipeline(
            model_name=training_config.contextual_model_name,
            encoder_dir=CONTEXTUAL_ENCODER_DIR,
            max_length=training_config.contextual_max_length,
            train_batch_size=training_config.contextual_batch_size,
            eval_batch_size=training_config.contextual_eval_batch_size,
            epochs=training_config.contextual_epochs,
            learning_rate=training_config.contextual_learning_rate,
            weight_decay=training_config.contextual_weight_decay,
            freeze_base_model=training_config.contextual_freeze_base_model,
            warmup_ratio=training_config.contextual_warmup_ratio,
            gradient_accumulation_steps=training_config.contextual_gradient_accumulation_steps,
            early_stopping_patience=training_config.contextual_early_stopping_patience,
            use_class_weights=training_config.contextual_use_class_weights,
            random_state=training_config.random_state,
        )
    return builders


def _select_best_candidate(
    dataset_bundle: DatasetBundle,
    mode: str,
    config: TrainingConfig | None = None,
) -> tuple[str, Any, dict[str, dict[str, Any]]]:
    try:
        builders = _build_candidate_builders(mode, config=config)
    except TypeError:
        builders = _build_candidate_builders(mode)
    candidate_metrics: dict[str, dict[str, Any]] = {}
    best_name = ""
    best_pipeline = None
    best_score = (-1.0, -1.0)

    train_texts = dataset_bundle.train["text"].tolist()
    train_labels = dataset_bundle.train["label"].tolist()
    validation_texts = dataset_bundle.validation["text"].tolist()
    validation_labels = dataset_bundle.validation["label"].tolist()

    for candidate_name, builder in builders.items():
        pipeline = builder()
        fit_kwargs: dict[str, Any] = {}
        if getattr(pipeline, "model_family", "") == "contextual_transformer":
            fit_kwargs = {
                "validation_texts": validation_texts,
                "validation_labels": validation_labels,
            }
        pipeline.fit(train_texts, train_labels, **fit_kwargs)
        predictions, probabilities = _evaluate_pipeline(pipeline, validation_texts, validation_labels)
        metrics = compute_binary_classification_metrics(validation_labels, predictions, probabilities)
        candidate_metrics[candidate_name] = metrics

        score = (metrics["accuracy"], metrics["f1_score"])
        if score > best_score:
            best_name = candidate_name
            best_pipeline = pipeline
            best_score = score

    if best_pipeline is None:
        raise RuntimeError("No candidate pipeline was successfully trained.")
    return best_name, best_pipeline, candidate_metrics


def _refit_best_pipeline(
    best_name: str,
    dataset_bundle: DatasetBundle,
    mode: str,
    config: TrainingConfig | None = None,
):
    builders = _build_candidate_builders(mode, config=config)
    pipeline = builders[best_name]()
    combined = pd.concat([dataset_bundle.train, dataset_bundle.validation], ignore_index=True)
    pipeline.fit(combined["text"].tolist(), combined["label"].tolist())
    return pipeline


def _build_training_metadata(
    dataset_bundle: DatasetBundle,
    best_model: str,
    fitted_pipeline: Any,
    validation_results: dict[str, dict[str, Any]],
    test_metrics: dict[str, Any],
    mode: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    artifacts = {
        "model": str(MODEL_ARTIFACT_PATH),
        "metadata": str(MODEL_METADATA_PATH),
    }
    if getattr(fitted_pipeline, "model_family", "") == "contextual_transformer":
        artifacts["encoder_dir"] = str(CONTEXTUAL_ENCODER_DIR)

    return {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_model,
        "model_family": getattr(fitted_pipeline, "model_family", "classical_ml"),
        "selected_training_mode": mode,
        "contextual_backbone": getattr(fitted_pipeline, "model_name", None),
        "class_names": ["ham", "spam"],
        "dataset_summary": dataset_bundle.summary,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "config": config.as_dict(),
        "artifacts": artifacts,
    }


def _save_training_outputs(pipeline, metadata: dict[str, Any]) -> None:
    ensure_project_directories()
    joblib.dump(pipeline, MODEL_ARTIFACT_PATH)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    TRAINING_RESULTS_PATH.write_text(
        metrics_to_markdown("Training Results", metadata["test_metrics"]),
        encoding="utf-8",
    )


def train_and_save(
    force_download: bool = False,
    mode: str = "contextual",
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    training_config = config or DEFAULT_TRAINING_CONFIG
    dataset_bundle = prepare_dataset(force_download=force_download, config=training_config)
    best_name, fitted_pipeline, validation_results = _select_best_candidate(
        dataset_bundle,
        mode=mode,
        config=training_config,
    )

    if mode != "contextual":
        fitted_pipeline = _refit_best_pipeline(
            best_name,
            dataset_bundle,
            mode=mode,
            config=training_config,
        )

    test_texts = dataset_bundle.test["text"].tolist()
    test_labels = dataset_bundle.test["label"].tolist()
    predictions, probabilities = _evaluate_pipeline(fitted_pipeline, test_texts, test_labels)
    test_metrics = compute_binary_classification_metrics(test_labels, predictions, probabilities)

    metadata = _build_training_metadata(
        dataset_bundle=dataset_bundle,
        best_model=best_name,
        fitted_pipeline=fitted_pipeline,
        validation_results=validation_results,
        test_metrics=test_metrics,
        mode=mode,
        config=training_config,
    )
    _save_training_outputs(fitted_pipeline, metadata)
    return metadata


def _build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    overrides: dict[str, Any] = {}
    argument_to_field = {
        "model_name": "contextual_model_name",
        "max_length": "contextual_max_length",
        "batch_size": "contextual_batch_size",
        "eval_batch_size": "contextual_eval_batch_size",
        "epochs": "contextual_epochs",
        "learning_rate": "contextual_learning_rate",
        "gradient_accumulation_steps": "contextual_gradient_accumulation_steps",
        "early_stopping_patience": "contextual_early_stopping_patience",
    }
    for argument_name, field_name in argument_to_field.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            overrides[field_name] = value

    if args.freeze_base_model is not None:
        overrides["contextual_freeze_base_model"] = args.freeze_base_model
    if args.disable_class_weights:
        overrides["contextual_use_class_weights"] = False
    if args.no_spamassassin_ham:
        overrides["spamassassin_ham_sources"] = ()

    return replace(DEFAULT_TRAINING_CONFIG, **overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the spam email classifier.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download and re-extract the SpamAssassin corpus again.",
    )
    parser.add_argument(
        "--mode",
        choices=["contextual", "baseline", "all"],
        default="contextual",
        help="Training mode: contextual encoder, legacy baseline, or compare all candidates.",
    )
    parser.add_argument(
        "--model-name",
        help="Hugging Face transformer backbone for contextual mode.",
    )
    parser.add_argument("--max-length", type=int, help="Maximum token length for contextual training.")
    parser.add_argument("--batch-size", type=int, help="Contextual train batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Contextual evaluation batch size.")
    parser.add_argument("--epochs", type=int, help="Number of contextual training epochs.")
    parser.add_argument("--learning-rate", type=float, help="Contextual optimizer learning rate.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Number of batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Validation epochs without improvement before stopping; -1 disables stopping.",
    )
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze-base-model",
        dest="freeze_base_model",
        action="store_true",
        default=None,
        help="Freeze the pretrained encoder and train only the classification head.",
    )
    freeze_group.add_argument(
        "--fine-tune-base-model",
        dest="freeze_base_model",
        action="store_false",
        help="Fine-tune the full pretrained encoder.",
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable class-weighted loss for contextual training.",
    )
    parser.add_argument(
        "--no-spamassassin-ham",
        action="store_true",
        help="Train only on Kaggle + Enron and skip SpamAssassin ham augmentation.",
    )
    args = parser.parse_args()

    training_config = _build_config_from_args(args)
    metadata = train_and_save(
        force_download=args.force_download,
        mode=args.mode,
        config=training_config,
    )
    metrics = metadata["test_metrics"]
    print(f"Selected model: {metadata['best_model']}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-score: {metrics['f1_score']:.4f}")
    print(f"Artifact saved to: {Path(metadata['artifacts']['model'])}")


if __name__ == "__main__":
    main()
