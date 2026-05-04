from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.api.app as app_module
import src.api.services as services_module
from src.models import contextual as contextual_module
from src.models import inference as inference_module
from src.api.app import create_app
from src.api.services import ApiError, SpamDetectionService
from src.models.contextual import ContextualTransformerSpamPipeline
from src.models import train as train_module
from tests.test_api import FakeService


class EmptyBatchAwareService(FakeService):
    def __init__(self) -> None:
        super().__init__()
        self.seen_texts = None

    def predict_batch_texts(self, texts, source="json"):
        self.seen_texts = texts
        if texts == []:
            raise ApiError(
                "Khong tim thay dong text hop le de du doan.",
                status_code=400,
                code="empty_batch",
            )
        return super().predict_batch_texts(texts=texts, source=source)


class BrokenContextualPipeline:
    model_family = "contextual_transformer"
    encoder_dir = Path("models/missing-contextual-encoder")

    def _load_inference_components(self):
        raise FileNotFoundError(
            f"Fine-tuned transformer directory not found at {self.encoder_dir}."
        )


class CapturingContextualPipeline:
    model_family = "contextual_transformer"

    def __init__(self) -> None:
        self.fit_kwargs: dict[str, object] = {}

    def fit(self, texts, labels, **kwargs):
        self.fit_kwargs = kwargs
        self._last_texts = list(texts)
        self._last_labels = list(labels)
        return self

    def predict(self, texts):
        return np.asarray([1 if "free" in text.casefold() else 0 for text in texts], dtype=int)

    def predict_proba(self, texts):
        probabilities = []
        for text in texts:
            if "free" in text.casefold():
                probabilities.append([0.01, 0.99])
            else:
                probabilities.append([0.99, 0.01])
        return np.asarray(probabilities, dtype=float)


def test_batch_predict_endpoint_preserves_empty_texts_field(monkeypatch):
    service = EmptyBatchAwareService()
    monkeypatch.setattr(app_module, "SpamDetectionService", lambda: service)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post("/api/batch-predict", json={"texts": []})
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["error"]["code"] == "empty_batch"
    assert service.seen_texts == []


def test_batch_page_embeds_full_export_payload(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(app_module, "SpamDetectionService", lambda: service)

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/batch",
        data={"file": (io.BytesIO(b"text\nFree bonus now\nProject meeting later\n"), "emails.csv")},
        content_type="multipart/form-data",
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "data-export-text=" in body
    assert "data-export-spam-probability=" in body
    assert "data-export-top-signals=" in body


def test_refresh_model_marks_broken_contextual_artifact_missing(monkeypatch):
    monkeypatch.setattr(
        services_module,
        "load_model_metadata",
        lambda: {"best_model": "contextual_bert_tiny"},
    )
    monkeypatch.setattr(
        services_module,
        "load_trained_pipeline",
        lambda: BrokenContextualPipeline(),
    )

    service = SpamDetectionService()

    assert service.model_status == "missing-artifact"
    assert service.pipeline is None
    assert service.model_name == "contextual_bert_tiny"
    assert service.load_errors
    assert "Fine-tuned transformer directory not found" in service.load_errors[0]


def test_decode_upload_rejects_binary_payload():
    service = SpamDetectionService.__new__(SpamDetectionService)

    with pytest.raises(ApiError) as exc_info:
        service._decode_upload(b"\x00\x01\x02PK\x03\x04not-a-text-upload")

    assert exc_info.value.code == "decode_error"


def test_select_best_candidate_passes_validation_to_contextual_pipeline(monkeypatch):
    pipeline = CapturingContextualPipeline()
    monkeypatch.setattr(
        train_module,
        "_build_candidate_builders",
        lambda mode: {"contextual_bert_tiny": lambda: pipeline},
    )

    dataset_bundle = SimpleNamespace(
        train=pd.DataFrame(
            {
                "text": ["free money now", "project update"],
                "label": [1, 0],
            }
        ),
        validation=pd.DataFrame(
            {
                "text": ["free prize", "team meeting"],
                "label": [1, 0],
            }
        ),
    )

    best_name, best_pipeline, _candidate_metrics = train_module._select_best_candidate(
        dataset_bundle,
        mode="contextual",
    )

    assert best_name == "contextual_bert_tiny"
    assert best_pipeline is pipeline
    assert pipeline.fit_kwargs["validation_texts"] == ["free prize", "team meeting"]
    assert pipeline.fit_kwargs["validation_labels"] == [1, 0]


def test_contextual_stable_softmax_returns_finite_probabilities():
    pipeline = ContextualTransformerSpamPipeline(
        model_name="microsoft/deberta-v3-base",
        encoder_dir=Path("models/test-contextual-encoder"),
    )

    probabilities = pipeline._stable_softmax(
        np.asarray(
            [
                [10000.0, -10000.0],
                [np.nan, 2.0],
                [np.inf, -np.inf],
            ],
            dtype=np.float16,
        )
    )

    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))


def test_contextual_loader_rejects_transformers_version_mismatch(monkeypatch, tmp_path):
    encoder_dir = tmp_path / "spam_contextual_encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(
        json.dumps({"transformers_version": "5.0.0"}),
        encoding="utf-8",
    )
    pipeline = ContextualTransformerSpamPipeline(
        model_name="microsoft/deberta-v3-base",
        encoder_dir=encoder_dir,
    )

    monkeypatch.setattr(contextual_module, "version", lambda _package: "5.7.0")

    with pytest.raises(RuntimeError, match="Transformer artifact version mismatch"):
        pipeline._validate_transformers_artifact_version()


def test_load_trained_pipeline_normalizes_colab_encoder_path(monkeypatch, tmp_path):
    artifact_path = tmp_path / "spam_detector.joblib"
    artifact_path.write_bytes(b"fake joblib payload")
    local_encoder_dir = tmp_path / "spam_contextual_encoder"
    local_encoder_dir.mkdir()
    pipeline = SimpleNamespace(
        model_family="contextual_transformer",
        encoder_dir=Path("/content/BTL-TTNT/models/spam_contextual_encoder"),
    )
    observed: dict[str, bool] = {}

    def fake_joblib_load(path):
        observed["posix_path_patched"] = (
            inference_module.pathlib.PosixPath is inference_module.pathlib.WindowsPath
        )
        assert path == artifact_path
        return pipeline

    monkeypatch.setattr(inference_module.os, "name", "nt", raising=False)
    monkeypatch.setattr(inference_module.joblib, "load", fake_joblib_load)
    monkeypatch.setattr(inference_module, "CONTEXTUAL_ENCODER_DIR", local_encoder_dir)

    loaded = inference_module.load_trained_pipeline(artifact_path)

    assert loaded is pipeline
    assert loaded.encoder_dir == local_encoder_dir
    assert observed["posix_path_patched"] is True
