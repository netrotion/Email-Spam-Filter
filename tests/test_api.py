from dataclasses import dataclass
import io

from src.api.app import create_app


@dataclass
class FakeService:
    model_status: str = "ready"
    model_source: str = "tests.fake"
    model_name: str = "fake-model"

    def predict_text(self, text: str):
        label = "spam" if "free" in text.lower() else "ham"
        label_id = 1 if label == "spam" else 0
        return {
            "label": label,
            "label_id": label_id,
            "confidence": 0.97,
            "decision_score": 0.81 if label == "spam" else -0.72,
            "spam_probability": 0.97 if label == "spam" else 0.04,
            "ham_probability": 0.03 if label == "spam" else 0.96,
            "scores": {"spam": 0.97 if label == "spam" else 0.04, "ham": 0.03 if label == "spam" else 0.96},
            "top_signals": [{"feature": "free", "contribution": 0.42, "direction": label}],
            "model_name": self.model_name,
            "model_source": self.model_source,
        }

    def predict_batch_texts(self, texts, source="json"):
        rows = []
        for idx, text in enumerate(texts, start=1):
            prediction = self.predict_text(text)
            rows.append(
                {
                    "row_number": idx,
                    "text": text,
                    "text_preview": text[:30],
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "scores": prediction["scores"],
                    "spam_probability": prediction["spam_probability"],
                    "ham_probability": prediction["ham_probability"],
                    "top_signals": prediction["top_signals"],
                }
            )
        spam_count = sum(1 for row in rows if row["label"] == "spam")
        return {
            "rows": rows,
            "summary": {
                "total": len(rows),
                "spam": spam_count,
                "ham": len(rows) - spam_count,
                "source": source,
            },
            "warnings": [],
        }

    def predict_uploaded_file(self, filename: str, content: bytes):
        lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
        if filename.endswith(".csv") and lines and lines[0].strip().casefold() == "text":
            lines = lines[1:]
        texts = lines
        return self.predict_batch_texts(texts=texts, source=filename)

    def get_model_info(self):
        return {
            "status": self.model_status,
            "source": self.model_source,
            "model_name": self.model_name,
            "artifact_path": "models/fake.joblib",
            "artifact_exists": True,
            "metadata_path": "models/fake.json",
            "metadata_exists": True,
            "labels": ["ham", "spam"],
            "supports_probabilities": True,
            "metrics": {"accuracy": 0.99, "precision": 0.98, "recall": 0.97, "f1": 0.975},
            "validation_metrics": {"accuracy": 0.98, "f1": 0.97},
            "metadata": {"trained_at": "2026-04-29T00:00:00Z", "best_model": self.model_name},
            "notes": [],
            "load_errors": [],
        }

    def health_payload(self):
        return {
            "status": "ok",
            "model_status": self.model_status,
            "model_source": self.model_source,
            "model_name": self.model_name,
            "artifact_exists": True,
        }


def _build_client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SPAM_SERVICE"] = FakeService()
    return app.test_client()


def test_health_endpoint_returns_service_status():
    client = _build_client()

    response = client.get("/api/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["model_status"] == "ready"


def test_home_page_renders_sidebar_dashboard():
    client = _build_client()

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Sidebar dashboard for quick email testing" in body
    assert 'id="single-predict"' in body
    assert 'id="download-batch-results"' in body


def test_model_info_page_renders_metrics_dashboard():
    client = _build_client()

    response = client.get("/model-info")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Metrics, metadata, and runtime state" in body
    assert "Saved test-set performance" in body


def test_predict_endpoint_validates_json_body():
    client = _build_client()

    response = client.post("/api/predict", json={})
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == "missing_text"


def test_predict_endpoint_returns_prediction_payload():
    client = _build_client()

    response = client.post("/api/predict", json={"text": "Get free bonus right now"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["prediction"]["label"] == "spam"
    assert "top_signals" in payload["prediction"]


def test_predict_page_renders_prediction_result():
    client = _build_client()

    response = client.post("/predict", data={"text": "Get free bonus right now"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Prediction snapshot" in body
    assert "spam" in body
    assert "Top signals" in body


def test_batch_predict_endpoint_supports_json_input():
    client = _build_client()

    response = client.post(
        "/api/batch-predict",
        json={"texts": ["Free bonus now", "Project meeting later"]},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["summary"]["total"] == 2
    assert len(payload["rows"]) == 2


def test_batch_predict_endpoint_supports_csv_upload():
    client = _build_client()

    response = client.post(
        "/api/batch-predict",
        data={"file": (io.BytesIO(b"text\nFree bonus now\nProject meeting later\n"), "emails.csv")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["summary"]["source"] == "emails.csv"
    assert payload["summary"]["total"] == 2


def test_batch_page_renders_results_table_and_download_action():
    client = _build_client()

    response = client.post(
        "/batch",
        data={"file": (io.BytesIO(b"text\nFree bonus now\nProject meeting later\n"), "emails.csv")},
        content_type="multipart/form-data",
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Upload CSV or TXT and score in bulk" in body
    assert "Download results CSV" in body
    assert 'id="batch-results-body"' in body


def test_missing_page_renders_html_error_dashboard():
    client = _build_client()

    response = client.get("/missing-page")
    body = response.get_data(as_text=True)

    assert response.status_code == 404
    assert "Current server response" in body
    assert "Back to dashboard" in body
