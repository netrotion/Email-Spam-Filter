"""
Enhanced webapp testing with automatic test dataset validation.

This module extends the basic API tests with:
- Automatic test dataset loading and validation
- Sample-based prediction testing using real test data
- Performance benchmarking on test samples
- Error handling validation with edge cases from test data
"""

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.api.app import create_app
from tests.test_api import FakeService


@dataclass
class DatasetSample:
    """Represents a single test sample from the dataset."""
    email_id: str
    source: str
    label: int
    label_name: str
    text: str
    text_length: int


class DatasetLoader:
    """Loads and manages test dataset samples."""

    def __init__(self, test_csv_path: str | Path):
        self.test_csv_path = Path(test_csv_path)
        self.samples: list[DatasetSample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load samples from test.csv file."""
        if not self.test_csv_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {self.test_csv_path}")

        # Increase CSV field size limit to handle large text fields
        csv.field_size_limit(10 * 1024 * 1024)  # 10MB limit

        with open(self.test_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(DatasetSample(
                    email_id=row["email_id"],
                    source=row["source"],
                    label=int(row["label"]),
                    label_name=row["label_name"],
                    text=row["text"],
                    text_length=int(row["text_length"]),
                ))

    def get_random_samples(self, n: int = 10, label: int | None = None) -> list[DatasetSample]:
        """Get n random samples, optionally filtered by label."""
        if label is not None:
            filtered = [s for s in self.samples if s.label == label]
            return random.sample(filtered, min(n, len(filtered)))
        return random.sample(self.samples, min(n, len(self.samples)))

    def get_spam_samples(self, n: int = 10) -> list[DatasetSample]:
        """Get n random spam samples."""
        return self.get_random_samples(n=n, label=1)

    def get_ham_samples(self, n: int = 10) -> list[DatasetSample]:
        """Get n random ham samples."""
        return self.get_random_samples(n=n, label=0)

    def get_edge_cases(self) -> dict[str, list[DatasetSample]]:
        """Get edge case samples for testing."""
        return {
            "very_short": [s for s in self.samples if s.text_length < 50][:5],
            "very_long": sorted(self.samples, key=lambda s: s.text_length, reverse=True)[:5],
            "medium": [s for s in self.samples if 200 <= s.text_length <= 500][:5],
        }

    @property
    def total_samples(self) -> int:
        """Total number of samples in dataset."""
        return len(self.samples)

    @property
    def spam_count(self) -> int:
        """Number of spam samples."""
        return sum(1 for s in self.samples if s.label == 1)

    @property
    def ham_count(self) -> int:
        """Number of ham samples."""
        return sum(1 for s in self.samples if s.label == 0)

    def get_dataset_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total": self.total_samples,
            "spam": self.spam_count,
            "ham": self.ham_count,
            "spam_ratio": self.spam_count / self.total_samples if self.total_samples > 0 else 0,
            "avg_length": sum(s.text_length for s in self.samples) / self.total_samples if self.total_samples > 0 else 0,
            "min_length": min(s.text_length for s in self.samples) if self.samples else 0,
            "max_length": max(s.text_length for s in self.samples) if self.samples else 0,
        }


@pytest.fixture(scope="module")
def test_dataset():
    """Fixture to load test dataset once per module."""
    repo_root = Path(__file__).resolve().parents[1]
    test_csv = repo_root / "data" / "processed" / "test.csv"

    if not test_csv.exists():
        pytest.skip(f"Test dataset not found at {test_csv}")

    return DatasetLoader(test_csv)


@pytest.fixture
def client_with_fake_service():
    """Create test client with fake service."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["SPAM_SERVICE"] = FakeService()
    return app.test_client()


def test_dataset_exists_and_loads(test_dataset):
    """Verify test dataset exists and can be loaded."""
    assert test_dataset.total_samples > 0, "Test dataset should contain samples"
    assert test_dataset.spam_count > 0, "Test dataset should contain spam samples"
    assert test_dataset.ham_count > 0, "Test dataset should contain ham samples"


def test_dataset_statistics(test_dataset):
    """Verify dataset statistics are reasonable."""
    stats = test_dataset.get_dataset_stats()

    assert stats["total"] > 1000, f"Expected >1000 samples, got {stats['total']}"
    assert 0.1 < stats["spam_ratio"] < 0.9, f"Spam ratio should be balanced, got {stats['spam_ratio']:.2f}"
    assert stats["min_length"] > 0, "Minimum text length should be positive"
    assert stats["max_length"] > stats["avg_length"], "Max length should exceed average"

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Spam: {stats['spam']} ({stats['spam_ratio']:.1%})")
    print(f"  Ham: {stats['ham']} ({1-stats['spam_ratio']:.1%})")
    print(f"  Avg length: {stats['avg_length']:.0f} chars")
    print(f"  Length range: {stats['min_length']} - {stats['max_length']} chars")


def test_predict_with_real_spam_samples(client_with_fake_service, test_dataset):
    """Test prediction endpoint with real spam samples from test dataset."""
    spam_samples = test_dataset.get_spam_samples(n=5)

    for sample in spam_samples:
        response = client_with_fake_service.post(
            "/api/predict",
            json={"text": sample.text}
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert "prediction" in payload
        assert payload["prediction"]["label"] in ["spam", "ham"]
        assert "confidence" in payload["prediction"]
        assert 0 <= payload["prediction"]["confidence"] <= 1


def test_predict_with_real_ham_samples(client_with_fake_service, test_dataset):
    """Test prediction endpoint with real ham samples from test dataset."""
    ham_samples = test_dataset.get_ham_samples(n=5)

    for sample in ham_samples:
        response = client_with_fake_service.post(
            "/api/predict",
            json={"text": sample.text}
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert "prediction" in payload
        assert payload["prediction"]["label"] in ["spam", "ham"]


def test_batch_predict_with_mixed_samples(client_with_fake_service, test_dataset):
    """Test batch prediction with mixed spam/ham samples."""
    spam_samples = test_dataset.get_spam_samples(n=3)
    ham_samples = test_dataset.get_ham_samples(n=3)
    mixed_samples = spam_samples + ham_samples
    random.shuffle(mixed_samples)

    texts = [s.text for s in mixed_samples]

    response = client_with_fake_service.post(
        "/api/batch-predict",
        json={"texts": texts}
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["summary"]["total"] == len(texts)
    assert len(payload["rows"]) == len(texts)

    for row in payload["rows"]:
        assert row["label"] in ["spam", "ham"]
        assert "confidence" in row
        assert "text_preview" in row


def test_edge_case_very_short_texts(client_with_fake_service, test_dataset):
    """Test prediction with very short text samples."""
    edge_cases = test_dataset.get_edge_cases()
    short_samples = edge_cases.get("very_short", [])

    if not short_samples:
        pytest.skip("No very short samples found in dataset")

    for sample in short_samples[:3]:
        response = client_with_fake_service.post(
            "/api/predict",
            json={"text": sample.text}
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True


def test_edge_case_very_long_texts(client_with_fake_service, test_dataset):
    """Test prediction with very long text samples."""
    edge_cases = test_dataset.get_edge_cases()
    long_samples = edge_cases.get("very_long", [])

    if not long_samples:
        pytest.skip("No very long samples found in dataset")

    for sample in long_samples[:3]:
        response = client_with_fake_service.post(
            "/api/predict",
            json={"text": sample.text}
        )

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True


def test_batch_upload_csv_with_real_samples(client_with_fake_service, test_dataset):
    """Test CSV upload with real samples from test dataset."""
    samples = test_dataset.get_random_samples(n=10)

    csv_content = "text\n"
    for sample in samples:
        escaped_text = sample.text.replace('"', '""')
        csv_content += f'"{escaped_text}"\n'

    response = client_with_fake_service.post(
        "/api/batch-predict",
        data={"file": (io.BytesIO(csv_content.encode("utf-8")), "test_samples.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["summary"]["total"] == len(samples)
    assert payload["summary"]["source"] == "test_samples.csv"


def test_batch_upload_txt_with_real_samples(client_with_fake_service, test_dataset):
    """Test TXT upload with real samples from test dataset."""
    samples = test_dataset.get_random_samples(n=10)

    txt_content = "\n".join(s.text for s in samples)

    response = client_with_fake_service.post(
        "/api/batch-predict",
        data={"file": (io.BytesIO(txt_content.encode("utf-8")), "test_samples.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["summary"]["total"] == len(samples)
    assert payload["summary"]["source"] == "test_samples.txt"


def test_web_predict_page_with_real_sample(client_with_fake_service, test_dataset):
    """Test web prediction page with real sample."""
    sample = test_dataset.get_random_samples(n=1)[0]

    response = client_with_fake_service.post(
        "/predict",
        data={"text": sample.text}
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Prediction snapshot" in body
    assert sample.text[:30] in body or "..." in body


def test_web_batch_page_with_real_samples(client_with_fake_service, test_dataset):
    """Test web batch page with real samples."""
    samples = test_dataset.get_random_samples(n=5)

    csv_content = "text\n"
    for sample in samples:
        escaped_text = sample.text.replace('"', '""')
        csv_content += f'"{escaped_text}"\n'

    response = client_with_fake_service.post(
        "/batch",
        data={"file": (io.BytesIO(csv_content.encode("utf-8")), "test_batch.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Upload CSV or TXT and score in bulk" in body
    assert "Download results CSV" in body
    assert 'id="batch-results-body"' in body


def test_dataset_sample_diversity(test_dataset):
    """Verify dataset contains diverse samples from multiple sources."""
    sources = set(s.source for s in test_dataset.samples)

    assert len(sources) > 1, "Dataset should contain samples from multiple sources"

    print(f"\nDataset sources found: {len(sources)}")
    for source in sorted(sources):
        count = sum(1 for s in test_dataset.samples if s.source == source)
        print(f"  {source}: {count} samples")


def test_prediction_consistency_across_samples(client_with_fake_service, test_dataset):
    """Test that predictions are consistent for the same text."""
    sample = test_dataset.get_random_samples(n=1)[0]

    results = []
    for _ in range(3):
        response = client_with_fake_service.post(
            "/api/predict",
            json={"text": sample.text}
        )
        payload = response.get_json()
        results.append(payload["prediction"]["label"])

    assert len(set(results)) == 1, "Predictions should be consistent for same text"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
