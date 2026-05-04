"""
Tests for dataset sampling functionality in webapp.
"""

import pytest

from src.api.app import create_app
from src.api.dataset_sampler import TestDatasetSampler
from tests.test_api import FakeService


@pytest.fixture
def client_with_dataset():
    """Create test client with fake service and dataset sampler."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["SPAM_SERVICE"] = FakeService()

    # Use real dataset sampler if available
    sampler = TestDatasetSampler()
    if sampler.is_available():
        sampler.load_samples()
    app.config["DATASET_SAMPLER"] = sampler

    return app.test_client()


def test_api_dataset_samples_endpoint(client_with_dataset):
    """Test API endpoint for getting dataset samples."""
    response = client_with_dataset.get("/api/dataset-samples")

    if response.status_code == 404:
        pytest.skip("Test dataset not available")

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["ok"] is True
    assert "samples" in payload
    assert "total" in payload
    assert "spam_count" in payload
    assert "ham_count" in payload
    assert isinstance(payload["samples"], list)


def test_api_dataset_sample_by_id_endpoint(client_with_dataset):
    """Test API endpoint for getting specific sample by ID."""
    # First get list of samples
    response = client_with_dataset.get("/api/dataset-samples")

    if response.status_code == 404:
        pytest.skip("Test dataset not available")

    payload = response.get_json()
    if not payload["samples"]:
        pytest.skip("No samples available")

    # Get first sample ID
    sample_id = payload["samples"][0]["email_id"]

    # Get specific sample
    response = client_with_dataset.get(f"/api/dataset-sample/{sample_id}")

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["ok"] is True
    assert "sample" in payload
    assert payload["sample"]["email_id"] == sample_id
    assert "text" in payload["sample"]
    assert "label" in payload["sample"]
    assert "label_name" in payload["sample"]


def test_predict_sample_page_random(client_with_dataset):
    """Test predicting with random sample from dataset."""
    response = client_with_dataset.post(
        "/predict-sample",
        data={"label_filter": ""}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        # Should render the page
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_predict_sample_page_spam_filter(client_with_dataset):
    """Test predicting with spam sample from dataset."""
    response = client_with_dataset.post(
        "/predict-sample",
        data={"label_filter": "1"}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_predict_sample_page_ham_filter(client_with_dataset):
    """Test predicting with ham sample from dataset."""
    response = client_with_dataset.post(
        "/predict-sample",
        data={"label_filter": "0"}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_predict_sample_page_by_id(client_with_dataset):
    """Test predicting with specific sample by ID."""
    # First get a sample ID
    response = client_with_dataset.get("/api/dataset-samples")

    if response.status_code == 404:
        pytest.skip("Test dataset not available")

    payload = response.get_json()
    if not payload["samples"]:
        pytest.skip("No samples available")

    sample_id = payload["samples"][0]["email_id"]

    # Predict with that sample
    response = client_with_dataset.post(
        "/predict-sample",
        data={"email_id": sample_id}
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Prediction snapshot" in body
    assert sample_id in body


def test_home_page_shows_dataset_samples_when_available(client_with_dataset):
    """Test that home page shows dataset samples section when available."""
    response = client_with_dataset.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200

    # Check if dataset section is present or not based on availability
    if "Test Dataset Samples" in body:
        assert "Test nhanh voi du lieu mau da xu ly" in body or "Test nhanh voi du lieu mau co san" in body
        assert "Random mau bat ky" in body
    else:
        # Dataset not available, section should not be shown
        assert "test-dataset-samples" not in body


def test_dataset_sampler_initialization():
    """Test TestDatasetSampler initialization."""
    sampler = TestDatasetSampler()

    # Should not be loaded initially
    assert sampler.total_samples == 0

    # Check if dataset is available
    is_available = sampler.is_available()

    if is_available:
        # Try to load
        loaded = sampler.load_samples()
        assert loaded is True
        assert sampler.total_samples > 0
        assert sampler.spam_count > 0
        assert sampler.ham_count > 0


def test_dataset_sampler_get_random_sample():
    """Test getting random samples from dataset."""
    sampler = TestDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Test dataset not available")

    sampler.load_samples()

    # Get random sample
    sample = sampler.get_random_sample()
    assert sample is not None
    assert hasattr(sample, "email_id")
    assert hasattr(sample, "text")
    assert hasattr(sample, "label")

    # Get spam sample
    spam_sample = sampler.get_random_sample(label=1)
    if spam_sample:
        assert spam_sample.label == 1

    # Get ham sample
    ham_sample = sampler.get_random_sample(label=0)
    if ham_sample:
        assert ham_sample.label == 0


def test_dataset_sampler_get_samples_for_display():
    """Test getting formatted samples for display."""
    sampler = TestDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Test dataset not available")

    sampler.load_samples()

    samples = sampler.get_samples_for_display(5)

    assert isinstance(samples, list)
    if samples:
        assert len(samples) <= 5
        for sample in samples:
            assert "email_id" in sample
            assert "label_name" in sample
            assert "text_preview" in sample
            assert "text_length" in sample
            assert "source" in sample


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
