"""
Tests for raw dataset sampling functionality in webapp.
"""

import pytest

from src.api.app import create_app
from src.api.raw_dataset_sampler import RawDatasetSampler
from tests.test_api import FakeService


@pytest.fixture
def client_with_raw_dataset():
    """Create test client with fake service and raw dataset sampler."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["SPAM_SERVICE"] = FakeService()

    # Use real raw dataset sampler if available
    raw_sampler = RawDatasetSampler()
    if raw_sampler.is_available():
        raw_sampler.load_samples(max_samples_per_source=100)  # Limit for faster tests
    app.config["RAW_DATASET_SAMPLER"] = raw_sampler

    return app.test_client()


def test_raw_dataset_sampler_initialization():
    """Test RawDatasetSampler initialization."""
    sampler = RawDatasetSampler()

    # Should not be loaded initially
    assert sampler.total_samples == 0

    # Check if raw dataset is available
    is_available = sampler.is_available()

    if is_available:
        # Try to load with limit
        loaded = sampler.load_samples(max_samples_per_source=50)
        assert loaded is True
        assert sampler.total_samples > 0


def test_raw_dataset_sampler_get_random_sample():
    """Test getting random samples from raw dataset."""
    sampler = RawDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Raw dataset not available")

    sampler.load_samples(max_samples_per_source=50)

    # Get random sample
    sample = sampler.get_random_sample()
    if sample:
        assert hasattr(sample, "email_id")
        assert hasattr(sample, "text")
        assert hasattr(sample, "label")
        assert sample.source in ["kaggle_raw", "enron_raw"]


def test_raw_dataset_sampler_filter_by_label():
    """Test filtering samples by label."""
    sampler = RawDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Raw dataset not available")

    sampler.load_samples(max_samples_per_source=50)

    # Get spam sample
    spam_sample = sampler.get_random_sample(label=1)
    if spam_sample:
        assert spam_sample.label == 1
        assert spam_sample.label_name == "spam"

    # Get ham sample
    ham_sample = sampler.get_random_sample(label=0)
    if ham_sample:
        assert ham_sample.label == 0
        assert ham_sample.label_name == "ham"


def test_raw_dataset_sampler_filter_by_source():
    """Test filtering samples by source."""
    sampler = RawDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Raw dataset not available")

    sampler.load_samples(max_samples_per_source=50)

    available_sources = sampler.get_available_sources()

    for source in available_sources:
        sample = sampler.get_random_sample(source=f"{source}_raw")
        if sample:
            assert sample.source == f"{source}_raw"


def test_raw_dataset_sampler_get_stats_by_source():
    """Test getting statistics by source."""
    sampler = RawDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Raw dataset not available")

    sampler.load_samples(max_samples_per_source=50)

    stats = sampler.get_stats_by_source()

    assert isinstance(stats, dict)
    for source, source_stats in stats.items():
        assert "total" in source_stats
        assert "spam" in source_stats
        assert "ham" in source_stats
        assert source_stats["total"] == source_stats["spam"] + source_stats["ham"]


def test_api_raw_dataset_samples_endpoint(client_with_raw_dataset):
    """Test API endpoint for getting raw dataset samples."""
    response = client_with_raw_dataset.get("/api/raw-dataset-samples")

    if response.status_code == 404:
        pytest.skip("Raw dataset not available")

    assert response.status_code == 200
    payload = response.get_json()

    assert payload["ok"] is True
    assert "samples" in payload
    assert "total" in payload
    assert "spam_count" in payload
    assert "ham_count" in payload
    assert "available_sources" in payload
    assert "stats_by_source" in payload
    assert isinstance(payload["samples"], list)


def test_predict_raw_sample_page_random(client_with_raw_dataset):
    """Test predicting with random sample from raw dataset."""
    response = client_with_raw_dataset.post(
        "/predict-raw-sample",
        data={"label_filter": ""}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_predict_raw_sample_page_spam_filter(client_with_raw_dataset):
    """Test predicting with spam sample from raw dataset."""
    response = client_with_raw_dataset.post(
        "/predict-raw-sample",
        data={"label_filter": "1"}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_predict_raw_sample_page_ham_filter(client_with_raw_dataset):
    """Test predicting with ham sample from raw dataset."""
    response = client_with_raw_dataset.post(
        "/predict-raw-sample",
        data={"label_filter": "0"}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_predict_raw_sample_page_source_filter(client_with_raw_dataset):
    """Test predicting with source filter from raw dataset."""
    response = client_with_raw_dataset.post(
        "/predict-raw-sample",
        data={"source_filter": "kaggle_raw"}
    )

    if response.status_code == 200:
        body = response.get_data(as_text=True)
        assert "Prediction snapshot" in body or "dataset khong kha dung" in body.lower()


def test_home_page_shows_raw_dataset_samples_when_available(client_with_raw_dataset):
    """Test that home page shows raw dataset samples section when available."""
    response = client_with_raw_dataset.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200

    # Check if raw dataset section is present or not based on availability
    if "Raw Dataset Samples" in body:
        assert "Test voi du lieu tho chua qua xu ly" in body
        assert "Random Kaggle" in body or "Random Enron" in body


def test_raw_dataset_sampler_get_samples_for_display():
    """Test getting formatted samples for display."""
    sampler = RawDatasetSampler()

    if not sampler.is_available():
        pytest.skip("Raw dataset not available")

    sampler.load_samples(max_samples_per_source=50)

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
