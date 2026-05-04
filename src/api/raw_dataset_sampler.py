"""
Utility for loading sample emails from raw (unprocessed) datasets for webapp testing.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RawSampleEmail:
    """Represents a sample email from raw dataset."""
    email_id: str
    source: str
    label: int
    label_name: str
    text: str
    text_length: int


class RawDatasetSampler:
    """Loads and provides sample emails from raw (unprocessed) datasets for webapp testing."""

    def __init__(self):
        repo_root = Path(__file__).resolve().parents[2]
        self.data_raw_dir = repo_root / "data" / "raw"

        # Define raw dataset paths
        self.raw_datasets = {
            "kaggle": self.data_raw_dir / "kaggle" / "spam_ham_dataset.csv",
            "enron": self.data_raw_dir / "enron" / "enron_spam.csv",
        }

        self.samples: list[RawSampleEmail] = []
        self._loaded = False
        self._available_datasets: list[str] = []

    def load_samples(self, max_samples_per_source: int = 1000) -> bool:
        """
        Load samples from raw datasets.
        Limits to max_samples_per_source per dataset to avoid memory issues.
        Returns True if at least one dataset loaded successfully.
        """
        if self._loaded:
            return True

        try:
            # Increase CSV field size limit for large text fields
            csv.field_size_limit(10 * 1024 * 1024)

            # Load from Kaggle dataset
            if self._load_kaggle_dataset(max_samples_per_source):
                self._available_datasets.append("kaggle")

            # Load from Enron dataset
            if self._load_enron_dataset(max_samples_per_source):
                self._available_datasets.append("enron")

            self._loaded = len(self.samples) > 0
            return self._loaded

        except Exception as e:
            print(f"Error loading raw datasets: {e}")
            return False

    def _load_kaggle_dataset(self, max_samples: int) -> bool:
        """Load samples from Kaggle spam_ham_dataset.csv"""
        kaggle_path = self.raw_datasets["kaggle"]

        if not kaggle_path.exists():
            return False

        try:
            with open(kaggle_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)

                # Read all rows first to sample randomly
                all_rows = list(reader)

                # Sample randomly if too many
                if len(all_rows) > max_samples:
                    sampled_rows = random.sample(all_rows, max_samples)
                else:
                    sampled_rows = all_rows

                for idx, row in enumerate(sampled_rows):
                    # Kaggle format: ,label,text,label_num
                    label_text = row.get("label", "").strip().lower()
                    text = row.get("text", "").strip()

                    if not text:
                        continue

                    # Map label
                    if label_text == "spam":
                        label = 1
                        label_name = "spam"
                    elif label_text == "ham":
                        label = 0
                        label_name = "ham"
                    else:
                        continue

                    self.samples.append(RawSampleEmail(
                        email_id=f"kaggle_raw_{idx}",
                        source="kaggle_raw",
                        label=label,
                        label_name=label_name,
                        text=text,
                        text_length=len(text),
                    ))

            return len([s for s in self.samples if s.source == "kaggle_raw"]) > 0

        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            return False

    def _load_enron_dataset(self, max_samples: int) -> bool:
        """Load samples from Enron enron_spam.csv"""
        enron_path = self.raw_datasets["enron"]

        if not enron_path.exists():
            return False

        try:
            with open(enron_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)

                # Read all rows first to sample randomly
                all_rows = list(reader)

                # Sample randomly if too many
                if len(all_rows) > max_samples:
                    sampled_rows = random.sample(all_rows, max_samples)
                else:
                    sampled_rows = all_rows

                for idx, row in enumerate(sampled_rows):
                    # Enron format: message_id,text,label,label_text,subject,message,date
                    label_num = row.get("label", "").strip()
                    text = row.get("text", "").strip()

                    if not text:
                        continue

                    # Map label (0=ham, 1=spam)
                    try:
                        label = int(label_num)
                        label_name = "spam" if label == 1 else "ham"
                    except (ValueError, TypeError):
                        continue

                    self.samples.append(RawSampleEmail(
                        email_id=f"enron_raw_{idx}",
                        source="enron_raw",
                        label=label,
                        label_name=label_name,
                        text=text,
                        text_length=len(text),
                    ))

            return len([s for s in self.samples if s.source == "enron_raw"]) > 0

        except Exception as e:
            print(f"Error loading Enron dataset: {e}")
            return False

    def get_random_sample(self, label: int | None = None, source: str | None = None) -> RawSampleEmail | None:
        """
        Get a random sample, optionally filtered by label (0=ham, 1=spam) and/or source.

        Args:
            label: Filter by label (0=ham, 1=spam)
            source: Filter by source ("kaggle_raw", "enron_raw")
        """
        if not self._loaded and not self.load_samples():
            return None

        if not self.samples:
            return None

        # Apply filters
        filtered = self.samples

        if label is not None:
            filtered = [s for s in filtered if s.label == label]

        if source is not None:
            filtered = [s for s in filtered if s.source == source]

        if not filtered:
            return None

        return random.choice(filtered)

    def get_samples_for_display(self, n: int = 10) -> list[dict[str, Any]]:
        """Get n random samples formatted for display in webapp."""
        if not self._loaded and not self.load_samples():
            return []

        if not self.samples:
            return []

        selected = random.sample(self.samples, min(n, len(self.samples)))

        return [
            {
                "email_id": s.email_id,
                "label_name": s.label_name,
                "text_preview": s.text[:100] + "..." if len(s.text) > 100 else s.text,
                "text_length": s.text_length,
                "source": s.source,
            }
            for s in selected
        ]

    def is_available(self) -> bool:
        """Check if any raw dataset is available."""
        return any(path.exists() for path in self.raw_datasets.values())

    def get_available_sources(self) -> list[str]:
        """Get list of available raw dataset sources."""
        if not self._loaded:
            self.load_samples()
        return self._available_datasets

    @property
    def total_samples(self) -> int:
        """Total number of loaded samples."""
        return len(self.samples)

    @property
    def spam_count(self) -> int:
        """Number of spam samples."""
        return sum(1 for s in self.samples if s.label == 1)

    @property
    def ham_count(self) -> int:
        """Number of ham samples."""
        return sum(1 for s in self.samples if s.label == 0)

    def get_stats_by_source(self) -> dict[str, dict[str, int]]:
        """Get statistics grouped by source."""
        stats = {}

        for source in set(s.source for s in self.samples):
            source_samples = [s for s in self.samples if s.source == source]
            stats[source] = {
                "total": len(source_samples),
                "spam": sum(1 for s in source_samples if s.label == 1),
                "ham": sum(1 for s in source_samples if s.label == 0),
            }

        return stats
