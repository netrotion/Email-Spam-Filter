"""
Utility for loading sample emails from test dataset for webapp testing.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SampleEmail:
    """Represents a sample email from the test dataset."""
    email_id: str
    source: str
    label: int
    label_name: str
    text: str
    text_length: int


class TestDatasetSampler:
    """Loads and provides sample emails from test dataset for webapp testing."""

    def __init__(self, test_csv_path: str | Path | None = None):
        if test_csv_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            test_csv_path = repo_root / "data" / "processed" / "test.csv"

        self.test_csv_path = Path(test_csv_path)
        self.samples: list[SampleEmail] = []
        self._loaded = False

    def load_samples(self) -> bool:
        """Load samples from test.csv file. Returns True if successful."""
        if self._loaded:
            return True

        if not self.test_csv_path.exists():
            return False

        try:
            # Increase CSV field size limit for large text fields
            csv.field_size_limit(10 * 1024 * 1024)

            with open(self.test_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append(SampleEmail(
                        email_id=row["email_id"],
                        source=row["source"],
                        label=int(row["label"]),
                        label_name=row["label_name"],
                        text=row["text"],
                        text_length=int(row["text_length"]),
                    ))

            self._loaded = True
            return True
        except Exception:
            return False

    def get_random_sample(self, label: int | None = None) -> SampleEmail | None:
        """Get a random sample, optionally filtered by label (0=ham, 1=spam)."""
        if not self._loaded and not self.load_samples():
            return None

        if not self.samples:
            return None

        if label is not None:
            filtered = [s for s in self.samples if s.label == label]
            if not filtered:
                return None
            return random.choice(filtered)

        return random.choice(self.samples)

    def get_sample_by_id(self, email_id: str) -> SampleEmail | None:
        """Get a specific sample by email_id."""
        if not self._loaded and not self.load_samples():
            return None

        for sample in self.samples:
            if sample.email_id == email_id:
                return sample
        return None

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
        """Check if test dataset is available."""
        return self.test_csv_path.exists()

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
