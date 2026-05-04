from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
KAGGLE_RAW_DIR = RAW_DATA_DIR / "kaggle"
ENRON_RAW_DIR = RAW_DATA_DIR / "enron"
SPAMASSASSIN_RAW_DIR = RAW_DATA_DIR / "spamassassin"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


@dataclass(frozen=True)
class DatasetSplitConfig:
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def validate(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if round(total, 10) != 1.0:
            raise ValueError(f"Split sizes must sum to 1.0, received {total}.")


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    split: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    spamassassin_ham_sources: tuple[str, ...] = ("easy_ham", "easy_ham_2", "hard_ham")
    max_word_features: int = 100_000
    max_char_features: int = 80_000
    max_iterations: int = 4_000
    logistic_c: float = 4.0
    linear_svc_c: float = 1.5
    top_signal_count: int = 8
    contextual_model_name: str = "microsoft/deberta-v3-base"
    contextual_max_length: int = 256
    contextual_batch_size: int = 8
    contextual_eval_batch_size: int = 16
    contextual_epochs: int = 3
    contextual_learning_rate: float = 1e-5
    contextual_weight_decay: float = 0.01
    contextual_freeze_base_model: bool = False
    contextual_warmup_ratio: float = 0.10
    contextual_gradient_accumulation_steps: int = 2
    contextual_early_stopping_patience: int = 1
    contextual_use_class_weights: bool = True

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["split"] = self.split.as_dict()
        return payload


def ensure_project_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        KAGGLE_RAW_DIR,
        ENRON_RAW_DIR,
        SPAMASSASSIN_RAW_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


DEFAULT_TRAINING_CONFIG = TrainingConfig()
MODEL_ARTIFACT_PATH = MODELS_DIR / "spam_detector.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "spam_detector_metadata.json"
TRAINING_RESULTS_PATH = DOCS_DIR / "TRAINING_RESULTS.md"
CONTEXTUAL_ENCODER_DIR = MODELS_DIR / "spam_contextual_encoder"
