from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_TRAINING_CONFIG, PROCESSED_DATA_DIR, TrainingConfig, ensure_project_directories
from src.data.download import (
    ensure_enron_spam_dataset,
    ensure_kaggle_spam_dataset,
    ensure_spamassassin_corpus,
)
from src.preprocessing.text import clean_email_text, html_to_text, normalize_for_deduplication


@dataclass
class DatasetBundle:
    full: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summary: dict[str, Any]


def _decode_part(part) -> str:
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        content = payload.decode(charset, errors="ignore")
    if part.get_content_type() == "text/html":
        return html_to_text(content)
    return content


def extract_email_text_from_file(path: Path) -> str:
    with path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    subject = message.get("subject", "") or ""
    body_parts: list[str] = []

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue
            body_parts.append(_decode_part(part))
    else:
        body_parts.append(_decode_part(message))

    return clean_email_text(f"{subject}\n" + "\n".join(body_parts))


def _finalize_dataframe(
    dataframe: pd.DataFrame,
    *,
    source_name: str,
    email_id_column: str,
    text_column: str,
    label_column: str,
    label_name_column: str | None = None,
) -> pd.DataFrame:
    working = dataframe.copy()
    working["email_id"] = working[email_id_column].fillna("").astype(str).str.strip()
    working["text"] = working[text_column].fillna("").astype(str).map(clean_email_text)
    if label_name_column is not None and label_name_column in working.columns:
        working["label_name"] = working[label_name_column].fillna("").astype(str).str.strip().str.lower()
    else:
        working["label_name"] = working[label_column].map({1: "spam", 0: "ham"}).fillna("")
    working["label"] = working[label_column].astype(int)
    working["source"] = source_name
    working["text_length"] = working["text"].str.len()

    filtered = working[
        working["text"].str.len().gt(0)
        & working["label"].isin([0, 1])
    ][["email_id", "source", "label", "label_name", "text", "text_length"]]
    return filtered.reset_index(drop=True)


def _load_kaggle_dataframe(force_download: bool = False) -> pd.DataFrame:
    csv_path = ensure_kaggle_spam_dataset(force=force_download)
    dataframe = pd.read_csv(csv_path)
    dataframe["label_text"] = dataframe["label"].astype(str).str.strip().str.lower()
    if "label_num" in dataframe.columns:
        dataframe["label"] = dataframe["label_num"].astype(int)
    else:
        dataframe["label"] = dataframe["label_text"].map({"ham": 0, "spam": 1})
    return _finalize_dataframe(
        dataframe,
        source_name="kaggle_venky73_spam_mails",
        email_id_column="Unnamed: 0" if "Unnamed: 0" in dataframe.columns else dataframe.columns[0],
        text_column="text",
        label_column="label",
        label_name_column="label_text",
    )


def _load_enron_dataframe(force_download: bool = False) -> pd.DataFrame:
    csv_path = ensure_enron_spam_dataset(force=force_download)
    dataframe = pd.read_csv(csv_path)
    text_series = dataframe["text"].fillna("")
    if "subject" in dataframe.columns:
        subject_series = dataframe["subject"].fillna("").astype(str).str.strip()
        text_series = (subject_series + "\n" + text_series.astype(str)).str.strip()
    dataframe["combined_text"] = text_series
    return _finalize_dataframe(
        dataframe,
        source_name="enron_spam_setfit",
        email_id_column="message_id",
        text_column="combined_text",
        label_column="label",
        label_name_column="label_text" if "label_text" in dataframe.columns else None,
    )


def _load_spamassassin_ham_dataframe(
    force_download: bool = False,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    training_config = config or DEFAULT_TRAINING_CONFIG
    corpora = ensure_spamassassin_corpus(force=force_download)
    frames: list[pd.DataFrame] = []

    for corpus_name in training_config.spamassassin_ham_sources:
        folder = corpora.get(corpus_name)
        if folder is None or not folder.exists():
            continue

        rows: list[dict[str, Any]] = []
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            try:
                text = extract_email_text_from_file(path)
            except Exception:
                continue
            if not text.strip():
                continue
            rows.append(
                {
                    "email_id": path.name,
                    "text": text,
                    "label": 0,
                    "label_name": "ham",
                }
            )

        if not rows:
            continue

        frame = pd.DataFrame(rows)
        frames.append(
            _finalize_dataframe(
                frame,
                source_name=f"spamassassin_{corpus_name}",
                email_id_column="email_id",
                text_column="text",
                label_column="label",
                label_name_column="label_name",
            )
        )

    if not frames:
        return pd.DataFrame(columns=["email_id", "source", "label", "label_name", "text", "text_length"])
    return pd.concat(frames, ignore_index=True)


def load_corpus_dataframe(
    force_download: bool = False,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    frames = [
        _load_kaggle_dataframe(force_download=force_download),
        _load_enron_dataframe(force_download=force_download),
        _load_spamassassin_ham_dataframe(force_download=force_download, config=config),
    ]
    dataframe = pd.concat(frames, ignore_index=True)
    if dataframe.empty:
        raise RuntimeError(
            "No email samples were loaded from the configured Kaggle, Enron, and SpamAssassin sources."
        )
    return dataframe


def deduplicate_samples(dataframe: pd.DataFrame) -> pd.DataFrame:
    working = dataframe.copy()
    working["dedupe_key"] = working["text"].map(normalize_for_deduplication)
    deduped = working.drop_duplicates(subset=["dedupe_key"]).drop(columns=["dedupe_key"])
    return deduped.reset_index(drop=True)


def split_dataset(dataframe: pd.DataFrame, config: TrainingConfig | None = None):
    training_config = config or DEFAULT_TRAINING_CONFIG
    split_config = training_config.split
    split_config.validate()

    train_validation, test = train_test_split(
        dataframe,
        test_size=split_config.test_size,
        stratify=dataframe["label"],
        random_state=training_config.random_state,
    )
    validation_ratio = split_config.validation_size / (
        split_config.train_size + split_config.validation_size
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_ratio,
        stratify=train_validation["label"],
        random_state=training_config.random_state,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def save_processed_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    ensure_project_directories()
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


def build_dataset_summary(
    full: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    return {
        "total_samples": int(len(full)),
        "spam_samples": int(full["label"].sum()),
        "ham_samples": int((full["label"] == 0).sum()),
        "average_length": float(full["text_length"].mean()),
        "train_size": int(len(train)),
        "validation_size": int(len(validation)),
        "test_size": int(len(test)),
        "source_counts": {str(key): int(value) for key, value in full["source"].value_counts().to_dict().items()},
    }


def prepare_dataset(
    force_download: bool = False,
    config: TrainingConfig | None = None,
) -> DatasetBundle:
    raw = load_corpus_dataframe(force_download=force_download, config=config)
    deduped = deduplicate_samples(raw)
    train, validation, test = split_dataset(deduped, config=config)
    save_processed_splits(train, validation, test)
    summary = build_dataset_summary(deduped, train, validation, test)
    return DatasetBundle(full=deduped, train=train, validation=validation, test=test, summary=summary)
