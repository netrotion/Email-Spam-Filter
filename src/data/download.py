from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

import kagglehub
import pandas as pd
from datasets import concatenate_datasets, load_dataset

from src.config import (
    ENRON_RAW_DIR,
    KAGGLE_RAW_DIR,
    SPAMASSASSIN_RAW_DIR,
    ensure_project_directories,
)


CORPUS_URLS: dict[str, str] = {
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "easy_ham_2": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "hard_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
    "spam_2": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
}
KAGGLE_DATASET_HANDLE = "venky73/spam-mails-dataset"
KAGGLE_DATASET_FILENAME = "spam_ham_dataset.csv"
ENRON_DATASET_HANDLE = "SetFit/enron_spam"
ENRON_DATASET_FILENAME = "enron_spam.csv"


def _archive_path(url: str) -> Path:
    return SPAMASSASSIN_RAW_DIR / Path(url).name


def _candidate_directory_names(archive_path: Path) -> list[Path]:
    stem = archive_path.name
    for suffix in (".tar.bz2", ".tar.gz", ".tgz", ".zip"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return [
        SPAMASSASSIN_RAW_DIR / stem,
        SPAMASSASSIN_RAW_DIR / stem.replace("20030228_", ""),
        SPAMASSASSIN_RAW_DIR / stem.replace("20050311_", ""),
    ]


def _safe_extract(archive_path: Path, output_dir: Path) -> Path:
    before = {path.name for path in output_dir.iterdir() if path.is_dir()}
    with tarfile.open(archive_path, "r:bz2") as archive:
        members = archive.getmembers()
        for member in members:
            target_path = (output_dir / member.name).resolve()
            if output_dir.resolve() not in target_path.parents and output_dir.resolve() != target_path:
                raise ValueError(f"Unsafe archive entry detected: {member.name}")
        archive.extractall(output_dir)

    new_directories = [
        path for path in output_dir.iterdir() if path.is_dir() and path.name not in before
    ]
    if new_directories:
        return new_directories[0]

    for candidate in _candidate_directory_names(archive_path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate extracted folder for {archive_path.name}.")


def _download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        return
    urllib.request.urlretrieve(url, destination)


def ensure_spamassassin_corpus(force: bool = False) -> dict[str, Path]:
    ensure_project_directories()
    extracted_directories: dict[str, Path] = {}
    for corpus_name, url in CORPUS_URLS.items():
        archive_path = _archive_path(url)
        if not force:
            for candidate in _candidate_directory_names(archive_path):
                if candidate.exists():
                    extracted_directories[corpus_name] = candidate
                    break
            if corpus_name in extracted_directories:
                continue
        _download_file(url, archive_path, force=force)
        extracted_directories[corpus_name] = _safe_extract(archive_path, SPAMASSASSIN_RAW_DIR)
    return extracted_directories


def ensure_kaggle_spam_dataset(force: bool = False) -> Path:
    ensure_project_directories()
    target_csv = KAGGLE_RAW_DIR / KAGGLE_DATASET_FILENAME
    if target_csv.exists() and not force:
        return target_csv

    download_root = Path(
        kagglehub.dataset_download(KAGGLE_DATASET_HANDLE, force_download=force)
    )
    csv_candidates = list(download_root.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV file found after downloading Kaggle dataset {KAGGLE_DATASET_HANDLE}."
        )

    source_csv = next(
        (path for path in csv_candidates if path.name.casefold() == KAGGLE_DATASET_FILENAME.casefold()),
        csv_candidates[0],
    )
    shutil.copy2(source_csv, target_csv)
    return target_csv


def ensure_enron_spam_dataset(force: bool = False) -> Path:
    ensure_project_directories()
    target_csv = ENRON_RAW_DIR / ENRON_DATASET_FILENAME
    if target_csv.exists() and not force:
        return target_csv

    dataset = load_dataset(
        ENRON_DATASET_HANDLE,
        cache_dir=str(ENRON_RAW_DIR / ".hf_cache"),
        download_mode="force_redownload" if force else None,
    )
    combined = concatenate_datasets([dataset[split] for split in dataset.keys()])
    dataframe = pd.DataFrame(combined)
    dataframe.to_csv(target_csv, index=False)
    return target_csv
