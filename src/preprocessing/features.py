from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocessing.text import URL_PATTERN, clean_email_text


UPPERCASE_PATTERN = re.compile(r"[A-Z]")
DIGIT_PATTERN = re.compile(r"\d")
WORD_PATTERN = re.compile(r"\b\w+\b")
HTML_LIKE_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
SUSPICIOUS_KEYWORD_PATTERN = re.compile(
    r"\b(free|winner|offer|credit|urgent|money|bonus|click|remove|unsubscribe)\b",
    flags=re.IGNORECASE,
)
CURRENCY_PATTERN = re.compile(r"[$€£]|usd|eur|vnd", flags=re.IGNORECASE)


def extract_email_statistics(text: str) -> dict[str, float]:
    raw_text = text or ""
    cleaned_text = clean_email_text(raw_text)
    word_matches = WORD_PATTERN.findall(cleaned_text)
    word_count = len(word_matches)
    char_count = len(cleaned_text)
    avg_word_length = (
        float(sum(len(word) for word in word_matches)) / word_count if word_count else 0.0
    )
    uppercase_ratio = len(UPPERCASE_PATTERN.findall(raw_text)) / max(len(raw_text), 1)
    digit_ratio = len(DIGIT_PATTERN.findall(raw_text)) / max(len(raw_text), 1)

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "avg_word_length": float(avg_word_length),
        "uppercase_ratio": float(uppercase_ratio),
        "digit_ratio": float(digit_ratio),
        "exclamation_count": float(raw_text.count("!")),
        "question_count": float(raw_text.count("?")),
        "url_count": float(len(URL_PATTERN.findall(raw_text))),
        "currency_count": float(len(CURRENCY_PATTERN.findall(raw_text))),
        "html_tag_count": float(len(HTML_LIKE_PATTERN.findall(raw_text))),
        "suspicious_keyword_hits": float(len(SUSPICIOUS_KEYWORD_PATTERN.findall(raw_text))),
    }


class EmailStatsTransformer(BaseEstimator, TransformerMixin):
    feature_names = np.array(
        [
            "char_count",
            "word_count",
            "avg_word_length",
            "uppercase_ratio",
            "digit_ratio",
            "exclamation_count",
            "question_count",
            "url_count",
            "currency_count",
            "html_tag_count",
            "suspicious_keyword_hits",
        ],
        dtype=object,
    )

    def fit(self, X: Iterable[str], y: Iterable[int] | None = None) -> "EmailStatsTransformer":
        return self

    def transform(self, X: Iterable[str]) -> csr_matrix:
        rows = []
        for text in X:
            stats = extract_email_statistics(text)
            rows.append([stats[name] for name in self.feature_names])
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return self.feature_names.copy()
