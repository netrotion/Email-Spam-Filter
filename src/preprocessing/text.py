from __future__ import annotations

import re


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style).*?>.*?</\1>", flags=re.IGNORECASE | re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r"\s+([!?.,;:])")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def html_to_text(value: str) -> str:
    no_script = SCRIPT_STYLE_PATTERN.sub(" ", value)
    return HTML_TAG_PATTERN.sub(" ", no_script)


def normalize_whitespace(value: str) -> str:
    compact = WHITESPACE_PATTERN.sub(" ", value).strip()
    return SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r"\1", compact)


def clean_email_text(value: str) -> str:
    if value is None:
        return ""
    text = value.replace("\x00", " ")
    text = html_to_text(text)
    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = EMAIL_PATTERN.sub(" EMAILTOKEN ", text)
    return normalize_whitespace(text)


def normalize_for_deduplication(value: str) -> str:
    return clean_email_text(value).lower()
