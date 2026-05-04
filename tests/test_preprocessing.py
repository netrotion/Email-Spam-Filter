from src.preprocessing.features import extract_email_statistics
from src.preprocessing.text import clean_email_text, normalize_for_deduplication


def test_clean_email_text_removes_html_and_normalizes_tokens():
    text = "<html><body>Hello <b>team</b>! Visit https://example.com now.</body></html>"
    cleaned = clean_email_text(text)

    assert "<b>" not in cleaned
    assert "URLTOKEN" in cleaned
    assert "Hello team!" in cleaned


def test_normalize_for_deduplication_is_case_insensitive():
    first = "FREE Offer!!!"
    second = "free offer!!!"

    assert normalize_for_deduplication(first) == normalize_for_deduplication(second)


def test_extract_email_statistics_counts_suspicious_markers():
    stats = extract_email_statistics("WIN money now!!! Click here for free credit.")

    assert stats["exclamation_count"] == 3.0
    assert stats["suspicious_keyword_hits"] >= 4.0
    assert stats["uppercase_ratio"] > 0.0
