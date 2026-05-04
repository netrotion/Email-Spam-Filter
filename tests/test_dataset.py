import pandas as pd

from src.data.dataset import _load_spamassassin_ham_dataframe, deduplicate_samples, split_dataset


def test_deduplicate_samples_drops_normalized_duplicates():
    dataframe = pd.DataFrame(
        [
            {"text": "FREE offer now", "label": 1, "label_name": "spam", "source": "spam"},
            {"text": "free offer now", "label": 1, "label_name": "spam", "source": "spam"},
            {"text": "Project meeting at 3pm", "label": 0, "label_name": "ham", "source": "ham"},
        ]
    )

    deduped = deduplicate_samples(dataframe)

    assert len(deduped) == 2


def test_split_dataset_preserves_all_rows():
    dataframe = pd.DataFrame(
        [
            {"text": f"spam message {idx}", "label": 1, "label_name": "spam", "source": "spam"}
            for idx in range(20)
        ]
        + [
            {"text": f"ham message {idx}", "label": 0, "label_name": "ham", "source": "ham"}
            for idx in range(20)
        ]
    )

    train, validation, test = split_dataset(dataframe)

    assert len(train) + len(validation) + len(test) == len(dataframe)
    assert set(train["label"].unique()) == {0, 1}
    assert set(validation["label"].unique()) == {0, 1}
    assert set(test["label"].unique()) == {0, 1}


def test_load_spamassassin_ham_dataframe_maps_all_rows_to_ham(monkeypatch, tmp_path):
    easy_ham_dir = tmp_path / "easy_ham"
    easy_ham_dir.mkdir()
    (easy_ham_dir / "0001.txt").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "src.data.dataset.ensure_spamassassin_corpus",
        lambda force=False: {"easy_ham": easy_ham_dir, "easy_ham_2": tmp_path / "missing", "hard_ham": tmp_path / "missing2"},
    )
    monkeypatch.setattr(
        "src.data.dataset.extract_email_text_from_file",
        lambda path: "Hello team, weekly project update.",
    )

    dataframe = _load_spamassassin_ham_dataframe(force_download=False)

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["label"] == 0
    assert dataframe.iloc[0]["label_name"] == "ham"
    assert dataframe.iloc[0]["source"] == "spamassassin_easy_ham"
