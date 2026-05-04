from src.config import TrainingConfig
from src.models.inference import predict_texts
from src.models.pipeline import build_logistic_regression_pipeline


class _NoExplanationPipeline:
    supports_explanations = False

    def predict_proba(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            spam_score = 0.9 if "free" in text.lower() else 0.1
            rows.append([1.0 - spam_score, spam_score])
        return np.asarray(rows, dtype=float)

    def predict(self, texts):
        import numpy as np

        return np.asarray([1 if "free" in text.lower() else 0 for text in texts], dtype=int)

    def decision_function(self, texts):
        import numpy as np

        return np.asarray([0.8 if "free" in text.lower() else -0.7 for text in texts], dtype=float)


def _toy_dataset():
    texts = [
        "Free credit available now click here",
        "Claim your bonus prize free money today",
        "Urgent winner offer waiting for you",
        "Congratulations free coupon available",
        "Team meeting moved to tomorrow morning",
        "Please review the attached project notes",
        "Lunch with client has been confirmed",
        "The sprint retrospective starts at noon",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    return texts, labels


def test_logistic_pipeline_can_fit_and_predict():
    texts, labels = _toy_dataset()
    pipeline = build_logistic_regression_pipeline(TrainingConfig(max_word_features=500, max_char_features=500))

    pipeline.fit(texts, labels)
    predictions = pipeline.predict(texts).tolist()

    assert len(predictions) == len(labels)
    assert set(predictions).issubset({0, 1})


def test_predict_texts_returns_confidence_and_signals():
    texts, labels = _toy_dataset()
    pipeline = build_logistic_regression_pipeline(TrainingConfig(max_word_features=500, max_char_features=500))
    pipeline.fit(texts, labels)

    results = predict_texts(
        [
            "Free prize waiting for you click now",
            "Project meeting moved to next week",
        ],
        pipeline=pipeline,
        metadata={"best_model": "toy_logistic"},
        top_n=4,
    )

    assert results[0]["label"] == "spam"
    assert results[1]["label"] == "ham"
    assert 0.0 <= results[0]["confidence"] <= 1.0
    assert 0.0 <= results[1]["confidence"] <= 1.0
    assert results[0]["top_signals"]
    assert results[1]["top_signals"]


def test_predict_texts_supports_contextual_pipelines_without_linear_explanations():
    results = predict_texts(
        [
            "Free prize waiting for you click now",
            "Project meeting moved to next week",
        ],
        pipeline=_NoExplanationPipeline(),
        metadata={"best_model": "contextual_bert_tiny"},
        top_n=4,
    )

    assert results[0]["label"] == "spam"
    assert results[1]["label"] == "ham"
    assert results[0]["top_signals"] == []
    assert results[1]["top_signals"] == []
