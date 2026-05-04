from __future__ import annotations

from collections.abc import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from src.config import TrainingConfig
from src.preprocessing.features import EmailStatsTransformer
from src.preprocessing.text import clean_email_text


def build_feature_union(config: TrainingConfig) -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_email_text,
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=config.max_word_features,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_email_text,
                    strip_accents="unicode",
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=config.max_char_features,
                ),
            ),
            (
                "email_stats",
                Pipeline(
                    steps=[
                        ("stats", EmailStatsTransformer()),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
            ),
        ]
    )


def build_logistic_regression_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_feature_union(config)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=config.max_iterations,
                    C=config.logistic_c,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def build_calibrated_linear_svc_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_feature_union(config)),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        C=config.linear_svc_c,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                    cv=3,
                ),
            ),
        ]
    )


def build_candidate_pipelines(config: TrainingConfig) -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic_regression": lambda: build_logistic_regression_pipeline(config),
        "calibrated_linear_svc": lambda: build_calibrated_linear_svc_pipeline(config),
    }
