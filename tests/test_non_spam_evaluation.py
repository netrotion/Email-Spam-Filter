from src.evaluation.non_spam import recommend_next_step


def test_recommend_next_step_prefers_retraining_when_ham_fpr_stays_high():
    rows = [
        {"threshold": 0.50, "all_ham_fpr": 0.53},
        {"threshold": 0.95, "all_ham_fpr": 0.29},
    ]

    recommendation = recommend_next_step(rows)

    assert recommendation["decision"] == "augment_ham_data_and_retrain"


def test_recommend_next_step_allows_threshold_tuning_when_problem_is_moderate():
    rows = [
        {"threshold": 0.50, "all_ham_fpr": 0.12},
        {"threshold": 0.95, "all_ham_fpr": 0.08},
    ]

    recommendation = recommend_next_step(rows)

    assert recommendation["decision"] == "tune_threshold_then_retest"


def test_recommend_next_step_can_keep_model_when_ham_fpr_is_low():
    rows = [
        {"threshold": 0.50, "all_ham_fpr": 0.05},
        {"threshold": 0.95, "all_ham_fpr": 0.02},
    ]

    recommendation = recommend_next_step(rows)

    assert recommendation["decision"] == "keep_current_model"
