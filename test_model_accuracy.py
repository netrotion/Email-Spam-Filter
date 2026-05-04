"""
Script để test và verify model accuracy issue.

Chạy script này để:
1. Test model với nhiều samples
2. Phân tích distribution của predictions
3. Tạo báo cáo chi tiết
"""

from src.api.dataset_sampler import TestDatasetSampler
from src.api.services import SpamDetectionService
from collections import Counter
import statistics


def test_model_accuracy(num_samples=100):
    """Test model accuracy với nhiều samples."""
    print("=" * 70)
    print("  MODEL ACCURACY TEST")
    print("=" * 70)
    print()

    # Load dataset
    sampler = TestDatasetSampler()
    if not sampler.load_samples():
        print("ERROR: Cannot load dataset")
        return

    print(f"Dataset loaded: {sampler.total_samples} samples")
    print(f"  Spam: {sampler.spam_count}")
    print(f"  Ham: {sampler.ham_count}")
    print()

    # Load service
    service = SpamDetectionService()

    # Test samples
    print(f"Testing with {num_samples} random samples...")
    print()

    results = {
        "correct": 0,
        "wrong": 0,
        "spam_correct": 0,
        "spam_wrong": 0,
        "ham_correct": 0,
        "ham_wrong": 0,
        "confidences": [],
        "spam_probs": [],
        "predictions": [],
        "actuals": [],
    }

    errors = []

    for i in range(num_samples):
        sample = sampler.get_random_sample()
        if not sample:
            continue

        try:
            result = service.predict_text(sample.text)
            predicted = result["label"]
            actual = sample.label_name
            confidence = result["confidence"]
            spam_prob = result["spam_probability"]

            is_correct = (predicted == actual)

            # Update stats
            results["confidences"].append(confidence)
            results["spam_probs"].append(spam_prob)
            results["predictions"].append(predicted)
            results["actuals"].append(actual)

            if is_correct:
                results["correct"] += 1
                if actual == "spam":
                    results["spam_correct"] += 1
                else:
                    results["ham_correct"] += 1
            else:
                results["wrong"] += 1
                if actual == "spam":
                    results["spam_wrong"] += 1
                else:
                    results["ham_wrong"] += 1

                # Save error for analysis
                errors.append({
                    "text": sample.text[:100],
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": confidence,
                    "spam_prob": spam_prob,
                })

        except Exception as e:
            print(f"ERROR on sample {i+1}: {e}")

    # Print results
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    total = results["correct"] + results["wrong"]
    accuracy = results["correct"] / total * 100 if total > 0 else 0

    print(f"Total tested: {total}")
    print(f"Correct: {results['correct']} ({results['correct']/total*100:.1f}%)")
    print(f"Wrong: {results['wrong']} ({results['wrong']/total*100:.1f}%)")
    print()

    print(f"Accuracy: {accuracy:.1f}%")
    print()

    # Spam/Ham breakdown
    spam_total = results["spam_correct"] + results["spam_wrong"]
    ham_total = results["ham_correct"] + results["ham_wrong"]

    if spam_total > 0:
        spam_acc = results["spam_correct"] / spam_total * 100
        print(f"Spam accuracy: {results['spam_correct']}/{spam_total} = {spam_acc:.1f}%")

    if ham_total > 0:
        ham_acc = results["ham_correct"] / ham_total * 100
        print(f"Ham accuracy: {results['ham_correct']}/{ham_total} = {ham_acc:.1f}%")

    print()

    # Confidence analysis
    if results["confidences"]:
        avg_conf = statistics.mean(results["confidences"])
        min_conf = min(results["confidences"])
        max_conf = max(results["confidences"])

        print(f"Confidence stats:")
        print(f"  Average: {avg_conf:.3f}")
        print(f"  Min: {min_conf:.3f}")
        print(f"  Max: {max_conf:.3f}")
        print()

    # Spam probability analysis
    if results["spam_probs"]:
        avg_spam_prob = statistics.mean(results["spam_probs"])
        min_spam_prob = min(results["spam_probs"])
        max_spam_prob = max(results["spam_probs"])

        print(f"Spam probability stats:")
        print(f"  Average: {avg_spam_prob:.3f}")
        print(f"  Min: {min_spam_prob:.3f}")
        print(f"  Max: {max_spam_prob:.3f}")
        print()

    # Prediction distribution
    pred_counter = Counter(results["predictions"])
    print(f"Prediction distribution:")
    for label, count in pred_counter.items():
        print(f"  {label}: {count} ({count/total*100:.1f}%)")
    print()

    # Show some errors
    if errors:
        print("=" * 70)
        print("  SAMPLE ERRORS (first 10)")
        print("=" * 70)
        print()

        for i, error in enumerate(errors[:10], 1):
            print(f"{i}. Actual: {error['actual']:4s} | Predicted: {error['predicted']:4s} | Conf: {error['confidence']:.3f}")
            print(f"   Text: {error['text']}...")
            print()

    # Diagnosis
    print("=" * 70)
    print("  DIAGNOSIS")
    print("=" * 70)
    print()

    if accuracy < 60:
        print("[CRITICAL] Accuracy qua thap (<60%)!")
        print("  - Model co the chua duoc train dung")
        print("  - Hoac model bi loi trong inference")
        print("  - Can re-train model ngay lap tuc")
    elif accuracy < 80:
        print("[WARNING] Accuracy thap (<80%)")
        print("  - Model can duoc improve")
        print("  - Xem xet re-train hoac tune hyperparameters")
    elif accuracy < 90:
        print("[OK] Accuracy chap nhan duoc (80-90%)")
        print("  - Model hoat dong on")
        print("  - Co the improve them")
    else:
        print("[EXCELLENT] Accuracy cao (>90%)")
        print("  - Model hoat dong tot")

    print()

    if avg_conf < 0.6:
        print("[WARNING] Confidence trung binh qua thap (<0.6)")
        print("  - Model khong chac chan ve predictions")
        print("  - Co the can train them hoac adjust threshold")

    print()

    # Check bias
    spam_pred_ratio = pred_counter.get("spam", 0) / total
    if spam_pred_ratio > 0.7:
        print("[WARNING] Model bias ve spam (>70% predictions la spam)")
        print("  - Model co xu huong predict spam qua nhieu")
        print("  - Can balance training data hoac adjust threshold")
    elif spam_pred_ratio < 0.3:
        print("[WARNING] Model bias ve ham (>70% predictions la ham)")
        print("  - Model co xu huong predict ham qua nhieu")
        print("  - Can balance training data hoac adjust threshold")

    print()
    print("=" * 70)

    return results


if __name__ == "__main__":
    import sys

    num_samples = 100
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}, using default 100")

    test_model_accuracy(num_samples)
