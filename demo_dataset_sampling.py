"""
Demo script for testing the new dataset sampling feature.

This script demonstrates how to use the dataset sampling functionality
both programmatically and via the web interface.
"""

from pathlib import Path
import sys

# Add project root to path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.api.dataset_sampler import TestDatasetSampler


def demo_dataset_sampler():
    """Demo the TestDatasetSampler class."""
    print("=" * 70)
    print("  DEMO: Dataset Sampling Feature")
    print("=" * 70)
    print()

    # Initialize sampler
    print("1. Initializing TestDatasetSampler...")
    sampler = TestDatasetSampler()

    # Check availability
    print(f"   Dataset available: {sampler.is_available()}")

    if not sampler.is_available():
        print("   [ERROR] Test dataset not found at data/processed/test.csv")
        print("   Please ensure the dataset file exists.")
        return

    print("   [OK] Dataset file found")
    print()

    # Load samples
    print("2. Loading samples from dataset...")
    loaded = sampler.load_samples()

    if not loaded:
        print("   [ERROR] Failed to load dataset")
        return

    print(f"   [OK] Loaded {sampler.total_samples} samples")
    print(f"   - Spam: {sampler.spam_count} ({sampler.spam_count/sampler.total_samples*100:.1f}%)")
    print(f"   - Ham: {sampler.ham_count} ({sampler.ham_count/sampler.total_samples*100:.1f}%)")
    print()

    # Get random sample
    print("3. Getting random sample...")
    sample = sampler.get_random_sample()

    if sample:
        print(f"   Email ID: {sample.email_id}")
        print(f"   Label: {sample.label_name}")
        print(f"   Source: {sample.source}")
        print(f"   Length: {sample.text_length} chars")
        print(f"   Preview: {sample.text[:100]}...")
    print()

    # Get spam sample
    print("4. Getting random spam sample...")
    spam_sample = sampler.get_random_sample(label=1)

    if spam_sample:
        print(f"   Email ID: {spam_sample.email_id}")
        print(f"   Label: {spam_sample.label_name}")
        print(f"   Preview: {spam_sample.text[:100]}...")
    print()

    # Get ham sample
    print("5. Getting random ham sample...")
    ham_sample = sampler.get_random_sample(label=0)

    if ham_sample:
        print(f"   Email ID: {ham_sample.email_id}")
        print(f"   Label: {ham_sample.label_name}")
        print(f"   Preview: {ham_sample.text[:100]}...")
    print()

    # Get samples for display
    print("6. Getting samples for display (5 samples)...")
    display_samples = sampler.get_samples_for_display(5)

    print(f"   Retrieved {len(display_samples)} samples:")
    for i, s in enumerate(display_samples, 1):
        print(f"   {i}. {s['email_id'][:20]}... | {s['label_name']} | {s['text_length']} chars")
    print()

    # Get specific sample by ID
    if display_samples:
        print("7. Getting specific sample by ID...")
        first_id = display_samples[0]['email_id']
        specific = sampler.get_sample_by_id(first_id)

        if specific:
            print(f"   Found sample: {specific.email_id}")
            print(f"   Label: {specific.label_name}")
            print(f"   Text length: {specific.text_length}")
    print()

    print("=" * 70)
    print("  Demo completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Start the webapp: python -m src.api.app")
    print("  2. Open browser: http://localhost:5000")
    print("  3. Look for 'Test Dataset Samples' section")
    print("  4. Click buttons to test with real samples")
    print()


def demo_api_usage():
    """Demo API usage examples."""
    print()
    print("=" * 70)
    print("  API Usage Examples")
    print("=" * 70)
    print()

    print("1. Get list of dataset samples:")
    print("   curl http://localhost:5000/api/dataset-samples")
    print()

    print("2. Get specific sample by ID:")
    print("   curl http://localhost:5000/api/dataset-sample/<email_id>")
    print()

    print("3. Test with random sample:")
    print("   curl -X POST http://localhost:5000/predict-sample -d 'label_filter='")
    print()

    print("4. Test with random spam:")
    print("   curl -X POST http://localhost:5000/predict-sample -d 'label_filter=1'")
    print()

    print("5. Test with random ham:")
    print("   curl -X POST http://localhost:5000/predict-sample -d 'label_filter=0'")
    print()

    print("6. Test with specific sample:")
    print("   curl -X POST http://localhost:5000/predict-sample -d 'email_id=<id>'")
    print()


if __name__ == "__main__":
    try:
        demo_dataset_sampler()
        demo_api_usage()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
