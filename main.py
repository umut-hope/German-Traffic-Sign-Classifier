# =============================================================
# main.py - Traffic Sign Classification Pipeline
# =============================================================
# This is the main entry point for the project. It orchestrates
# the full classification pipeline:
#
#   1. Load the GTSRB dataset
#   2. Preprocess images (resize, grayscale)
#   3. Visualize preprocessing steps
#   4. Extract HOG features
#   5. Split data into training and test sets
#   6. Train SVM classifier
#   7. Train KNN classifier (for comparison)
#   8. Evaluate both classifiers
#   9. Generate confusion matrices
#  10. Display sample predictions
#  11. Compare classifier performance
#
# Usage:
#   python main.py
#   python main.py --max-per-class 200  (limit images for faster testing)
#
# Team Members:
#   - Okan Umut ÖZEN    (2200674)
#   - Deniz Yaman Denizci (2104118)
# =============================================================

import os
import sys
import time
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# Import project modules
from src.config import (
    OUTPUT_DIR,
    TEST_SIZE,
    RANDOM_STATE,
    IMG_SIZE,
    NUM_CLASSES,
)
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dataset
from src.feature_extraction import extract_features_batch
from src.classifier import (
    train_svm,
    train_knn,
    predict,
    save_model,
)
from src.evaluation import (
    calculate_accuracy,
    generate_classification_report,
    plot_confusion_matrix,
    display_sample_predictions,
    plot_class_distribution,
    plot_preprocessing_pipeline,
    plot_accuracy_comparison,
)


def print_header():
    """Print a formatted project header."""
    print("\n" + "=" * 60)
    print("  TRAFFIC SIGN IMAGE CLASSIFICATION")
    print("  Using Computer Vision & Machine Learning")
    print("=" * 60)
    print("  Team: Okan Umut ÖZEN (2200674)")
    print("        Deniz Yaman Denizci (2104118)")
    print("  Dataset: GTSRB (German Traffic Sign Recognition)")
    print("  Method:  HOG Features + SVM/KNN Classifiers")
    print("=" * 60)


def parse_arguments():
    """
    Parse command-line arguments for configurable parameters.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Traffic Sign Classification using HOG + SVM/KNN'
    )
    parser.add_argument(
        '--max-per-class',
        type=int,
        default=None,
        help='Maximum images per class (default: all images). '
             'Use 200-500 for faster testing.'
    )
    parser.add_argument(
        '--skip-knn',
        action='store_true',
        help='Skip KNN training (only train SVM).'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Path to the GTSRB dataset directory.'
    )
    return parser.parse_args()


def main():
    """
    Main function that runs the complete classification pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Print project header
    print_header()

    # Create output directory for saving results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[INFO] Output will be saved to: {OUTPUT_DIR}")

    # Track total execution time
    total_start = time.time()

    # ==========================================================
    # STEP 1: Load the Dataset
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 1: Loading Dataset")
    print("-" * 60)

    try:
        images, labels = load_dataset(
            dataset_dir=args.dataset_dir,
            max_per_class=args.max_per_class,
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        print("\n[TIP] Download the GTSRB dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        print(f"\nExtract the 'Train' folder into the 'dataset/' directory.")
        sys.exit(1)

    # ==========================================================
    # STEP 2: Visualize Class Distribution
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 2: Analyzing Class Distribution")
    print("-" * 60)

    plot_class_distribution(labels)

    # ==========================================================
    # STEP 3: Visualize Preprocessing Pipeline
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 3: Demonstrating Preprocessing Pipeline")
    print("-" * 60)

    # Select one random sample from each of 5 different classes
    unique_classes = np.unique(labels)
    rng = np.random.RandomState(RANDOM_STATE)
    sample_classes = rng.choice(
        unique_classes, min(5, len(unique_classes)), replace=False
    )
    sample_indices = []
    for cls in sample_classes:
        cls_indices = np.where(labels == cls)[0]
        sample_indices.append(rng.choice(cls_indices))

    sample_images = [images[i] for i in sample_indices]
    sample_labels = [labels[i] for i in sample_indices]

    plot_preprocessing_pipeline(sample_images, sample_labels)

    # ==========================================================
    # STEP 4: Preprocess All Images
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 4: Preprocessing Images")
    print("-" * 60)
    print(f"  -> Resize to {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"  -> Convert to grayscale")

    preprocessed_images = preprocess_dataset(images)

    # ==========================================================
    # STEP 5: Extract HOG Features
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 5: Extracting HOG Features")
    print("-" * 60)

    features = extract_features_batch(preprocessed_images)

    # ==========================================================
    # STEP 6: Split Data into Train and Test Sets
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 6: Splitting Data (Train/Test)")
    print("-" * 60)

    X_train, X_test, y_train, y_test, idx_train, idx_test = \
        train_test_split(
            features,
            labels,
            np.arange(len(labels)),  # Track indices for visualization
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,  # Maintain class proportions in both sets
        )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"  Split ratio:  {(1 - TEST_SIZE) * 100:.0f}% train / "
          f"{TEST_SIZE * 100:.0f}% test")

    # Get the corresponding preprocessed images for test set visualization
    test_images = [preprocessed_images[i] for i in idx_test]

    # ==========================================================
    # STEP 7: Train SVM Classifier
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 7: Training SVM Classifier")
    print("-" * 60)

    svm_model, svm_scaler = train_svm(X_train, y_train)

    # ==========================================================
    # STEP 8: Evaluate SVM Classifier
    # ==========================================================
    print("\n" + "-" * 60)
    print("  STEP 8: Evaluating SVM Classifier")
    print("-" * 60)

    print("\n  Making predictions on test set...")
    svm_predictions = predict(svm_model, svm_scaler, X_test)

    svm_accuracy = calculate_accuracy(y_test, svm_predictions, "SVM")
    generate_classification_report(y_test, svm_predictions, "SVM")
    plot_confusion_matrix(y_test, svm_predictions, "SVM")
    display_sample_predictions(test_images, y_test, svm_predictions, "SVM")

    # Save the SVM model
    print("\n  Saving SVM model...")
    save_model(svm_model, svm_scaler, "svm")

    # Dictionary to store results for comparison
    results = {'SVM': svm_accuracy}

    # ==========================================================
    # STEP 9: Train and Evaluate KNN Classifier (optional)
    # ==========================================================
    if not args.skip_knn:
        print("\n" + "-" * 60)
        print("  STEP 9: Training KNN Classifier")
        print("-" * 60)

        knn_model, knn_scaler = train_knn(X_train, y_train)

        print("\n  Making predictions on test set...")
        knn_predictions = predict(knn_model, knn_scaler, X_test)

        knn_accuracy = calculate_accuracy(y_test, knn_predictions, "KNN")
        generate_classification_report(y_test, knn_predictions, "KNN")
        plot_confusion_matrix(y_test, knn_predictions, "KNN")
        display_sample_predictions(
            test_images, y_test, knn_predictions, "KNN"
        )

        # Save the KNN model
        print("\n  Saving KNN model...")
        save_model(knn_model, knn_scaler, "knn")

        results['KNN'] = knn_accuracy

    # ==========================================================
    # STEP 10: Compare Classifiers
    # ==========================================================
    if len(results) > 1:
        print("\n" + "-" * 60)
        print("  STEP 10: Comparing Classifiers")
        print("-" * 60)

        plot_accuracy_comparison(results)

        print("\n  +-----------------------------------+")
        print("  |    Classifier Comparison          |")
        print("  +-------------+---------------------+")
        print("  |  Classifier |  Accuracy           |")
        print("  +-------------+---------------------+")
        for name, acc in results.items():
            print(f"  |  {name:<11}|  {acc * 100:.2f}%              |")
        print("  +-------------+---------------------+")

        best = max(results, key=results.get)
        print(f"\n  Best classifier: {best} "
              f"({results[best] * 100:.2f}%)")

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total execution time: {total_time:.1f} seconds")
    print(f"  Results saved to:     {OUTPUT_DIR}")
    print(f"\n  Generated files:")

    # List all output files
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            filepath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"    - {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("  Thank you for using Traffic Sign Classifier!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
