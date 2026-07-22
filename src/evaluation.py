# =============================================================
# evaluation.py - Evaluation and Visualization Module
# =============================================================
# This module provides functions for evaluating classifier
# performance and creating visualizations:
#
#   - Accuracy calculation
#   - Classification report (precision, recall, F1-score)
#   - Confusion matrix heatmap
#   - Sample predictions display
#   - Class distribution chart
#   - Preprocessing pipeline visualization
#
# All plots are saved to the output/ directory as PNG files
# for use in reports and presentations.
# =============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from src.config import OUTPUT_DIR, CLASS_NAMES, IMG_SIZE


def calculate_accuracy(y_true, y_pred, model_name="Model"):
    """
    Calculate and display the overall classification accuracy.

    Accuracy = (Number of correct predictions) / (Total predictions)

    Parameters
    ----------
    y_true : numpy.ndarray
        True class labels.
    y_pred : numpy.ndarray
        Predicted class labels.
    model_name : str
        Name of the classifier (for display purposes).

    Returns
    -------
    float
        Accuracy as a fraction between 0 and 1.
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n  {model_name} Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    return accuracy


def generate_classification_report(y_true, y_pred, model_name="Model"):
    """
    Generate a detailed classification report showing precision,
    recall, and F1-score for each class.

    Metrics explained:
    - Precision: Of all predictions for class X, how many were correct?
    - Recall:    Of all actual class X samples, how many were found?
    - F1-score:  Harmonic mean of precision and recall (balanced metric)

    Parameters
    ----------
    y_true : numpy.ndarray
        True class labels.
    y_pred : numpy.ndarray
        Predicted class labels.
    model_name : str
        Name of the classifier.

    Returns
    -------
    str
        The classification report as a formatted string.
    """
    print(f"\n{'=' * 55}")
    print(f"  Classification Report - {model_name}")
    print(f"{'=' * 55}")

    # Get unique class labels present in the data
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    # Create target names for the classes present
    target_names = [
        f"{label}: {CLASS_NAMES.get(label, 'Unknown')[:25]}"
        for label in unique_labels
    ]

    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0,
    )
    print(report)

    # Save report to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(
        OUTPUT_DIR, f"classification_report_{model_name.lower()}.txt"
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 55 + "\n")
        f.write(report)
    print(f"  Report saved to: {report_path}")

    return report


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Generate and save a confusion matrix heatmap.

    The confusion matrix shows how many samples of each true class
    were predicted as each class. Diagonal values are correct
    predictions; off-diagonal values are misclassifications.

    Parameters
    ----------
    y_true : numpy.ndarray
        True class labels.
    y_pred : numpy.ndarray
        Predicted class labels.
    model_name : str
        Name of the classifier.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Compute the confusion matrix
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    n_classes = len(unique_labels)

    # Normalize the confusion matrix (show percentages instead of counts)
    # This makes it easier to compare performance across classes
    # with different sample counts.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Handle division by zero for classes with no samples
    cm_normalized = np.nan_to_num(cm_normalized)

    # Create the figure
    fig_size = max(12, n_classes * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Plot the heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set axis labels
    tick_labels = [str(label) for label in unique_labels]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)

    # Add text annotations for smaller matrices
    if n_classes <= 25:
        for i in range(n_classes):
            for j in range(n_classes):
                color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                        ha='center', va='center', color=color, fontsize=5)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(
        f'Confusion Matrix - {model_name}\n'
        f'(Normalized, {n_classes} classes)',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(
        OUTPUT_DIR, f"confusion_matrix_{model_name.lower()}.png"
    )
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix saved to: {save_path}")


def display_sample_predictions(images, y_true, y_pred, model_name="Model",
                                n_samples=20):
    """
    Display a grid of sample test images with their true and
    predicted labels. Correct predictions are shown with green
    borders, incorrect ones with red borders.

    Parameters
    ----------
    images : list of numpy.ndarray
        Preprocessed grayscale test images.
    y_true : numpy.ndarray
        True class labels.
    y_pred : numpy.ndarray
        Predicted class labels.
    model_name : str
        Name of the classifier.
    n_samples : int
        Number of sample images to display.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Limit to available samples
    n_samples = min(n_samples, len(images))

    # Select a mix of correct and incorrect predictions
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # Try to show half correct, half incorrect
    n_incorrect = min(n_samples // 2, len(incorrect_indices))
    n_correct = min(n_samples - n_incorrect, len(correct_indices))

    # Randomly sample indices
    rng = np.random.RandomState(42)
    selected = []
    if n_incorrect > 0:
        selected.extend(
            rng.choice(incorrect_indices, n_incorrect, replace=False)
        )
    if n_correct > 0:
        selected.extend(
            rng.choice(correct_indices, n_correct, replace=False)
        )

    rng.shuffle(selected)

    # Create the figure grid
    n_cols = 5
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

    if n_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        f'Sample Predictions - {model_name}',
        fontsize=16, fontweight='bold', y=1.02
    )

    for idx, ax in enumerate(axes.flat):
        if idx < len(selected):
            sample_idx = selected[idx]
            img = images[sample_idx]
            true_label = y_true[sample_idx]
            pred_label = y_pred[sample_idx]
            is_correct = true_label == pred_label

            # Display the image
            ax.imshow(img, cmap='gray')

            # Set title color based on correctness
            color = '#2ecc71' if is_correct else '#e74c3c'  # green or red
            symbol = '[OK]' if is_correct else '[X]'

            ax.set_title(
                f'{symbol} True: {true_label}\nPred: {pred_label}',
                fontsize=9,
                color=color,
                fontweight='bold',
            )

            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        else:
            ax.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR, f"sample_predictions_{model_name.lower()}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sample predictions saved to: {save_path}")


def plot_class_distribution(labels, title="Dataset Class Distribution"):
    """
    Plot the distribution of samples across classes as a bar chart.

    This visualization helps identify class imbalance in the dataset,
    which can affect classifier performance.

    Parameters
    ----------
    labels : numpy.ndarray
        Array of class labels.
    title : str
        Title for the plot.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique_labels)))
    bars = ax.bar(unique_labels, counts, color=colors, edgecolor='white',
                  linewidth=0.5)

    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(unique_labels)
    ax.set_xticklabels(unique_labels, fontsize=7, rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Add total count annotation
    total = len(labels)
    ax.text(
        0.98, 0.95,
        f'Total: {total:,} images\n{len(unique_labels)} classes',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8)
    )

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Class distribution plot saved to: {save_path}")


def plot_preprocessing_pipeline(sample_images, sample_labels):
    """
    Visualize the preprocessing pipeline on sample images.

    Shows the transformation from original → grayscale → edges
    for several sample images, which is useful for reports.

    Parameters
    ----------
    sample_images : list of numpy.ndarray
        List of raw BGR images to demonstrate the pipeline on.
    sample_labels : list of int
        Class labels for the sample images.
    """
    from src.preprocessing import get_preprocessing_visualization

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_samples = min(5, len(sample_images))
    stages = ['original', 'grayscale', 'blurred', 'edges']
    stage_titles = ['Original (Color)', 'Grayscale', 'Gaussian Blur',
                    'Canny Edges']

    fig, axes = plt.subplots(
        n_samples, len(stages),
        figsize=(3 * len(stages), 3 * n_samples)
    )

    if n_samples == 1:
        axes = np.array([axes])

    fig.suptitle(
        'Image Preprocessing Pipeline',
        fontsize=16, fontweight='bold', y=1.02
    )

    for i in range(n_samples):
        # Get all intermediate preprocessing results
        viz = get_preprocessing_visualization(sample_images[i])

        for j, (stage_key, stage_title) in enumerate(
            zip(stages, stage_titles)
        ):
            ax = axes[i][j]
            img = viz[stage_key]

            # Use appropriate colormap
            if stage_key == 'original':
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')

            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles on first row
            if i == 0:
                ax.set_title(stage_title, fontsize=11, fontweight='bold')

            # Row labels (class name) on first column
            if j == 0:
                class_name = CLASS_NAMES.get(sample_labels[i], '?')
                # Truncate long names
                if len(class_name) > 20:
                    class_name = class_name[:18] + '...'
                ax.set_ylabel(
                    f'Class {sample_labels[i]}\n{class_name}',
                    fontsize=8, rotation=0, labelpad=70, va='center'
                )

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "preprocessing_pipeline.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Preprocessing pipeline visualization saved to: {save_path}")


def plot_accuracy_comparison(results):
    """
    Create a bar chart comparing accuracy of different classifiers.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to accuracy values.
        Example: {'SVM': 0.92, 'KNN': 0.85}
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = list(results.keys())
    accuracies = [results[m] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(models, accuracies, color=colors[:len(models)],
                  edgecolor='white', linewidth=2, width=0.5)

    # Add accuracy labels on top of bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{acc:.2f}%',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classifier Accuracy Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Accuracy comparison chart saved to: {save_path}")
