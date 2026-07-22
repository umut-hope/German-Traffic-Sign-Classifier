# =============================================================
# classifier.py - Machine Learning Classifier Module
# =============================================================
# This module implements two traditional ML classifiers:
#
# 1. SVM (Support Vector Machine):
#    - Finds the optimal hyperplane that separates classes
#    - Uses the "kernel trick" (RBF kernel) to handle non-linear
#      boundaries by mapping features to a higher-dimensional space
#    - Generally achieves higher accuracy than KNN for this task
#    - Training is slower but prediction is fast
#
# 2. KNN (K-Nearest Neighbors):
#    - Classifies a sample based on the majority vote of its
#      K closest neighbors in the feature space
#    - No explicit training phase (lazy learner)
#    - Simple to understand and implement
#    - Prediction can be slow for large datasets
#
# Both classifiers are imported from scikit-learn (sklearn).
# =============================================================

import os
import time
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.config import (
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    SVM_MAX_ITER,
    KNN_N_NEIGHBORS,
    KNN_WEIGHTS,
    MODEL_DIR,
)


def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine classifier on HOG features.

    SVM works by finding the hyperplane that best separates different
    classes while maximizing the margin (distance between the
    hyperplane and the nearest data points of each class).

    The RBF (Radial Basis Function) kernel is used because:
    - HOG features are not linearly separable in the original space
    - RBF maps features to an infinite-dimensional space where
      a linear separator can be found
    - It has only two hyperparameters to tune (C and gamma)

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train : numpy.ndarray
        Training labels of shape (n_samples,).

    Returns
    -------
    tuple (SVC, StandardScaler)
        - Trained SVM classifier
        - Fitted scaler (needed for transforming test data)
    """
    print("\n" + "=" * 55)
    print("  Training SVM Classifier")
    print("=" * 55)
    print(f"  Kernel:      {SVM_KERNEL}")
    print(f"  C:           {SVM_C}")
    print(f"  Gamma:       {SVM_GAMMA}")
    print(f"  Train size:  {X_train.shape[0]} samples")
    print(f"  Features:    {X_train.shape[1]} per sample")

    # Step 1: Normalize features using StandardScaler
    # This scales each feature to have zero mean and unit variance.
    # SVM is sensitive to feature scales - without normalization,
    # features with larger ranges would dominate the distance
    # calculations in the kernel.
    print("\n  [1/2] Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 2: Create and train the SVM classifier
    print("  [2/2] Training SVM model (this may take a few minutes)...")
    svm_model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        max_iter=SVM_MAX_ITER,
        random_state=42,
        decision_function_shape='ovr',  # One-vs-Rest for multi-class
    )

    start_time = time.time()
    svm_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    print(f"\n  [SUCCESS] SVM training complete!")
    print(f"  Training time: {training_time:.1f} seconds")
    print(f"  Support vectors: {svm_model.n_support_.sum()}")

    return svm_model, scaler


def train_knn(X_train, y_train):
    """
    Train a K-Nearest Neighbors classifier on HOG features.

    KNN classifies new samples by finding the K most similar
    training samples (nearest neighbors) and using majority voting.
    Distance weighting gives closer neighbors more influence.

    KNN is a "lazy learner" - it doesn't learn a model during
    training. Instead, it stores the training data and does all
    computation at prediction time.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix of shape (n_samples, n_features).
    y_train : numpy.ndarray
        Training labels of shape (n_samples,).

    Returns
    -------
    tuple (KNeighborsClassifier, StandardScaler)
        - Trained KNN classifier
        - Fitted scaler (needed for transforming test data)
    """
    print("\n" + "=" * 55)
    print("  Training KNN Classifier")
    print("=" * 55)
    print(f"  K (neighbors): {KNN_N_NEIGHBORS}")
    print(f"  Weights:       {KNN_WEIGHTS}")
    print(f"  Train size:    {X_train.shape[0]} samples")
    print(f"  Features:      {X_train.shape[1]} per sample")

    # Step 1: Normalize features
    # KNN uses distance metrics (e.g., Euclidean), which are
    # also sensitive to feature scales.
    print("\n  [1/2] Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 2: Create and train the KNN classifier
    print("  [2/2] Fitting KNN model...")
    knn_model = KNeighborsClassifier(
        n_neighbors=KNN_N_NEIGHBORS,
        weights=KNN_WEIGHTS,
        algorithm='auto',  # Let sklearn choose the best algorithm
        n_jobs=-1,          # Use all CPU cores for prediction
    )

    start_time = time.time()
    knn_model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    print(f"\n  [SUCCESS] KNN training complete!")
    print(f"  Training time: {training_time:.2f} seconds")

    return knn_model, scaler


def predict(model, scaler, X_test):
    """
    Make predictions using a trained classifier.

    Parameters
    ----------
    model : sklearn classifier
        Trained SVM or KNN model.
    scaler : StandardScaler
        Fitted scaler to transform test features (must be the same
        scaler used during training).
    X_test : numpy.ndarray
        Test feature matrix of shape (n_samples, n_features).

    Returns
    -------
    numpy.ndarray
        Predicted class labels.
    """
    # Apply the same scaling transformation used during training
    # IMPORTANT: Use transform(), NOT fit_transform() on test data
    # to avoid data leakage (the scaler should only learn from
    # training data).
    X_test_scaled = scaler.transform(X_test)

    start_time = time.time()
    predictions = model.predict(X_test_scaled)
    pred_time = time.time() - start_time

    print(f"  Prediction time: {pred_time:.2f}s "
          f"({len(X_test)} samples)")

    return predictions


def save_model(model, scaler, filename):
    """
    Save a trained model and its scaler to disk using joblib.

    Parameters
    ----------
    model : sklearn classifier
        Trained model to save.
    scaler : StandardScaler
        Associated scaler to save alongside the model.
    filename : str
        Base filename (without extension).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"{filename}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{filename}_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"  Model saved to:  {model_path}")
    print(f"  Scaler saved to: {scaler_path}")


def load_model(filename):
    """
    Load a previously saved model and scaler from disk.

    Parameters
    ----------
    filename : str
        Base filename (without extension).

    Returns
    -------
    tuple (model, scaler)
        Loaded classifier and its associated scaler.
    """
    model_path = os.path.join(MODEL_DIR, f"{filename}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{filename}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"  Model loaded from:  {model_path}")
    print(f"  Scaler loaded from: {scaler_path}")

    return model, scaler
