# =============================================================
# feature_extraction.py - Feature Extraction Module
# =============================================================
# This module extracts HOG (Histogram of Oriented Gradients)
# features from preprocessed images.
#
# HOG Feature Extraction - How It Works:
# ─────────────────────────────────────
# HOG captures the shape and structure of objects by analyzing
# the distribution of gradient (edge) directions in local
# regions of the image.
#
# Steps:
#   1. Compute the gradient (direction and magnitude) at each pixel
#      using Sobel filters (horizontal and vertical derivatives).
#   2. Divide the image into small spatial regions called "cells"
#      (e.g., 8×8 pixels each).
#   3. For each cell, create a histogram of gradient orientations
#      (e.g., 9 bins covering 0°–180°), weighted by gradient magnitude.
#   4. Group adjacent cells into larger "blocks" (e.g., 2×2 cells)
#      and normalize the histograms within each block to handle
#      variations in illumination and contrast.
#   5. Concatenate all block histograms into a single feature vector.
#
# Why HOG works well for traffic signs:
# - Traffic signs have distinctive shapes (circles, triangles, etc.)
# - HOG captures edge orientations which define these shapes
# - Block normalization provides robustness to lighting changes
# - The feature is compact and efficient for traditional ML models
# =============================================================

import cv2
import numpy as np
from src.config import (
    IMG_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    HOG_BLOCK_STRIDE,
)


def create_hog_descriptor():
    """
    Create and configure an OpenCV HOG descriptor.

    The HOG descriptor is configured with parameters from config.py.
    Using OpenCV's HOGDescriptor provides efficient C++ implementation
    for fast feature computation.

    Returns
    -------
    cv2.HOGDescriptor
        Configured HOG descriptor ready for feature computation.
    """
    # Window size must match the image size
    win_size = IMG_SIZE  # (64, 64)

    # Block size in pixels: cells_per_block * pixels_per_cell
    block_size = (
        HOG_CELLS_PER_BLOCK[0] * HOG_PIXELS_PER_CELL[0],
        HOG_CELLS_PER_BLOCK[1] * HOG_PIXELS_PER_CELL[1],
    )  # (16, 16)

    # Block stride: how much the block window slides
    block_stride = HOG_BLOCK_STRIDE  # (8, 8)

    # Cell size in pixels
    cell_size = HOG_PIXELS_PER_CELL  # (8, 8)

    # Number of orientation bins
    n_bins = HOG_ORIENTATIONS  # 9

    # Create the HOG descriptor with these parameters
    hog = cv2.HOGDescriptor(
        win_size,       # Window size (must match image size)
        block_size,     # Block size
        block_stride,   # Block stride
        cell_size,      # Cell size
        n_bins,         # Number of orientation bins
    )

    return hog


def extract_hog_features(image, hog_descriptor=None):
    """
    Extract HOG features from a single preprocessed grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Preprocessed grayscale image (must be IMG_SIZE dimensions).
    hog_descriptor : cv2.HOGDescriptor, optional
        Pre-created HOG descriptor. If None, a new one is created.
        Passing a pre-created descriptor is more efficient for
        batch processing.

    Returns
    -------
    numpy.ndarray
        1D feature vector containing HOG features.
        For 64×64 image with default params: 1764 features.

    Feature vector size calculation:
        cells_per_dim = 64 / 8 = 8  (image_size / cell_size)
        blocks_per_dim = (8 - 2) / 1 + 1 = 7
            (cells - cells_per_block) / stride_in_cells + 1
        total_blocks = 7 × 7 = 49
        features_per_block = 2 × 2 × 9 = 36
            (cells_per_block² × n_bins)
        total_features = 49 × 36 = 1764
    """
    if hog_descriptor is None:
        hog_descriptor = create_hog_descriptor()

    # Ensure image is the correct size
    if image.shape != (IMG_SIZE[1], IMG_SIZE[0]):
        raise ValueError(
            f"Image size {image.shape} does not match "
            f"expected size {(IMG_SIZE[1], IMG_SIZE[0])}. "
            f"Make sure to preprocess images first."
        )

    # Ensure image is uint8 (required by OpenCV HOG)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Compute HOG features
    # Returns a column vector, so we flatten it to 1D
    features = hog_descriptor.compute(image)
    features = features.flatten()

    return features


def extract_features_batch(images):
    """
    Extract HOG features from a batch of preprocessed images.

    This function creates a single HOG descriptor and reuses it
    for all images, which is more efficient than creating a new
    one for each image.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of preprocessed grayscale images.

    Returns
    -------
    numpy.ndarray
        2D feature matrix of shape (n_images, n_features).
        Each row is the HOG feature vector for one image.
    """
    total = len(images)
    print(f"\n[INFO] Extracting HOG features from {total} images...")

    # Create HOG descriptor once (reuse for efficiency)
    hog_descriptor = create_hog_descriptor()

    # Extract features for the first image to determine vector size
    first_features = extract_hog_features(images[0], hog_descriptor)
    n_features = len(first_features)
    print(f"[INFO] HOG feature vector size: {n_features}")

    # Pre-allocate the feature matrix for efficiency
    # (much faster than appending to a list)
    feature_matrix = np.zeros((total, n_features), dtype=np.float32)
    feature_matrix[0] = first_features

    # Extract features for remaining images
    for i in range(1, total):
        feature_matrix[i] = extract_hog_features(
            images[i], hog_descriptor
        )

        # Print progress every 20%
        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  Progress: {i + 1}/{total} ({pct:.0f}%)")

    print(f"[SUCCESS] Extracted features: matrix shape = {feature_matrix.shape}")

    return feature_matrix
