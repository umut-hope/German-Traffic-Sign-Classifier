# =============================================================
# preprocessing.py - Image Preprocessing Module
# =============================================================
# This module handles all image preprocessing steps:
#   1. Resizing images to a fixed size
#   2. Converting color images to grayscale
#   3. Applying Gaussian blur for noise reduction
#   4. Applying Canny edge detection
#
# Why these steps?
# - Resizing ensures all images have the same dimensions,
#   which is required for feature extraction and ML models.
# - Grayscale reduces the image from 3 channels to 1,
#   simplifying computation while keeping key information.
# - Edge detection highlights boundaries and shapes of signs,
#   which is useful for visualization and understanding.
# =============================================================

import cv2
import numpy as np
from src.config import (
    IMG_SIZE,
    GAUSSIAN_KERNEL,
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
)


def resize_image(image, target_size=None):
    """
    Resize an image to the specified target size.

    Uses cv2.INTER_AREA interpolation for shrinking (better quality)
    and cv2.INTER_LINEAR for enlarging.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (any size, color or grayscale).
    target_size : tuple of int, optional
        Target dimensions as (width, height).
        Defaults to config.IMG_SIZE.

    Returns
    -------
    numpy.ndarray
        Resized image.
    """
    if target_size is None:
        target_size = IMG_SIZE

    # Choose interpolation method based on whether we're
    # shrinking or enlarging the image
    h, w = image.shape[:2]
    target_w, target_h = target_size

    if h > target_h or w > target_w:
        # Shrinking - INTER_AREA gives best results
        interpolation = cv2.INTER_AREA
    else:
        # Enlarging - INTER_LINEAR is smooth and fast
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(image, target_size, interpolation=interpolation)
    return resized


def convert_to_grayscale(image):
    """
    Convert a BGR color image to grayscale.

    Grayscale conversion uses the formula:
        gray = 0.299*R + 0.587*G + 0.114*B
    This weighted sum reflects human perception of brightness.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR color image (3 channels).

    Returns
    -------
    numpy.ndarray
        Grayscale image (1 channel).
    """
    # Check if image is already grayscale (single channel)
    if len(image.shape) == 2:
        return image

    # Convert from BGR (OpenCV default) to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def apply_gaussian_blur(image, kernel_size=None):
    """
    Apply Gaussian blur to reduce image noise.

    Gaussian blur smooths the image by convolving with a Gaussian
    kernel. This helps reduce noise which can cause false edges
    in Canny edge detection.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    kernel_size : tuple of int, optional
        Size of the Gaussian kernel (must be odd numbers).
        Defaults to config.GAUSSIAN_KERNEL.

    Returns
    -------
    numpy.ndarray
        Blurred image.
    """
    if kernel_size is None:
        kernel_size = GAUSSIAN_KERNEL

    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred


def apply_canny_edge_detection(image, low_threshold=None, high_threshold=None):
    """
    Apply Canny edge detection to find edges in the image.

    The Canny algorithm works in these steps:
        1. Apply Gaussian filter to smooth the image
        2. Find intensity gradients using Sobel operators
        3. Apply non-maximum suppression to thin edges
        4. Apply double threshold to detect strong/weak edges
        5. Track edges by hysteresis: keep weak edges connected
           to strong edges

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    low_threshold : int, optional
        Lower threshold for hysteresis. Defaults to config value.
    high_threshold : int, optional
        Upper threshold for hysteresis. Defaults to config value.

    Returns
    -------
    numpy.ndarray
        Binary edge image (white edges on black background).
    """
    if low_threshold is None:
        low_threshold = CANNY_LOW_THRESHOLD
    if high_threshold is None:
        high_threshold = CANNY_HIGH_THRESHOLD

    # Apply Gaussian blur first to reduce noise
    blurred = apply_gaussian_blur(image)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def preprocess_image(image):
    """
    Apply the full preprocessing pipeline to a single image.

    Pipeline:
        1. Resize to fixed dimensions (e.g., 64x64)
        2. Convert to grayscale

    Note: Canny edge detection is NOT applied before feature
    extraction because HOG (Histogram of Oriented Gradients)
    already computes gradient information internally. Applying
    edge detection before HOG would discard gradient magnitude
    information that HOG uses for building its histograms.
    Canny is demonstrated separately for visualization purposes.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR color image (any size).

    Returns
    -------
    numpy.ndarray
        Preprocessed grayscale image (fixed size).
    """
    # Step 1: Resize to uniform dimensions
    resized = resize_image(image)

    # Step 2: Convert to grayscale
    gray = convert_to_grayscale(resized)

    return gray


def preprocess_dataset(images):
    """
    Apply preprocessing to all images in the dataset.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of raw BGR images.

    Returns
    -------
    list of numpy.ndarray
        List of preprocessed grayscale images (all same size).
    """
    total = len(images)
    print(f"\n[INFO] Preprocessing {total} images...")

    preprocessed = []
    for i, img in enumerate(images):
        processed = preprocess_image(img)
        preprocessed.append(processed)

        # Print progress every 20%
        if (i + 1) % max(1, total // 5) == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  Progress: {i + 1}/{total} ({pct:.0f}%)")

    print(f"[SUCCESS] Preprocessed {total} images "
          f"(size: {IMG_SIZE[0]}x{IMG_SIZE[1]}, grayscale)")

    return preprocessed


def get_preprocessing_visualization(image):
    """
    Generate all intermediate preprocessing results for
    visualization purposes. This is used to show the
    preprocessing pipeline in reports and presentations.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR color image.

    Returns
    -------
    dict
        Dictionary containing intermediate images:
        - 'original': original color image (resized)
        - 'grayscale': grayscale version
        - 'blurred': Gaussian blurred version
        - 'edges': Canny edge detection result
    """
    resized = resize_image(image)
    gray = convert_to_grayscale(resized)
    blurred = apply_gaussian_blur(gray)
    edges = apply_canny_edge_detection(gray)

    return {
        'original': cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
        'grayscale': gray,
        'blurred': blurred,
        'edges': edges,
    }
