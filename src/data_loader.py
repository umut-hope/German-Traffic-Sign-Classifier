# =============================================================
# data_loader.py - Dataset Loading Module
# =============================================================
# This module handles loading the GTSRB (German Traffic Sign
# Recognition Benchmark) dataset from disk. It supports two
# common folder structures:
#
#   Format A (Kaggle):   dataset/Train/0/, dataset/Train/1/, ...
#   Format B (Official): dataset/GTSRB/Final_Training/Images/00000/, ...
#
# Each subfolder represents one class and contains the traffic
# sign images for that class.
# =============================================================

import os
import cv2
import numpy as np
from src.config import DATASET_DIR, MAX_IMAGES_PER_CLASS, NUM_CLASSES


def detect_dataset_format(dataset_dir):
    """
    Auto-detect the GTSRB dataset format by checking which folder
    structure exists on disk.

    Parameters
    ----------
    dataset_dir : str
        Path to the root dataset directory.

    Returns
    -------
    str or None
        Path to the directory containing class subfolders,
        or None if no supported format is found.

    Raises
    ------
    FileNotFoundError
        If the dataset directory does not exist at all.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset directory not found: {dataset_dir}\n"
            f"Please download the GTSRB dataset and extract it into:\n"
            f"  {dataset_dir}\n"
            f"\nSee README.md for detailed download instructions."
        )

    # Check Format A: Kaggle format → dataset/Train/
    kaggle_path = os.path.join(dataset_dir, "Train")
    if os.path.isdir(kaggle_path):
        # Verify it has class subfolders (numbered 0, 1, 2, ...)
        subfolders = [f for f in os.listdir(kaggle_path)
                      if os.path.isdir(os.path.join(kaggle_path, f))]
        if len(subfolders) > 0:
            print(f"[INFO] Detected Kaggle dataset format at: {kaggle_path}")
            return kaggle_path

    # Check Format B: Official format → dataset/GTSRB/Final_Training/Images/
    official_path = os.path.join(
        dataset_dir, "GTSRB", "Final_Training", "Images"
    )
    if os.path.isdir(official_path):
        print(f"[INFO] Detected official GTSRB format at: {official_path}")
        return official_path

    # Check if class folders are directly in the dataset directory
    # (user may have extracted directly)
    subfolders = [f for f in os.listdir(dataset_dir)
                  if os.path.isdir(os.path.join(dataset_dir, f))]
    numeric_folders = [f for f in subfolders if f.isdigit()]
    if len(numeric_folders) >= 10:  # At least 10 class folders
        print(f"[INFO] Detected class folders directly in: {dataset_dir}")
        return dataset_dir

    return None


def load_dataset(dataset_dir=None, max_per_class=None):
    """
    Load all images and their labels from the GTSRB dataset.

    This function:
    1. Auto-detects the dataset format
    2. Iterates through each class subfolder
    3. Loads images using OpenCV
    4. Stores raw images (before preprocessing) and their labels

    Parameters
    ----------
    dataset_dir : str, optional
        Path to the dataset root. Defaults to config.DATASET_DIR.
    max_per_class : int or None, optional
        Maximum number of images to load per class.
        None means load all images. Defaults to config.MAX_IMAGES_PER_CLASS.

    Returns
    -------
    images : list of numpy.ndarray
        List of loaded images (BGR color format, original size).
    labels : numpy.ndarray
        Array of integer class labels corresponding to each image.

    Raises
    ------
    FileNotFoundError
        If the dataset directory or class folders cannot be found.
    ValueError
        If no images could be loaded from the dataset.
    """
    # Use default values from config if not specified
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    if max_per_class is None:
        max_per_class = MAX_IMAGES_PER_CLASS

    # Step 1: Find the directory containing class subfolders
    data_path = detect_dataset_format(dataset_dir)
    if data_path is None:
        raise FileNotFoundError(
            f"\n[ERROR] Could not find GTSRB class folders in: {dataset_dir}\n"
            f"Expected one of these structures:\n"
            f"  {dataset_dir}/Train/0/, /Train/1/, ...\n"
            f"  {dataset_dir}/GTSRB/Final_Training/Images/00000/, ...\n"
            f"\nSee README.md for download instructions."
        )

    # Step 2: Find all class subdirectories
    # Each subdirectory name is the class ID (e.g., '0', '00012')
    all_entries = sorted(os.listdir(data_path))
    class_dirs = [
        entry for entry in all_entries
        if os.path.isdir(os.path.join(data_path, entry))
    ]

    if len(class_dirs) == 0:
        raise FileNotFoundError(
            f"\n[ERROR] No class subdirectories found in: {data_path}\n"
            f"Each class should have its own folder (e.g., 0/, 1/, 2/, ...)."
        )

    print(f"[INFO] Found {len(class_dirs)} class folders")
    print(f"[INFO] Loading images from: {data_path}")
    if max_per_class:
        print(f"[INFO] Limiting to {max_per_class} images per class")

    # Step 3: Load images from each class folder
    images = []
    labels = []
    total_loaded = 0
    # Supported image file extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.ppm', '.bmp')

    for class_dir in class_dirs:
        # Extract the class ID from the folder name
        # Handle both '5' and '00005' naming conventions
        try:
            class_id = int(class_dir)
        except ValueError:
            # Skip non-numeric directories (e.g., '__MACOSX')
            continue

        # Skip if class ID exceeds expected range
        if class_id >= NUM_CLASSES:
            continue

        class_path = os.path.join(data_path, class_dir)
        class_images_loaded = 0

        # Get all image files in this class folder
        image_files = [
            f for f in sorted(os.listdir(class_path))
            if f.lower().endswith(valid_extensions)
        ]

        # Limit the number of images if specified
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        # Load each image
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)

            try:
                # Read the image using OpenCV (BGR format)
                # Use np.fromfile to handle Windows paths with non-ASCII characters (e.g. Masaüstü)
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    # cv2.imread returns None for corrupted/unsupported files
                    print(f"  [WARNING] Could not read: {img_path}")
                    continue

                images.append(img)
                labels.append(class_id)
                class_images_loaded += 1
                total_loaded += 1

            except Exception as e:
                print(f"  [WARNING] Error loading {img_path}: {e}")
                continue

        # Print progress every class
        if class_images_loaded > 0:
            print(
                f"  Class {class_id:2d}: loaded {class_images_loaded} images "
                f"({total_loaded} total)"
            )

    # Step 4: Validate that we loaded some data
    if total_loaded == 0:
        raise ValueError(
            f"\n[ERROR] No images could be loaded from: {data_path}\n"
            f"Please verify that the dataset contains image files "
            f"(.png, .jpg, .ppm)."
        )

    # Convert labels to numpy array for easier manipulation
    labels = np.array(labels, dtype=np.int32)

    print(f"\n[SUCCESS] Loaded {total_loaded} images across "
          f"{len(np.unique(labels))} classes")

    return images, labels


def get_class_name(class_id):
    """
    Get the human-readable name for a traffic sign class.

    Parameters
    ----------
    class_id : int
        The class ID (0-42).

    Returns
    -------
    str
        The traffic sign name, or 'Unknown' if the ID is invalid.
    """
    from src.config import CLASS_NAMES
    return CLASS_NAMES.get(class_id, f"Unknown ({class_id})")
