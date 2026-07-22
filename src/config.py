# =============================================================
# config.py - Project Configuration
# =============================================================
# This file contains all configurable parameters for the
# traffic sign classification project. Centralizing settings
# here makes it easy to modify parameters without changing
# the core logic in other modules.
# =============================================================

import os

# ============================================================
# Path Configuration
# ============================================================
# Root directory of the project (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset directory - the user should place the GTSRB dataset here.
# Supported structures:
#   Option A (Kaggle format):  dataset/Train/0/, dataset/Train/1/, ...
#   Option B (Official format): dataset/GTSRB/Final_Training/Images/00000/, ...
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

# Directory where output results (plots, reports) will be saved
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Directory for saving trained models
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# Image Preprocessing Parameters
# ============================================================
# All images will be resized to this fixed size (width, height).
# 64x64 provides a good balance between detail and processing speed.
IMG_SIZE = (64, 64)

# Gaussian blur kernel size for noise reduction before edge detection.
# Must be an odd number. (5, 5) provides moderate smoothing.
GAUSSIAN_KERNEL = (5, 5)

# Canny edge detection thresholds.
# Low threshold: edges below this are discarded.
# High threshold: edges above this are definite edges.
# Pixels between the thresholds are kept if connected to strong edges.
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# ============================================================
# HOG (Histogram of Oriented Gradients) Feature Parameters
# ============================================================
# HOG divides the image into cells and computes a histogram of
# gradient orientations in each cell. Cells are grouped into
# blocks for normalization.
#
# Number of orientation bins (directions to quantize gradients into).
# 9 bins means each bin covers 20 degrees (180/9), which is standard.
HOG_ORIENTATIONS = 9

# Size of each cell in pixels. (8, 8) is the standard choice.
# Smaller cells capture finer detail but increase feature vector size.
HOG_PIXELS_PER_CELL = (8, 8)

# Number of cells per block for local normalization.
# (2, 2) means each block is a 2x2 group of cells.
HOG_CELLS_PER_BLOCK = (2, 2)

# Block stride in pixels - how much the block window moves.
# Using cell_size as stride means blocks overlap by 1 cell.
HOG_BLOCK_STRIDE = (8, 8)

# ============================================================
# Machine Learning Parameters
# ============================================================
# Fraction of data to reserve for testing (0.2 = 20% test, 80% train)
TEST_SIZE = 0.2

# Random seed for reproducibility - ensures the same split each run.
RANDOM_STATE = 42

# --- SVM (Support Vector Machine) Parameters ---
# Kernel type: 'rbf' (Radial Basis Function) works well for
# non-linearly separable data like image features.
SVM_KERNEL = 'rbf'

# Regularization parameter C: higher values mean less regularization
# (model fits training data more closely, risk of overfitting).
SVM_C = 10

# Gamma parameter for RBF kernel: 'scale' uses 1/(n_features * X.var())
# which is a good default that adapts to the data.
SVM_GAMMA = 'scale'

# Maximum iterations for SVM solver (-1 means no limit).
SVM_MAX_ITER = -1

# --- KNN (K-Nearest Neighbors) Parameters ---
# Number of neighbors to consider for classification.
# k=5 is a standard choice - odd number avoids ties.
KNN_N_NEIGHBORS = 5

# Weight function: 'distance' means closer neighbors have more
# influence on the prediction than farther ones.
KNN_WEIGHTS = 'distance'

# ============================================================
# Dataset Parameters
# ============================================================
# Maximum number of images to load per class (for faster testing).
# Set to None to use ALL images in each class.
# Recommended: None for final evaluation, 200-500 for development.
MAX_IMAGES_PER_CLASS = None

# Total number of traffic sign classes in GTSRB
NUM_CLASSES = 43

# ============================================================
# GTSRB Class Names
# ============================================================
# Human-readable names for all 43 traffic sign categories.
# These are used for labeling plots and displaying results.
CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing (vehicles > 3.5t)',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles > 3.5t prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing (vehicles > 3.5t)',
}
