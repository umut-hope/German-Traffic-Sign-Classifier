# Traffic Sign Image Classification Using Computer Vision

> **Course Project** — Computer Vision  
> **Team Members:**  
> - Okan Umut ÖZEN (2200674)  
> - Deniz Yaman Denizci (2104118)

---

## Problem Statement

Traffic signs are critical for road safety and driver awareness. This project builds an image classification system that can recognize different traffic signs using **traditional Computer Vision** and **Machine Learning** techniques — without deep learning, CNNs, or neural networks.

---

## Project Architecture

```
traffic-sign-classification/
│
├── main.py                    # Main pipeline (entry point)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── src/                       # Source code modules
│   ├── __init__.py            # Package initialization
│   ├── config.py              # All configuration parameters
│   ├── data_loader.py         # Dataset loading & format detection
│   ├── preprocessing.py       # Image preprocessing (resize, grayscale, edges)
│   ├── feature_extraction.py  # HOG feature extraction
│   ├── classifier.py          # SVM & KNN classifiers
│   └── evaluation.py          # Accuracy metrics & visualizations
│
├── dataset/                   # GTSRB dataset (user must download)
│   └── Train/                 # Training images organized by class
│       ├── 0/                 # Speed limit 20km/h
│       ├── 1/                 # Speed limit 30km/h
│       └── ...                # Classes 2-42
│
├── output/                    # Generated results (auto-created)
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_knn.png
│   ├── sample_predictions_svm.png
│   ├── sample_predictions_knn.png
│   ├── class_distribution.png
│   ├── preprocessing_pipeline.png
│   ├── accuracy_comparison.png
│   └── classification_report_*.txt
│
└── models/                    # Saved trained models (auto-created)
    ├── svm_model.pkl
    └── knn_model.pkl
```

---

## Setup Instructions

### Prerequisites

- **Python 3.8+** installed
- **Visual Studio Code** installed
- **pip** (Python package manager)

### Step 1: Clone or Download the Project

Place the project folder somewhere accessible, for example:
```
C:\Users\YourName\Projects\traffic-sign-classification\
```

### Step 2: Create a Virtual Environment (Recommended)

Open a terminal in VS Code (`Ctrl + ~`) and run:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the GTSRB Dataset

1. Go to **Kaggle**: [GTSRB German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
2. Download the dataset (you need a free Kaggle account)
3. Extract the ZIP file
4. Copy the **`Train`** folder into the `dataset/` directory:

```
dataset/
└── Train/
    ├── 0/
    ├── 1/
    ├── 2/
    └── ... (up to 42)
```

### Step 5: Run the Project

```bash
python main.py
```

For faster testing with limited images:
```bash
python main.py --max-per-class 200
```

To skip KNN training (SVM only):
```bash
python main.py --skip-knn
```

---

## How the Algorithm Works

### Pipeline Overview

```
Raw Images → Resize → Grayscale → HOG Features → Train/Test Split → SVM/KNN → Evaluation
```

### Step-by-Step Explanation

#### 1. Image Loading
Images are loaded from the GTSRB dataset, which contains **43 classes** of German traffic signs (speed limits, warnings, prohibitions, etc.) with over **39,000 training images**.

#### 2. Image Preprocessing
- **Resize** all images to a fixed size (64×64 pixels) so they have uniform dimensions
- **Grayscale conversion** reduces 3 color channels to 1, simplifying computation
- **Canny edge detection** is demonstrated to show how edges reveal sign shapes (used for visualization, not for features)

#### 3. HOG (Histogram of Oriented Gradients) Feature Extraction
HOG is the core feature extraction technique used in this project:

1. **Compute gradients**: Calculate the direction and strength of edges at each pixel
2. **Divide into cells**: Split the image into 8×8 pixel cells
3. **Create histograms**: For each cell, count how many edges point in each of 9 directions (0°-180°)
4. **Normalize in blocks**: Group cells into 2×2 blocks and normalize to handle lighting changes
5. **Concatenate**: Combine all block histograms into a single feature vector (1764 values per image)

**Why HOG works for traffic signs:**
- Traffic signs have distinctive shapes (circles, triangles, octagons)
- HOG captures the distribution of edge directions that define these shapes
- It is robust to small changes in position and lighting

#### 4. Classification

**SVM (Support Vector Machine):**
- Finds the optimal boundary (hyperplane) between classes
- Uses the RBF kernel to handle non-linear relationships
- Generally achieves higher accuracy
- Good for high-dimensional feature spaces

**KNN (K-Nearest Neighbors):**
- Classifies based on the majority vote of the 5 closest training samples
- Simple and intuitive algorithm
- Uses distance weighting (closer neighbors have more influence)
- No explicit training phase needed

#### 5. Evaluation
- **Accuracy**: Percentage of correctly classified images
- **Precision**: Of all images predicted as class X, how many actually are?
- **Recall**: Of all actual class X images, how many did we find?
- **Confusion Matrix**: Visual map showing which classes are confused with each other

---

## Expected Results

With the full GTSRB dataset and default parameters:

| Classifier | Expected Accuracy |
|:----------:|:-----------------:|
| SVM (RBF)  | ~85-92%           |
| KNN (k=5)  | ~80-88%           |

*Actual results depend on the number of images used and parameter tuning.*

---

## Suggestions for Improving Accuracy

1. **Data Augmentation**: Rotate, flip, and add noise to training images to increase dataset diversity
2. **Color Histograms**: Combine HOG features with color histogram features (traffic signs use distinctive colors like red, blue, yellow)
3. **Parameter Tuning**: Use GridSearchCV to find optimal SVM parameters (C, gamma)
4. **Image Enhancement**: Apply histogram equalization (CLAHE) to improve contrast
5. **Multi-Scale HOG**: Extract HOG at multiple image scales and concatenate
6. **Feature Selection**: Use PCA to reduce feature dimensionality and remove noise
7. **Ensemble Methods**: Combine predictions from multiple classifiers (voting)
8. **Better Preprocessing**: Apply adaptive thresholding or morphological operations

---

## Tools and Technologies

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Programming language |
| **OpenCV** | Image loading, preprocessing, HOG computation |
| **NumPy** | Numerical operations and array manipulation |
| **Matplotlib** | Visualization (plots, confusion matrices) |
| **Scikit-learn** | ML classifiers (SVM, KNN), metrics, data splitting |

---

## References

- GTSRB Dataset: [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- HOG Paper: Dalal, N. and Triggs, B. (2005). "Histograms of Oriented Gradients for Human Detection"
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
