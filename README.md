## Title:
# Autism Spectrum Disorder Behavioral Gesture Classification Using Machine Learning
## Project Overview:
This project implements an automated system for classifying Autism Spectrum Disorder (ASD) behavioral gestures using a machine learning approach. Instead of relying on deep learning, the system utilizes handcrafted feature engineering techniques combined with classical machine learning to achieve reliable performance on a small and imbalanced dataset.

The goal of the project is to recognize and classify ASD-related behavioral gestures from images into three categories:

- Crying
- Screaming
- Stimming

## Problem Statement:

Autism Spectrum Disorder (ASD) often involves distinctive behavioral gestures that reflect emotional or psychological states. Manual monitoring of these behaviors is difficult, subjective, and time-consuming. This project aims to develop an automated system capable of recognizing ASD behavioral gestures from images using machine learning techniques, even with limited training data.

## Methodology:
The project follows a structured machine learning pipeline consisting of:

**1. Feature Engineering:**

For each image, three types of handcrafted features are extracted:

**Gabor Features:**

- Capture texture and edge-based information
- Detect orientation-specific patterns
- Extract statistical measures (mean and standard deviation)

**Blob Features:**

- Detect prominent regions and shapes
- Extract structural information such as position and size

**Pose Features:**

- Use MediaPipe Hand Landmarks
- Capture hand gesture geometry
- Extract 21 landmark points (x, y, z) → 63 features

**2. Feature Fusion:**

All extracted features are combined into a single feature vector:
<pre>
[Gabor Features] + [Blob Features] + [Pose Features] → Unified Feature Vector
</pre>

This fusion creates a comprehensive representation of each image.

**3. Feature Processing:**

- Feature Scaling using StandardScaler
- Initial Feature Selection using Mutual Information
- Hybrid Optimization using PSO + CWO to select the most relevant subset of features

**4. Classification Model:**

- Algorithm: Support Vector Machine (SVM)
- Kernel: RBF
- Evaluation Metrics:
  - Accuracy
  - Classification Report
  - Confusion Matrix

## System Flow:

For every input image:

<pre>
  
  Image
  ↓
Gabor Feature Extraction
  ↓
Blob Feature Extraction
  ↓
Pose Feature Extraction
  ↓
Feature Fusion
  ↓
Scaling
  ↓
Feature Selection
  ↓
SVM Classification
  ↓
Predicted Behavior Class

</pre>

## Results:

The final model achieved:

- Validation Accuracy: 87.5%
- Test Accuracy: 84%

The system demonstrated consistent performance across all three ASD behavioral classes despite the limited dataset size.

## Dataset Structure:
The dataset is organized as:
<pre>
  
Datasets/
 ├── train/
 │   ├── crying/
 │   ├── screaming/
 │   └── stimming/
 ├── val/
 │   ├── crying/
 │   ├── screaming/
 │   └── stimming/
 └── test/
     ├── crying/
     ├── screaming/
     └── stimming/

</pre>

## Technologies Used:

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Matplotlib

## Limitations:

- Dataset size is small
- Slight class imbalance
- Model performance may vary on unseen real-world data

## Future Enhancements:

- Increase dataset size
- Use balanced data collection
- Explore deep learning models when more data is available
- Extend system to real-time video-based gesture recognition

## How to Run:
1. Clone the repository
2. Extract the dataset zip files
3. Open the notebook
4. Run all cells sequentially
