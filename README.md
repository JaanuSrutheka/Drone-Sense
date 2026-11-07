# 🚁 Drone Sense: Intelligent Drone Detection Through Sound

## 📌 Project Overview

**Drone Sense** is a machine learning-based audio classification project designed to detect drone sounds by distinguishing them from surrounding background noise. The system analyses acoustic characteristics of audio recordings and uses a supervised learning approach to classify sounds into two categories: **Drone** and **Background Noise**.

Instead of relying on visual cameras or manually monitoring audio, Drone Sense extracts meaningful acoustic features from recordings and uses a trained machine learning model to identify patterns associated with drone sounds.

The project implements an end-to-end audio processing and classification pipeline using Python, Librosa, NumPy, Pandas, and Scikit-learn.

## 🎯 Objectives

- Develop an audio-based system for drone sound detection.
- Preprocess audio recordings to maintain consistency.
- Extract meaningful acoustic features from audio signals.
- Prepare a labelled dataset for supervised machine learning.
- Train a Random Forest classifier to distinguish drone sounds from background noise.
- Evaluate the model using appropriate classification metrics.
- Save the trained model for reuse in predicting new audio recordings.

## ⚙️ System Workflow

The project follows these major stages:

1. **Audio Input:** Drone recordings and background-noise samples are provided as input.
2. **Audio Preprocessing:** Audio is converted to mono, resampled to 16 kHz, trimmed to remove silent portions, and normalized.
3. **Feature Extraction:** Acoustic characteristics are extracted from each processed recording.
4. **Dataset Preparation:** Extracted features are organized into a labelled dataset, with drone samples assigned label `1` and background-noise samples assigned label `0`.
5. **Model Training:** A Random Forest classifier is trained using the extracted features and corresponding labels.
6. **Model Evaluation:** The trained classifier is evaluated using F1-score and ROC-AUC.
7. **Prediction:** The saved model can be reused to classify new audio recordings.

## 🧠 Feature Extraction

The system extracts **32 acoustic features** from each audio recording:

| Feature Category | Number of Features |
|---|---:|
| MFCC Means | 13 |
| MFCC Standard Deviations | 13 |
| Spectral Centroid | 1 |
| Spectral Bandwidth | 1 |
| Spectral Rolloff | 1 |
| Spectral Flatness | 1 |
| Zero-Crossing Rate | 1 |
| RMS Energy | 1 |
| **Total** | **32** |

These features capture the spectral, frequency-related, and energy characteristics of audio signals, providing a numerical representation for machine learning.

## 🤖 Machine Learning Model

Drone Sense uses a **Random Forest Classifier** from Scikit-learn.

The current implementation uses the following configuration:

- Number of decision trees: 100
- Maximum tree depth: 12
- Minimum samples required for splitting: 4
- Class weighting: Balanced
- Random state: 42

Random Forest combines predictions from multiple decision trees using majority voting to determine whether an audio recording contains drone sound or background noise.

## 📊 Current Dataset

The current training pipeline generates a dataset of 600 samples using two base audio recordings and audio augmentation techniques.

| Category | Number of Samples | Label |
|---|---:|---:|
| Drone | 150 | 1 |
| Background Noise | 450 | 0 |
| **Total** | **600** | |

The dataset is divided into:
- 75% training data
- 25% testing data

**Note:** The generated samples are variations of two base recordings and should not be considered 600 independent real-world recordings. Evaluation on a larger collection of independent recordings is required to establish real-world generalization.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Librosa** – Audio loading, preprocessing, and feature extraction
- **NumPy** – Numerical computations
- **Pandas** – Dataset preparation and manipulation
- **Scikit-learn** – Random Forest model training and evaluation
- **Joblib** – Saving and loading the trained model
- **Matplotlib** – Audio and feature visualization

## 📁 Project Structure

```text
Drone-Sense/
│
├── data/
│   ├── raw/                  # Original audio samples
│   └── processed/            # Extracted feature dataset
│
├── models/                   # Saved trained model
│
├── notebooks/
│   └── 01_eda_and_features.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── robustness.py
│   ├── test_feature_extraction.py
│   └── testing_preprocessing.py
│
├── requirements.txt
└── README.md
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone <repository-url>
cd Drone-Sense
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python -m src.train
```

This executes the preprocessing, feature extraction, dataset preparation, and Random Forest training pipeline, and saves the trained model.

## 📈 Evaluation Metrics

The project uses:

- **F1-score:** Evaluates the balance between precision and recall.
- **ROC-AUC:** Measures the classifier's ability to distinguish between the two classes across different classification thresholds.

## 🔮 Future Enhancements

- Expand the dataset with diverse, independently recorded drone and environmental sounds.
- Compare multiple machine learning algorithms.
- Perform hyperparameter tuning and cross-validation.
- Improve robustness against varying environmental conditions.
- Integrate real-time microphone-based drone sound detection.
- Develop a user-friendly interface for audio upload and prediction.

## 👥 Project Team

**Project:** Drone Sense – Intelligent Drone Detection Through Sound

A machine learning project focused on acoustic feature extraction and audio-based drone classification.

## 👥 Team
-Jaanu Srutheka 
-Swati
