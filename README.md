# Voice is our most primal form of communication but computers have historically only understood what we say, not how we say it today we change that.#

# Speech-Emotion-Recognition

> Multi-dataset, multi-model Speech Emotion Recognition system using classical ML and deep learning on RAVDESS, CREMA-D, TESS, and SAVEE.

---

## Overview

Speech-Emotion-Recognition is a comprehensive Speech Emotion Recognition (SER) pipeline that classifies human emotions from raw audio. It combines four benchmark datasets, rich audio feature engineering, data augmentation, and a comparative evaluation across ten machine learning models — from classical classifiers to a deep 1D CNN.

**Recognized Emotions:** `neutral` · `happy` · `sad` · `angry` · `fear` · `disgust` · `surprise`

---

## Datasets

| Dataset | Description | Samples |
|---|---|---|
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 24 professional actors, controlled studio recordings | ~1,440 |
| [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) | 91 actors, diverse demographics | ~7,442 |
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2 female speakers, Toronto Emotional Speech Set | ~2,800 |
| [SAVEE](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) | 4 male British English speakers | ~480 |

> **Note:** RAVDESS classes 1 (neutral) and 2 (calm) are merged into a single `neutral` class due to their acoustic similarity.

---

## Feature Engineering

Each audio file is loaded with a 2.5s window (offset 0.6s) using `librosa`. Three acoustic feature types are extracted and concatenated:

| Feature | Description |
|---|---|
| **MFCC** | 40 Mel-Frequency Cepstral Coefficients — captures spectral shape |
| **ZCR** | Zero Crossing Rate — measures signal sharpness/noisiness |
| **RMSE** | Root Mean Square Energy — measures loudness/energy over time |

### Data Augmentation (4x expansion)
Each audio sample generates 4 feature vectors:
1. Original
2. + Gaussian Noise
3. + Pitch Shift
4. + Noise & Pitch combined

---

## Models

### Deep Learning
| Model | Architecture |
|---|---|
| **1D CNN** | 4x Conv1D blocks (256→256→128→64 filters), BatchNorm, MaxPool, Dropout, Dense(32) → Softmax(7) |

Trained with:
- Optimizer: Adam (lr=0.0005)
- Loss: Categorical Cross-Entropy
- Epochs: 80, Batch size: 64
- Callback: ReduceLROnPlateau

### Classical ML (GPU-accelerated via cuML where applicable)
- Support Vector Machine (RBF kernel) — `cuml.svm.SVC`
- Random Forest (200 trees, max_depth=20)
- XGBoost (200 estimators, gpu_hist)
- Gradient Boosting
- K-Nearest Neighbors (k=5)
- Naive Bayes (Gaussian)
- Extra Trees
- AdaBoost
- Decision Tree

---

## Visualizations

The notebook generates:
- Waveform plots per emotion class
- Spectrograms (STFT-based)
- MFCC heatmaps
- Emotion class distribution bar chart
- Training/Validation accuracy & loss curves
- Confusion matrices for every model

---

## How to Run

### On Kaggle (Recommended)

1. Clone or upload the notebook to Kaggle
2. Add the following datasets to your Kaggle notebook:
   - `uwrfkaggler/ravdess-emotional-speech-audio`
   - `ejlok1/speech-emotion-recognition-en`
   - `ejlok1/toronto-emotional-speech-set-tess`
3. Enable GPU accelerator (P100 recommended)
4. Run all cells

### On Google Colab

```bash
# Install required packages
pip install librosa soundfile torchaudio cuml-cu11
```

Update dataset paths at the top of the notebook to match your mounted Google Drive.

---

## Requirements
