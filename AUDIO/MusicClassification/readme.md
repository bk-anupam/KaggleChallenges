# Music Classification - Kaggle Challenge

This directory contains Jupyter notebooks related to a Kaggle challenge on music classification. The goal of the challenge is to classify music tracks into predefined genres using various machine learning and deep learning techniques. Below is a summary of the notebooks and their purposes.

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Audio Processing: `librosa`, `audiomentations`, `soundfile`
  - Visualization: `matplotlib`, `seaborn`, `librosa.display`
  - Machine Learning: `scikit-learn`, `torch`, `pytorch-lightning`, `torchmetrics`
  - Deep Learning Models: `timm`, `torchvision`
  - Parallel Processing: `joblib`, `tqdm`
  - Logging: `logging`
  - Others: `IPython`, `Pillow`, `skimage`

---

## Notebooks Overview

### 1. `data_preprocessing.ipynb`
- **Purpose**: Prepares the raw audio data for further processing.
- **Key Tasks**:
  - Reads audio files and trims silence.
  - Generates mel spectrograms and saves them as images.
  - Visualizes spectrograms and analyzes the target genre distribution.
  - Extracts MFCC (Mel-Frequency Cepstral Coefficients) features for audio tracks.

---

### 2. `preprocess_augment_train.ipynb`
- **Purpose**: Preprocesses audio data, applies augmentations, and trains a deep learning model.
- **Key Tasks**:
  - Splits the dataset into stratified k-folds for cross-validation.
  - Applies audio augmentations (e.g., Gaussian noise, pitch shift) and mel spectrogram augmentations.
  - Defines a PyTorch dataset and dataloader for training and validation.
  - Implements a deep learning model using `timm` for feature extraction and classification.
  - Trains the model with PyTorch Lightning and logs metrics.

---

### 3. `train_baseline.ipynb`
- **Purpose**: Implements a baseline model for music classification.
- **Key Tasks**:
  - Loads and preprocesses the dataset.
  - Defines a PyTorch Lightning model with a backbone (e.g., ResNet) and a classifier.
  - Trains the model using stratified k-fold cross-validation.
  - Logs training and validation metrics, including F1 score and loss.

---

### 4. `mfcc_baseline.ipynb`
- **Purpose**: Implements a baseline model using MFCC features.
- **Key Tasks**:
  - Extracts MFCC features from audio files.
  - Defines a PyTorch dataset and dataloader for MFCC-based training.
  - Implements a simple feedforward neural network for classification.
  - Trains the model using PyTorch Lightning and evaluates performance with F1 score.

---

### 5. `pog_musicclf_inference.ipynb`
- **Purpose**: Performs inference on test data using trained models.
- **Key Tasks**:
  - Loads trained models and their checkpoints.
  - Applies preprocessing and feature extraction to test data.
  - Generates predictions for test samples and saves them to a CSV file.
  - Combines predictions from multiple folds for final submission.

---

### 6. `lr_meta_model.ipynb`
- **Purpose**: Implements a meta-model using logistic regression to combine predictions from multiple models.
- **Key Tasks**:
  - Processes out-of-fold (OOF) predictions and test predictions from base models.
  - Trains a logistic regression model to combine predictions from multiple models (e.g., ResNext50, EfficientNetB4).
  - Evaluates the meta-model using cross-validation and calculates the F1 score.
  - Generates final predictions for the test set and saves them for submission.

---

## Usage
Each notebook is designed to handle a specific part of the pipeline, from data preprocessing to training, inference, and meta-modeling. Follow the order of the notebooks to reproduce the results or adapt them for your use case.