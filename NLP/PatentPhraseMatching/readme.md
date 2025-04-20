# Patent Phrase Matching - Kaggle Challenge

This directory contains Jupyter notebooks related to the Kaggle competition on [Patent Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching). The goal of the competition is to train models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents. This task is crucial for determining semantic similarity during patent searches and examinations, helping the patent community connect the dots between millions of patent documents.

---

## Tech Stack
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Machine Learning: `scikit-learn`
  - Deep Learning: `transformers`, `torch`, `datasets`
  - Optimization: `bitsandbytes`
  - Visualization: `matplotlib`
  - Logging and Experiment Tracking: `wandb`
  - Parallel Processing: `torch.multiprocessing`
  - Others: `google.colab`, `pydrive`

---

## Notebooks Overview

### 1. `USPPPM_train_hf.ipynb`
- **Purpose**: Fine-tunes a `deberta-v3-base` model on the competition dataset.
- **Key Tasks**:
  - Implements stratified group k-fold cross-validation.
  - Adds patent section as a special token to the tokenizer vocabulary.
  - Trains the model using the Hugging Face `Trainer` API.
  - Generates out-of-fold (OOF) predictions and calculates the Pearson correlation coefficient as the evaluation metric.

---

### 2. `USPPPM_train_deberta_v2_xlarge.ipynb`
- **Purpose**: Fine-tunes a `deberta-v2-xlarge` model with advanced optimization techniques.
- **Key Tasks**:
  - Uses 8-bit Adam optimizer for memory-efficient training.
  - Adds patent section as a special token to the tokenizer vocabulary.
  - Implements gradient checkpointing and warmup scheduling.
  - Tracks experiments using `wandb`.
  - Generates OOF predictions and evaluates the model using the Pearson correlation coefficient.

---

### 3. `USPPPM_train_deberta_addvocab.ipynb`
- **Purpose**: Fine-tunes a `deberta-v3-small` model by adding missing anchor and target words to the tokenizer vocabulary.
- **Key Tasks**:
  - Identifies and adds missing words from the anchor and target phrases to the tokenizer vocabulary.
  - Trains the model using stratified k-fold cross-validation.
  - Compares the performance of the model with and without the additional vocabulary.

---

### 4. `us-pppm-inference.ipynb`
- **Purpose**: Generates predictions on the test dataset using a `deberta-v2-xlarge` model.
- **Key Tasks**:
  - Loads the trained model and tokenizer.
  - Prepares the test dataset by adding special tokens and tokenizing inputs.
  - Generates predictions for each fold and averages them to create the final submission.

---

### 5. `us-pppm-meta-inference.ipynb`
- **Purpose**: Implements a meta-model to combine predictions from multiple base models.
- **Key Tasks**:
  - Loads OOF predictions from multiple models (`deberta-v3-large`, `bert-for-patents`, `deberta-v2-xlarge`).
  - Trains a level-2 linear regression model (Ridge) to combine predictions.
  - Generates final predictions on the test dataset using the meta-model.

---

## Usage
Each notebook is designed to handle a specific part of the pipeline, from data preprocessing to training, inference, and meta-modeling. Follow the order of the notebooks to reproduce the results or adapt them for your use case.