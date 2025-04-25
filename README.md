# Deepfake Audio Detection Using Mel Spectrograms and Res2Net Classifier

This repository presents a robust deep learning framework for detecting AI-generated (TTS) speech, commonly referred to as audio deepfakes. Leveraging Mel spectrograms as the primary feature representation, the system employs a modified ResNet architecture—specifically, [Res2Net](https://ieeexplore.ieee.org/abstract/document/9413828)—to perform binary classification between authentic and synthetic audio samples.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model & Approach](#model--approach)
4. [Implementation Process](#implementation-process)
5. [Training & Evaluation](#training--evaluation)
6. [Results & Performance](#results--performance)
7. [Installation & Usage](#installation--usage)
8. [Acknowledgments](#acknowledgments)

---

## Project Overview

The proliferation of AI-generated speech has introduced new challenges in audio forensics and content authentication. This project addresses these challenges by transforming audio signals into Mel spectrograms, which visually encode frequency and temporal information. These spectrograms are then analyzed by a Res2Net-based classifier, fine-tuned for the nuanced task of distinguishing real human speech from synthetic audio.

### Key Features

- **Mel Spectrogram Feature Extraction:** Converts audio into a time-frequency representation that captures subtle artifacts introduced by generative models.
- **Advanced Deep Learning Backbone:** Utilizes a Res2Net-enhanced ResNet50, enabling multi-scale feature extraction to improve detection accuracy.
- **End-to-End Pipeline:** From data preprocessing to model evaluation, the repository provides a complete workflow for deepfake audio detection.

---

## Dataset

- **Source:** ["In the Wild"](https://deepfake-total.com/in_the_wild) dataset
- **Contents:** Includes both bona-fide (real) and spoofed (fake) speech recordings, with comprehensive metadata (`meta.csv`) for supervised learning.
- **Preprocessing:**
  - Audio files are converted to Mel spectrograms using the `librosa` library.
  - Spectrograms are resized to 224x224 pixels for compatibility with the Res2Net model.

### Data Preparation Workflow

1. Parse metadata (`meta.csv`) to associate audio files with labels.
2. Convert each audio sample to a Mel spectrogram (128 Mel bins, 16 kHz sampling rate).
3. Resize spectrograms to 224x224 pixels.
4. Stratified split into training (80%), validation (10%), and test (10%) sets.

**Label Encoding:**
- `0`: Bona-fide (real)
- `1`: Spoofed (fake)

---

## Model & Approach

### Res2Net Classifier

- **Architecture:** Builds on ResNet50, augmented with Res2Net blocks for enhanced multi-scale feature extraction—crucial for capturing the subtle spectral differences between genuine and synthetic speech[1][4].
- **Input:** Mel spectrogram images (224x224).
- **Output:** Binary classification (real vs. fake).

---

## Implementation Process

### Preprocessing

- **Audio Conversion:** Audio files are transformed into Mel spectrograms using `librosa`, ensuring consistent representation across the dataset.
- **Dataset Class:** A custom `MelSpectrogramDataset` class manages loading and transformation of spectrogram images.
- **Dataset Split:** Data is partitioned into training, validation, and test sets for robust evaluation.

### Model Training

- **Base Model:** Modified ResNet50 with Res2Net blocks for binary classification.
- **Optimizer:** Adam, with a learning rate of 0.0001.
- **Loss Function:** Binary cross-entropy.
- **Training Regimen:** 10 epochs, with accuracy monitored on both training and validation sets.

#### Example Training Code

```python
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
```

---

## Training & Evaluation

### Model Training

To initiate training, execute:

```python
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
```

### Model Evaluation

Assess model performance on the test set:

```python
evaluate_model(model, test_loader)
```

### Model Persistence

Save the trained model for future inference:

```python
torch.save(model.state_dict(), "path_to_save_model.pth")
```

---

## Results & Performance

- **Test Accuracy:** Achieves approximately 99% accuracy on the held-out test set, demonstrating high reliability in distinguishing between bona-fide and deepfake audio samples.
- **Generalization:** The model shows strong performance on real-world, diverse audio data, indicating robust generalization beyond the training set[1][4][5].

---

## Installation & Usage

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare Data:** Place the dataset and metadata in the designated directories as described in the documentation.
2. **Train the Model:** Run the training script to begin model training.
3. **Evaluate:** Use the provided evaluation scripts to assess model performance.

---

## Acknowledgments

This project acknowledges the creators of the "In the Wild" dataset for their contribution to the field of audio deepfake detection research.

---

**Note:**  
This repository is intended for research and educational purposes only. For any commercial use, please contact the repository owner.
