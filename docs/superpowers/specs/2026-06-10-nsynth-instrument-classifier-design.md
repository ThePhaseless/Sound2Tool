# NSynth Instrument Classifier Design

## Overview

Build an instrument family classification system using the NSynth dataset. Given a 4-second audio note, predict which instrument family produced it (11 classes).

## Dataset

**NSynth** (Neural Audio Synthesis):
- 305,979 musical notes from 1,006 instruments
- 16kHz mono audio, 4 seconds each (3s held + 1s decay)
- 11 instrument families: bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal
- MIDI pitches 21-108, velocities 25/50/75/100/127

**Splits:**
- Train: 289,205 examples
- Valid: 12,678 examples
- Test: 4,096 examples

## Data Pipeline

```
NSynth dataset (305k notes, 16kHz mono, 4s each)
  ↓
Load JSON metadata + WAV files
  ↓
Extract log-mel spectrograms (128 mels, hop=512)
  ↓
Cache to .npy files (avoid re-computation)
  ↓
PyTorch Dataset/DataLoader
```

**Features:**
- Input: log-mel spectrogram (128 × 251 × 1)
- Labels: instrument_family (11 classes), instrument_str (1006 classes)

## Model Architecture

Three models with progressive complexity (all from course material):

### Model 1: Baseline (sklearn)
- Features: MFCC statistics (mean, std per coefficient → 40 features)
- Classifier: Random Forest (100 estimators)
- Purpose: quick baseline, no GPU needed

### Model 2: CNN (from Ćw. 5)
```
Input: (batch, 1, 128, 251)  # mel spectrogram
↓
Conv2d(1, 32, 3x3) → BatchNorm → ReLU → MaxPool(2)
Conv2d(32, 64, 3x3) → BatchNorm → ReLU → MaxPool(2)
Conv2d(64, 128, 3x3) → BatchNorm → ReLU → MaxPool(2)
Conv2d(128, 256, 3x3) → BatchNorm → ReLU → AdaptiveAvgPool(4,4)
↓
Flatten → Linear(256*4*4, 512) → ReLU → Dropout(0.5) → Linear(512, 11)
```

### Model 3: CNN + LSTM (from Ćw. 5 + Ćw. 8)
```
Input: (batch, 1, 128, 251)  # mel spectrogram
↓
CNN feature extractor (same as Model 2, but without final pooling)
  → Output: (batch, 256, 16, 31)
↓
Reshape to sequence: (batch, 16, 256*31)  # 16 timesteps, 7936 features each
↓
LSTM(input_size=7936, hidden_size=256, num_layers=2, batch_first=True)
↓
Take last hidden state: (batch, 256)
↓
Linear(256, 11)
```
CNN extracts spatial features from spectrogram frames, LSTM models temporal dependencies.

## Classification Task

**Single-stage family classification:**
- Predict which instrument family produced the note (11 classes)
- bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal

## Training & Evaluation

**Training setup:**
- Optimizer: Adam, lr=1e-3
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7 epochs
- Batch size: 64
- Max epochs: 30
- Device: GPU (CUDA)

**Evaluation metrics:**
- Overall accuracy
- Per-class precision/recall/F1
- Confusion matrix (11×11 for family classification)
- Per-class accuracy bar chart

**Inference:**
- `predict_instrument(audio_path)` → top-k predictions with confidences
- Interactive widget: input audio path → play audio + show spectrogram + predictions

## Implementation Structure

Single notebook: `notebooks/02_instrument_classifier.ipynb`

Sections:
1. Data download & loading
2. Exploratory analysis
3. Feature extraction (mel spectrograms)
4. Baseline model (sklearn)
5. CNN model (from Ćw. 5)
6. CNN + LSTM model (from Ćw. 5 + Ćw. 8)
7. Training loop
8. Evaluation
9. Inference with interactive UI

## Constraints

- **Course material only:** Only use concepts from labs (CNN from Ćw. 5, LSTM from Ćw. 8, PyTorch basics)
- **Compute:** GPU available (RTX 3060 or better)
- **Time:** Training should complete in reasonable time (< 2 hours per model)
- **Complexity:** School project - understandable but demonstrates good methodology
