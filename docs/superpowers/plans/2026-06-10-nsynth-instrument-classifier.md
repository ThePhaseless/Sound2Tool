# NSynth Instrument Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook that classifies instrument families from audio using NSynth dataset, with baseline (sklearn), CNN, and CNN+LSTM models.

**Architecture:** Single notebook with progressive model complexity. Data pipeline: NSynth WAV → mel spectrograms → PyTorch Dataset. Three models compared: sklearn RF on MFCCs, CNN on spectrograms, CNN+LSTM hybrid.

**Tech Stack:** Python 3.12, PyTorch, librosa, scikit-learn, matplotlib, numpy, pandas

**Course Material:** CNN (Ćw. 5), LSTM (Ćw. 8), PyTorch basics

---

## File Structure

| File | Responsibility |
|------|----------------|
| `notebooks/02_instrument_classifier.ipynb` | Main notebook with all code |
| `data/` | NSynth dataset (downloaded) |
| `data/cache/` | Cached mel spectrograms (.npy) |

---

### Task 1: Notebook Header & Imports

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown header cell**

```markdown
# NSynth Instrument Classifier

Classify instrument families from audio using the [NSynth dataset](https://magenta.withgoogle.com/datasets/nsynth).

**Models compared:**
1. Baseline: Random Forest on MFCC statistics
2. CNN: Convolutional neural network on mel spectrograms (from Ćw. 5)
3. CNN + LSTM: CNN features + LSTM temporal modeling (from Ćw. 5 + Ćw. 8)

**11 Instrument Families:** bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal
```

- [ ] **Step 2: Create imports cell**

```python
import json
import tarfile
import urllib.request
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "import torch; import librosa; import sklearn; print('OK')"`
Expected: `OK`

---

### Task 2: Constants

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create constants cell**

```python
SAMPLE_RATE = 16000
NOTE_DURATION = 4.0
N_MELS = 128
HOP_LENGTH = 512
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 7

INSTRUMENT_FAMILIES = [
    "bass", "brass", "flute", "guitar", "keyboard",
    "mallet", "organ", "reed", "string", "synth_lead", "vocal",
]

DATA_DIR = Path("../data")
TRAIN_DIR = DATA_DIR / "nsynth-train.jsonwav"
VALID_DIR = DATA_DIR / "nsynth-valid.jsonwav"
TEST_DIR = DATA_DIR / "nsynth-test.jsonwav"
CACHE_DIR = DATA_DIR / "cache"

BASE_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/"
```

---

### Task 3: Data Download Helper

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Data Download

Download and extract NSynth JSON+WAV splits. Each split is ~4GB.
```

- [ ] **Step 2: Create download function cell**

```python
def download_and_extract(split: str, data_dir: Path):
    """Download and extract an NSynth split (train/valid/test)."""
    filename = f"nsynth-{split}.jsonwav.tar.gz"
    tar_path = data_dir / filename
    extract_dir = data_dir / f"nsynth-{split}.jsonwav"

    if extract_dir.exists():
        print(f"{split} already extracted, skipping.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + filename
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, tar_path)
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    tar_path.unlink()
    print(f"Done: {extract_dir}")


def download_all():
    for split in ["train", "valid", "test"]:
        download_and_extract(split, DATA_DIR)
```

- [ ] **Step 3: Create download execution cell (commented by default)**

```python
# Uncomment to download (~12GB total):
# download_all()
```

---

### Task 4: Data Loading Functions

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Data Loading
```

- [ ] **Step 2: Create load_nsynth_json function**

```python
def load_nsynth_json(split_dir: Path) -> pd.DataFrame:
    """Load NSynth metadata from examples.json into a DataFrame."""
    json_path = split_dir / "examples.json"
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for note_str, meta in data.items():
        rows.append({
            "note_str": note_str,
            "instrument_family": meta["instrument_family"],
            "instrument_family_str": meta["instrument_family_str"],
            "instrument_source": meta["instrument_source"],
            "instrument_source_str": meta["instrument_source_str"],
            "instrument_str": meta["instrument_str"],
            "pitch": meta["pitch"],
            "velocity": meta["velocity"],
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 3: Create load_audio function**

```python
def load_audio(note_str: str, split_dir: Path) -> np.ndarray:
    """Load a single WAV file and return the waveform."""
    wav_path = split_dir / "audio" / f"{note_str}.wav"
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    return audio
```

- [ ] **Step 4: Create data loading execution cell**

```python
df_train = load_nsynth_json(TRAIN_DIR)
df_valid = load_nsynth_json(VALID_DIR)
df_test = load_nsynth_json(TEST_DIR)
print(f"Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")
```

---

### Task 5: Exploratory Analysis

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Exploratory Analysis
```

- [ ] **Step 2: Create family distribution plot**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (df, title) in zip(
    axes, [(df_train, "Train"), (df_valid, "Valid"), (df_test, "Test")], strict=True
):
    counts = df["instrument_family_str"].value_counts().reindex(INSTRUMENT_FAMILIES)
    counts.plot.bar(ax=ax, color="steelblue")
    ax.set_title(f"{title} ({len(df):,} notes)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Create spectrogram gallery**

```python
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, family in enumerate(INSTRUMENT_FAMILIES[:10]):
    sample = df_train[df_train["instrument_family_str"] == family].iloc[0]
    audio = load_audio(sample["note_str"], TRAIN_DIR)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", ax=axes[i], cmap="magma")
    axes[i].set_title(f"{family} (pitch={sample['pitch']})")

axes[-1].axis("off")
plt.tight_layout()
plt.show()
```

---

### Task 6: Feature Extraction

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Feature Extraction

Extract log-mel spectrograms and cache them to disk for fast loading during training.
```

- [ ] **Step 2: Create extract_mel_spectrogram function**

```python
def extract_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE,
                            n_mels: int = N_MELS, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """Extract log-mel spectrogram from audio waveform."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max)
```

- [ ] **Step 3: Create cache_mel_spectrograms function**

```python
def cache_mel_spectrograms(df: pd.DataFrame, split_dir: Path, cache_name: str) -> np.ndarray:
    """Precompute and cache mel spectrograms as .npy files."""
    cache_path = CACHE_DIR / cache_name
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached spectrograms from {cache_path}")
        return np.load(cache_path)

    specs = []
    for i, row in df.iterrows():
        if i % 5000 == 0:
            print(f"  Processing {i}/{len(df)}...")
        audio = load_audio(row["note_str"], split_dir)
        mel = extract_mel_spectrogram(audio)
        specs.append(mel)

    specs = np.array(specs, dtype=np.float32)
    np.save(cache_path, specs)
    print(f"Cached {specs.shape} to {cache_path}")
    return specs
```

- [ ] **Step 4: Create extraction execution cell**

```python
print("Extracting train spectrograms...")
X_train = cache_mel_spectrograms(df_train, TRAIN_DIR, "mel_train.npy")
print("Extracting valid spectrograms...")
X_valid = cache_mel_spectrograms(df_valid, VALID_DIR, "mel_valid.npy")
print("Extracting test spectrograms...")
X_test = cache_mel_spectrograms(df_test, TEST_DIR, "mel_test.npy")

le = LabelEncoder()
y_train = le.fit_transform(df_train["instrument_family_str"])
y_valid = le.transform(df_valid["instrument_family_str"])
y_test = le.transform(df_test["instrument_family_str"])

print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print(f"Classes: {le.classes_}")
```

---

### Task 7: PyTorch Dataset & DataLoader

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## PyTorch Dataset & DataLoader
```

- [ ] **Step 2: Create NSynthDataset class**

```python
class NSynthDataset(Dataset):
    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray):
        self.spectrograms = torch.from_numpy(spectrograms).unsqueeze(1)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]
```

- [ ] **Step 3: Create DataLoader instances**

```python
train_dataset = NSynthDataset(X_train, y_train)
valid_dataset = NSynthDataset(X_valid, y_valid)
test_dataset = NSynthDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
)

sample_x, sample_y = next(iter(train_loader))
print(f"Batch shape: {sample_x.shape}, Labels shape: {sample_y.shape}")
```

---

### Task 8: Baseline Model (sklearn)

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Model 1: Baseline (Random Forest on MFCCs)

Quick baseline using handcrafted features. No GPU needed.
```

- [ ] **Step 2: Create MFCC extraction function**

```python
def extract_mfcc_stats(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 20) -> np.ndarray:
    """Extract MFCC statistics (mean and std per coefficient)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
```

- [ ] **Step 3: Create baseline training cell**

```python
from sklearn.ensemble import RandomForestClassifier

print("Extracting MFCC features for baseline (this may take a while)...")

# Use subset for speed
MAX_BASELINE_SAMPLES = 10000
train_subset = df_train.iloc[:MAX_BASELINE_SAMPLES]
X_train_mfcc = np.array([
    extract_mfcc_stats(load_audio(row["note_str"], TRAIN_DIR))
    for _, row in train_subset.iterrows()
])
y_train_subset = y_train[:MAX_BASELINE_SAMPLES]

print(f"Training RF on {X_train_mfcc.shape[0]} samples...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_mfcc, y_train_subset)

# Quick validation
val_subset = df_valid.iloc[:2000]
X_val_mfcc = np.array([
    extract_mfcc_stats(load_audio(row["note_str"], VALID_DIR))
    for _, row in val_subset.iterrows()
])
y_val_subset = y_valid[:2000]

rf_accuracy = rf_model.score(X_val_mfcc, y_val_subset)
print(f"Baseline RF accuracy: {rf_accuracy:.2%}")
```

---

### Task 9: CNN Model (from Ćw. 5)

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Model 2: CNN (from Ćw. 5)

Convolutional neural network on mel spectrograms.
```

- [ ] **Step 2: Create InstrumentCNN class**

```python
class InstrumentCNN(nn.Module):
    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


cnn_model = InstrumentCNN(num_classes=len(INSTRUMENT_FAMILIES)).to(device)
total_params = sum(p.numel() for p in cnn_model.parameters())
print(f"CNN parameters: {total_params:,}")
print(cnn_model)
```

---

### Task 10: CNN + LSTM Model (from Ćw. 5 + Ćw. 8)

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Model 3: CNN + LSTM (from Ćw. 5 + Ćw. 8)

CNN extracts spatial features from spectrogram frames, LSTM models temporal dependencies.
```

- [ ] **Step 2: Create CNNLSTM class**

```python
class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256 * 16 * 31,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 16, -1)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        return self.classifier(x)


cnnlstm_model = CNNLSTM(num_classes=len(INSTRUMENT_FAMILIES)).to(device)
total_params = sum(p.numel() for p in cnnlstm_model.parameters())
print(f"CNN+LSTM parameters: {total_params:,}")
print(cnnlstm_model)
```

---

### Task 11: Training Function

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Training
```

- [ ] **Step 2: Create train_model function**

```python
def train_model(model, train_loader, valid_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """Train a model with validation and early stopping."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}"
              f"{' *' if val_acc == best_val_acc else ''}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    return model, history
```

---

### Task 12: Train CNN Model

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create training execution cell**

```python
print("Training CNN...")
cnn_model = InstrumentCNN(num_classes=len(INSTRUMENT_FAMILIES))
cnn_model, cnn_history = train_model(cnn_model, train_loader, valid_loader)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cnn_history["train_loss"], label="Train")
axes[0].plot(cnn_history["val_loss"], label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("CNN Loss")
axes[0].legend()

axes[1].plot(cnn_history["train_acc"], label="Train")
axes[1].plot(cnn_history["val_acc"], label="Validation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("CNN Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

### Task 13: Train CNN+LSTM Model

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create training execution cell**

```python
print("Training CNN+LSTM...")
cnnlstm_model = CNNLSTM(num_classes=len(INSTRUMENT_FAMILIES))
cnnlstm_model, cnnlstm_history = train_model(cnnlstm_model, train_loader, valid_loader)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cnnlstm_history["train_loss"], label="Train")
axes[0].plot(cnnlstm_history["val_loss"], label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("CNN+LSTM Loss")
axes[0].legend()

axes[1].plot(cnnlstm_history["train_acc"], label="Train")
axes[1].plot(cnnlstm_history["val_acc"], label="Validation")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("CNN+LSTM Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

### Task 14: Evaluation on Test Set

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Evaluation on Test Set
```

- [ ] **Step 2: Create evaluate_model function**

```python
@torch.no_grad()
def evaluate_model(model, test_loader):
    """Evaluate model on test set and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)
```

- [ ] **Step 3: Create evaluation execution cell**

```python
print("Evaluating CNN...")
cnn_preds, cnn_labels = evaluate_model(cnn_model, test_loader)
print(classification_report(cnn_labels, cnn_preds, target_names=le.classes_))

print("\n" + "="*60 + "\n")

print("Evaluating CNN+LSTM...")
cnnlstm_preds, cnnlstm_labels = evaluate_model(cnnlstm_model, test_loader)
print(classification_report(cnnlstm_labels, cnnlstm_preds, target_names=le.classes_))
```

---

### Task 15: Confusion Matrices

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create confusion matrix visualization**

```python
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

cm_cnn = confusion_matrix(cnn_labels, cnn_preds)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=le.classes_)
disp_cnn.plot(ax=axes[0], cmap="Blues", xticks_rotation=45)
axes[0].set_title("CNN Confusion Matrix")

cm_cnnlstm = confusion_matrix(cnnlstm_labels, cnnlstm_preds)
disp_cnnlstm = ConfusionMatrixDisplay(confusion_matrix=cm_cnnlstm, display_labels=le.classes_)
disp_cnnlstm.plot(ax=axes[1], cmap="Greens", xticks_rotation=45)
axes[1].set_title("CNN+LSTM Confusion Matrix")

plt.tight_layout()
plt.show()
```

---

### Task 16: Model Comparison

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create comparison visualization**

```python
results = {
    "Baseline (RF)": rf_accuracy if "rf_accuracy" in dir() else 0.0,
    "CNN": (cnn_preds == cnn_labels).mean(),
    "CNN+LSTM": (cnnlstm_preds == cnnlstm_labels).mean(),
}

fig, ax = plt.subplots(figsize=(10, 5))
pd.Series(results).plot.bar(ax=ax, color=["lightblue", "steelblue", "darkblue"])
ax.set_ylabel("Test Accuracy")
ax.set_title("Model Comparison")
ax.set_ylim(0, 1)
for i, v in enumerate(results.values()):
    ax.text(i, v + 0.02, f"{v:.2%}", ha="center")
plt.tight_layout()
plt.show()
```

---

### Task 17: Inference Function

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create markdown cell**

```markdown
## Inference: Classify a Given Sound
```

- [ ] **Step 2: Create predict_instrument function**

```python
@torch.no_grad()
def predict_instrument(audio_path: str | Path, model=None, top_k: int = 5) -> list[dict]:
    """Predict instrument family for a given audio file."""
    if model is None:
        model = cnnlstm_model
    model.eval()
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    if len(audio) > SAMPLE_RATE * NOTE_DURATION:
        audio = audio[:int(SAMPLE_RATE * NOTE_DURATION)]
    elif len(audio) < SAMPLE_RATE * NOTE_DURATION:
        audio = np.pad(audio, (0, int(SAMPLE_RATE * NOTE_DURATION) - len(audio)))

    mel = extract_mel_spectrogram(audio)
    mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)

    logits = model(mel_tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for idx in top_indices:
        predictions.append({
            "family": le.classes_[idx],
            "confidence": float(probs[idx]),
        })
    return predictions
```

- [ ] **Step 3: Create visualize_prediction function**

```python
def visualize_prediction(audio_path: str | Path, top_k: int = 5):
    """Predict and visualize instrument classification for an audio file."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    predictions = predict_instrument(audio_path, top_k=top_k)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel", ax=axes[0], cmap="magma")
    axes[0].set_title(f"Input: {Path(audio_path).name}")

    families = [p["family"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]
    colors = ["steelblue" if i > 0 else "coral" for i in range(len(families))]
    axes[1].barh(families[::-1], confidences[::-1], color=colors[::-1])
    axes[1].set_xlabel("Confidence")
    top = predictions[0]
    axes[1].set_title(f"Prediction: {top['family']} ({top['confidence']:.1%})")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

    print("\nTop predictions:")
    for p in predictions:
        print(f"  {p['family']:12s} {p['confidence']:.1%}")

    return predictions
```

---

### Task 18: Interactive Inference Widget

**Files:**
- Modify: `notebooks/02_instrument_classifier.ipynb`

- [ ] **Step 1: Create interactive widget**

```python
import ipywidgets as widgets
from IPython.display import Audio, display

file_input = widgets.Text(
    value=str(TEST_DIR / "audio" / f"{df_test.iloc[0]['note_str']}.wav"),
    description="Audio path:",
    layout=widgets.Layout(width="80%"),
)
classify_btn = widgets.Button(description="Classify", button_style="primary")
output = widgets.Output()


def on_classify(btn):
    output.clear_output()
    with output:
        path = Path(file_input.value)
        if not path.exists():
            print(f"File not found: {path}")
            return
        display(Audio(filename=str(path)))
        visualize_prediction(path)


classify_btn.on_click(on_classify)
display(widgets.VBox([file_input, classify_btn, output]))
```

---

## Summary

This plan creates a single notebook with:
- Data download and loading
- Exploratory analysis
- Three models: Baseline (RF), CNN, CNN+LSTM
- Training with early stopping
- Evaluation with confusion matrices
- Interactive inference widget

All concepts are from course material (Ćw. 5 for CNN, Ćw. 8 for LSTM).
