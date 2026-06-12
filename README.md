# EEG-Based Epileptic Seizure Prediction
### EEGNet + Temporal Attention + Causal LSTM with Knowledge Distillation — CHB-MIT LOPO Benchmark

---

## Overview

This repository implements a patient-agnostic epileptic seizure prediction system evaluated on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). The pipeline combines spatial-temporal feature extraction (EEGNet), causal multi-head self-attention, and a unidirectional LSTM — all with strictly causal operations — making it compatible with real-time streaming deployment.

A mutual Knowledge Distillation (KD) fine-tuning stage adapts the general model to the target patient using a small support set, without requiring full retraining.

Evaluation follows **Leave-One-Patient-Out (LOPO)** cross-validation over 14 CHB-MIT patients, with temporal voting post-processing to produce clinically meaningful alarm events.

---

## Architecture

```
EEG Input (23 ch × 1280 samples)
        │
        ▼
┌──────────────────┐
│  EEGNet Block 1  │  Temporal conv  (1 × 64, F1=8 filters)
│  EEGNet Block 2  │  Depthwise conv (n_channels × 1, D=2)
│  EEGNet Block 3  │  Separable conv (1 × 16, F2=16 filters)
└──────────────────┘
        │  (B, T', 16)
        ▼
┌─────────────────────────────┐
│  Linear Projection (16→16) │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Causal Multi-Head Attention    │  4 heads, causal mask (no future leakage)
│  + LayerNorm + Residual         │
└─────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Causal LSTM (2 layers, h=128)  │  bidirectional=False
└──────────────────────────────────┘
        │  last timestep
        ▼
┌────────────────────────────┐
│  MLP Classifier (128→64→2) │
└────────────────────────────┘
        │
        ▼
  Preictal / Interictal
```

> **Note on naming:** The class is called `EEGNetBiLSTM` for legacy reasons. The LSTM is unidirectional (`bidirectional=False`), which is required for causal streaming.

---

## Key Design Choices

| Component | Choice | Rationale |
|---|---|---|
| Channels | 22 bipolar + 1 avg reference = **23** | Standardises across patients with varying montages |
| Window | **5 s** @ 256 Hz = 1280 samples | Standard preictal prediction unit |
| Preictal lookback | **60 minutes** before seizure onset | Conservative, reduces false negatives |
| Interictal ratio | **1:1** (balanced) | Avoids majority-class collapse |
| Preictal stride | **2 s** (dense overlap) | More preictal windows per seizure |
| Interictal stride | **5 s** (non-overlapping) | Reduces redundancy |
| Filter | Causal 101-tap FIR, **0.5–40 Hz** | Linear phase, no future samples |
| KD Temperature | **T = 4.0** | Softens logits for peer distillation |
| KD Support ratio | **15%** of target patient data | Small but balanced support set |
| Temporal voting | 10-min window, **40% threshold** | Converts window preds → alarms |
| Alarm merge gap | **30 s** | Suppresses duplicate alarm events |

---

## Training Pipeline

### Phase 1 — Cross-Patient Pretraining

Train on all patients **except** the held-out test patient. Produces a general seizure predictor.

```
Optimizer  : Adam, lr=1e-3
Scheduler  : ReduceLROnPlateau (patience=3, factor=0.5)
Loss       : CrossEntropyLoss (label_smoothing=0.1)
Batch      : 64 (effective 256 via gradient accumulation, step=4)
Grad clip  : 1.0
Early stop : patience=7 on val loss
Max epochs : 40
AMP        : torch.amp.autocast + GradScaler (CUDA)
```

### Phase 2 — Knowledge Distillation Adaptation

Peer-network mutual KD: the pretrained model (P) and a freshly initialised student (C) train simultaneously on a 15% support set from the target patient.

```
Loss_P = CE(logits_P, y) + KL(softmax(logits_P/T) ‖ softmax(logits_C/T)) × T²
Loss_C = CE(logits_C, y) + KL(softmax(logits_C/T) ‖ softmax(logits_P/T)) × T²

P and C alternate updates every batch (i % 2 == 0 → update P, else → update C)
lr_P = 1e-4  (fine-tune)
lr_C = 1e-3  (train from scratch guided by P)
KD epochs: 15
```

---

## LOPO Results (14 Patients)

| Patient | Accuracy | F1 | AUC-ROC | True Alarms | False Alarms | Sensitivity | FAR/h |
|---|---|---|---|---|---|---|---|
| chb01 | 0.9378 | 0.9410 | 0.9918 | 6 | 1 | 0.8571 | 0.153 |
| chb02 | 0.9284 | 0.9307 | 0.9199 | 2 | 1 | 0.6667 | 0.262 |
| chb03 | 0.9139 | 0.9058 | 0.9149 | 3 | 0 | 1.0000 | 0.000 |
| chb04 | 0.8809 | 0.8835 | 0.9442 | 3 | 0 | 1.0000 | 0.000 |
| chb05 | 0.8102 | 0.8397 | 0.8558 | 2 | 1 | 0.6667 | 0.197 |
| chb07 | 0.8613 | 0.8779 | 0.9019 | 1 | 2 | 0.3333 | 0.324 |
| chb09 | 0.9895 | 0.9880 | 0.8760 | 4 | 0 | 1.0000 | 0.000 |
| chb10 | 0.8303 | 0.7956 | 0.8162 | 8 | 0 | 1.0000 | 0.000 |
| chb14 | 0.6230 | 0.6951 | 0.5989 | 9 | 8 | 0.5294 | 0.797 |
| chb18 | 0.8857 | 0.8765 | 0.8899 | 5 | 1 | 0.8333 | 0.159 |
| chb20 | 0.8928 | 0.8944 | 0.9382 | 3 | 2 | 0.6000 | 0.290 |
| chb21 | 0.7604 | 0.8053 | 0.8244 | 3 | 3 | 0.5000 | 0.601 |
| chb22 | 0.6437 | 0.7360 | 0.6954 | 3 | 3 | 0.5000 | 0.655 |
| chb23 | 0.9456 | 0.9256 | 0.9691 | 2 | 1 | 0.6667 | 0.081 |
| **MEAN** | **0.8502** | **0.8639** | **0.8669** | — | — | **0.7252** | **0.251** |
| **STD** | **0.1052** | **0.0786** | **0.1034** | — | — | **0.2155** | **0.253** |

> chb14 is the known failure case (highly irregular montage, extra unnamed channels). chb23 and chb09 are the strongest performers.

---

## Repository Structure

```
.
├── chbmit_cnnlstm.py        # Preprocessing: EDF → .pt windows
├── train_lopo.py            # Main: LOPO loop, Phase 1 + Phase 2
│
├── /kaggle/temp/data/       # Preprocessed .pt windows (auto-created)
│   ├── preictal_chb01_*.pt
│   └── interictal_chb01_*.pt
│
└── /kaggle/working/         # Outputs
    ├── phase1_{patient}.pt  # Phase 1 checkpoints
    ├── phase2_{patient}.pt  # Phase 2 (KD-adapted) checkpoints
    ├── fold_{patient}_curves.png
    └── lopo_results_all_folds.pt
```

---

## Setup

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mne scipy scikit-learn matplotlib numpy
```

### Dataset

Download the CHB-MIT Scalp EEG Database from PhysioNet:

```
https://physionet.org/content/chbmit/1.0.0/
```

Set `ROOT_DIR` in the preprocessing script to your local path.

---

## Usage

### Step 1 — Preprocess EDF files into windowed `.pt` tensors

```python
# Edit ROOT_DIR and SAVE_DIR at the top of the preprocessing block
python chbmit_cnnlstm.py
```

Each saved `.pt` file contains:
```python
{
    "data"      : torch.Tensor,   # shape (23, 1280), float32
    "label"     : int,            # 1 = preictal, 0 = interictal
    "patient_id": str,            # e.g. "chb01"
}
```

Expected output:
```
  chb01:  5547 Preictal |  5547 Interictal
  chb02:  3230 Preictal |  3230 Interictal
  ...
  TOTAL : 85194 Preictal | 85194 Interictal
  GRAND : 170388 windows
  Shape per window: (23, 1280) — float32
```

### Step 2 — Run LOPO cross-validation

```python
python train_lopo.py
```

This iterates over all 14 patients, running Phase 1 + Phase 2 + evaluation for each fold. Checkpoints and training curves are saved automatically.

### Step 3 — Load a saved model for inference

```python
import torch
import torch.nn.functional as F
from train_lopo import EEGNetBiLSTM

ckpt = torch.load("phase2_chb01.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["cfg"]

model = EEGNetBiLSTM(
    n_channels   = cfg["n_channels"],
    n_timepoints = cfg["n_timepoints"],
    F1           = cfg["F1"],
    D            = cfg["D"],
    dropout      = 0.0,          # disable at inference
    lstm_hidden  = cfg["lstm_hidden"],
    lstm_layers  = cfg["lstm_layers"],
    lstm_dropout = 0.0,
    attn_heads   = cfg.get("attn_heads", 4),
    attn_dim     = cfg.get("attn_dim", 16),
    attn_dropout = cfg.get("attn_dropout", 0.1),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# window: torch.Tensor of shape (1, 23, 1280)
with torch.no_grad():
    prob = F.softmax(model(window), dim=1)[0, 1].item()
    print(f"P(preictal) = {prob:.4f}")
```

---

## Temporal Voting

Raw window-level probabilities are post-processed with a sliding vote window to reduce false alarms:

```
vote_window  = 120 windows  (~10 minutes at 2s stride)
vote_thresh  = 0.40         (40% of windows in the block must predict preictal)
vote_step    = 12           (step between vote blocks)
alarm_threshold = 0.50      (window-level probability cutoff before voting)
merge_gap    = 30 s         (merge adjacent alarm events within 30s)
```

An alarm event is counted as a **true alarm** if any window in the event overlaps a ground-truth preictal period. False alarm rate (FAR/h) is computed over interictal hours only.

---

## Preprocessing Details

| Parameter | Value |
|---|---|
| Sampling frequency | 256 Hz |
| Filter | Causal FIR (101 taps), 0.5–40 Hz, `scipy.signal.firwin` |
| Channels | 22 canonical bipolar + 1 average reference = **23** |
| Missing channels | Zero-filled |
| Normalization | Per-window instance normalization (mean=0, std=1 per channel) |
| Preictal window | Up to 60 min before seizure onset, 2 s stride |
| Interictal window | Non-overlapping (5 s stride) from seizure-free files |
| Class balance | 1:1 interictal-to-preictal (random file selection) |

---

## Limitations & Known Issues

- **chb14**: Contains extra unnamed channels (`-`, `.`) and non-standard scaling factors. The model struggles on this patient (Acc=0.62, FAR/h=0.80). Consider excluding or applying patient-specific preprocessing.
- **chb07**: Low event sensitivity (0.33) — only 1 true alarm detected out of 3 events.
- **KD phase instability**: For some patients, KD loss plateaus early. Increasing `kd_epochs` or adjusting `T_temp` may help.
- **No real-time streaming demo**: The real-time simulation code (commented out) requires a working display and an installed `mne` + matplotlib environment with `plt.ion()` support. It is not runnable on headless Kaggle kernels.

---

## Citation

If you use this work, please cite the following:

```bibtex
@article{wu2022knowledge,
  title     = {A Knowledge Distillation Framework for Enhancing 
               EEG-Based Emotion Recognition},
  author    = {Wu, Yue and others},
  journal   = {Journal of Neural Engineering},
  year      = {2022},
}

@misc{chbmit,
  title  = {CHB-MIT Scalp EEG Database},
  author = {Shoeb, A. and Guttag, J.},
  year   = {2010},
  url    = {https://physionet.org/content/chbmit/1.0.0/}
}

@article{lawhern2018eegnet,
  title   = {EEGNet: a compact convolutional neural network for EEG-based
             brain–computer interfaces},
  author  = {Lawhern, Vernon J and others},
  journal = {Journal of Neural Engineering},
  year    = {2018},
}
```

---

## Authors

- **Rainzler** (lead implementation)
- **Adila Taib** (co-author, thesis)

Master's thesis project — EEG-Based Epileptic Seizure Prediction using Deep Learning, 2024–2025.
