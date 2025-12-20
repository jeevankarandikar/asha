# CLAUDE.md

Complete project guide for Claude Code when working with Project Asha.

**Last Updated**: December 16, 2025

---

## CRITICAL RULES FOR CLAUDE CODE

1. **DO NOT create new .md files** unless explicitly requested by the user
2. **All summaries and explanations go directly in chat** - do NOT write them to files
3. **Only create .md files when the user explicitly says "create a .md file" or "write to file"**
4. All documentation updates go in this file (CLAUDE.md) or existing READMEs
5. When asked for summaries, explanations, or advice - provide them in chat responses, NOT in new files

### Code Style Guidelines

- **Comments**: Write clear, concise comments explaining WHY not WHAT
- **No emojis**: Unless explicitly requested by user
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Type hints**: Use Python type hints for function signatures
- **Naming**: Descriptive variable names (e.g., `emg_window` not `x`)

---

## Project Overview

**Project Asha** is a real-time EMG-to-hand-pose system for prosthetic control and gesture recognition.

**Goal**: Predict 3D hand poses from surface EMG muscle signals, enabling camera-free hand tracking.

**Current Status**: ⭐ **Camera-free hand tracking achieved!** Joints model: **14.92mm MPJPE** on 20 min training data - competitive with vision-based IK (9.71mm) and better than all parametric approaches.

### What Makes This Project Unique

1. **Low-cost ground truth**: Uses webcam + IK (9.71mm accuracy) instead of $100K+ mocap systems
2. **Direct EMG training**: Simple Conv1D+LSTM architecture beats complex transfer learning
3. **Real-time capable**: Full pipeline runs at 30fps with <10ms inference latency
4. **Camera-free**: Final system requires only EMG armband (no visual input)

---

## System Architecture Overview

```
Webcam (Training Only)
    ↓
MediaPipe (21 keypoints)
    ↓
MANO IK Optimization (45 joint angles θ)
    ↓ [Creates pseudo-labels for EMG training]
EMG Signals (8 channels @ 500Hz)
    ↓
SimpleEMGModel (Conv1D + LSTM)
    ↓
MANO θ → 3D Hand Mesh
    ↓
Real-time Visualization (30fps)
```

### Key Insight

We bootstrap from expensive ground truth → cheap ground truth → no visual input:
1. **FreiHAND**: Motion capture data (130K images)
2. **IK Labels**: Webcam + MANO IK (9.71mm error)
3. **EMG Model**: Direct prediction from muscle signals (12-15mm MPJPE)

---

## Version Roadmap

### System Versions (Major Phases)

| Version | Description | Status | Key Metrics |
|---------|-------------|--------|-------------|
| **v0** | MediaPipe baseline | ✅ Complete | 60fps, 21 keypoints, ~16-17mm error |
| **v1** | MANO IK + LBS | ✅ Complete | **9.71mm MPJPE**, 25fps, 99.7% convergence |
| **v2** | EMG integration + recording | ✅ Complete | 8ch @ 500Hz, HDF5 format, dual-thread sync |
| **v2.5** | Image → θ training (paused) | ⏸️ Paused | ResNet/Transformer plateaued at 47mm |
| **v3 (Transfer)** | Transfer learning attempts | ❌ Abandoned | emg2pose adapter failed (hardware mismatch) |
| **v4** | EMG → MANO θ (SimpleEMGModel) | ✅ Complete | 34.92mm MPJPE (20 min data) |
| **v5** | EMG → Joints (EMGToJointsModel) | ✅ Complete | **14.92mm MPJPE** (20 min data) ⭐ BEST |
| **v6** | Multi-sensor fusion (future) | 📋 Planned | IMU + PPG integration |

---

## Training Results Summary

### Image → θ Training (v2.5 - Paused)

**Goal**: Train neural network to predict MANO θ from RGB images

| Model | Architecture | Dataset | MPJPE | Status | Issue |
|-------|--------------|---------|-------|--------|-------|
| resnet_v1 | ResNet-50 + MLP | FreiHAND | 47mm | ❌ Failed | Model collapse (SimpleMANOLoss) |
| resnet_v2 | ResNet-50 + MLP | FreiHAND | 47.44mm | ❌ Plateaued | Early plateau (epoch 22) |
| transformer_v1 | ResNet + Transformer | FreiHAND | 47.41mm | ❌ Plateaued | Loss imbalance |
| transformer_v2 | ResNet + Transformer | FreiHAND | 47.40mm | ❌ No improvement | lambda_joint=25 too high |
| transformer_v3 | Enhanced losses + priors | FreiHAND | 47.46mm | ❌ Model collapse | Predicts identical poses, θ variance=0.0002 |
| transformer_v3.1 | Rebalanced losses | FreiHAND | - | 📋 Planned | lambda_joint=1.0, lambda_param=1.0 |

**Key Findings**:
- All models plateaued at ~47mm MPJPE (not viable for pseudo-labeling, need <15mm)
- Loss imbalance (lambda_joint too high) causes parameter suppression
- Model collapse: Predicts single average pose to minimize dominant loss term
- Diversity metrics (θ variance, entropy) critical for detecting collapse early

**Decision**: Pivoted to direct EMG training with IK labels (9.71mm quality, better than image models)

### EMG Training (v2/v3 - SUCCESS!)

**Two Approaches**:

#### 1. EMG → MANO θ (SimpleEMGModel)

**Goal**: Predict MANO pose parameters (45 values) directly from EMG signals

| Model | Data | MPJPE | Val Loss | Epochs | Status |
|-------|------|-------|----------|--------|--------|
| **v3** (SimpleEMGModel) | 3 min | **12-15mm** (est) | 0.1379 | 48 | ✅ Best |
| **v2** (SimpleEMGModel) | 20 min | 34.92mm | 0.078480 | 100 | ✅ Complete |

**Output**: 45 MANO θ parameters → MANO forward pass → 3D mesh

#### 2. EMG → Joints (EMGToJointsModel) ⭐ BEST RESULTS

**Goal**: Predict MediaPipe joint positions (63 values) directly from EMG signals

| Model | Data | MPJPE | Val Loss | Epochs | Status |
|-------|------|-------|----------|--------|--------|
| **joints_v1** | 20 min | 11.95mm | 11.95 | 200 | ✅ (Dec 14) |
| **joints_v2** | 20 min | **14.92mm** | 14.92 | 100 | ✅ **CURRENT** (Dec 16) |

**Output**: 63 values (21 joints × 3 coords) → direct joint positions

**Training Details (v2 - Latest)**:
- Started: 19.84mm → Final: **14.92mm** (25% improvement!)
- Learning rate schedule: 1e-3 (epochs 1-77) → 5e-4 (epoch 78-100)
- Best epoch: 99 (saved at 14.92mm)
- Hardware: A100 GPU, 100 epochs (~1.5 hours)
- Data: 5 sessions, 23,961 training windows (data/v4/emg_recordings/)
- **Critical fix**: Checkpoint now includes global EMG statistics (emg_mean, emg_std) for proper confidence calculation

**Key Findings**:
- **Joints model (14.92mm) is BEST**: Direct joint prediction beats parametric θ approach (34.92mm)
- v5 joints (14.92mm) competitive with IK ground truth (9.71mm) - only 5mm gap!
- Global EMG statistics critical for confidence calculation in real-time inference
- Direct training with IK labels (9.71mm) beats transfer learning completely
- Simple architecture (Conv1D+LSTM) sufficient, no need for complex TDS blocks
- **Two approaches compared**:
  - Joints (11.95mm) ⭐: Simpler supervision, direct optimization
  - θ params (12-34mm): Parametric constraints, needs MANO forward pass

### Transfer Learning (Abandoned)

**Attempt**: Adapt pre-trained emg2pose model (16ch@2kHz) to MindRove (8ch@500Hz)

**Result**: ❌ Failed - Hardware mismatch too fundamental to bridge

**Lessons**:
1. Can't create missing spatial info (8→16 channels)
2. Can't create missing temporal info (500Hz→2kHz)
3. Direct training with IK labels beats 80M frame pre-training
4. Simplicity wins: 1 day implementation > 2 weeks overengineering

See `docs/TRANSFER_LEARNING_POSTMORTEM.md` for full analysis.

---

## EMG Model Architectures

### Model 1: SimpleEMGModel (EMG → MANO θ)

**Overview**:

```python
Input:  EMG window [batch, 8 channels, 50 samples] = 100ms @ 500Hz
Output: MANO θ [batch, 45 params] = 15 joints × 3 DoF
```

**Purpose**: Predicts MANO pose parameters for parametric hand modeling

### Layers

```
Conv1D Block 1:
  - Conv1d(8→64, kernel=5, padding=2)
  - BatchNorm1d(64)
  - ReLU activation
  - MaxPool1d(2) → downsample 50→25 samples

Conv1D Block 2:
  - Conv1d(64→128, kernel=5, padding=2)
  - BatchNorm1d(128)
  - ReLU activation
  - MaxPool1d(2) → downsample 25→12 samples

LSTM Block:
  - LSTM(128→256, num_layers=2, dropout=0.3)
  - Batch-first mode
  - Use last timestep output (most recent context)

Output Head:
  - Linear(256→128)
  - ReLU + Dropout(0.3)
  - Linear(128→45) → MANO θ
```

### Total Parameters

~1,004,397 parameters - lightweight, trains fast (~1.5 hours for 100 epochs on A100)

### Why This Architecture Works

1. **Conv1D**: Captures spatial patterns (which muscles fire together)
2. **LSTM**: Captures temporal dynamics (how muscles transition)
3. **No TDS blocks**: Simpler than emg2pose, sufficient for our hardware
4. **Direct θ prediction**: No velocity integration needed for initial model

### Training Details

- **Window size**: 50 samples (100ms @ 500Hz)
- **Batch size**: 32-64
- **Loss**: MSE(θ_pred, θ_gt) - simple L2 loss on joint angles
- **Optimizer**: Adam, lr=1e-3 with ReduceLROnPlateau
- **Data augmentation**: Time warping, noise injection (future)
- **Quality filtering**: IK error <15mm, MediaPipe confidence >0.5

### Model 2: EMGToJointsModel (EMG → Joints)

**Overview**:

```python
Input:  EMG window [batch, 8 channels, 50 samples] = 100ms @ 500Hz
Output: Joint positions [batch, 63] = 21 joints × 3 coords
```

**Purpose**: Directly predicts MediaPipe joint coordinates without MANO parametric model

**Architecture**: Same as SimpleEMGModel (Conv1D + LSTM) but with different output:
- FC output: Linear(128→63) instead of Linear(128→45)
- No MANO forward pass needed - direct 3D joint positions

**Advantages**:
- Simpler pipeline (no MANO layer required)
- Direct supervision from MediaPipe labels
- May be more interpretable for downstream tasks

**Disadvantages**:
- No parametric smoothness constraints
- Harder to enforce anatomical validity
- May produce unrealistic joint configurations

**Status**: Trained and working in `asha/v5/train_colab.py` (11.95mm MPJPE, needs retraining with fixed normalization)

---

## Quick Start

### Setup

```bash
# 1. Create runtime environment
python3 -m venv asha_env
source asha_env/bin/activate

# 2. Install package in editable mode
pip install -e .

# 3. Download MANO models from https://mano.is.tue.mpg.de
# Place in models/mano/ as MANO_RIGHT.pkl, MANO_LEFT.pkl

# 4. Convert MANO to numpy format (one-time setup)
python -m asha.shared.convert_mano_to_numpy
```

### Running Programs

```bash
source asha_env/bin/activate

# Data recording (EMG + video with IK pseudo-labels)
python -m asha.shared.record

# Real-time EMG inference v5 (BEST: 11.95mm, camera-free!)
python -m asha.v5.inference --model models/v5/emg_joints_best.pth

# Real-time EMG inference v4 (34.92mm)
python -m asha.v4.inference --model models/v4/emg_model_v2_best.pth

# MANO IK visualization only
python -m asha.v1.mano

# MediaPipe baseline demo
python -m asha.v0.mediapipe_demo
```

### Training EMG Model (Google Colab)

```bash
# 1. Upload to Google Drive:
#    - MyDrive/Colab Notebooks/asha/data/*.h5 (recording files)
#    - MyDrive/Colab Notebooks/asha/MANO_RIGHT_numpy.pkl

# 2. Upload asha/v4/train_colab.py to Colab

# 3. Run in Colab cell:
!python train_colab.py

# Expected: Val loss <0.15, MPJPE ~12-15mm after 50-100 epochs
```

---

## Experimental Validation (v1 IK System)

**Dataset**: 5,330 frames (3 min video @ 29.3fps)

### Experiment 1: Loss Ablation

| Configuration | Mean Error | Change | Key Finding |
|---------------|-----------|--------|-------------|
| Baseline (all losses) | 9.71mm | - | Balanced performance |
| No bone direction | 12.48mm | +28% | **CRITICAL LOSS** |
| No temporal | 9.66mm | -0.5% | Minimal mean impact |
| No regularization | 8.49mm | -13% | **Better without!** |
| Position only | 7.04mm | -27% | Best numerically, no guarantees |

**Recommended config**: Position + bone direction + temporal smoothness, NO regularization

### Experiment 2: Alignment Method

| Method | Mean Error | Scale Factor |
|--------|-----------|--------------|
| **Umeyama scaled** | **9.71mm** | 0.981 ± 0.121 |
| Umeyama rigid | 10.31mm | 1.000 (fixed) |
| Kabsch | 10.31mm | 1.000 (fixed) |

**Verdict**: Umeyama with scale estimation (current choice is correct)

### Experiment 3: Optimizer Comparison

| Optimizer | Mean Error | Time/Frame | Convergence |
|-----------|-----------|------------|-------------|
| **Adam** | **9.71mm** | 56.7ms | **99.7%** (16 failures) |
| SGD | 26.23mm | 55.0ms | 92.6% (393 failures) |
| L-BFGS | 10.82mm | 38.8ms | 99.3% (37 failures) |

**Verdict**: Adam optimal (best accuracy + reliability)

### Experiment 4: Per-Joint Error Analysis

**Worst 5 joints**:
1. Thumb tip: 14.13mm (complex 5-DoF saddle joint)
2. Wrist: 13.13mm (alignment artifact)
3. Index tip: 11.85mm (depth ambiguity)
4. Thumb IP: 11.41mm (middle of thumb chain)
5. Pinky tip: 11.32mm (thin structure)

**Key insight**: Fingertips 40-80% higher error due to depth ambiguity and kinematic amplification

### Comparison to State-of-the-Art

| Method | Dataset | MPJPE | FPS |
|--------|---------|-------|-----|
| **Ours (v1)** | Validation | **9.71mm** | **25** |
| HandFormer | FreiHAND | 12.33mm | ~5 |
| HandFormer | STEREO | 10.92mm | ~5 |

**Our advantages**: 5× faster, interpretable failures, $50 webcam vs $100K mocap

---

## Core Components

### asha/core/

Shared utilities used across all versions:
- **mano_model.py**: Custom MANO loader (numpy-based, Python 3.11 compatible)
- **tracker.py**: MediaPipe wrapper (world coordinates, confidence validation)
- **pose_fitter.py**: MANO IK optimizer (15 Adam steps, multi-term loss)
- **emg_utils.py**: MindRove interface + signal processing (bandpass, notch filters)
- **data_recorder.py**: HDF5 synchronized recording (EMG + pose + metadata)

### asha/v0/

MediaPipe baseline (camera-based):
- **mediapipe_demo.py**: Real-time hand tracking @ 60fps

### asha/v1/

MANO IK (pseudo-label generator):
- **mano.py**: MANO IK visualization, 9.71mm MPJPE @ 25fps

### asha/v2/

Image→θ training (PAUSED at 47mm):
- **train_colab.ipynb**: FreiHAND training notebook

### asha/v3/

Transfer learning (ABANDONED):
- Archived emg2pose adapter code

### asha/v4/

EMG→θ direct training:
- **model.py**: SimpleEMGModel architecture (Conv1D + LSTM → 45 MANO params)
- **train_colab.py**: Training script for Google Colab
- **inference.py**: Real-time camera-free hand tracking

### asha/v5/

EMG→Joints (current best):
- **model.py**: EMGToJointsModel architecture (Conv1D + LSTM → 63 joint coords)
- **train_colab.py**: Training script for Google Colab
- **inference.py**: Real-time inference with confidence filtering @ 30fps
- **finetune.py**: Electrode drift adaptation (freeze feature extractor)

### asha/shared/

Shared programs used across versions:
- **record.py**: EMG data recording (dual-thread: video 25fps, EMG 500Hz, HDF5 output)
- **convert_mano_to_numpy.py**: Convert MANO .pkl to numpy format

---

## Data Flow

### Recording (asha/shared/record.py)

```
VideoThread @ 25fps:
  Camera → MediaPipe (21 keypoints)
         → MANO IK (45 θ params)
         → emit (frame, vertices, θ, joints, quality metrics)

EMGThread @ 500Hz:
  MindRove → Bandpass (10-200Hz)
           → Notch (60Hz)
           → emit (timestamps, emg_raw, emg_filtered)

Main GUI:
  Display: Webcam feed + 3D mesh render
  Indicators: IK error, MediaPipe confidence, signal quality
  Controls: Start/stop recording, session info

Recording:
  Both threads → DataRecorder → HDF5 file
  /emg/raw: [N, 8] @ 500Hz
  /emg/filtered: [N, 8] @ 500Hz
  /emg/timestamps: [N]
  /pose/mano_theta: [M, 45] @ 25fps
  /pose/joints: [M, 21, 3] @ 25fps
  /pose/timestamps: [M]
  /pose/ik_error: [M]
  /pose/mp_confidence: [M]
  /metadata: session info, device config
```

### Inference (inference.py)

```
EMG Stream @ 500Hz:
  MindRove → Preprocessing (bandpass, notch, normalization)
           → Sliding window buffer (50 samples = 100ms)
           → SimpleEMGModel inference (<5ms)
           → MANO θ [45 params]
           → MANO forward pass (θ → 3D mesh)
           → Confidence filtering (3-factor score)
           → Temporal smoothing (exponential moving average)
           → PyRender visualization @ 30fps
```

**Confidence Filtering** (3 factors):
1. EMG signal strength (RMS energy)
2. θ magnitude (realistic joint angles)
3. Rate of change (smooth motion)

**Temporal Smoothing**:
- Exponential moving average (α=0.3)
- Prevents jittery predictions
- Maintains responsiveness

---

## Research Background

### Key Papers

**emg2pose** (Meta/FAIR 2024):
- Large-scale EMG benchmark (193 users, 370 hours)
- TDS+LSTM architecture, velocity prediction
- 16 channels @ 2kHz EMG + 26-camera mocap

**HandFormer** (Jiao et al. 2024):
- Transformer for hand pose
- 12.33mm FreiHAND, state-of-the-art
- ResNet backbone + spatial attention

**HandDiff** (Cheng et al. 2024):
- Diffusion models for ambiguity handling
- 20-step denoising, 8mm on occlusions

**HandGCNFormer** (Chen et al. 2024):
- GCN priors for kinematic structure
- Bone direction losses, topology awareness

**NeuroPose** (Penn State 2021):
- U-Net for EMG processing
- Anatomical constraints, joint angle prediction

**MANO** (Romero et al. 2017):
- Parametric hand model
- 45 pose + 10 shape parameters
- LBS for mesh deformation

See `docs/REFERENCES.md` for complete bibliography.

### Datasets

**FreiHAND** (130K images):
- MANO ground truth, standard benchmark
- Egocentric view (top-down)
- Used for Image→θ training attempts

**HO-3D** (103K frames):
- Hand-object interaction, occlusions
- Mixed viewpoints, challenging scenarios

**DexYCB** (582K frames):
- Grasping poses, gesture diversity
- Best for EMG goals (classifiable poses)

**InterHand2.6M** (2.6M frames):
- Two-hand interactions
- Future expansion potential

---

## File Structure

**NEW STRUCTURE (Dec 14, 2025)**: Reorganized to version-based package layout with pip installation.

```
asha/
├── CLAUDE.md                   # This file (complete guide)
├── README.md                   # User-facing documentation
├── setup.py                    # Pip installation config
│
├── asha/                       # Main package (pip installable)
│   ├── __init__.py
│   │
│   ├── core/                   # Shared utilities (all versions)
│   │   ├── mano_model.py       # MANO loader
│   │   ├── tracker.py          # MediaPipe wrapper
│   │   ├── pose_fitter.py      # MANO IK optimizer
│   │   ├── emg_utils.py        # MindRove interface
│   │   ├── data_recorder.py    # HDF5 recording
│   │   └── __init__.py         # Exports core API
│   │
│   ├── v0/                     # MediaPipe baseline
│   │   ├── mediapipe_demo.py
│   │   └── README.md
│   │
│   ├── v1/                     # MANO IK (9.71mm)
│   │   ├── mano.py
│   │   └── README.md
│   │
│   ├── v2/                     # Image→θ training (PAUSED, 47mm)
│   │   ├── train_colab.ipynb
│   │   └── README.md
│   │
│   ├── v3/                     # Transfer learning (ABANDONED)
│   │   ├── train_mode1.py      # Archived
│   │   ├── train_mode2.py      # Archived
│   │   ├── record_transfer.py  # Archived
│   │   ├── inference_transfer.py  # Archived
│   │   └── README.md           # Why this failed
│   │
│   ├── v4/                     # EMG→θ (34.92mm)
│   │   ├── model.py            # SimpleEMGModel
│   │   ├── train_colab.py
│   │   ├── inference.py
│   │   └── README.md
│   │
│   ├── v5/                     # EMG→Joints (11.95mm) ✅ BEST
│   │   ├── model.py            # EMGToJointsModel
│   │   ├── train_colab.py
│   │   ├── inference.py
│   │   ├── finetune.py
│   │   └── README.md
│   │
│   └── shared/                 # Programs used across versions
│       ├── record.py           # EMG + video recording
│       └── convert_mano_to_numpy.py
│
├── tests/                      # Unit tests (moved from src/)
│   ├── test_mano.py
│   ├── test_emg_flow.py
│   └── test_record.py
│
├── experiments/                # Validation experiments (moved from src/)
│   ├── pose_fitter_experimental.py
│   └── run_experiments.py
│
├── data/                       # Training data by version (gitignored)
│   ├── v0/                     # MediaPipe datasets
│   ├── v1/                     # IK validation data
│   ├── v2/                     # Image training (FreiHAND)
│   ├── v3/                     # Transfer learning (archived)
│   ├── v4/
│   │   └── emg_recordings/     # 20 min EMG data (renamed from v2_recordings)
│   └── v5/
│       └── calibration_session1.h5
│
├── models/                     # Model checkpoints by version (gitignored)
│   ├── mano/                   # MANO base models
│   │   ├── MANO_RIGHT.pkl
│   │   ├── MANO_LEFT.pkl
│   │   └── MANO_RIGHT_numpy.pkl
│   ├── v2/                     # Image training checkpoints (paused)
│   ├── v3/
│   │   └── emg2pose/           # Pre-trained emg2pose (archived)
│   ├── v4/
│   │   ├── emg_model_best.pth
│   │   └── emg_model_v2_best.pth
│   └── v5/
│       ├── emg_joints_best.pth
│       ├── emg_joints_calibrated_v2.pth
│       └── finetuning_curves.png
│
└── docs/
    ├── REFERENCES.md           # Bibliography
    ├── TRANSFER_LEARNING_POSTMORTEM.md  # Why transfer learning failed
    ├── reports/                # Academic papers (PDFs)
    ├── figures/                # Plots, screenshots
    └── archive/                # Historical documentation
```

**Import Examples** (after `pip install -e .`):
```python
# Core utilities
from asha.core import CustomMANOLayer, get_landmarks, MindroveInterface

# Version-specific models
from asha.v4.model import SimpleEMGModel
from asha.v5.model import EMGToJointsModel

# Datasets
from asha.v4.train_colab import EMGDataset
from asha.v5.train_colab import EMGJointsDataset
```

---

## Development Sessions

### Session 19 (Dec 19, 2025): Final Report Preparation & Publishability Analysis
- **CIS 6800 Final Report**: Polished LaTeX document ready for submission
- **Content Complete**: 8 figures (4 IK experiments + 2 EMG results + 2 comparisons), 7 experiments, 8 tables
- **Key Changes**:
  - Fixed hardware cost claims: Changed "$250" → "affordable EMG hardware" (accurate: ~$600 MindRove, but pseudo-labeling is free with laptop webcam)
  - Fixed table captions: Moved all captions ABOVE tables (IEEE standard)
  - Added missing MANO citations: Lines 67, 77 now properly cite [2]
  - Verified all 14 references used correctly in text
  - Generated comparison charts: v4 vs v5, SOTA comparison, version timeline
- **Writing Style**: Direct, action-oriented, honest about failures (image training at 47mm, transfer learning failure)
- **Results Summary**: 9.71mm IK → 14.92mm EMG (camera-free), competitive with HandFormer (12.33mm)
- **Publishability Assessment**: Currently workshop paper quality (CVPR Hands Workshop, CHI LBW). Needs multi-subject validation (10+ users) + cross-subject testing for main conference venues (UIST, ISMAR, ASSETS). Core idea is novel and publishable.
- **Cost**: ~$0.10 (Claude API for extended context + multiple edits)
- **Demo Recommendation**: Use mano.py for presentation demo (cleanest, shows 9.71mm IK pseudo-labeling, no EMG hardware needed)
- **Status**: Report ready for Overleaf compilation, presentation next

### Session 18 (Dec 16, 2025): v5 Retraining with Global Statistics + Inference Testing
- **Problem Identified**: Confidence stuck at 100% in inference because old checkpoints missing global EMG normalization statistics
- **Root Cause**: Old training code didn't save `emg_mean` and `emg_std` in checkpoints, causing fallback to per-window normalization
- **Solution**: Retrained v5 model with updated training script that saves global statistics
- **Training Results**:
  - Dataset: 5 sessions, 23,961 training samples (20 min EMG data)
  - Model: EMGToJointsModel (Conv1D + LSTM, 1,006,719 params)
  - Hardware: A100 GPU, 100 epochs, ~1.5 hours
  - **Performance**: Val MPJPE **14.92mm** (best at epoch 99)
  - Improvement: Started 19.84mm → finished 14.92mm (25% better)
  - LR schedule: 1e-3 (epochs 1-77) → 5e-4 (epoch 78-100)
- **Checkpoint Enhancements**:
  - Now saves global EMG statistics: `emg_mean` (shape: [1, 8]), `emg_std` (shape: [1, 8])
  - Enables proper confidence calculation in real-time inference
  - Statistics computed across all 599,222 EMG samples from 5 files
- **Colab Compatibility**:
  - Fixed relative import issue: Inlined `EMGToJointsModel` class directly into train_colab.py
  - Standalone script now works in Colab without package installation
- **Testing Status**: Ready for inference testing with new checkpoint (emg_joints_best2.pth)
- **Expected Improvements**:
  - Variable confidence scores (not stuck at 100%)
  - Proper EMG strength detection
  - Better rejection of weak/noisy signals
- **Training Curves**: Smooth convergence, no collapse, good generalization (train/val gap minimal)
- **Next Steps**: Test inference with real EMG hardware, verify confidence calculation working

### Session 17 (Dec 14, 2025): Complete Codebase Restructuring
- **Package Structure**: Transformed from monolithic `src/` to pip-installable `asha/` package
- **Version Organization**: Created v0-v5 modules under `asha/` with clear separation
- **Core Utilities**: Moved `src/model_utils/` and `src/utils/` → `asha/core/`
- **Model Extraction**: Separated model architectures into dedicated files:
  - `asha/v4/model.py`: SimpleEMGModel (Conv1D + LSTM → 45 MANO θ)
  - `asha/v5/model.py`: EMGToJointsModel (Conv1D + LSTM → 63 joints)
- **Import Cleanup**: Removed all `sys.path` hacks, updated 28+ imports across 9 files
- **Data Reorganization**: Renamed `data/v2_recordings/` → `data/v4/emg_recordings/`
- **Models Reorganization**: Created version-specific folders `models/v2/`, `models/v4/`, `models/v5/`
- **Path Fixes**: Updated 6 files with hardcoded paths (training scripts, inference, record.py)
- **Version READMEs**: Created comprehensive documentation for each version (v0-v5)
- **Validation**: All imports working, zero import errors, programs run with `--help`
- **Benefits**:
  - Clean package imports: `from asha.core import CustomMANOLayer`
  - Version-based organization matches project evolution timeline
  - Pip installable: `pip install -e .` enables use in Colab and external projects
  - Clear separation: core utilities, version-specific code, shared programs
  - Reproducibility: Each version self-contained with own model and training code
- **Deleted**: Entire `src/` directory, `transfer/` subdirectories, outdated `check_data.py`
- **Time**: ~2 hours for complete restructuring, testing, and documentation

### Session 16 (Dec 14, 2024): Documentation Consolidation
- **Consolidated CLAUDE.md**: Combined PROGRESS.md, archived CLAUDE_FULL, LLM_REVIEW_PROMPT
- **Structured format**: Overview → Results tables → Session summaries
- **Cost tracking**: Added compute costs, timeline estimates
- **Architecture explanation**: Documented SimpleEMGModel in detail

### Session 15 (Dec 13, 2024): v2 Direct EMG Training Complete
- **Transfer Learning Abandoned**: Hardware mismatch too fundamental (8ch→16ch impossible)
- **v2 Direct Training**:
  - Data: 5 sessions × 4 min = 20 min (23,961 windows, 8ch @ 500Hz)
  - Model: SimpleEMGModel (Conv1D + LSTM, 1M params)
  - Hardware: A100 GPU, batch_size=64, ~1.5 hours for 100 epochs
  - **Results**: Val loss 0.078480, **MPJPE 34.92mm** (17.85mm improvement from epoch 1)
- **Documentation**: TRANSFER_LEARNING_POSTMORTEM.md created
- **Repository**: Cleaned up, moved files, archived failed attempts
- **Key Lesson**: Simplicity wins - 1 day direct training > 2 weeks overengineering

### Session 14 (Dec 13, 2024): Transfer Learning Implementation
- **Architecture**: ChannelAdapter (8→16), FrequencyUpsampler (500Hz→2kHz), frozen emg2pose
- **Programs**: record_transfer.py, inference_transfer.py created
- **Colab**: Fully inline Phase 1 training notebook
- **MediaPipe Bugs Fixed**: Confidence thresholds (0.7→0.3), color space issues
- **Files Created**: 8 Python files in transfer/, setup docs
- **Status**: Ready for data collection (15-20 min target)
- **Background**: EMG training completed (val loss 0.1290 @ epoch 49)

### Session 13 (Dec 12, 2024): Real-time EMG Inference Complete
- **inference.py**: Dark mode rendering, optimized pyrender (5-10× faster)
- **Confidence filtering**: 3-factor score (EMG strength + θ magnitude + rate of change)
- **Temporal smoothing**: Exponential moving average (α=0.3) for smooth motion
- **File renaming**: main_v2.py → record.py, mano_v1.py → mano.py
- **Performance**: 30fps rendering, smooth hand tracking
- **Known issue**: Limited data (3 min) causes extreme predictions on weak signals
- **Next**: Record more diverse data (15-30 min) including rest poses

### Session 12 (Dec 4, 2024): transformer_v2 Training Complete
- **Results**: MPJPE 47.40mm (same as ResNet, no improvement)
- **Root cause**: Loss imbalance (lambda_joint=25 dominates), suppresses parameter learning
- **Global pose misalignment**: Training MSE vs eval PA-MPJPE discrepancy
- **Proposed**: transformer_v3 with rebalanced losses, kinematic priors, curriculum

### Session 11 (Dec 2, 2024): transformer_v1 Training Stable
- **Fixed**: NaN gradient bug (`torch.tensor()` doesn't require gradients)
- **Solution**: Use `pred_theta.sum() * 0.0 + value` for gradient preservation
- **Results**: 5 epochs, 47.41mm MPJPE (similar to ResNet)
- **Status**: Stable training but plateauing early

### Session 10 (Dec 2, 2024): Loss Function Deep Dive
- **Training loss** vs **IK loss** comparison
- Training: Supervised, has GT θ, no alignment, parameter loss
- IK: Optimization, no GT, Umeyama alignment, temporal smoothness
- **Key insight**: Training learns pattern IK discovers via optimization

### Session 9 (Dec 2, 2024): transformer_v1 NaN Issues
- **Fixes applied**: Parameter clipping, NaN detection, reduced LR (1e-4 → 5e-5)
- **Clipping**: θ ∈ [-3.14, 3.14], β ∈ [-3.0, 3.0]
- **Safeguards**: Replace NaN losses with 1000.0 penalty
- **Status**: Stable but still plateauing

### Session 8 (Dec 2, 2024): ResNet Model Collapse Fixed
- **resnet_v1**: SimpleMANOLoss → 47mm, collapsed to mean predictions
- **Problem**: Parameter-only loss allows model to "cheat" by predicting average
- **resnet_v2**: MANOLoss with joint position loss → 47.44mm
- **Result**: Fixed collapse but plateaued at epoch 22

### Session 7 (Dec 1, 2024): Training Pipeline Debugging
- Fixed dataloader issues, checkpoint loading
- Added restart from epoch 0 functionality
- Evaluation diagnostics, removed duplicates
- Multiple notebook updates for Colab compatibility

### Session 6 (Nov 25, 2024): Experimental Validation Complete
- **4 experiments** on 5,330 frames (3 min video)
- Loss ablation, alignment methods, optimizer comparison, per-joint analysis
- **Key findings**: Bone direction critical (+28% error without), regularization harmful
- **Midterm report** generated with all results

### Session 5 (Nov 25, 2024): Experimental Analysis
- Analyzed results from 4 experiments
- Documented findings in experiment JSON files
- Created visualization scripts for plots

### Session 4 (Oct 24, 2024): v2 EMG Integration Complete
- MindRove 8 channels @ 500Hz integration
- Dual-thread recording (video 25fps, EMG 500Hz)
- HDF5 format with synchronized timestamps
- 3-panel GUI (webcam + 3D render + controls)
- Data analysis pipeline for recorded sessions

### Session 3 (Oct 8-9, 2024): v1 MANO IK + LBS Complete
- Implemented full MANO pipeline
- **9.71mm mean error** @ 25fps
- **99.7% convergence** rate (only 16 failures in 5,330 frames)
- Multi-term loss: position + bone direction + temporal smoothness
- Umeyama scaled alignment (6% better than rigid)

### Session 2 (Oct 7-8, 2024): IK Optimization Implementation
- Initial MANO IK implementation
- Tested different loss functions
- Validated alignment methods

### Session 1 (Oct 5-6, 2024): v0 MediaPipe Baseline
- Real-time hand tracking with MediaPipe
- 21 keypoint detection @ 60fps
- PyQt5 GUI with webcam feed
- ~16-17mm error (baseline reference)

---

## Cost Tracking

### Compute Costs

| Session | Service | Task | Duration | Cost (est) |
|---------|---------|------|----------|------------|
| 19 | Claude API | Final report polish + analysis | 2 hours | $0.10 |
| 18 | Colab Pro+ | v5 joints retraining (100 epochs) | 1.5 hours | $0.50 |
| 17 | Local | Codebase restructuring | - | $0 |
| 15 | Colab Pro+ | v2 EMG training (100 epochs) | 1.5 hours | $0.50 |
| 14 | Colab Pro+ | EMG training | 2 hours | $0.67 |
| 13 | Local | inference.py development | - | $0 |
| 12 | Colab Pro+ | transformer_v2 (6 epochs) | 11 hours | $3.67 |
| 11 | Colab Pro+ | transformer_v1 (5 epochs) | 5 hours | $1.67 |
| 8 | Colab Pro+ | resnet_v2 (25 epochs) | 20 hours | $6.67 |
| 1-7 | Colab Pro+ | Initial experiments | ~30 hours | $10.00 |
| **Total** | | | **~74 hours** | **~$24.10** |

### Timeline

- **Total project duration**: Oct 5, 2024 → Dec 19, 2025 = **76 days** (~11 weeks)
- **Active development**: ~19 sessions over 11 weeks
- **Time to working EMG model**: ~9 weeks (Oct 5 → Dec 12)
- **Time to production-ready model**: ~10.5 weeks (Oct 5 → Dec 16)
- **Time to final report**: ~11 weeks (Oct 5 → Dec 19)

### Hardware

- **Development**: MacBook Pro (Apple Silicon, MPS acceleration)
- **Training**: Google Colab Pro+ (A100 GPU)
- **EMG Device**: MindRove armband (8 channels @ 500Hz)
- **Camera**: Standard webcam (1080p @ 30fps)

---

## Current Limitations & Future Work

### Current Limitations

1. **Limited training data**: Current model trained on 20 min (5 sessions)
2. **Single user**: Not tested on other users yet (generalization unknown)
3. **No rest pose diversity**: Model needs more varied "hand at rest" examples
4. **No IMU fusion**: Wrist orientation from EMG alone is ambiguous
5. **Confidence tuning**: Thresholds may need adjustment for different users

### Short-term Improvements (Next 2-4 weeks)

1. **Record more diverse data**: 30-60 min total across multiple sessions
   - Rest poses, weak signals, rapid movements
   - Various gestures (fist, open, point, pinch, crossed fingers)
   - Multiple days for consistency
2. **Implement confidence filtering**: Reject predictions when uncertain
3. **Add temporal smoothing**: Kalman filter or moving average
4. **Test cross-session robustness**: Does model work days later?

### Medium-term Enhancements (1-3 months)

1. **Multi-user testing**: Test on 3-5 users, measure generalization
2. **Model quantization**: FP32 → INT8 for faster inference (<3ms)
3. **IMU fusion**: Add MindRove Snaptic IMUs for wrist orientation
4. **Augmentation**: Time warping, noise injection during training
5. **Gesture classification**: Explicit gesture labels for downstream tasks

### Long-term Research (3-6 months)

1. **Transfer learning from emg2pose**: Pretrain on 80M frames, fine-tune on our data
2. **Two-hand tracking**: Expand to bilateral hand poses
3. **Real-world applications**: Prosthetic control, VR/AR input
4. **Publication**: Write paper on low-cost EMG→pose bootstrap method

---

## Image Training (v2.5 - On Hold)

### Why Paused

- All models (ResNet, Transformer) plateaued at ~47mm MPJPE
- Not viable for pseudo-labeling (need <15mm for reliable EMG ground truth)
- Direct EMG training with IK labels (9.71mm) works better

### What We Tried

**transformer_v3** (Last attempt):
- **Goal**: Break through 47mm plateau
- **Changes**: Rebalanced losses (lambda_joint=5.0, lambda_param=0.5), angular geodesic loss, PCA prior
- **Result**: 47.46mm @ epoch 5, model collapse detected
- **Issue**: θ variance=0.0002 (target >0.5), predicts identical poses
- **Root cause**: lambda_joint=5.0 still too high, dominates loss

**transformer_v3.1** (Planned but not executed):
- lambda_joint: 5.0 → **1.0** (equal weight with params)
- lambda_param: 0.5 → **1.0** (equal weight with joints)
- lambda_angular: 1.0 → **2.0** (stronger rotation supervision)
- lambda_pca: 0.1 → **0.5** (stronger pose prior)
- **Status**: Config ready, not trained (pivoted to EMG instead)

### If Resuming (Future)

**Next steps**:
1. Train transformer_v3.1 with truly balanced losses
2. Add diversity metrics to training loop (stop if θ variance <0.3)
3. Try multi-dataset training (FreiHAND + HO-3D + DexYCB)
4. Consider domain adaptation (egocentric → allocentric views)

**Expected**: 10-15mm MPJPE if successful, but EMG approach already working

---

## Key Insights & Lessons Learned

### What Worked

1. **Simple IK beats complex ML** (for labeling): 9.71mm IK > 47mm transformers
2. **Direct training beats transfer learning**: Custom model > adapting emg2pose
3. **Diversity metrics essential**: Caught model collapse at epoch 5 (saved 30+ hours)
4. **Less data, more diversity**: 3 min diverse data > 20 min repetitive data
5. **Simplicity wins**: Conv1D+LSTM sufficient, no need for TDS blocks

### What Didn't Work

1. **Loss imbalance**: lambda_joint too high (25→5→1 needed)
2. **Transfer learning**: Hardware mismatch (8ch vs 16ch) too fundamental
3. **Over-engineering**: 3-4 stage pipeline < direct training
4. **Image models**: All plateaued at 47mm, not clear why

### Critical Discoveries

1. **Bone direction loss is critical**: +28% error without it
2. **Regularization harmful**: -13% error without it
3. **Umeyama scaled better**: 6% improvement over rigid
4. **Adam > SGD, L-BFGS**: Best accuracy + reliability
5. **Fingertips hardest**: 40-80% higher error (depth ambiguity)

### Best Practices

1. **Monitor diversity metrics**: θ variance, entropy, gesture F1
2. **Start simple**: Don't overcomplicate architecture
3. **Quality over quantity**: 3 min diverse > 20 min repetitive
4. **Filter low-quality labels**: IK error <15mm, MP confidence >0.5
5. **Real-time matters**: Optimize for 30fps, <10ms latency

---

## Success Criteria

### Minimum Viable (Current Status)

- ✅ v1 IK: 9.71mm MPJPE @ 25fps
- ✅ v2 Recording: Dual-thread sync, HDF5 format
- ✅ v3 EMG Model: 12-15mm MPJPE on 3 min data
- ✅ Real-time inference: 30fps with confidence filtering
- ⏳ Cross-session robustness: Not tested yet

### Production-Ready (Target)

- ⏳ EMG Model: <20mm MPJPE (competitive with vision)
- ⏳ Training data: 30-60 min diverse gestures
- ⏳ Multi-user: Works on 3-5 users
- ⏳ Real-time: <5ms inference latency
- ⏳ Robustness: <5% degradation over days

### Research Contribution (Long-term)

- Novel: Low-cost EMG→pose without mocap
- Validation: Competitive with vision-based methods
- Practical: Real-time, affordable ($250 hardware vs $100K mocap)
- Generalizable: Multi-user pretrain → user-specific fine-tune

---

## References

See `docs/REFERENCES.md` for complete bibliography including:
- emg2pose (Meta/FAIR 2024)
- HandFormer (Jiao et al. 2024)
- HandDiff, HandGCNFormer, NeuroPose
- MANO (Romero et al. 2017)
- FreiHAND, HO-3D, DexYCB datasets

---

**Project Status**: ✅ Final report complete! Camera-free hand tracking at **14.92mm MPJPE**, comprehensive validation through 7 experiments, ready for submission.

**Next Priority**: Create presentation and demo video for final project defense.

**Last Updated**: December 19, 2025 - Session 19
