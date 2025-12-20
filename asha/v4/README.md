# v4: EMG → MANO θ (Direct Regression)

**Status**: ✅ Trained | ❌ Inference Broken (needs retraining after normalization fix)

**Approach**: Direct prediction of MANO pose parameters from EMG signals using Conv1D + LSTM architecture.

## Overview

v4 represents the **direct EMG-to-pose training approach** after abandoning transfer learning (v3). Instead of adapting a pre-trained model (emg2pose), v4 trains from scratch using IK-generated pseudo-labels.

**Key Innovation**: Using v1 MANO IK (9.71mm MPJPE) as pseudo-labels for supervised learning, bypassing the need for expensive hand-object datasets.

## Architecture

**SimpleEMGModel** (1,004,397 parameters):
```
Input: [batch, 8, 50] EMG (8 channels × 50 samples @ 500Hz = 100ms window)
├── Conv1D(8→64, k=5) + BatchNorm + ReLU + MaxPool(2)
├── Conv1D(64→128, k=5) + BatchNorm + ReLU + MaxPool(2)
├── LSTM(128→256, 2 layers, dropout=0.3)
├── FC(256→128) + ReLU + Dropout(0.3)
└── FC(128→45) → MANO θ
Output: [batch, 45] MANO pose parameters
```

**Design Rationale**:
- Simpler than emg2pose (no TDS blocks, no velocity prediction)
- Temporal modeling via LSTM (captures gesture dynamics)
- Direct θ prediction (no intermediate joint representation)

## Results

**Training Data** (Session 1-5, Dec 13, 2024):
- 5 sessions × 4 min = **20 min** recorded data
- 23,961 training windows (8ch @ 500Hz, 50-sample sliding window)
- Generated via v1 IK (9.71mm quality labels)

**Training** (Google Colab A100):
- 100 epochs, batch_size=64, ~1.5 hours
- Best checkpoint: epoch 100
- Val loss: 0.078480

**Performance**:
- **MPJPE: 34.92mm** (±18.65mm std)
- Improvement from epoch 1: 52.77mm → 34.92mm (17.85mm / 34% better)
- Comparison: v5 (joints) = 11.95mm ✅ | v4 (θ) = 34.92mm ❌

**Known Issues**:
- ❌ **Inference broken**: Normalization bug fixed Dec 14, 2024 (training used global stats, inference used per-window stats)
- ❌ **Needs retraining**: Checkpoint pre-dates bug fix, predictions stuck/extreme
- Limited diversity: 20 min data insufficient for robust generalization
- Higher error than v5: Direct θ prediction harder than joint positions

## Files

- **model.py**: SimpleEMGModel architecture (Conv1D + LSTM → 45 params)
- **train_colab.py**: Training script for Google Colab (100 epochs, A100 GPU)
- **inference.py**: Real-time camera-free inference (❌ broken, needs retrained checkpoint)

## Quick Start

### Training (Google Colab)

```python
# Upload this directory to Google Drive: MyDrive/Colab Notebooks/asha/
# Open train_colab.py in Colab, run all cells

from asha.v4.train_colab import train_model
train_model(
    data_dir="/content/drive/MyDrive/Colab Notebooks/asha/data/v4/emg_recordings",
    output_dir="/content/drive/MyDrive/Colab Notebooks/asha/models/v4",
    epochs=100,
    batch_size=64
)
```

### Inference (CURRENTLY BROKEN)

```bash
# ❌ DO NOT USE until retrained with fixed normalization
python -m asha.v4.inference --model models/v4/emg_model_v2_best.pth
```

**To fix inference**:
1. Retrain model using updated train_colab.py (has normalization fix)
2. Replace checkpoint: models/v4/emg_model_v2_best.pth
3. Test inference with new checkpoint

## v4 vs v5 Comparison

| Metric | v4 (EMG→θ) | v5 (EMG→Joints) |
|--------|-----------|----------------|
| **Output** | 45 MANO θ params | 63 joint positions (21×3) |
| **MPJPE** | 34.92mm ❌ | 11.95mm ✅ |
| **Architecture** | Conv1D + LSTM | Conv1D + LSTM (same) |
| **Training Data** | 20 min (5 sessions) | 20 min (5 sessions) |
| **Training Time** | ~1.5 hours | ~1.5 hours |
| **Inference** | ❌ Broken | ✅ Working |
| **Why Worse?** | Pose manifold complexity | Direct 3D supervision |

**Conclusion**: v5 (joints) is clearly superior. Direct joint prediction with 3D supervision outperforms parametric θ prediction.

## Why v4 Exists

v4 was trained **before** discovering that direct joint prediction (v5) works better. Kept for:
1. Historical record of evolution
2. Comparison baseline for v5
3. Demonstrates why direct joint supervision > parametric supervision

**Recommendation**: Use v5 for all future work (11.95mm vs 34.92mm).

## Timeline

- **Dec 13, 2024**: Training complete (34.92mm MPJPE, 5 sessions × 4 min data)
- **Dec 14, 2024**: Normalization bug discovered and fixed (inference broken)
- **Dec 14, 2024**: Codebase restructure (v4 extracted from src/training/train_v2_colab.py)
- **Status**: Awaiting retraining with fixed normalization

## Next Steps

1. **Retrain model**: Run train_colab.py with updated normalization (prioritize v5 first)
2. **Replace checkpoint**: models/v4/emg_model_v2_best.pth
3. **Test inference**: Verify predictions are stable and realistic
4. **Collect more data**: 15-30 min target for robustness (if v5 insufficient)

**Priority**: LOW (v5 is superior, focus efforts there)
