# Transfer Learning Setup Guide

Complete implementation of transfer learning from Meta FAIR's emg2pose to Project Asha's MindRove hardware.

## What Was Implemented

### ✅ Core Components

1. **Transfer Modules** (`transfer/training/transfer_modules.py`)
   - ChannelAdapter: 8ch → 16ch learned projection
   - FrequencyUpsampler: 500Hz → 2kHz interpolation
   - DifferentiableIKLayer: Joint angles → MANO θ conversion (optional)

2. **Pre-trained Model Loader** (`transfer/training/load_emg2pose.py`)
   - Loads emg2pose checkpoint with exact architecture
   - **Status**: ✅ Working! All 68 parameters loaded successfully
   - Input: [batch, time, 16] @ 2kHz
   - Output: [batch, time', 20] joint angles @ 50Hz

3. **Dataset Loaders** (`transfer/training/emg2pose_loader.py`)
   - EMG2PoseDataset: Loads emg2pose HDF5 files (30 files, 21,078 samples)
   - MindRoveDataset: Loads Project Asha recordings
   - MixedEMGDataset: 70/30 weighted mixing for Phase 2

4. **Data Recorder** (`transfer/utils/data_recorder.py`)
   - Extended version that saves BOTH joint angles AND MANO params
   - Records EMG (8ch @ 500Hz), MANO θ (from IK), and joint angles (from transfer model)
   - HDF5 structure:
     ```
     /emg/filtered         [N, 8]
     /pose/mano_theta      [M, 45]
     /transfer/joint_angles [N, 20]  ← NEW!
     ```

### ✅ Programs

1. **Recording Program** (`transfer/programs/record_transfer.py`)
   - Real-time EMG acquisition + transfer model inference
   - Dual output: Saves both joint angles and MANO params
   - GUI with webcam feed and recording controls
   - Usage:
     ```bash
     python transfer/programs/record_transfer.py
     python transfer/programs/record_transfer.py --no-emg  # Test without EMG
     ```

2. **Inference Program** (`transfer/programs/inference_transfer.py`)
   - Real-time camera-free hand tracking
   - Uses transfer learning pipeline: 8ch → 16ch → 2kHz → emg2pose → 20 angles
   - 3D visualization with pyrender
   - Usage:
     ```bash
     python transfer/programs/inference_transfer.py
     python transfer/programs/inference_transfer.py --no-emg  # Test mode
     ```

### ✅ Training Scripts

1. **Phase 1 Training** (`transfer/training/train_phase1.py`)
   - Trains ChannelAdapter (8→16ch mapping)
   - Freezes emg2pose model
   - Uses MindRove recordings only
   - Target: 15-20mm MPJPE
   - Usage:
     ```bash
     python transfer/training/train_phase1.py \
         --data transfer/data/transfer_session_*.h5 \
         --epochs 50 \
         --batch-size 32 \
         --lr 1e-3
     ```

2. **Colab Notebook** (`transfer/training/train_transfer_colab.ipynb`)
   - Complete inline implementation (no external imports)
   - Automatically downloads emg2pose checkpoint
   - Supports Google Drive integration
   - Phase 1 training ready
   - Opens in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeevankarandikar/asha/blob/main/transfer/training/train_transfer_colab.ipynb)

## File Structure

```
transfer/
├── README.md                              # Project overview
├── SETUP.md                               # This file
├── training/
│   ├── transfer_modules.py                # ChannelAdapter, FrequencyUpsampler, IKLayer
│   ├── load_emg2pose.py                   # ✅ Pre-trained model loader (WORKING!)
│   ├── emg2pose_loader.py                 # Dataset loaders
│   ├── emg2pose_pretrained.py             # Checkpoint inspector
│   ├── emg2pose_wrapper.py                # Alternative loader (unused)
│   ├── train_phase1.py                    # Phase 1 training script
│   └── train_transfer_colab.ipynb         # Complete Colab notebook
├── programs/
│   ├── record_transfer.py                 # Recording with dual output
│   └── inference_transfer.py              # Real-time inference
├── utils/
│   ├── __init__.py
│   └── data_recorder.py                   # Extended recorder with joint angles
├── emg2pose/                              # Cloned emg2pose repo
└── checkpoints/                           # Saved model weights (gitignored)

models/
└── emg2pose/
    └── tracking_vemg2pose.ckpt            # Pre-trained checkpoint (236MB)

data/
└── emg2pose/
    └── emg2pose_dataset_mini/             # Mini dataset (30 files, 21K samples)
```

## Quick Start

### 1. Download Pre-trained Checkpoint

```bash
cd /Users/jeevankarandikar/Documents/GitHub/asha

# Download emg2pose checkpoint (236MB)
curl -L -o models/emg2pose/emg2pose_model_checkpoints.tar.gz \
    https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz

# Extract
tar -xzf models/emg2pose/emg2pose_model_checkpoints.tar.gz -C models/emg2pose/
rm models/emg2pose/emg2pose_model_checkpoints.tar.gz
```

**Result**: `models/emg2pose/tracking_vemg2pose.ckpt` (68M parameters)

### 2. Clone emg2pose Repository

```bash
# Clone for architecture definitions
git clone https://github.com/facebookresearch/emg2pose.git transfer/emg2pose
```

### 3. Test Pre-trained Model

```bash
cd transfer/training
python load_emg2pose.py
```

**Expected Output**:
```
Loading emg2pose from: ../../models/emg2pose/tracking_vemg2pose.ckpt
✅ All weights loaded successfully!
✅ Model ready for inference

Test input: torch.Size([2, 2000, 16])
Output shape: torch.Size([2, 210, 20])
Output range: [-0.033, 0.163]
✅ Model loaded and tested successfully!
```

### 4. Record MindRove Data

```bash
# Activate environment
source asha_env/bin/activate

# Record with EMG hardware
python transfer/programs/record_transfer.py

# Or test without EMG
python transfer/programs/record_transfer.py --no-emg
```

**Saves to**: `transfer/data/transfer_session_YYYYMMDD_HHMMSS.h5`

### 5. Train Phase 1 (ChannelAdapter)

**Option A: Local Training**
```bash
python transfer/training/train_phase1.py \
    --data transfer/data/transfer_session_*.h5 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --device cpu  # or mps for Apple Silicon
```

**Option B: Colab Training (Recommended)**
1. Upload MindRove recordings to Google Drive: `MyDrive/Colab Notebooks/asha/mindrove_recordings/`
2. Open `transfer/training/train_transfer_colab.ipynb` in Colab
3. Run all cells

**Outputs**:
- `transfer/checkpoints/phase1_best.pth` (best model)
- `transfer/checkpoints/phase1_epoch{N}.pth` (periodic checkpoints)
- `transfer/checkpoints/phase1_training_curves.png` (loss plot)

### 6. Run Real-time Inference

```bash
# With trained ChannelAdapter (after Phase 1)
python transfer/programs/inference_transfer.py \
    --model models/emg2pose/tracking_vemg2pose.ckpt

# Test mode (no EMG hardware)
python transfer/programs/inference_transfer.py --no-emg
```

## Next Steps

### Phase 2 Training (Optional)

Fine-tune entire model on mixed dataset (70% emg2pose + 30% MindRove):

1. **Download emg2pose dataset** (50GB recommended, or full 431GB):
   ```bash
   # Download full dataset
   wget https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar

   # Extract subset (first 50GB / 5,000 files)
   # TODO: Add extraction script
   ```

2. **Update Colab notebook** to include Phase 2 training loop

3. **Train with differential learning rates**:
   - emg2pose: 1e-5 (low, preserve pre-training)
   - ChannelAdapter: 1e-4 (high, adapt to new hardware)

## Architecture Details

### Transfer Pipeline

```
Input: 8ch @ 500Hz (MindRove EMG)
    ↓
ChannelAdapter (trainable)
    ↓ [batch, 50, 16]
FrequencyUpsampler (fixed)
    ↓ [batch, 200, 16]
emg2pose TDS + LSTM (frozen in Phase 1)
    ↓ [batch, time', 20]
Output: 20 joint angles @ 50Hz
```

### emg2pose Architecture

**Loaded Successfully! ✅**

- **Conv blocks**: 2 layers (16→256, 256→256)
- **TDS stages**: 2 stages (256→256→64)
- **LSTM decoder**: 2 layers, hidden=512, input=84 (64 TDS + 20 state)
- **Output**: 20 joint angles (velocity-based tracking)

**Total parameters**: 68 parameter tensors

### ChannelAdapter Architecture

```python
class ChannelAdapter(nn.Module):
    def __init__(self):
        self.projection = nn.Linear(8, 16)      # Anatomical initialization
        self.bn = nn.BatchNorm1d(16)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [batch, time, 8] → [batch, time, 16]
```

**Initialization**: Anatomical priors (electrode interpolation)
- Each input channel contributes to 2 output channels
- Neighbor contributions added (circular)

## Troubleshooting

### Issue: "Checkpoint not found"

**Solution**: Download checkpoint first (see Quick Start #1)

### Issue: "No MindRove data found"

**Solution**: Record data using `record_transfer.py` or upload to Drive for Colab

### Issue: "emg2pose module not found"

**Solution**: Clone emg2pose repo (see Quick Start #2)

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use CPU/MPS

## Performance Targets

| Phase | Dataset | Target MPJPE | Status |
|-------|---------|--------------|--------|
| Phase 1 | MindRove only | 15-20mm | 📋 Ready to train |
| Phase 2 | 70% emg2pose + 30% MindRove | 10-15mm | ⏳ Future work |

**Baseline**: Current from-scratch EMG model = 47mm MPJPE

## Credits

- **emg2pose**: Meta FAIR Research ([GitHub](https://github.com/facebookresearch/emg2pose))
- **MANO**: MPI-IS ([Website](https://mano.is.tue.mpg.de))
- **Project Asha**: Jeevan Karandikar ([GitHub](https://github.com/jeevankarandikar/asha))

## References

1. emg2pose: [Unsupervised Learning for Precise Hand Pose Estimation](https://research.facebook.com/publications/unsupervised-learning-for-precise-hand-pose-estimation/)
2. MANO: [Embodied Hands: Modeling and Capturing Hands and Bodies Together](https://mano.is.tue.mpg.de)

---

**Last Updated**: December 13, 2025
**Status**: ✅ Phase 1 implementation complete, ready for training
