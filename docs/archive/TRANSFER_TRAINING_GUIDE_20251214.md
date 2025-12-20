# Transfer Learning Training Guide (Mode 2)

## Quick Start

### 1. Prerequisites

```bash
# Clone emg2pose repo (if not done already)
git clone https://github.com/facebookresearch/emg2pose.git transfer/emg2pose

# Verify checkpoints exist (should already be there!)
ls models/emg2pose/tracking_vemg2pose.ckpt  # ✅ Already downloaded!
ls models/MANO_RIGHT_numpy.pkl              # ✅ From previous work
```

### 2. Record Data

```bash
source asha_env/bin/activate

# Record 15-20 minutes total (5-6 sessions × 4 min each)
python transfer/programs/record_transfer.py
```

**Follow the 5-phase guidance:**
1. Rest/Calibration (30s)
2. Dynamic Movements (2 min) - **Most important!**
3. Gesture Holds (30s)
4. Random Exploration (30s)
5. Cool Down (30s)

**Saved to:** `transfer/data/transfer_session_YYYYMMDD_HHMMSS.h5`

### 3. Train Model (M3 Mac)

```bash
# Train on all recorded sessions
python transfer/training/train_mode2.py \
    --data transfer/data/transfer_session_*.h5 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3

# Or specify specific files
python transfer/training/train_mode2.py \
    --data transfer/data/transfer_session_20251213_*.h5 \
    --epochs 100
```

**Training time:** ~2-3 hours on M3 Pro (100 epochs, 20 min data)

**Outputs:**
- `transfer/checkpoints/transfer_mode2_best.pth` - Best model
- `transfer/checkpoints/transfer_mode2_epoch{N}.pth` - Periodic checkpoints
- `transfer/checkpoints/training_curves.png` - Loss plots

### 4. Check Results

**Success criteria:**
- ✅ **MPJPE < 15mm** → SUCCESS! Ready for deployment
- ⚠️ **MPJPE 15-20mm** → Good but not great (collect more data or train longer)
- ❌ **MPJPE > 20mm** → Needs improvement (check EMG quality, collect diverse data)

---

## Architecture (Mode 2: emg2pose → MANO)

```
MindRove EMG (8ch @ 500Hz, 2sec window = 1000 samples)
  ↓
ChannelAdapter: 8→16ch [TRAINABLE, ~5K params]
  ↓
FrequencyUpsampler: 500Hz→2kHz (linear interp) [FIXED]
  ↓
emg2pose: 16ch@2kHz → 20 angles [FROZEN, 68M params]
  ↓
AngleConverter: 20 angles → 45 MANO θ [TRAINABLE, ~15K params]
  ↓
MANO Forward: 45 θ → 21 joints [FROZEN, ~5K params]
  ↓
Output: MANO joints (21, 3)

Loss: MSE(pred_θ, IK_θ) + MSE(pred_joints, IK_joints)
Trainable: ~20K params (ChannelAdapter + AngleConverter)
Frozen: ~68M params (emg2pose + MANO)
```

---

## What's Happening During Training

### Data Loading
- Loads EMG (8ch @ 500Hz) and MANO θ (45D from IK)
- Filters low-quality frames: `ik_error < 15mm` and `mp_confidence > 0.5`
- Interpolates pose data to match EMG rate
- Normalizes EMG (z-score per channel)
- Creates sliding windows: 2sec @ 500Hz = 1000 samples, 50% overlap

### Training Loop
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingWarmRestarts (restarts every 20 epochs)
- **Batch size:** 32 (reduce to 16 if memory issues)
- **Gradient clipping:** max_norm=1.0
- **Loss:** MSE(θ) + MSE(joints) - dual supervision!

### What's Being Learned
- **ChannelAdapter:** Learns mapping from MindRove 8ch → emg2pose's expected 16ch layout
- **AngleConverter:** Learns mapping from emg2pose's 20 joint angles → MANO's 45 parameters
  - Wrist orientation differences (emg2pose 2 DOF vs MANO 3 DOF)
  - Finger coupling patterns (e.g., ring/pinky coordination)
  - Scale differences (generic hand vs personalized MANO)

### What's Frozen
- **emg2pose:** Pre-trained on 80M frames @ 2kHz - too valuable to modify in Phase 1!
- **MANO:** Parametric hand model - just used for forward pass to compute joints

---

## Troubleshooting

### Error: "emg2pose not found"
```bash
git clone https://github.com/facebookresearch/emg2pose.git transfer/emg2pose
```

### Error: "emg2pose checkpoint not found"
```bash
curl -L -o models/emg2pose/tracking_vemg2pose.ckpt \
    https://dl.fbaipublicfiles.com/emg2pose/tracking_vemg2pose.ckpt
```

### Error: "MANO model not found"
```bash
# MANO should exist from v1 work
ls models/MANO_RIGHT_numpy.pkl

# If missing, check src/utils/convert_mano_to_numpy.py
```

### Error: "MPS not available"
- Script will fall back to CPU (slower but works)
- M3 should have MPS available by default

### Training is slow
- Reduce batch size: `--batch-size 16`
- Reduce epochs: `--epochs 50` (for quick testing)
- Check Activity Monitor for GPU usage

### MPJPE not improving
1. **Check data quality:**
   ```python
   import h5py
   f = h5py.File('transfer/data/transfer_session_*.h5', 'r')
   print(f"IK error mean: {f['pose/ik_error'][:].mean():.2f}mm")
   print(f"MP confidence mean: {f['pose/mp_confidence'][:].mean():.2f}")
   ```
   - IK error should be < 15mm on average
   - MP confidence should be > 0.5

2. **Check EMG signal:**
   - Electrode placement correct?
   - Good skin contact?
   - Cable noise?

3. **Collect more data:**
   - Target: 15-30 min total
   - More diverse movements (fast/slow, partial gestures)

4. **Train longer:**
   - Try 200 epochs: `--epochs 200`
   - Watch for overfitting (val loss increasing)

---

## Next Steps After Training

### If Training Succeeds (MPJPE < 15mm)

1. **Test real-time inference:**
   ```bash
   # TODO: Create inference_transfer.py with Mode 2 pipeline
   python transfer/programs/inference_transfer.py \
       --model transfer/checkpoints/transfer_mode2_best.pth
   ```

2. **Evaluate on held-out data:**
   - Record a new session (different day/time)
   - Evaluate without training
   - Check if performance holds

3. **Deploy for prosthetic control:**
   - Integrate with control system
   - Add safety checks (confidence thresholds)
   - User testing

### If Training Fails (MPJPE > 20mm)

1. **Collect more data** (most important!)
   - Current: 5 sessions × 4 min = 20 min
   - Target: 10 sessions × 3 min = 30 min
   - Focus on dynamic movements (60% of time)

2. **Improve data quality:**
   - Better electrode placement
   - Reduce noise (shielded cables, ground strap)
   - Higher MP confidence (better lighting, camera angle)

3. **Try Mode 1 first** (simpler):
   - Train ChannelAdapter only
   - Output landmarks (not MANO)
   - See if issue is data or architecture

4. **Increase model capacity:**
   - Wider AngleConverter: `nn.Linear(20, 256)` → `nn.Linear(20, 512)`
   - More layers: 3-4 layers instead of 3

---

## Training Progress Example

```
Epoch 1/100
  Train Loss: 0.523412 (θ: 0.312345, joints: 0.211067)
  Val Loss:   0.487234 (θ: 0.289123, joints: 0.198111)
  Val MPJPE:  42.31mm (median: 38.45mm)
  LR:         0.001000

Epoch 20/100
  Train Loss: 0.123456 (θ: 0.067890, joints: 0.055566)
  Val Loss:   0.134567 (θ: 0.072345, joints: 0.062222)
  Val MPJPE:  18.23mm (median: 16.11mm)
  ✅ Saved best model (val_loss=0.134567, MPJPE=18.23mm)
  LR:         0.000873

Epoch 50/100
  Train Loss: 0.089012 (θ: 0.045123, joints: 0.043889)
  Val Loss:   0.095678 (θ: 0.048234, joints: 0.047444)
  Val MPJPE:  13.45mm (median: 11.89mm)
  ✅ Saved best model (val_loss=0.095678, MPJPE=13.45mm)
  LR:         0.000234

Training complete!
Best val MPJPE: 13.45mm
✅ SUCCESS! MPJPE (13.45mm) < 15mm target
```

---

## Key Design Decisions

### Why freeze emg2pose?
- Pre-trained on 80M frames @ 2kHz with expensive sensors
- Our data: ~20 min @ 500Hz with 8ch MindRove
- Fine-tuning would overfit and destroy pre-training

### Why train AngleConverter?
- emg2pose uses generic 20-DOF hand model
- MANO uses personalized 45-DOF model
- Need learnable mapping between representations

### Why dual loss (θ + joints)?
- θ loss: Learn MANO parameter space directly
- Joints loss: Ensure 3D joint accuracy
- Both together: Better than either alone

### Why 2sec windows?
- emg2pose trained with 2kHz × 2sec = 4000 samples
- After upsampling: 500Hz × 2sec × 4 = 4000 samples (matches!)
- Shorter windows: Less temporal context
- Longer windows: More memory, slower training

---

**Last Updated:** December 13, 2025
**Status:** Ready for training! 🚀
