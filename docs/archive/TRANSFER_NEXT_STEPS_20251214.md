# Transfer Learning - Next Steps & Session Handoff

**Date**: December 13, 2025
**Status**: Phase 1 implementation complete, GUI debugging in progress

---

## 🎯 Current Issue: MediaPipe Not Detecting Hand

**Symptom**: `record_transfer.py --no-emg` runs but MediaPipe doesn't detect hand
```
[debug] Frame 0: landmarks=False, world=False
[debug] Frame 30: landmarks=False, world=False
```

**What works:**
- ✅ Camera feed displays correctly
- ✅ GUI opens without errors
- ✅ All imports working
- ✅ Transfer model loads successfully

**What doesn't work:**
- ❌ MediaPipe hand detection (always returns None)
- ❌ No landmarks appear when hand shown to camera
- ❌ MANO model never loads (because no hand detected)

---

## 🔍 Debug Steps to Try Next

### 1. Verify MediaPipe Configuration

Check if MediaPipe Hands is initialized correctly in `src/model_utils/tracker.py`:

```bash
grep -A 20 "_hands = " src/model_utils/tracker.py
```

**Look for:**
- `max_num_hands=1` (should be 1)
- `model_complexity=1` (default is fine)
- `min_detection_confidence=0.5` (might be too high)
- `min_tracking_confidence=0.5` (might be too high)

**Try lowering confidence thresholds:**
```python
_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3,  # Lower from 0.5
    min_tracking_confidence=0.3    # Lower from 0.5
)
```

### 2. Test MediaPipe Directly

Create a minimal test script to verify MediaPipe works:

```python
# test_mediapipe.py
import cv2
import sys
sys.path.insert(0, 'src')

from model_utils.tracker import get_landmarks, get_landmarks_world

cap = cv2.VideoCapture(0)
print("Show your hand to the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = get_landmarks(frame_rgb)
    landmarks_world = get_landmarks_world(frame_rgb)

    if landmarks is not None:
        print(f"✅ Hand detected! Landmarks shape: {landmarks[0].shape}")
        break
    else:
        print("❌ No hand detected")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run: `python test_mediapipe.py`

### 3. Check Camera Permissions

MediaPipe might be failing silently due to permissions:
- **macOS**: System Preferences → Security & Privacy → Camera
- Verify Python has camera access

### 4. Compare with Working Code

The original `src/programs/record.py` works. Compare:

```bash
# Run the original to verify MediaPipe works
python src/programs/record.py
```

If original works but transfer version doesn't, compare the VideoThread implementations.

### 5. Check OpenCV Version

```bash
python -c "import cv2; print(cv2.__version__)"
```

Should be 4.x. If 3.x, might have compatibility issues.

---

## 📋 What Was Accomplished This Session

### ✅ Complete Transfer Learning Implementation

**Core Components Created:**
1. `transfer/training/transfer_modules.py` - ChannelAdapter, FrequencyUpsampler
2. `transfer/training/load_emg2pose.py` - Pre-trained model loader (**WORKING!**)
3. `transfer/training/emg2pose_loader.py` - Dataset loaders
4. `transfer/utils/data_recorder.py` - Extended recorder with joint angle saving
5. `transfer/programs/record_transfer.py` - Recording program (needs MediaPipe fix)
6. `transfer/programs/inference_transfer.py` - Real-time inference
7. `transfer/training/train_phase1.py` - Phase 1 training script
8. `transfer/training/train_transfer_colab.ipynb` - Complete inline Colab notebook

**Documentation:**
- `transfer/README.md` - Project overview
- `transfer/SETUP.md` - Complete setup guide
- `transfer/NEXT_STEPS.md` - This file

**Fixes Applied:**
- ✅ Added `__init__.py` files to make Python packages
- ✅ Fixed import path conflicts (src/ vs transfer/)
- ✅ Fixed `mano_from_landmarks()` signature (no `side` parameter)
- ✅ Fixed return value unpacking (verts, joints, theta, ik_error)
- ✅ Fixed landmarks tuple unpacking (landmarks, confidence)
- ✅ Changed to always emit frames (camera feed shows even without hand)

### ✅ Pre-trained Model Verification

```bash
cd transfer/training
python load_emg2pose.py
```

**Output:**
```
✅ All weights loaded successfully!
Test input: torch.Size([2, 2000, 16])
Output shape: torch.Size([2, 210, 20])
✅ Model loaded and tested successfully!
```

**Conclusion**: emg2pose model loads correctly and produces valid outputs!

---

## 🚀 Once MediaPipe is Fixed: Recording Protocol

### Recording Strategy (15-20 minutes total)

**Session Structure** (5-6 sessions × 3-4 minutes each):

```
Session Template:
┌─────────────────────────────────────┐
│ 0:00-0:30  │ Rest/Calibration       │
│ 0:30-2:30  │ Dynamic movements      │
│ 2:30-3:00  │ Gesture holds (3-4)    │
│ 3:00-3:30  │ Random exploration     │
│ 3:30-4:00  │ Cool down (slow moves) │
└─────────────────────────────────────┘
```

**Movement Distribution:**
- **60% Dynamic movements** (most important for velocity-based tracking!)
  - Smooth transitions: Open → Fist → Open
  - Finger waves: Sequential finger lifts
  - Pinch sequences: Thumb to each finger
  - Wrist rotations with different hand shapes

- **20% Gesture holds with entry/exit**
  - Fist, Open, Pinch, Point, OK sign, Peace, Thumbs up
  - Focus on smooth transitions, not the hold

- **10% Rest states**
  - Hand relaxed on table
  - Baseline calibration

- **10% Random exploration**
  - Weird combinations
  - Fast/slow variations
  - Partial gestures

**Key Points:**
- ✅ RIGHT HAND ONLY (no left hand, no IMU for now)
- ✅ Keep moving smoothly (velocity-based model learns from motion)
- ✅ Natural speed (not too slow, not too fast)
- ✅ Multiple short sessions (not one long recording)
- ✅ Take breaks (avoid muscle fatigue)

**Recording Command:**
```bash
source asha_env/bin/activate
python transfer/programs/record_transfer.py  # With EMG hardware
# OR
python transfer/programs/record_transfer.py --no-emg  # Camera-only test
```

**Saves to:** `transfer/data/transfer_session_YYYYMMDD_HHMMSS.h5`

**Data Structure:**
```
/emg/filtered         [N, 8]      # EMG data @ 500Hz
/pose/mano_theta      [M, 45]     # MANO params from IK
/transfer/joint_angles [N, 20]    # Transfer model predictions
/pose/joints_3d       [M, 21, 3]  # MediaPipe landmarks
```

---

## 🎓 Phase 1 Training (After Recording)

### Option A: Local Training

```bash
python transfer/training/train_phase1.py \
    --data transfer/data/transfer_session_*.h5 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --device cpu  # or mps for Apple Silicon
```

**Expected:**
- Training time: 30-50 epochs to converge
- Target MPJPE: 15-20mm (vs 47mm from-scratch baseline)
- Checkpoints saved to: `transfer/checkpoints/phase1_best.pth`

### Option B: Colab Training (Recommended)

1. **Upload data to Google Drive:**
   ```
   MyDrive/Colab Notebooks/asha/mindrove_recordings/
   ```

2. **Open notebook in Colab:**
   ```
   transfer/training/train_transfer_colab.ipynb
   ```

3. **Run all cells** (everything is inline, no external imports!)

**Notebook includes:**
- ✅ Auto-setup (clones repo, installs deps)
- ✅ Downloads emg2pose checkpoint
- ✅ Mounts Drive and loads your data
- ✅ Defines all modules inline (ChannelAdapter, etc.)
- ✅ Trains Phase 1 (ChannelAdapter only, emg2pose frozen)
- ✅ Plots training curves
- ✅ Saves checkpoints to Drive

---

## 📊 Expected Results

### Phase 1 (After Training)
- **Target MPJPE**: 15-20mm
- **Baseline comparison**: Much better than 47mm from-scratch
- **Trainable parameters**: ChannelAdapter only (~2K params)
- **Frozen parameters**: emg2pose model (~68M params)

### Phase 2 (Future)
- **Target MPJPE**: 10-15mm
- **Dataset**: 70% emg2pose + 30% MindRove (mixed)
- **Training**: Fine-tune all parameters with differential LR
- **Status**: Not implemented yet (Phase 1 first!)

---

## 🐛 Known Issues & Quirks

### Working
- ✅ Pre-trained emg2pose loads correctly (68 parameters)
- ✅ ChannelAdapter architecture implemented
- ✅ FrequencyUpsampler (500Hz→2kHz) working
- ✅ Dataset loaders ready (EMG2PoseDataset, MindRoveDataset)
- ✅ DataRecorder saves dual output (joint angles + MANO params)
- ✅ Colab notebook fully inline (no external imports)
- ✅ Camera feed displays correctly

### Not Working
- ❌ MediaPipe hand detection in record_transfer.py (current blocker)

### Not Tested Yet
- ⏳ Real-time inference (inference_transfer.py)
- ⏳ Phase 1 training script (waiting for recorded data)
- ⏳ Colab notebook (waiting for recorded data)

### Design Decisions
- **RIGHT HAND ONLY**: Using MANO_RIGHT, MediaPipe configured for right hand
- **NO IMU**: Not using IMU data for now (EMG only)
- **VELOCITY-BASED**: Using tracking_vemg2pose.ckpt (predicts velocities, not positions)
- **DUAL OUTPUT**: Saves both joint angles and MANO params for training flexibility

---

## 🔧 File Structure

```
transfer/
├── README.md                          # Project overview
├── SETUP.md                           # Setup guide
├── NEXT_STEPS.md                      # This file
├── training/
│   ├── __init__.py                    # Package marker
│   ├── transfer_modules.py            # ChannelAdapter, FrequencyUpsampler
│   ├── load_emg2pose.py               # ✅ Pre-trained loader (WORKING!)
│   ├── emg2pose_loader.py             # Dataset loaders
│   ├── train_phase1.py                # Phase 1 training script
│   └── train_transfer_colab.ipynb     # Colab notebook (inline)
├── programs/
│   ├── __init__.py                    # Package marker
│   ├── record_transfer.py             # Recording (needs MediaPipe fix)
│   └── inference_transfer.py          # Real-time inference
├── utils/
│   ├── __init__.py                    # Package marker
│   └── data_recorder.py               # Extended recorder
├── emg2pose/                          # Cloned repo (for architecture)
└── checkpoints/                       # Model weights (gitignored)

models/
├── mano/
│   └── MANO_RIGHT.pkl                 # Right hand model
└── emg2pose/
    └── tracking_vemg2pose.ckpt        # Pre-trained checkpoint (236MB)

data/
├── emg2pose/
│   └── emg2pose_dataset_mini/         # Mini dataset (30 files)
└── transfer_session_*.h5              # Your recordings (to be created)
```

---

## 💬 Prompt for Next Claude Session

```
Hi! I'm continuing the transfer learning implementation for Project Asha.

CURRENT ISSUE: MediaPipe hand detection not working in transfer/programs/record_transfer.py

WHAT'S DONE:
- ✅ Complete transfer learning implementation (modules, loaders, training scripts)
- ✅ Pre-trained emg2pose model loads successfully (68 parameters)
- ✅ Colab notebook fully inline and ready
- ✅ Camera feed displays correctly

WHAT'S BROKEN:
- ❌ MediaPipe always returns None (landmarks=False, world=False)
- ❌ Hand detection doesn't work even when hand clearly visible

DEBUGGING STEPS TO TRY:
1. Check MediaPipe configuration in src/model_utils/tracker.py (confidence thresholds?)
2. Test MediaPipe directly with minimal script (does it work standalone?)
3. Compare with working src/programs/record.py (what's different?)
4. Verify camera permissions and OpenCV version

FILES TO CHECK:
- transfer/programs/record_transfer.py (VideoThread.run() method)
- src/model_utils/tracker.py (MediaPipe configuration)
- transfer/NEXT_STEPS.md (full context and debugging steps)

ONCE FIXED:
1. Record 15-20 minutes of data (see recording protocol in NEXT_STEPS.md)
2. Train Phase 1 using Colab notebook (transfer/training/train_transfer_colab.ipynb)
3. Target: 15-20mm MPJPE (vs 47mm baseline)

Please help debug the MediaPipe issue and get the recording program working!
```

---

## 📚 Key References

- **emg2pose paper**: https://research.facebook.com/publications/unsupervised-learning-for-precise-hand-pose-estimation/
- **emg2pose repo**: https://github.com/facebookresearch/emg2pose
- **MANO**: https://mano.is.tue.mpg.de
- **MediaPipe Hands**: https://google.github.io/mediapipe/solutions/hands

---

**Last Updated**: December 13, 2025
**Session Duration**: ~3 hours
**Status**: Implementation complete, debugging in progress
**Next Session**: Fix MediaPipe detection → Record data → Train Phase 1
