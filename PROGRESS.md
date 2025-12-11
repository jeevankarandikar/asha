# Project Asha - Progress Tracker

**Last Updated**: December 10, 2025
**Current Focus**: Phase 0 - Image → MANO θ Training (v2.5)

---

## Project Vision

Build a **camera-free hand tracking system** using EMG signals for prosthetic control and gesture recognition, achieving real-time 3D hand pose estimation without visual input.

**End Goal**: Real-time EMG-only hand tracking @ <10ms latency, 25fps

---

## Pipeline Overview

```
Phase 0: Image → θ (v2.5)     ← WE ARE HERE
    ↓ (generates pseudo-labels)
Phase 1: Data Collection (v2)  ✅ COMPLETE
    ↓ (synchronized EMG + video)
Phase 2: Pseudo-Labeling (v2.5 → v3 bridge)
    ↓ (apply Phase 0 model to videos)
Phase 3: EMG → θ Training (v3)
    ↓ (learn from pseudo-labeled data)
Phase 4: Camera-Free Inference (v4)
    ✓ Real-time hand tracking from EMG only!
```

---

## Phase 0: Image → MANO θ (v2.5 - IN PROGRESS)

### Purpose
Train a neural network to predict MANO θ parameters directly from RGB images, creating a "cheap motion capture" system to generate ground truth for EMG training.

### Why This Matters
- **Pseudo-labeling**: This model will label our EMG recordings (no $100K mocap needed!)
- **Quality requirement**: Must achieve 10-12mm MPJPE for reliable EMG training
- **Current blocker**: Model plateaued at 47mm (not good enough for pseudo-labels)

### Model Architecture Status

**Current State: transformer_v2 (TRAINED BUT INADEQUATE)**

```python
Architecture: ResNet-50 backbone + Transformer encoder
  - 4 layers, 8 heads, d_model=256
  - Spatial attention on 7×7 feature patches (49 patches)
  - Global pooling → MLP head → MANO θ [45]

Hyperparameters:
  - lambda_joint=25.0  ← TOO HIGH (suppresses parameter learning)
  - lambda_param=0.5
  - lr=7.5e-5  ← TOO LOW (stuck in local minimum)
  - Cosine LR schedule decaying from epoch 1

Training Results:
  ✗ MPJPE: 47.40mm (same as ResNet - no improvement!)
  ✗ Training loss: plateaued after epoch 1
  ✗ Root cause: Loss imbalance, insufficient learning rate

Status: ❌ NOT GOOD ENOUGH FOR PSEUDO-LABELS
```

**Next Attempt: transformer_v3 (ARCHITECTURE READY, NOT TRAINED YET)**

```python
Architecture: Same transformer base, ENHANCED components

Loss Function Improvements (losses.py - ✅ IMPLEMENTED):
  ✓ Rebalanced MANOLoss: lambda_joint=5.0 (reduced from 25)
  ✓ Angular geodesic loss: lambda_angular=1.0 (captures twists)
  ✓ Per-joint weights: proximal 2.0, distal 1.0, pinky/thumb 1.5×
  ✓ Kinematic priors: soft tanh clipping + PCA pose prior
  ✓ NaN handling: gradient-preserving penalties

Training Strategy Enhancements (ready to integrate):
  ✓ Data augmentation: CutMix occlusions (p=0.3), rotations ±30°, scaling
  ✓ Cosine LR scheduler: Restarts every 20 epochs (NOT decaying from epoch 1)
  ✓ Higher initial LR: 1e-4 → 1e-6 (better escape from minima)
  ✓ Curriculum learning: Easy poses → complex → multi-dataset
  ✓ Multi-task losses: 2D keypoint heatmaps (weight 0.1)
  ✓ Self-supervised pretraining: HandMIM (mask 40% patches)

Evaluation Metrics (evaluate.py - ✅ IMPLEMENTED):
  ✓ Diversity: θ variance (>0.5), entropy (>2.0 bits)
  ✓ Gesture F1: 5-class classifier (fist/open/point/pinch/rest, >0.6)
  ✓ Per-dataset MPJPE: FreiHAND/HO-3D/DexYCB breakdown

Expected Performance:
  → FreiHAND: 10-12mm MPJPE (vs current 47mm)
  → HO-3D occlusions: 15-20mm
  → DexYCB grasping: 12-15mm
  → Diversity: θ_std >0.5, entropy >2.0, gesture F1 >0.8

Status: 🔄 READY TO TRAIN
```

### Research Synthesis Alignment

Based on your synthesis, transformer_v3 incorporates:

**From Gemini/ChatGPT/Grok:**
- ✅ Hybrid approach: ResNet backbone + Transformer (efficient local + global)
- ✅ Loss rebalancing: Reduced λ_joint from 25→5 to prevent suppression
- ✅ Kinematic priors: Tanh clipping, PCA Gaussian prior
- ✅ Curriculum learning: Simple FreiHAND → full → HO-3D/DexYCB
- ✅ Diversity metrics: Entropy, variance, gesture F1
- ✅ Multi-task: 2D keypoints for attention guidance

**Recommended Next:**
- 🔄 Self-supervised pretraining: HandMIM (mask 40% patches, 20 epochs)
- 🔄 GCN fusion: Add HandGCNFormer-style kinematic graphs (10-15% MPJPE reduction)
- ⏳ Diffusion refiner: HandDiff-inspired 20-step denoising (Phase 5)

### Training Plan

**Immediate (1-3 days):**
1. ✅ Verify architecture files compatible (DONE - see TENSOR_COMPATIBILITY_VERIFICATION.md)
2. 🔄 Integrate enhanced components into train_colab.ipynb
   - Import enhanced MANOLoss from losses.py
   - Add augmentations (CutMix, rotations ±30°, scaling)
   - Update scheduler (cosine with restarts, NOT decay from epoch 1)
   - Add diversity metrics to evaluation
3. 🔄 Train transformer_v3 on FreiHAND (100 epochs)
   - Target: <30mm MPJPE by epoch 30, <15mm by epoch 100
   - Monitor: Loss components, θ variance, gesture F1

**Short-term (3-7 days):**
4. ⏳ Add HO-3D dataset (occlusions)
   - Download HO-3D, integrate via dataset_loader.py
   - Mix 70/30 FreiHAND/HO-3D, fine-tune 20 epochs
5. ⏳ Add DexYCB subset (grasping diversity)
   - Sample 100K frames, mix 50/25/25
   - Target: High gesture diversity (F1 >0.8)

**Medium-term (1-2 weeks):**
6. ⏳ Self-supervised pretraining (HandMIM)
   - Mask 40% patches, reconstruct, 20 epochs
   - Expected: +5-10% MPJPE improvement
7. ⏳ Diffusion refiner (optional)
   - HandDiff-inspired 20-step denoising
   - Target: 8mm on occlusion scenarios

### Success Criteria

**Minimum viable (proceed to Phase 2):**
- ✓ MPJPE: 15-20mm on FreiHAND (acceptable for pseudo-labels)
- ✓ Diversity: θ_std >0.5, entropy >1.5, gesture F1 >0.6
- ✓ Robustness: <25mm on HO-3D occlusions

**Target (high-quality pseudo-labels):**
- ✓ MPJPE: 10-12mm on FreiHAND (HandFormer-level)
- ✓ Diversity: θ_std >0.7, entropy >2.0, gesture F1 >0.8
- ✓ Robustness: <20mm on HO-3D, <15mm on DexYCB

**Current Status: 47mm MPJPE - NOT VIABLE for pseudo-labeling**

---

## Phase 1: Data Collection (v2 - ✅ COMPLETE)

### Purpose
Record synchronized EMG signals + video of hand gestures to build training dataset for EMG→θ model.

### Status: INFRASTRUCTURE COMPLETE

**System: main_v2.py** ✅
```python
Dual-thread recording:
  - VideoThread: 25fps camera → MediaPipe → MANO IK → θ
  - EMGThread: 500Hz MindRove device → bandpass/notch filter
  - Synchronized: time.perf_counter() timestamps
  - Output: HDF5 format (EMG + pose + metadata)

Recording Protocols:
  - Session 1: Basic poses (3 min)
  - Session 2: Dynamic gestures (5 min)
  - Session 3: Continuous (10 min)
  - Session 4: Fast movements (5 min)
  - Session 5: Occlusions (3 min)

Data Format:
  /emg/raw: [N, 8] @ 500Hz
  /emg/filtered: [N, 8] @ 500Hz
  /emg/timestamps: [N]
  /pose/theta: [M, 45] @ 25fps
  /pose/joints: [M, 21, 3] @ 25fps
  /pose/timestamps: [M]
  /metadata: session_type, duration, etc.
```

**Quality Indicators:**
- ✅ IK error: 9.71mm mean (validated)
- ✅ MediaPipe confidence: >0.9 required
- ✅ EMG signal quality: Bandpass 10-200Hz, notch 60Hz
- ✅ Temporal alignment: <5ms sync error

**Current Data:**
- 🔄 Test recordings exist, production data pending
- ⏳ Target: 10-30 min per session, 5+ sessions
- ⏳ Diversity: All 5 protocols, multiple days

### Next Steps

**After Phase 0 succeeds (10-12mm model):**
1. ⏳ Clear test data, start production recordings
2. ⏳ Record 5 sessions using structured protocols
3. ⏳ Verify quality: IK error <15mm, MP confidence >0.9
4. ⏳ Backup to external storage (HDF5 files ~1GB/session)

---

## Phase 2: Pseudo-Labeling (v2.5 → v3 BRIDGE)

### Purpose
Apply the trained image→θ model (Phase 0) to recorded videos (Phase 1) to generate ground truth θ for EMG training.

### Status: BLOCKED (waiting on Phase 0)

**Process:**
```python
For each HDF5 recording:
  1. Load video frames [M, H, W, 3]
  2. Run transformer_v3 model: Image → θ_pseudo
  3. Align EMG windows (500ms) with video frames (40ms)
  4. Create training pairs: (emg_window [250, 8], θ_pseudo [45])
  5. Save to emg_training_data.h5

Quality checks:
  - Verify θ_pseudo smoothness (no jumps >0.5 rad)
  - Compare with IK-based θ (should be similar on simple poses)
  - Visual inspection: Render MANO mesh, check realism
```

**Expected Output:**
```
emg_training_data.h5:
  /train/emg: [N_train, 250, 8]      # 500ms windows @ 500Hz
  /train/theta: [N_train, 45]        # Pseudo-labels from transformer_v3
  /train/timestamps: [N_train]

  /val/emg: [N_val, 250, 8]
  /val/theta: [N_val, 45]
  /val/timestamps: [N_val]

  /metadata: train/val split, session IDs, quality metrics
```

**Quality Requirements:**
- Pseudo-label MPJPE: <15mm (verified on held-out test set with IK)
- Diversity: Multiple gesture types represented
- Coverage: 10,000+ training pairs (5 sessions × 3 min × 25fps × 80% train = 18,000)

### Next Steps

**When Phase 0 reaches 10-12mm:**
1. ⏳ Load transformer_v3 checkpoint
2. ⏳ Process all recorded sessions
3. ⏳ Generate pseudo-labels with quality checks
4. ⏳ Create train/val split (80/20)
5. ⏳ Save to emg_training_data.h5

---

## Phase 3: EMG → θ Training (v3 - ARCHITECTURE READY)

### Purpose
Train neural network to predict MANO θ from EMG signals alone, enabling camera-free hand tracking.

### Status: ARCHITECTURE COMPLETE, BLOCKED ON DATA

**Model: EMG2PoseModel** ✅
```python
Architecture (emg_model.py - IMPLEMENTED):
  ✓ TDS Featurizer: EMG [B, 500, 8] → features [B, 50, 16]
      - 10× downsampling (500Hz → 50Hz)
      - 4 TDS blocks (depthwise separable convolutions)
  ✓ 2-layer LSTM: [B, 50, 16] → [B, 50, 512]
      - Temporal modeling (remember past muscle activity)
  ✓ Velocity Prediction: [B, 512] → Δθ [B, 45]
      - Predict change in pose (smoother than absolute)
  ✓ Integration: θ_t = θ_{t-1} + Δθ
      - Accumulate changes over time
  ✓ Optional IMU Fusion: [B, 500, 18] → [B, 50, 16] → concat
      - Ready for v5 (wrist orientation)

Preprocessing (EMGPreprocessor - IMPLEMENTED):
  ✓ Bandpass filter: 20-450Hz (Butterworth order 2)
  ✓ Notch filter: 60Hz (power line noise)
  ✓ MVC normalization: Per-channel (baseline + max voluntary contraction)

Training Strategy (based on emg2pose):
  ⏳ Window size: 500ms (250 samples @ 500Hz)
  ⏳ Batch size: 32-64
  ⏳ Loss: MSE(θ_pred, θ_pseudo) + velocity regularization
  ⏳ Optimizer: Adam, lr=1e-3 → 1e-5 (cosine schedule)
  ⏳ Augmentation: Time warping, noise injection
  ⏳ Evaluation: MPJPE, gesture F1, jitter (acceleration variance)
```

**Research Alignment:**
- ✅ TDS conv + LSTM: Consensus recommendation (Gemini/ChatGPT/Grok)
- ✅ Velocity prediction: From emg2pose (smoother than absolute θ)
- ✅ IMU fusion ready: Per Zeng et al. 2025 (+20% accuracy)
- ⏳ Multi-user generalization: Pretrain on emg2pose dataset (193 users)

### Training Plan

**When Phase 2 completes (pseudo-labeled data ready):**

**Stage 1: Single-user baseline (1 week)**
1. ⏳ Train on own EMG data (10K+ pairs)
2. ⏳ Target: MPJPE <30mm, gesture F1 >0.7
3. ⏳ Evaluate: Per-gesture accuracy, temporal smoothness

**Stage 2: Multi-user generalization (2-3 weeks)**
4. ⏳ Download emg2pose dataset (370 hours, 193 users)
5. ⏳ Pretrain on emg2pose, fine-tune on own data
6. ⏳ Target: MPJPE <25mm, cross-user robustness

**Stage 3: IMU fusion (optional, 1 week)**
7. ⏳ Integrate MindRove Snaptic kit (2 IMUs @ 100Hz)
8. ⏳ Train with EMG+IMU features
9. ⏳ Target: MPJPE <20mm, better wrist orientation

### Success Criteria

**Minimum viable (proceed to Phase 4):**
- ✓ MPJPE: <30mm on validation set
- ✓ Gesture F1: >0.7 (5 classes)
- ✓ Real-time capable: <10ms inference on GPU
- ✓ Temporal smoothness: Low jitter

**Target (production-ready):**
- ✓ MPJPE: <20mm (competitive with vision-based)
- ✓ Gesture F1: >0.85
- ✓ Cross-session robust: <5% degradation over days
- ✓ Low latency: <5ms inference

---

## Phase 4: Camera-Free Inference (v4 - FUTURE)

### Purpose
Deploy real-time hand tracking using EMG signals only (no camera required).

### Status: PLANNED

**System Architecture:**
```python
Runtime Pipeline:
  MindRove device → EMG stream @ 500Hz
      ↓
  Preprocessing: bandpass/notch/MVC norm
      ↓
  Sliding window buffer: 500ms (250 samples)
      ↓
  EMG2PoseModel inference: <5ms
      ↓
  MANO forward pass: θ → 3D mesh
      ↓
  Render: PyRender/OpenGL @ 25fps
```

**Optimizations:**
- ⏳ Model quantization: FP32 → INT8 (4× faster, <2% accuracy loss)
- ⏳ Batch predictions: Process multiple windows in parallel
- ⏳ GPU offloading: MPS (Apple Silicon) or CUDA
- ⏳ Temporal smoothing: Kalman filter on θ predictions

**Target Performance:**
- Latency: <10ms end-to-end
- Throughput: 25fps minimum
- Accuracy: Maintain Phase 3 MPJPE (<20mm)
- Robustness: Handle sensor drift, muscle fatigue

**Applications:**
- Prosthetic control (real-time hand pose → motor commands)
- VR/AR input (gesture recognition without cameras)
- Accessibility (control devices via hand gestures)
- Medical (track hand function in rehabilitation)

---

## Phase 5: Advanced Enhancements (OPTIONAL)

### Multi-Sensor Fusion (v5)

**Hardware: MindRove Snaptic Kit**
- 2× 9-DOF IMUs @ 100Hz (accel + gyro + mag)
- 1× PPG sensor (blood flow/pulse)

**Fusion Strategy:**
```python
IMU 1 (forearm): Arm orientation
IMU 2 (hand): Wrist roll/pitch/yaw
PPG: Muscle activation timing (blood flow → muscle contraction)

Fusion: Concatenate features → EMG2PoseModel
  EMG: [B, 500, 8]
  IMU: [B, 100, 18]  # 2 IMUs × 9-DOF, interpolate to 500Hz
  → Concat: [B, 500, 26]
  → TDS/LSTM as before
```

**Expected Gains:**
- +10-20% gesture classification accuracy
- Better wrist orientation tracking
- Reduced EMG-only ambiguity (e.g., similar muscle patterns for different wrist angles)

### Diffusion Refinement

**Inspired by HandDiff (Cheng et al. 2024)**
- 20-step denoising on predicted joints
- Conditioned on image features (for v2.5) or EMG features (for v3)
- Target: 8mm MPJPE on occlusion scenarios

**Two-stage training:**
1. Train base model (transformer_v3 or EMG2Pose)
2. Train diffusion refiner on residual errors
3. Inference: Base prediction → diffusion denoising → final pose

### Self-Supervised Pretraining

**HandMIM (Masked Image Modeling)**
- Mask 40% of image patches
- Reconstruct with pose tokens
- Pretrain 20 epochs on FreiHAND
- Expected: +5-10% MPJPE improvement

---

## Current Blockers & Solutions

### Blocker 1: Phase 0 Plateaued at 47mm ❌

**Problem:** transformer_v2 stuck at 47mm MPJPE (not viable for pseudo-labels)

**Root Causes:**
- Loss imbalance: lambda_joint=25 suppresses parameter learning
- Learning rate too low: 7.5e-5, stuck in local minimum
- LR schedule wrong: Cosine decay from epoch 1 (should restart every 20 epochs)
- Insufficient augmentation: No occlusions/CutMix
- No diversity monitoring: Model may predict repetitive poses

**Solutions (transformer_v3):** ✅ IMPLEMENTED
- ✅ Rebalanced loss: lambda_joint=5, lambda_param=0.5
- ✅ Angular geodesic loss: lambda_angular=1.0
- ✅ Kinematic priors: tanh clipping, PCA prior
- ✅ Higher LR: 1e-4 initial, cosine restarts every 20 epochs
- ✅ Stronger augmentation: CutMix, rotations ±30°, scaling
- ✅ Diversity metrics: θ variance, entropy, gesture F1

**Next Action:** 🔄 TRAIN transformer_v3 now!

### Blocker 2: No Production EMG Data ⏳

**Problem:** Only test recordings exist, need production data for Phase 3

**Solution:** After Phase 0 succeeds (10-12mm model):
1. Clear test data
2. Record 5 sessions (25-50 minutes total)
3. Apply structured protocols (basic/dynamic/continuous/fast/occlusions)
4. Verify quality (IK error <15mm, MP confidence >0.9)

**Timeline:** 1-2 weeks after Phase 0 complete

### Blocker 3: No Multi-Dataset Training ⏳

**Problem:** Only using FreiHAND (lacks occlusions/interactions/diversity)

**Solution:** Integrate HO-3D and DexYCB
- ✅ Dataset loaders implemented (dataset_loader.py)
- ⏳ Download datasets (~119GB DexYCB, ~10GB HO-3D)
- ⏳ Implement curriculum mixing (70/30 → 50/25/25)

**Timeline:** 3-7 days (can overlap with Phase 0 training)

---

## Key Metrics Dashboard

### Phase 0 (Image → θ)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **FreiHAND MPJPE** | 47.40mm | 10-12mm | ❌ CRITICAL |
| **HO-3D MPJPE** | Not tested | 15-20mm | ⏳ |
| **DexYCB MPJPE** | Not tested | 12-15mm | ⏳ |
| **θ variance** | 0.16 (low) | >0.5 | ❌ |
| **Entropy** | Not measured | >2.0 bits | ⏳ |
| **Gesture F1** | Not measured | >0.8 | ⏳ |

### Phase 1 (Data Collection)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Session count** | Test only | 5+ prod | ⏳ |
| **Total duration** | ~10 min test | 25-50 min | ⏳ |
| **IK quality** | 9.71mm | <15mm | ✅ |
| **MP confidence** | >0.9 | >0.9 | ✅ |
| **Training pairs** | 0 | 10,000+ | ⏳ |

### Phase 3 (EMG → θ)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Architecture** | Complete | - | ✅ |
| **MPJPE** | Not trained | <20mm | ⏳ |
| **Gesture F1** | Not trained | >0.85 | ⏳ |
| **Inference latency** | Not measured | <10ms | ⏳ |

---

## Timeline Estimate

**Optimistic (everything works first try):**
- Phase 0: 1-2 weeks (train transformer_v3, achieve 10-12mm)
- Phase 1-2: 1 week (record + pseudo-label)
- Phase 3: 2-3 weeks (train EMG model)
- Phase 4: 1 week (deploy + optimize)
- **Total: 5-7 weeks**

**Realistic (some iterations needed):**
- Phase 0: 3-4 weeks (multiple training runs, dataset integration)
- Phase 1-2: 2 weeks (quality issues, re-recording)
- Phase 3: 4-5 weeks (architecture tuning, multi-user)
- Phase 4: 2 weeks (optimization, testing)
- **Total: 11-15 weeks (~3 months)**

**Conservative (major roadblocks):**
- Phase 0: 6-8 weeks (may need diffusion/GCN additions)
- Phase 1-2: 3 weeks (data quality issues)
- Phase 3: 6-8 weeks (emg2pose pretraining, IMU fusion)
- Phase 4: 3 weeks (extensive optimization)
- **Total: 18-22 weeks (~5 months)**

**Critical Path:** Phase 0 must succeed for everything else to work!

---

## Research Synthesis Integration

Based on synthesized review from Gemini/ChatGPT/Grok (December 2024):

### Key Recommendations Implemented ✅

1. **Loss Rebalancing**: λ_joint reduced from 25→5, added angular loss
2. **Kinematic Priors**: Tanh clipping, PCA Gaussian prior
3. **Diversity Metrics**: θ variance, entropy, gesture F1
4. **Dataset Integration**: HO-3D/DexYCB loaders ready
5. **EMG Architecture**: TDS+LSTM with velocity prediction (emg2pose-aligned)
6. **Multi-task Learning**: 2D keypoint auxiliary head ready

### Recommendations In Progress 🔄

7. **Curriculum Learning**: Dataset mixing strategy implemented, needs training
8. **Self-Supervised Pretraining**: HandMIM approach documented, not implemented
9. **GCN Fusion**: HandGCNFormer topology awareness (Phase 5)
10. **Diffusion Refinement**: HandDiff-inspired denoising (Phase 5)

### Gaps to Address ⏳

11. **emg2pose Pretraining**: Download dataset, pretrain EMG model (Phase 3)
12. **IMU Fusion**: Integrate Snaptic kit (Phase 5)
13. **Hybrid Efficiency**: Consider FastMETRO/FastViT for real-time (Phase 4)
14. **Temporal Consistency**: Add PoseFormer-style sequence modeling (Phase 4)

---

## Success Criteria (Overall Project)

### Minimum Viable Product (MVP)

- ✅ v1: MANO IK baseline (9.71mm) - COMPLETE
- ✅ v2: EMG recording infrastructure - COMPLETE
- ⏳ v2.5: Image→θ model (<20mm MPJPE)
- ⏳ v3: EMG→θ model (<30mm MPJPE, F1 >0.7)
- ⏳ v4: Real-time inference (<25fps, <10ms latency)

### Production-Ready System

- ✓ Image→θ: 10-12mm MPJPE (HandFormer-competitive)
- ✓ EMG→θ: 20mm MPJPE (vision-competitive, camera-free!)
- ✓ Gesture classification: F1 >0.85 (5 classes)
- ✓ Real-time: 25fps, <5ms latency
- ✓ Robustness: Cross-session, handles occlusions
- ✓ Diversity: High θ variance, entropy >2.0

### Research Contributions

- Novel: Low-cost EMG→pose without mocap (pseudo-labeling approach)
- Validation: Competitive with vision-based methods (10-20mm MPJPE)
- Practical: Real-time, affordable ($50 webcam + $200 EMG device vs $100K+ mocap)
- Generalizable: Multi-user (pretrain on emg2pose), multi-sensor (EMG+IMU)

---

## Documentation Index

**Architecture Verification:**
- `docs/TENSOR_COMPATIBILITY_VERIFICATION.md` - Complete tensor shape analysis
- `docs/IMPLEMENTATION_VERIFICATION_SUMMARY.md` - Executive summary
- `src/tests/test_tensor_compatibility.py` - Automated test suite

**Training Resources:**
- `src/training/train_colab.ipynb` - Complete training pipeline (Colab-ready)
- `src/training/losses.py` - Enhanced MANOLoss with all improvements
- `src/training/emg_model.py` - EMG2PoseModel architecture
- `src/training/dataset_loader.py` - Multi-dataset loaders with curriculum

**Integration Instructions:**
- `docs/archive/NOTEBOOK_UPDATES_20251210.md` - Step-by-step Colab integration
- `docs/archive/INTEGRATION_STATUS_20251210.md` - Component status tracking

**Project Documentation:**
- `CLAUDE.md` - Complete project guide (COMPACT VERSION)
- `docs/archive/CLAUDE_FULL_20251210.md` - Full archived documentation
- `README.md` - User-facing overview

---

## Next Immediate Actions (Priority Order)

**TODAY (December 10, 2025):**
1. 🔥 **TRAIN transformer_v3**
   - Open `src/training/train_colab.ipynb` in Google Colab
   - Verify FreiHAND dataset in Google Drive
   - Integrate enhanced losses from losses.py
   - Start training (100 epochs, ~20-30 hours on A100)
   - Monitor: Total loss, θ variance, gesture F1

**THIS WEEK:**
2. Monitor training progress
   - Check TensorBoard every 6 hours
   - Verify loss decreasing (not plateauing like v2)
   - Watch for NaN issues (should be fixed)
3. Download HO-3D dataset (~10GB)
   - Start background download during training
4. Document training observations
   - Note when loss starts decreasing
   - Track diversity metrics evolution

**NEXT WEEK:**
5. Evaluate transformer_v3 checkpoint
   - Compute MPJPE, diversity metrics
   - Visual inspection (render predictions)
   - Decide: Good enough (10-15mm) or retrain?
6. If good: Proceed to Phase 2 (pseudo-labeling)
7. If not: Iterate (add HO-3D, adjust hyperparameters)

---

**Status Summary**: Phase 0 is the current blocker. All architecture is ready and verified. Training transformer_v3 is the critical next step to unblock the entire pipeline.

**Estimated time to camera-free tracking**: 3-5 months (optimistic: 2 months, conservative: 6 months)

**Key insight**: We're bootstrapping from expensive ground truth (FreiHAND mocap) → cheap ground truth (webcam pseudo-labels) → no visual input (EMG only). The quality of Phase 0 directly determines success of Phase 3!
