# CLAUDE.md (ARCHIVED - December 10, 2025)

**NOTE**: This is an archived version of CLAUDE.md for historical reference. File names have been updated in the current codebase:
- `main_v2.py` → `record.py`
- `mano_v1.py` → `mano.py`
- `mediapipe_v0.py` → `mediapipe.py`
- Test files updated accordingly

Refer to `/CLAUDE.md` for current documentation.

---

This file provides complete guidance to Claude Code (claude.ai/code) when working with code in this repository. Everything you need to know about the project is documented here.

**IMPORTANT**: Do NOT create new .md files. All documentation should go in this file (CLAUDE.md) or existing README files.

## Project Overview

**Project Asha** is a real-time EMG-to-hand-pose system for prosthetic control and gesture recognition. The ultimate goal is to predict 3D hand poses from surface electromyography (EMG) muscle signals, enabling camera-free hand tracking.

### Research Tools Used

- **Consensus**: Used to find and review research papers and references that inspired the project architecture and methodology. Consensus helped identify key papers like HandFormer, emg2pose, NeuroPose, and other state-of-the-art approaches for hand pose estimation and EMG-based tracking.

### Current Phase: Image → MANO θ Training (v2.5 - In Progress)

**What's Working:**
1. Webcam → MediaPipe → 21 joint positions (v0)
2. MANO IK → 45 joint angle parameters θ (v1)
3. EMG integration + data recording (v2)
4. **Experimental validation complete** - 4 experiments on 5,330 frames
5. **Image → θ training pipeline** - Complete Colab notebook, first training run completed

**Key Results (v1 IK System):**
- **9.71mm mean error** (competitive with HandFormer: 10.92mm)
- **25fps real-time performance** (5× faster than transformers)
- **99.7% convergence rate** (only 16 failures in 5,330 frames)

**Training Progress (v2.5):**
- **resnet_v1**: 47mm PA-MPJPE (model collapse detected - low variance predictions)
  - Used SimpleMANOLoss (parameter-only loss)
  - Issue: Model learned to predict mean θ values (lowest parameter loss)
  - Result: All predictions collapsed to same pose, model didn't learn diverse hand shapes
- **resnet_v2**: Fixed version with proper MANOLoss (parameter + joint position loss)
  - Forces model to match actual joint positions, not just parameter values
  - Result: 47.44mm PA-MPJPE, loss plateaued early (epoch 22-25)
  - Issue: Loss plateauing suggests learning rate too low or architecture limitations
  - Conclusion: ResNet architecture may plateau around 30-40mm, insufficient for target 15-20mm
- **transformer_v1**: Switched to Transformer architecture (HandFormer-style)
  - Architecture: ResNet-50 backbone + Transformer encoder (4 layers, 8 heads, d_model=256)
  - How it works: ResNet extracts features → spatial patches (49 patches from 7×7 feature map) → self-attention (each patch attends to all others) → global pooling → MLP head → MANO θ
  - Key difference: Attention mechanisms allow model to explicitly model relationships between hand parts (e.g., finger tip can 'see' palm, index finger can 'see' middle finger)
  - Why transformer helps: Better capacity for complex relationships, better occlusion handling, better accuracy
  - Hyperparameters: lambda_joint=20.0, lambda_param=0.5, predict_beta=False, lr=5e-5, gradient_clip=1.0
  - Training: 100 epochs, gradient clipping enabled, weight_decay=1e-4
  - Expected: 12-15mm MPJPE (based on HandFormer achieving 12.33mm on FreiHAND)
  - Status: Training completed (epochs 1-5), stable but plateauing
  - **NaN Issue (Fixed)**: Training encountered NaN losses starting at epoch 6 in initial run. Root cause: Extreme MANO parameters (pred_theta/pred_beta) causing NaN in MANO forward pass. Fixes applied:
    - Parameter clipping: Clamp pred_theta to [-3.14, 3.14] and pred_beta to [-3.0, 3.0] before MANO forward pass
    - NaN detection: Check for NaN/inf in loss computation and MANO output, skip bad batches with warnings
    - Reduced learning rate: Changed from 1e-4 to 5e-5 for more stable training
    - Loss safeguards: Replace NaN losses with large finite penalty (1000.0) to prevent training crash
  - **Training Progress (Epochs 1-5)**:
    - Total loss: ~2.43 (train), ~2.42 (val) - stable, minimal decrease
    - Joint loss: ~0.114 → **~11.4mm MPJPE** (promising! within target 12-15mm range)
    - Param loss: ~0.28, Reg loss: ~0.07
    - Training stable (no NaN), best model at epoch 3 (val_loss: 2.4232)
    - Loss not decreasing much yet (similar to resnet_v2 pattern), but joint loss is much better
- **transformer_v2**: Increased learning rate and loss weights to address plateauing
  - Architecture: Same as transformer_v1 (ResNet-50 + Transformer encoder)
  - Hyperparameters: lambda_joint=25.0 (increased from 20.0), lambda_param=0.5, predict_beta=False, lr=7.5e-5 (increased from 5e-5), gradient_clip=1.5 (increased from 1.0)
  - Training: 100 epochs, cosine learning rate schedule
  - Status: Training completed (6 epochs), **CRITICAL ISSUE - NOT IMPROVING**
  - **Evaluation Results**: MPJPE: 47.40mm (same as resnet_v2: 47.44mm) - **NO IMPROVEMENT**
  - **Root Cause**: Loss imbalance (lambda_joint=25 dominates, suppresses learning), global pose misalignment (training MSE vs eval PA-MPJPE discrepancy), insufficient diversity metrics

- **transformer_v3** (proposed - based on synthesized review): Major architecture and training improvements
  - **Loss Function Refinements** (losses.py updates):
    - **Rebalanced MANOLoss**: lambda_joint=5-10 (reduced from 25), lambda_param=0.5, lambda_reg=0.001
      - Rationale: High joint weight (25) suppresses parameter learning, causes imbalance
    - **Angular geodesic loss**: Add quaternion-based loss for rotation accuracy (latent DoFs)
      - `loss_rot = geodesic_distance(pred_rot_quat, gt_rot_quat)` on per-joint rotations
      - Weight: lambda_rot=2.0 (explicit rotation supervision)
    - **Kinematic priors integration**:
      - Tanh clipping: `θ_clipped = 3.14 * tanh(θ_raw / 3.14)` (soft joint limits)
      - PCA pose prior: Gaussian prior on MANO pose space (fit PCA on training θ, penalize deviation)
      - `loss_prior = ||PCA_encode(θ) - μ||²` (weight: lambda_prior=0.1)
  - **Diffusion refinement** (optional, inspired by HandDiff):
    - 20-step denoising on predicted joints, conditioned on image features
    - Denoising model: small U-Net (4 layers, 64 channels)
    - Target: 8mm MPJPE on DexYCB-like occlusion scenarios
    - Training: Two-stage (transformer first, then diffusion refiner)
    - Implementation: Add `diffusion_refiner.py` stub in src/training/
  - **Training Strategy Enhancements** (train_colab.ipynb):
    - **Data augmentation**: CutMix occlusions (p=0.3), rotation/scaling (±15°, ±10%)
    - **Curriculum learning**: Start with FreiHAND easy subsets (open hands), gradually add complex poses, then HO-3D/DexYCB
      - Epoch 0-20: FreiHAND simple (50K samples)
      - Epoch 20-50: FreiHAND full (130K) + HO-3D (20K occlusions)
      - Epoch 50-100: Mixed FreiHAND + HO-3D + DexYCB (20K grasping)
    - **Multi-task auxiliary losses**: 2D keypoint projection (weight=0.1), hand segmentation mask (weight=0.05)
      - Improves feature learning, regularizes overfitting
    - **Self-supervised pretraining** (HandMIM): Mask 40% image patches, reconstruct with pose tokens
      - Pretrain 20 epochs before supervised fine-tuning
      - Improves robustness to occlusions
  - **Evaluation Metrics** (evaluate.py additions):
    - **Diversity metrics**: θ variance (target >0.5), prediction entropy (target >2.0)
    - **Gesture classification**: Train 5-10 gesture classifier on predictions, report F1 score (target >0.8)
    - **Per-dataset breakdown**: Separate MPJPE for FreiHAND/HO-3D/DexYCB/custom
  - **Expected Performance**:
    - FreiHAND: 10-12mm MPJPE (vs current 47mm)
    - HO-3D occlusions: 15-20mm MPJPE
    - DexYCB grasping: 12-15mm MPJPE
    - Diversity: θ_std > 0.5, entropy > 2.0, gesture F1 > 0.85
  - **Status**: 🔄 Architecture designed, ready for implementation

**Next phase (v3):** Train EMG → θ regression model
**Final phase (v4):** Real-time camera-free hand tracking

**System Comparison: mediapipe_v0 vs mano_v1 vs resnet_v2 vs transformer_v1 vs transformer_v2**

| Feature | **mediapipe_v0** | **mano_v1** | **resnet_v2** | **transformer_v1** | **transformer_v2** |
|---------|------------------|-------------|---------------|---------------------|---------------------|
| **System Type** | Baseline (CV) | IK Optimization | Learned (CV) | Learned (CV) | Learned (CV) |
| **Method** | MediaPipe only | MediaPipe + MANO IK | ResNet + MLP | ResNet + Transformer | ResNet + Transformer |
| **Input** | RGB Image | RGB Image | RGB Image | RGB Image | RGB Image |
| **Output** | 21 keypoints | MANO θ (45 params) | MANO θ (45 params) | MANO θ (45 params) | MANO θ (45 params) |
| **3D Model** | None | MANO mesh (778 verts) | MANO mesh (778 verts) | MANO mesh (778 verts) | MANO mesh (778 verts) |
| **Articulation** | None | Full (IK optimized) | Learned | Learned | Learned |
| **Training Required** | No | No | Yes (FreiHAND) | Yes (FreiHAND) | Yes (FreiHAND) |
| **Training Time** | N/A | N/A | ~20 hours | ~11 hours (100 epochs) | ~11 hours (100 epochs) |
| **Performance (MPJPE)** | ~16-17mm | 9.71mm | 47.44mm | 47.41mm (similar to resnet_v2) | 47.40mm (no improvement) |
| **Real-time FPS** | ~60fps | ~25fps | ~30fps | ~5fps (training) / ~20-40fps (inference) | ~5fps (training) / ~20-40fps (inference) |
| **Inference Speed** | Very fast | Medium (IK opt) | Fast (single pass) | Medium (transformer) | Medium (transformer) |
| **Accuracy** | Low | High | Medium | High (expected) | High (expected) |
| **Occlusion Handling** | Poor | Good | Learned | Better (attention) | Better (attention) |
| **Complex Poses** | Poor | Good | Limited | Better (attention) | Better (attention) |
| **Model Size** | N/A | N/A | ~25M params | ~30M+ params | ~30M+ params |
| **Use Case** | Baseline demo | Real-time tracking | Production (if improved) | Production (target) | Production (target) |
| **File/Notebook** | mediapipe_v0.py | mano_v1.py | train_colab.ipynb | train_colab.ipynb | train_colab.ipynb |
| **Status** | Complete | Complete | Plateaued | Plateaued (similar to resnet_v2) | ❌ Stuck (no improvement) |

**Key Insights:**
- **mediapipe_v0**: Fastest but least accurate, no 3D model
- **mano_v1**: Best current accuracy (9.71mm), real-time capable, no training needed
- **resnet_v2**: Learned approach but plateaued at 47mm (insufficient for target)
- **transformer_v1**: Expected best accuracy (12-15mm target), slower training (~4-5 it/s) but inference still real-time capable (~20-40 FPS on GPU), requires training

**Training Pipeline Overview:**
The complete training pipeline is in `src/training/train_colab.ipynb` - a self-contained Colab notebook with all code inline.

**Architecture Choices:**
1. **ResNet + MLP** (Baseline)
   - Simple, fast, proven baseline
   - Performance: Target ~15-20mm MPJPE on FreiHAND
   - Architecture: ResNet-50/101 backbone (ImageNet pretrained) + MLP head: 2048 → 512 → 45 (θ) + 10 (β)
   - Training: Faster, less memory
   - Use case: Initial experiments, baseline comparison
   - Results: resnet_v1 (47mm, collapsed), resnet_v2 (47.44mm, plateaued early)

2. **HandFormer-style Transformer** (Best Accuracy - Current)
   - Why: HandFormer (2024) achieves 12.33mm on FreiHAND
   - Architecture: ResNet backbone + Transformer encoder + MLP head
   - Performance: Target ~12-15mm MPJPE
   - Training: Slower, more memory
   - Use case: Final model for production
   - Current: transformer_v1 training in progress

3. **Vision Transformer (ViT)** (Alternative - Not Implemented)
   - Modern transformer architecture
   - Performance: Similar to HandFormer
   - Training: Requires more data
   - Use case: If you have large datasets

**Datasets:**

1. **FreiHAND** (Primary - Currently Using)
   - **Size**: 32K unique poses × 4 views = 130K images
   - **Ground truth**: MANO parameters (θ, β) via multi-view triangulation (~5mm accuracy)
   - **Quality**: High-quality annotations, standard benchmark for hand pose estimation
   - **Why we use it**: Standard benchmark, good quality, MANO GT available, sufficient for initial training
   - **Limitations**: Limited pose diversity (no hand-object interaction), no occlusions
   - **Our results**: ResNet plateaued at 47mm, transformer_v1 target 12-15mm (HandFormer achieved 12.33mm)

2. **HO-3D** (Secondary - Phase 3 Integration)
   - **Size**: 103K frames
   - **Ground truth**: 3D joint positions (may require MANO fitting)
   - **Challenges**: Severe occlusion (hand-object interaction), challenging scenarios
   - **Why we'll use it**: Improve robustness, handle occlusions better, test generalization
   - **When**: Phase 3 (epochs 20-50 of transformer_v3 curriculum)
   - **Integration strategy** (dataset_loader.py updates):
     - Subset selection: 20K hardest occlusion cases (hand-object contact regions)
     - MANO fitting: If GT θ unavailable, fit MANO to 3D joints (use pose_fitter optimization)
     - Multi-subject handling: Normalize per-subject, augment shape diversity
     - Mixing: Interleave with FreiHAND (70% FreiHAND, 30% HO-3D) to avoid catastrophic forgetting

3. **DexYCB** (Best Addition for Pose Diversity - Phase 4 Integration)
   - **Size**: 582K RGB-D frames, 1,000 sequences, 10 subjects, 8 views
   - **Ground truth**: Direct MANO parameters (θ, β) with fixed shape per subject
   - **Strengths**: Hand-object grasping interactions, excellent for pose diversity, classifiable gestures
   - **Why we want it**: Best for EMG goals (pose diversity > absolute MPJPE), 582K frames vs 103K HO-3D
   - **Barrier**: 119GB download, storage constraints (we'll use external SSD later)
   - **When**: Phase 4 (epochs 50-100 of transformer_v3 curriculum)
   - **Integration strategy** (dataset_loader.py updates):
     - Subset selection: 100K grasping frames (diverse grasp taxonomies: power, precision, pinch, etc.)
     - Pose filtering: Select frames with high θ variance (avoid repetitive poses)
     - Gesture labeling: Map DexYCB grasps to gesture classes (for diversity metrics)
     - Mixing: 50% FreiHAND + 25% HO-3D + 25% DexYCB (balanced diversity + accuracy)

4. **DexCanvas** (Very Large-Scale - Future Consideration)
   - **Size**: 7,000 hours of dexterous hand-object interactions
   - **Ground truth**: High-precision MANO parameters, contact points, force profiles
   - **Strengths**: Very large-scale, multi-view RGB-D synchronized, contact information
   - **Why we might use it**: If we need more data, contact information could help with EMG training
   - **When**: Long-term, if we need more training data

5. **HOGraspNet** (Shape Diversity - Future Consideration)
   - **Size**: Diverse hand shapes from 99 participants
   - **Ground truth**: MANO parameters fitted to multi-view RGB-D
   - **Strengths**: Diverse hand shapes, full grasp taxonomies, 3D meshes, keypoints, contact maps
   - **Why we might use it**: Shape diversity could help generalize across users (important for EMG)
   - **When**: If we need shape diversity beyond FreiHAND

6. **SHOWMe** (Rigid Scenarios - Future Consideration)
   - **Size**: 96 videos with detailed hand-object 3D meshes
   - **Ground truth**: Precise 3D reconstructions with MANO
   - **Strengths**: Rigid hand-object scenarios, precise reconstructions
   - **Why we might use it**: High precision, but limited size (96 videos)
   - **When**: If we need high-precision rigid scenarios

7. **AMASS** (Motion Data - Future Consideration)
   - **Size**: 40+ hours, 300+ subjects, unifies 15 motion capture datasets
   - **Ground truth**: MANO parameters (part of unified format), realistic motion data
   - **Strengths**: Large-scale unified framework, realistic motion, many subjects
   - **Why we might use it**: Realistic motion data, many subjects (good for generalization)
   - **When**: If we need motion/temporal data for EMG training

8. **HUMBI** (Natural Scenarios - Future Consideration)
   - **Size**: Multiview human body expressions
   - **Ground truth**: MANO parameters for hands, consistent shape parameters per subject
   - **Strengths**: Natural clothing scenarios, consistent shape per subject
   - **Why we might use it**: Natural scenarios, consistent shapes (good for user-specific training)
   - **When**: If we need natural, user-specific data

9. **InterHand2.6M** (Two-Hand - Future Consideration)
   - **Size**: 2.6M frames
   - **Ground truth**: Large-scale, two-hand interactions (may require MANO fitting)
   - **Strengths**: Very large-scale, two-hand interactions
   - **Why we might use it**: If we expand to two-hand tracking
   - **When**: Long-term, if we expand scope

**Training Strategy:**
1. **Phase 1**: Train on FreiHAND with ResNet - Quick baseline (~15-20mm), validate approach (COMPLETED: resnet_v1, resnet_v2)
2. **Phase 2**: Train Transformer on FreiHAND - Better accuracy (~12-15mm), compare with HandFormer (IN PROGRESS: transformer_v1)
3. **Phase 3**: Fine-tune on HO-3D (future) - Improve robustness, handle occlusions better
4. **Phase 4**: Fine-tune on collected data (future) - Domain adaptation, match camera/lighting

**Loss Functions:**
1. **SimpleMANOLoss**: Parameter L2 + regularization. Fast, no MANO forward pass needed. Good for initial training. Problem: Model learns to predict mean θ values (lowest parameter loss), causing model collapse.
2. **MANOLoss**: Multi-term loss with parameter loss (L2 on θ, β), joint position loss (MPJPE), vertex loss (optional), regularization. Requires MANO forward pass (slower but more accurate). Forces model to match actual joint positions, not just parameter values. Current config: lambda_param=0.5, lambda_joint=20.0, lambda_reg=0.001
   - **Comparison with IK Loss**: Training loss (MANOLoss) and IK loss (pose_fitter) solve different problems:
     - **Training**: Supervised learning (has GT θ and J), learns function Image → θ, no alignment needed (GT aligned), has parameter loss (learns MANO space)
     - **IK**: Optimization (no GT, only MediaPipe observations), finds θ for specific frame, uses Umeyama alignment (coordinate mismatch), has bone direction loss (monocular depth), has temporal smoothness (video tracking)
     - **Similarity**: Both penalize joint position errors, both use MANO forward pass, both use regularization
     - **Key Insight**: Training learns the pattern that IK discovers through optimization - both aim to produce θ that minimizes joint error, but through different methods
3. **Umeyama Alignment**: Closed-form solution for similarity transform (scale, rotation, translation) via SVD. Used in IK to align MediaPipe and MANO coordinate frames. Not needed in training (GT is aligned). Standard in hand pose estimation (FreiHAND, HO-3D papers).

**Expected Performance (Based on Literature):**
- MediaPipe + IK (current): ~16-17mm on FreiHAND, poor on complex poses
- HandFormer (2024): 12.33mm on FreiHAND
- Our ResNet: Target 15-20mm (competitive baseline) - Actual: 47mm (plateaued)
- Our Transformer: Target 12-15mm (close to HandFormer) - Training in progress

**Key Improvements Over Current System:**
1. Handles occlusions: Neural networks learn from diverse training data
2. Better on complex poses: Crossed fingers, self-occlusions
3. More robust: Less sensitive to MediaPipe detection failures
4. Faster inference: Single forward pass vs iterative IK optimization

**Usage (Google Colab):**
1. **Setup**: Open `train_colab.ipynb` in Google Colab. The notebook clones repository, installs dependencies, mounts Google Drive, copies dataset from Drive to Colab, sets up all training code inline.
2. **Prepare Dataset**: Upload FreiHAND dataset zip to Google Drive at `MyDrive/Colab Notebooks/asha/freihand.zip`. Notebook automatically mounts Drive, copies dataset to `/content/freihand/`, verifies dataset structure.
3. **Train Model**: Run all cells in order. Includes environment setup, dataset loading (130K images, 90/10 train/val split), model creation (ResNet or Transformer), training loop with TensorBoard logging, evaluation with detailed metrics, checkpoint saving to Google Drive.
4. **Export Results**: Checkpoints and results automatically saved to Google Drive: `MyDrive/Colab Notebooks/asha/{EXPERIMENT_NAME}/` and local Colab backup: `/content/asha/checkpoints/{EXPERIMENT_NAME}/`. Download from Drive or access via Colab file browser.

**Integration:**
After training, integrate the model into `main_v2.py`:
1. Load trained checkpoint
2. Replace MediaPipe + IK with model inference
3. Use as primary method, fallback to IK if confidence low

**EMG Data Collection Strategy (v3 preparation):**
Given storage/credit constraints, prioritize getting EMG→pose working ASAP:
1. **Train transformer now** (Priority 1): Run transformer_v1 training on FreiHAND (~20-30 hours on A100)
   - Expected: 12-15mm MPJPE (good enough for EMG ground truth)
   - Action: Open train_colab.ipynb, set EXPERIMENT_NAME='transformer_v1', train
2. **Record EMG + video now** (Priority 2 - parallel): Use existing main_v2.py recording pipeline
   - Format: EMG (8 channels @ 500Hz) + RGB video @ 25fps, synchronized
   - Current: Saves IK-based θ (MediaPipe + IK) - fine for now
   - Later: Replace IK θ with transformer predictions
   - Gestures: Focus on diverse, classifiable poses (fist, open palm, crossed fingers, pointing, pinch, etc.)
   - Sessions: 10-30 minutes per session, multiple sessions over days/weeks
3. **Generate ground truth** (After transformer trained): Run transformer on EMG video frames → get MANO θ
   - Process: Extract frames from EMG recordings, run transformer inference, align with EMG timestamps
   - Result: EMG signals → MANO θ pairs ready for training
4. **Train EMG→pose model** (Final step): Temporal CNN/LSTM, input EMG windows → output MANO θ

**Dataset Strategy for Pose Diversity:**
For EMG→pose goals, prioritize datasets that provide diverse, classifiable poses:
- **FreiHAND**: Baseline (130K images, good quality)
- **DexYCB**: Best addition for diversity (582K frames, grasping poses, hand-object interactions)
- **HO-3D**: Occlusions and robustness (103K frames)
- **HOGraspNet**: Shape diversity (99 participants, diverse hand shapes)
- **Other options**: DexCanvas (7K hours), SHOWMe (96 videos), AMASS (40+ hours)

**Important Consideration: MPJPE vs Pose Diversity**
For EMG→pose goals, pose diversity and classifiability may be more important than absolute MPJPE accuracy:
- **MPJPE measures**: Average joint position error (useful for benchmarking against papers)
- **What matters for EMG**: Ability to distinguish between different poses (crossed fingers, thumbs up, etc.)
- **Key insight**: A model with 20mm MPJPE but high pose diversity may be better for EMG training than 10mm MPJPE with low diversity
- **Why this matters**: 
  - EMG signals need to map to distinct, classifiable poses
  - If model predicts similar poses for different inputs, EMG can't learn meaningful mappings
  - Crossed fingers, complex gestures require model to predict diverse, distinguishable poses
  - Low MPJPE can come from predicting 'average' poses (model collapse)
  - EMG needs DISTINCT poses (fist vs open vs crossed), not just accurate average
- **Evaluation strategy**: Monitor both MPJPE (accuracy) and prediction variance (diversity) during training
- **Target**: Model should learn to predict diverse, classifiable poses that EMG can map to, not just minimize average error
- **Practical approach**: 
  - Use MPJPE as a sanity check (should be reasonable, <30mm)
  - Prioritize pose diversity metrics (variance in predictions, ability to classify different gestures)
  - Test on challenging poses (crossed fingers, occlusions) to verify model learns diverse outputs
  - Visual inspection: Do predictions look realistic and diverse?
  - Gesture classification: Can model distinguish between different poses?

**Transformer Architecture & Pose Diversity:**
- **How transformer helps**: Attention mechanisms allow explicit modeling of relationships between hand parts (finger tip can 'see' palm, index finger can 'see' middle finger). This helps with complex poses like crossed fingers.
- **What transformer doesn't guarantee**: Pose diversity depends on loss function (MANOLoss with lambda_joint=20.0) and training data diversity, not just architecture. A transformer with only MPJPE loss can still collapse.
- **What matters more for diversity**: 
  1. Loss function: MANOLoss (penalizes joint positions) > SimpleMANOLoss (only parameters)
  2. Training data diversity: FreiHAND has 32K unique poses (good diversity)
  3. Hyperparameters: lambda_joint weight (higher = more emphasis on joint positions)
  4. Evaluation metrics: Check prediction variance, not just MPJPE
- **Bottom line**: Transformer helps with accuracy and complex poses, but pose diversity comes from loss function + data. Don't hyperfixate on MPJPE - check diversity metrics!

### System Components

- MediaPipe for real-time hand landmark detection (21 keypoints)
- MANO parametric hand model for 3D mesh representation
- **Multi-term IK optimization** (position + bone direction + temporal smoothness)
- PyQt5 for dual-panel GUI (webcam feed + 3D render)
- PyTorch with Apple Silicon MPS acceleration
- Pyrender for offscreen 3D mesh rendering
- MindRove EMG hardware (8 channels @ 500Hz) - integrated in v2

## Why Joint Angles (Not Joint Positions)

### The Critical Distinction

**MediaPipe gives joint POSITIONS**:
- 21 joints × 3D coordinates = 63 values
- Example: "Index fingertip is at (0.05, 0.12, 0.03) meters"
- Depends on hand location/orientation in space

**MANO produces joint ANGLES**:
- 15 joints × 3 DoF = 45 rotation parameters (θ in axis-angle form)
- Example: "Index MCP joint bent 45° forward, 5° sideways"
- Independent of hand location (intrinsic to hand pose)

### Why Angles Matter for EMG→Pose

**Research consensus** (emg2pose, NeuroPose):
- Train EMG → joint angles, NOT joint positions
- Angles are anatomically meaningful (how much each finger bends)
- Angles generalize better across users and conditions
- Positions would require learning spatial location too

### What MANO IK Does (v1's Role)

MANO's inverse kinematics solver converts:
```
Input:  21 joint positions (from MediaPipe)
Output: 45 joint angles θ (axis-angle parameters)
```

This is **exactly what research papers do** with motion capture:
- emg2pose: 26-camera mocap → hand model → joint angles
- Our approach: 1 webcam + MediaPipe → MANO IK → joint angles

Same pipeline, lower-cost ground truth (\$50 vs \$100K+ mocap).

## Experimental Results Summary

### Validation Dataset
- **5,330 frames** (3 min video @ 29.3fps)
- Controlled lighting, 99.6% MediaPipe detection
- All 4 planned experiments completed

### Experiment 1: Loss Ablation Study
**What we tested:** Systematically removed each loss term

| Configuration | Mean Error | Key Finding |
|--------------|-----------|-------------|
| Baseline (all losses) | 9.71mm | Balanced performance |
| No bone direction | **12.48mm** | +28% worse! Critical loss |
| No temporal | 9.66mm | Minimal impact on mean |
| No regularization | **8.49mm** | Actually better without! |
| Position only | **7.04mm** | Best numerically, but no guarantees |

**Key insights:**
- **Bone direction loss critical** - ensures realistic finger lengths/orientations
- **Temporal smoothness** - minimal mean impact but reduces jitter 87%
- **Regularization harmful** - biases toward neutral poses, prevents finding true pose
- **Position-only paradox** - best numbers but lacks anatomical guarantees

**Recommended configuration:**
```python
λ_pos = 1.0          # Position alignment
λ_dir = 0.5          # Bone direction (CRITICAL)
λ_smooth = 0.1       # Temporal smoothness (for video quality)
λ_reg = 0.0          # Drop regularization (it hurts performance)
```

### Experiment 2: Alignment Method Comparison
**What we tested:** Different algorithms for aligning MANO ↔ MediaPipe coordinates

| Method | Mean Error | Scale Factor |
|--------|-----------|--------------|
| Umeyama scaled | **9.71mm** | 0.981 ± 0.121 |
| Umeyama rigid | 10.31mm | 1.000 (fixed) |
| Kabsch | 10.31mm | 1.000 (fixed) |

**Key insights:**
- **Scale estimation improves 6%** (10.31mm → 9.71mm)
- MediaPipe "world coordinates" not perfectly calibrated (~2% systematic error)
- Umeyama rigid = Kabsch (validates implementation)

**Verdict:** Keep Umeyama with scale estimation (current choice is correct)

### Experiment 3: Optimizer Comparison
**What we tested:** Adam vs SGD vs L-BFGS

| Optimizer | Mean Error | Time/Frame | Convergence |
|-----------|-----------|------------|-------------|
| **Adam** | **9.71mm** | 56.7ms | **99.7%** (16 failures) |
| SGD | 26.23mm | 55.0ms | 92.6% (393 failures) |
| L-BFGS | 10.82mm | **38.8ms** | 99.3% (37 failures) |

**Key insights:**
- **Adam wins** - best accuracy + highest reliability
- **SGD fails** - 2.7× worse error, fixed learning rate can't handle non-convex landscape
- **L-BFGS tempting** - 31% faster but 2.3× more failures (jarring visual artifacts)

**Verdict:** Adam is correct choice. Reliability > speed for real-time tracking.

### Experiment 4: Per-Joint Error Analysis
**What we found:** Error distribution non-uniform across hand topology

**Worst 5 joints:**
1. **Thumb tip:** 14.13mm (complex kinematics, saddle joint, 5 DoF)
2. **Wrist:** 13.13mm (alignment artifact, not tracking failure)
3. **Index tip:** 11.85mm (depth ambiguity + kinematic amplification)
4. **Thumb IP:** 11.41mm (middle joint of complex thumb chain)
5. **Pinky tip:** 11.32mm (thin structure, hard to localize)

**Key insights:**
- **Fingertips 40-80% higher error** than mean (9.71mm)
- **Depth ambiguity** - single camera can't reliably determine if fingertip is 5cm or 10cm from palm
- **Kinematic amplification** - small angle errors accumulate through wrist → metacarpal → proximal → middle → distal → TIP
- **Wrist error** - indicates alignment limitations (MANO/MediaPipe define wrist center differently)

**Why this matters:** Reveals systematic failure modes. Future improvements:
- Per-joint loss weighting (trust proximal joints more)
- Stereo camera or learned depth for fingertips
- Improved wrist alignment (use palm center as anchor)

### Comparison to State-of-the-Art

| Method | Dataset | Error (mm) | FPS |
|--------|---------|-----------|-----|
| **Ours (v1)** | **Validation** | **9.71** | **25** |
| HandFormer | STEREO | 10.92 | ~5 |
| HandFormer | FreiHAND | 12.33 | ~5 |
| Drosakis | 2D keypts | Competitive | - |

**Our competitive advantages:**
1. **Speed:** 25fps vs transformers' ~5fps (5× faster)
2. **Interpretability:** Know exactly why it fails (alignment, depth ambiguity)
3. **Cost:** \$50 webcam vs \$100K+ mocap
4. **Robustness:** 99.7% convergence rate

## Research Background

### emg2pose (Meta/FAIR, 2024)

**Paper**: "emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation"
- **Dataset**: 193 users, 370 hours, 29 gesture stages
- **Hardware**: 16 channels @ 2kHz EMG, 26-camera mocap
- **Representation**: Joint angles (degrees) as primary output
- **Models tested**: vemg2pose (best), NeuroPose, SensingDynamics

**Key findings**:
- sEMG signals strongly depend on user anatomy and sensor placement
- Requires diverse training data to generalize
- Joint angles + landmark positions used for loss
- Loss weights: 1.0 for angles, 0.01 for fingertip locations

**Preprocessing**:
- High-pass filter @ 40Hz
- Rescale so noise floor std = 1
- Sign flip for left hands

**Architecture (vemg2pose)**:
- Strided convolutional featurizer
- Time-Depth Separable (TDS) network
- LSTM decoder
- Predicts joint angular velocities → integrate to angles

**What we learned**:
- Validates our approach: EMG → joint angles (not positions) is the right representation
- Confirms need for diverse training data (we're collecting multi-session, multi-gesture data)
- Architecture guidance: Temporal models (LSTM) are important for EMG (we'll use LSTM/1D CNN+LSTM for v3)
- Loss weighting: Prioritize angles over positions (we use MANO θ as primary output)
- Preprocessing matters: High-pass filtering, normalization critical (we implement notch+bandpass filters)

### NeuroPose (Penn State, 2021)

**Paper**: "NeuroPose: 3D Hand Pose Tracking using EMG Wearables"
- **Approach**: RNN/U-Net architectures with anatomical constraints
- **Focus**: Wearable EMG for fine-grained finger motion
- **Key contribution**: Fusing anatomical constraints with ML

**Architecture**:
- U-Net encoder-decoder
- Residual bottleneck layers
- Direct joint angle prediction

**What we learned**:
- U-Net architecture effective for EMG (we'll consider for v3)
- Anatomical constraints help (we use MANO model which enforces anatomical validity)
- Direct angle prediction works (we predict MANO θ directly, not velocities)
- Wearable focus aligns with our use case (MindRove is wearable EMG)

### HandFormer (Jiao et al., 2024)

**Paper**: "HandFormer: A Transformer-based Approach for Hand Pose Estimation"
- **Architecture**: ResNet backbone + Transformer encoder + MLP head
- **Performance**: 10.92mm (stereo), 12.33mm (FreiHAND monocular)
- **Key innovation**: Self-attention for hand part relationships

**What we learned**:
- Transformer architecture achieves state-of-the-art accuracy (12.33mm on FreiHAND)
- Self-attention helps with complex poses (crossed fingers, occlusions)
- ResNet backbone + Transformer is effective combination (we use same architecture)
- Target accuracy: 12-15mm MPJPE is achievable (our transformer_v1 target)
- **Why we switched**: ResNet plateaued at 47mm, transformer expected to reach 12-15mm

### Drosakis & Argyros (Petra, 2023)

**Paper**: "MANO Optimization from 2D Keypoints"
- **Approach**: IK optimization from 2D keypoints (similar to our MediaPipe → MANO IK)
- **Key finding**: Optimization competitive with learning-based approaches
- **Method**: Multi-term loss with alignment

**What we learned**:
- Validates our IK approach: Optimization from landmarks works well (we achieve 9.71mm)
- Multi-term loss is critical (we validated bone direction loss is essential)
- Alignment matters (we use Umeyama scaled alignment)
- **Our extension**: Added bone direction loss (+28% error without) and temporal smoothness (87% jitter reduction)

### Tu et al. (IEEE TPAMI, 2022)

**Paper**: "Consistent 3D Hand Reconstruction in Video"
- **Approach**: Multi-term loss with temporal consistency
- **Loss terms**: 2D keypoints + motion + texture + shape

**What we learned**:
- Temporal smoothness loss justified (we use ||θ - θ_{t-1}||²)
- Multi-term losses improve video quality (we validated this experimentally)
- Motion consistency reduces jitter (we see 87% jitter reduction with temporal loss)

### MediaPipe (Google, 2020)

**Paper**: "MediaPipe Hands: On-device Real-time Hand Tracking"
- **Performance**: 21 keypoints @ 60fps
- **Method**: CNN-based detection + regression

**What we learned**:
- Fast, reliable baseline for hand detection (we use for v0-v2)
- World coordinates available (we use for better depth estimation)
- Good enough for our IK pipeline (9.71mm error)
- Limitations: Fails on complex poses (crossed fingers, occlusions) → why we're training transformer

### MANO (Romero et al., SIGGRAPH Asia 2017)

**Paper**: "Embodied Hands: Modeling and Capturing Hands and Bodies Together"
- **Model**: Parametric hand model (45 pose + 10 shape parameters)
- **Method**: Linear Blend Skinning (LBS) for realistic meshes

**What we learned**:
- Standard representation in hand pose estimation (used by FreiHAND, HO-3D, DexYCB)
- Anatomically validated (built from 3D scans)
- Enables IK optimization (we use for v1)
- Provides joint angles (exactly what EMG research uses - emg2pose, NeuroPose)

### Comparison to Our Approach

| Aspect | emg2pose/NeuroPose | Project Asha |
|--------|-------------------|--------------|
| Ground truth | 26-camera mocap | 1 webcam + MediaPipe |
| Accuracy | Very high | Moderate (single-camera depth) |
| Cost | $$$$ | $ |
| Representation | Joint angles | Joint angles (via MANO IK) |
| Pipeline | Mocap → model → angles | MediaPipe → MANO → angles |
| Training | EMG → θ (angles) | EMG → θ (angles) |
| Inference | EMG only | EMG only (v4) |

**Same fundamental approach, different ground truth quality**. Lower-cost pipeline with same core objective: EMG-based hand tracking.

## Version Roadmap

### v0 - MediaPipe Baseline (DONE)
- **Purpose**: Baseline hand tracking visualization
- **Output**: 21 joint positions (3D scatter plot)
- **Performance**: ~60 fps
- **Limitations**: No joint angles, no EMG integration
- **Status**: ✅ Complete

### v1 - MANO IK + LBS (DONE, VALIDATED)
- **Purpose**: Convert joint positions → joint angles
- **Features**:
  - Inverse kinematics optimization (15 Adam steps/frame)
  - Full linear blend skinning for realistic mesh
  - **Multi-term loss validated through experiments:**
    - Position alignment (Umeyama scaled)
    - **Bone direction (CRITICAL - +28% error without)**
    - Temporal smoothness (87% jitter reduction)
    - Regularization (HARMFUL - remove for v1.1)
  - Warm-start tracking
- **Output**: 45 joint angle parameters (θ) + 778-vertex mesh
- **Performance**: 9.71mm @ 25fps (competitive with HandFormer)
- **Status**: ✅ Complete + Validated (October 7, 2024)

### v2 - EMG Integration & Data Collection (DONE)
- **Purpose**: Build complete EMG → pose pipeline
- **Features**:
  - MindRove WiFi EMG integration (8 channels @ 500Hz)
  - Signal processing pipeline (notch filter, bandpass, zero-phase filtering)
  - Synchronized recording (EMG + MANO θ + MediaPipe joints)
  - Gesture labeling system
  - HDF5 data format with metadata
  - Real-time 3-panel GUI (webcam, 3D mesh, EMG plot)
  - PyQtGraph for 30fps+ EMG visualization
- **Output**: Synchronized HDF5 datasets (EMG + pose + gestures)
- **Performance**: ~25fps camera, 500Hz EMG
- **Status**: ✅ Complete (October 8, 2024)

### v3 - EMG → θ Training Pipeline (NEXT)
- **Purpose**: Train neural network to predict joint angles from EMG
- **Features**:
  - Data validation & inspection tools (load HDF5, visualize sessions)
  - Preprocessing pipeline (windowing, normalization, quality filtering)
  - Dataset preparation (train/val/test split, windowed samples)
  - Model architectures (LSTM, 1D CNN + LSTM, based on emg2pose)
  - Training loop with early stopping, learning rate scheduling
  - Evaluation metrics (angle error, position error, temporal smoothness)
  - Per-joint error analysis and visualization
- **Output**: Trained PyTorch model (.pt file) for real-time inference
- **Status**: 🔄 Planned

### v2.5 - Image → θ Training Pipeline (READY FOR TRAINING)
- **Purpose**: Improve pose estimation quality by training neural network to predict MANO θ directly from images, bypassing MediaPipe + IK
- **Motivation**: Current MediaPipe + IK achieves 9.71mm but fails on complex poses (crossed fingers, occlusions). Neural networks trained on diverse datasets (FreiHAND) should handle these better.
- **Features**:
  - FreiHAND dataset loader (32K poses × 4 views = 130K images with MANO ground truth)
  - Two architectures:
    - ResNet + MLP: Fast baseline (target ~15-20mm MPJPE)
    - Transformer (HandFormer-style): Better accuracy (target ~12-15mm MPJPE)
  - Multi-term loss functions (parameter L2, joint position, regularization)
  - Training pipeline with TensorBoard logging, checkpointing, evaluation
  - Complete self-contained Google Colab notebook (`src/training/train_colab.ipynb`)
- **Output**: Trained PyTorch model (.pth checkpoint) for image → θ prediction
- **Integration**: Replace MediaPipe + IK in `main_v2.py` with trained model inference
- **Expected improvements**: Better handling of complex poses, occlusions, crossed fingers
- **Status**: ✅ Training pipeline complete, ready for Colab training
- **Notebook Details**: `src/training/train_colab.ipynb`
  - **Self-contained**: All code inline (models, losses, dataset loader, MANO layer) - no external file dependencies
  - **Simple setup**: Clones public repo from GitHub (no token needed)
  - **Google Drive integration**: Unified folder structure (`MyDrive/Colab Notebooks/asha/`)
    - Checkpoints: `{EXPERIMENT_NAME}/` folders (e.g., `resnet_v1/`, `resnet_v2/`, `transformer_v1/`)
    - Dataset: `freihand.zip`
    - MANO model: `MANO_RIGHT_numpy.pkl`
  - **GPU optimized**: Configured for A100 (mixed precision FP16, batch size 32-64, gradient clipping)
  - **Architecture support**: Both ResNet and Transformer architectures (currently using Transformer)
  - **Easy debugging**: Each cell is clear and debuggable with simple error messages
  - **Dependencies**: All packages verified for Colab (torch, torchvision, albumentations, scipy, matplotlib, etc.)
  - **Usage**: Upload notebook to Colab or open from GitHub → Run cells in order → Train model → Checkpoints saved to Drive

### v4 - Real-Time EMG Inference System
- **Purpose**: Camera-free hand tracking using EMG only
- **Features**:
  - Load trained model from v3
  - Real-time EMG streaming → sliding window buffer → predict θ
  - MANO forward pass: θ → 3D mesh vertices
  - Visualization (same GUI as v2, but no camera needed)
  - Performance optimization (model quantization, batch predictions)
  - Latency target: <10ms inference @ 25fps
- **Output**: Standalone EMG-based hand tracking system
- **Evaluation**: Compare EMG→θ→mesh vs MediaPipe→θ→mesh
- **Status**: 🔄 Planned

### v5 - Multi-Sensor Fusion (OPTIONAL)
- **Purpose**: Enhance tracking with IMU + PPG from Snaptic kit
- **Hardware**: MindRove Snaptic kit (2 IMUs @ 100Hz, 1 PPG sensor)
- **Features**:
  - **IMU integration**: 2×9-DOF (accel + gyro + mag) @ 100Hz
    - IMU 1 (forearm): Arm orientation, acceleration
    - IMU 2 (back of hand): Hand orientation in space
  - **PPG integration**: Blood flow/pulse sensor
    - Potential: Detect muscle activation patterns via blood flow
    - Explore correlation with EMG signals
  - Sensor fusion architectures:
    - Early fusion: Concat EMG+IMU+PPG → model
    - Late fusion: Separate branches → fuse features
    - Hierarchical: IMU → wrist pose, EMG → finger articulation
  - Ablation studies: EMG-only vs EMG+IMU vs full sensor suite
- **Why after v4**: Validate EMG-only baseline first, then assess what IMUs/PPG add
- **Expected gains**:
  - IMU: Better hand orientation/position tracking (wrist roll/pitch/yaw)
  - PPG: Potentially improve temporal prediction (muscle activation timing)
- **Status**: 🔄 Planned

## Environment Setup

### Python Version
Python 3.11 (or 3.10 for Apple Silicon optimization)

### Creating Runtime Environment
```bash
./setup_runtime_env.sh
source asha_env/bin/activate
```

Or manually:
```bash
python3.11 -m venv asha_env
source asha_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### MANO Models Setup
MANO models are required but not included in the repository. They must be:
1. Downloaded from https://mano.is.tue.mpg.de/download.php
2. Placed in `models/`
3. Named exactly: `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`
4. Converted from chumpy to numpy format:
   ```bash
   ./setup_convert_env.sh
   source mano_convert_env/bin/activate
   python src/convert_mano_to_numpy.py
   deactivate
   ```

This creates `MANO_RIGHT_numpy.pkl` and `MANO_LEFT_numpy.pkl` which are loaded by the application.

## Running the Application

```bash
source asha_env/bin/activate
python src/programs/main_v2.py          # emg + mano ik (full system)
# or
python src/programs/mano_v1.py          # full ik + lbs articulation (no emg)
# or
python src/programs/mediapipe_v0.py     # mediapipe baseline (scatter plot)
```

**For main_v2:**
- Ensure webcam permissions are granted to the terminal application
- Connect MindRove WiFi board before starting
- Select session type from dropdown (5 structured protocols)
- Click "Start Recording" to begin data collection
- Timer shows progress: elapsed/target (turns green when target reached)

## Running Experiments

**Quick test (100 frames):**
```bash
cd src
python experiments/run_experiments.py --video path/to/video.mp4 --quick
```

**Full experiments (all 4):**
```bash
cd src
python experiments/run_experiments.py \
  --video path/to/validation.mp4 \
  --experiments 1,2,3,4 \
  --output-dir results/experiments
```

**Generate figures:**
```bash
cd src
python utils/visualize_experiments.py \
  --results results/experiments \
  --output ../docs/figures
```

**Results are saved to:**
- `src/results/experiments/exp1_loss_ablation.json`
- `src/results/experiments/exp2_alignment_methods.json`
- `src/results/experiments/exp3_optimizer_comparison.json`
- `src/results/experiments/exp4_per_joint_analysis.json`

**Figures generated:**
- `docs/figures/exp1_loss_ablation.png`
- `docs/figures/exp2_alignment.png`
- `docs/figures/exp3_optimizer.png`
- `docs/figures/exp4_per_joint.png`

## Project Architecture

### Directory Structure (After Reorganization)

```
asha/
├── README.md                          # Main project documentation
├── CLAUDE.md                          # This file (complete guide)
├── MIDTERM_REPORT_FINAL.tex          # Academic report with experiments
├── CIS6800_Proposal.tex              # Original project proposal
│
├── models/                            # MANO models (not in git)
│   ├── MANO_RIGHT.pkl
│   ├── MANO_LEFT.pkl
│   ├── MANO_RIGHT_numpy.pkl
│   └── MANO_LEFT_numpy.pkl
│
├── src/
│   ├── model_utils/                   # Core MANO, tracker, pose_fitter
│   │   ├── mano_model.py
│   │   ├── tracker.py
│   │   └── pose_fitter.py
│   │
│   ├── programs/                      # v0, v1, v2 applications
│   │   ├── mediapipe_v0.py
│   │   ├── mano_v1.py
│   │   └── main_v2.py
│   │
│   ├── experiments/                   # Experiment framework
│   │   ├── pose_fitter_experimental.py  # Configurable IK for experiments
│   │   ├── run_experiments.py           # Runs all 4 experiments
│   │   ├── freihand_loader.py           # FreiHAND dataset loader
│   │   └── ho3d_loader.py               # HO-3D dataset loader
│   │
│   ├── training/                      # Image → MANO θ training pipeline (v2.5)
│   │   ├── train_colab.ipynb           # Complete Colab training notebook
│   │   ├── dataset_loader.py           # FreiHAND/HO-3D data loading
│   │   ├── model.py                    # ResNet + Transformer architectures
│   │   ├── losses.py                   # Multi-term loss functions
│   │   ├── evaluate.py                 # Evaluation script
│   │   ├── README.md                   # Training documentation
│   │   └── ARCHITECTURE_NOTES.md       # Architecture recommendations
│   │
│   ├── utils/                         # Visualization, data tools
│   │   ├── emg_utils.py               # MindRove interface + signal processing
│   │   ├── data_recorder.py           # HDF5 synchronized recording
│   │   ├── data_loader.py             # Load HDF5 recordings
│   │   ├── visualize_experiments.py   # Generate publication-quality figures
│   │   ├── extract_results.py         # Metrics extraction from HDF5
│   │   └── generate_plots.py          # Plot generation utilities
│   │
│   ├── tests/                         # Test suites
│   │   ├── test.py
│   │   ├── test_mediapipe_v0.py
│   │   ├── test_mano_v1.py
│   │   └── test_main_v2.py
│   │
│   └── results/                       # Experiment results (JSON)
│       └── experiments/
│           ├── exp1_loss_ablation.json
│           ├── exp2_alignment_methods.json
│           ├── exp3_optimizer_comparison.json
│           └── exp4_per_joint_analysis.json
│
├── docs/
│   ├── REFERENCES.md                  # Bibliography
│   ├── reports/                       # Academic reports (PDFs)
│   │   ├── CIS_6800_Final_Project_Proposal.pdf
│   │   └── CIS_6800_Final_Project_Midterm_Report.pdf
│   ├── figures/                       # All figures (experiments + system screenshots)
│   │   ├── exp1_loss_ablation.png
│   │   ├── exp2_alignment.png
│   │   ├── exp3_optimizer.png
│   │   ├── exp4_per_joint.png
│   │   ├── ik_error_histogram.png
│   │   ├── quality_filtering.png
│   │   ├── ui_progression.png
│   │   ├── v0_mediapipe.png
│   │   ├── v1_mano.png
│   │   └── v2_emg.png
│   ├── references/                    # PDF papers
│   │   ├── freihand2019.pdf
│   │   └── HOnnotate2020.pdf
│   └── archive/                       # Historical/temporary files (not in git)
│       ├── SESSION_NOTES_NOV25.md
│       └── ...
│
├── data/                              # Recorded sessions (not in git)
│   └── session_*.h5                   # HDF5 recordings
│
├── requirements.txt                   # Runtime dependencies
├── requirements_convert.txt           # Conversion environment
├── setup_runtime_env.sh
└── setup_convert_env.sh
```

### Core Components

**model_utils/mano_model.py** - Custom MANO loader
- `CustomMANOLayer`: Loads converted numpy-based MANO models
- Avoids Python 3.11/chumpy compatibility issues
- Implements forward pass with shape/pose blend shapes
- Returns vertices (778, 3) and joints (21, 3)
- Loads from `models/MANO_{side}_numpy.pkl`

**model_utils/tracker.py** - MediaPipe hand tracking wrapper
- Singleton MediaPipe Hands instance configured for real-time tracking
- `get_landmarks()`: Extracts 21 hand landmarks from BGR frame (image coordinates, normalized 0-1)
- `get_landmarks_world()`: Extracts landmarks in world coordinates (metric, meters)
  - Preferred for mano_v1, more stable depth estimation
  - Includes validation (rejects >2m span or NaN)
- `draw_landmarks_on_frame()`: Renders green dots on detected keypoints

**model_utils/pose_fitter.py** - MANO interface + IK optimization
- Lazy-loads CustomMANOLayer (singleton pattern)
- `get_torch_device()`: Auto-detects MPS (Apple Silicon) or falls back to CPU
- `mano_from_landmarks()`: Full IK optimization (mano_v1)
  - Takes MediaPipe landmarks, optimizes 45 pose parameters (15 joints × 3 DoF)
  - 15 Adam iterations per frame with **validated multi-term loss:**
    - Joint position loss (L2 after Umeyama/Procrustes alignment)
    - **Bone direction loss (CRITICAL - +28% error without)**
    - Temporal smoothness (87% jitter reduction)
    - Regularization (REMOVE - harmful to accuracy)
  - Returns vertices (778, 3) and joints (21, 3)
- `mano_from_landmarks_simple()`: Neutral pose only (deprecated)
- `get_mano_faces()`: Returns face indices for mesh triangulation

**experiments/pose_fitter_experimental.py** - Experimental IK framework
- Configurable loss terms (for ablation studies)
- Multiple alignment methods (Umeyama scaled/rigid, Kabsch)
- Different optimizers (Adam, SGD, L-BFGS)
- Comprehensive metrics collection (per-joint errors, convergence, timing)
- Used for all 4 validation experiments

**experiments/run_experiments.py** - Experiment runner
- Runs all 4 experiments from validation study
- Processes video frame-by-frame with different configurations
- Saves results to JSON
- Generates comprehensive metrics

**utils/visualize_experiments.py** - Figure generator
- Generates publication-quality figures from experiment JSON
- Creates all 4 experiment plots
- Exports LaTeX summary table
- Uses seaborn for clean, professional styling

**programs/mediapipe_v0.py** - MediaPipe baseline
- Dual-panel PyQt5 GUI: webcam feed + 3D scatter plot
- Visualizes 21 MediaPipe landmarks directly
- No MANO model, fast (~60 fps)
- Useful for baseline comparison

**programs/mano_v1.py** - Full IK + LBS implementation
- Dual-panel PyQt5 GUI: webcam feed + 3D MANO mesh
- Full inverse kinematics optimization per frame
- Real finger articulation via linear blend skinning
- Pyrender offscreen rendering for 3D mesh
- ~25 fps with 15 Adam steps per frame
- **Validated: 9.71mm mean error @ 99.7% convergence**

**programs/main_v2.py** - Full system with EMG integration
- Minimal PyQt5 GUI: webcam feed + 3D MANO mesh (EMG runs in background)
- Dual-thread architecture: VideoThread (25fps) + EMGThread (500Hz)
- Synchronized recording with DataRecorder
- Structured session protocol selector (5 predefined session types)
- Quality indicators (IK error, MediaPipe confidence, EMG status)
- Recording controls with elapsed/target timer (turns green at target)

**utils/emg_utils.py** - MindRove interface + signal processing
- `MindroveInterface`: Hardware connection to MindRove WiFi board (8ch @ 500Hz)
  - connect(), get_data(), disconnect()
  - Automatic channel mapping and validation
- `filter_emg()`: Signal processing pipeline
  - Notch filter @ 60Hz (power line noise)
  - Bandpass 10-200Hz (Butterworth order 2)
  - Zero-phase filtering (scipy.signal.filtfilt)
  - Handles small batches gracefully (< 15 samples pass through unfiltered)

**utils/data_recorder.py** - HDF5 synchronized recording
- `DataRecorder`: Manages dual-rate data (EMG @ 500Hz, pose @ 25fps)
  - record_emg(): Buffers EMG samples with timestamps
  - record_pose(): Buffers MANO θ + joints with quality metrics
  - save(): Writes to HDF5 with compression
  - get_stats(): Recording statistics (duration, sample counts, rates)
- HDF5 structure: /emg, /pose, /metadata groups
- Timestamp synchronization via time.perf_counter()
- Session metadata includes session type for protocol tracking

**training/** - Image → MANO θ training pipeline (v2.5)
- `train_colab.ipynb`: Complete Colab notebook for training (setup, training, evaluation, checkpointing)
- `dataset_loader.py`: FreiHAND/HO-3D dataset loading with augmentation
- `model.py`: ResNet + MLP and Transformer architectures for image → θ prediction
- `losses.py`: SimpleMANOLoss (fast) and MANOLoss (accurate, requires MANO forward pass)
- `evaluate.py`: Evaluation script with MPJPE metrics
- Purpose: Train neural network to predict MANO θ directly from images, bypassing MediaPipe + IK
- Expected: Better handling of complex poses (crossed fingers, occlusions) than current MediaPipe + IK

### Data Flow (mano_v1)

1. `VideoThread` (QThread) captures frames from webcam (OpenCV)
2. MediaPipe extracts 21 hand landmarks in world coordinates (metric)
3. If hand detected:
   - Landmarks validated (span < 2m, no NaN)
   - Drawn on frame with green dots
   - Passed to `mano_from_landmarks()` for IK optimization
   - 15 Adam iterations optimize 45 pose parameters
   - MANO forward pass (Rodrigues → FK → LBS) generates mesh vertices (778, 3)
4. Both frame and vertices emitted via Qt signal to main thread
5. GUI updates both panels:
   - Left: webcam feed with landmarks overlaid
   - Right: 3D MANO mesh rendered via pyrender

### Data Flow (main_v2)

**Thread 1: VideoThread @ 25fps**
1. Capture frame from webcam (OpenCV)
2. MediaPipe → 21 landmarks (world coordinates)
3. MANO IK → θ (45 params) + vertices (778, 3)
4. Emit signal: (frame, verts, ik_error, mp_confidence, θ, joints)
5. If recording: DataRecorder.record_pose(θ, joints, ik_error, mp_confidence)

**Thread 2: EMGThread @ 500Hz**
1. MindroveInterface.get_data() → (timestamps, emg_raw)
2. filter_emg(emg_raw) → emg_filtered
3. Emit signal: (timestamps, emg_raw, emg_filtered)
4. If recording: DataRecorder.record_emg(emg_raw, emg_filtered)

**Thread 3: GUI (Main Thread)**
1. Update left panel: webcam feed with landmarks
2. Update middle panel: 3D MANO mesh (pyrender)
3. Update quality indicators (IK error, MP confidence, EMG status)
4. Update recording timer (elapsed/target, turns green at target)
5. Handle user interactions (session selector, record button)

**Recording Session:**
1. User selects session type (e.g., "Session 1: Basic Poses")
2. User clicks "Start Recording" → enter subject ID
3. DataRecorder initialized with output path (includes session number)
4. Both threads start buffering data with timestamps (time.perf_counter())
5. Timer shows progress toward target duration (e.g., "02:30 / 03:00")
6. User clicks "Stop Recording" → DataRecorder.save() → HDF5 file written

### Key Technical Details

**Thread Safety**: Video capture runs in separate QThread, communicates with GUI via Qt signals to avoid race conditions.

**3D Rendering Pipeline** (mano_v1):
- Uses pyrender OffscreenRenderer (800x600 viewport)
- Scene includes perspective camera, directional lights from multiple angles
- Mesh rebuilt each frame by removing/adding trimesh node with new vertices
- Rendered to numpy array, converted to QImage for display

**Device Handling**: Code detects and uses Apple Silicon MPS backend when available, falling back to CPU. This is critical for performance on M1/M2 Macs.

**MANO Integration**:
- Uses custom MANO loader with numpy-based `_numpy.pkl` files (avoids chumpy/Python 3.11 issues)
- Full LBS pipeline: Rodrigues rotations → forward kinematics → pose blend shapes → skinning
- **Validated IK optimization:** Umeyama scaled alignment + bone direction + temporal smoothness
- Warm-start tracking (uses previous frame pose as initial guess)

**Joint Mapping**: MediaPipe outputs 21 landmarks in specific order (wrist, thumb, index, middle, ring, pinky). MANO has different joint ordering. `_MANO21_FOR_MP` array maps between them.

## Strategic Considerations: MANO vs Custom Hand Model

### Long-Term IP Concerns

**MANO Licensing:**
- Research use: Free
- Commercial use: Requires licensing from Max Planck Institute
- Risk: Licensing terms may change; dependency on third-party model

**For a Startup:**
- MANO is not proprietary IP (competitors can use same model)
- Licensing costs/complexity for commercial use
- No differentiation from using MANO alone
- Your real IP is in EMG→θ pipeline, not hand model

**Recommendation:**
- Short-term: Use MANO to validate and ship quickly
- Long-term: Develop proprietary hand model for IP and differentiation

### Short-Term: Why MANO vs Alternatives

**Why MANO is Better Than Custom θ:**

1. **Datasets & Ground Truth**
   - FreiHAND, HO-3D, InterHand2.6M provide MANO ground truth
   - Custom θ would require creating your own dataset
   - Years of research validation

2. **Tooling & Ecosystem**
   - MANO has established tooling (PyTorch layers, visualization)
   - Research papers use MANO (easier comparisons)
   - Community support and documentation

3. **Proven Representation**
   - 45 pose + 10 shape parameters (well-validated)
   - Anatomically validated (built from 3D scans)
   - Linear Blend Skinning for realistic meshes

4. **EMG Research Alignment**
   - emg2pose, NeuroPose use joint angles
   - MANO provides standard angle representation
   - Easier to compare with published results

**Why MANO vs Other Models:**

| Model | Output | Issue |
|-------|--------|-------|
| MediaPipe | 21 joint positions | Need IK conversion → MANO |
| Custom skeleton | Joint positions | No parametric model, harder to animate |
| Other parametric models | Different format | Less dataset support, less research |

### Strategic Roadmap

**Phase 1 (Now - 6 months): Use MANO**
- Fast validation and MVP
- Leverage existing datasets (FreiHAND)
- Focus on EMG → θ training (your core IP)
- Your IP is in the EMG pipeline, not the hand model

**Phase 2 (6-12 months): Hybrid Approach**
- Keep MANO for training/validation
- Develop lightweight proprietary model for inference
- Train your model to predict both MANO θ and custom θ
- Gradually migrate to custom representation

**Phase 3 (12+ months): Proprietary Model**
- Custom hand model optimized for your use case
- Potentially better for EMG (trained on your data)
- Strong IP position
- Can license your model

### What to Protect (Your Real IP)

1. **EMG → θ Training Pipeline** (v3)
   - Your data collection protocol
   - Preprocessing (windowing, filtering)
   - Model architecture for EMG
   - This is your core IP

2. **Real-Time Inference System** (v4)
   - Latency optimizations
   - Sensor fusion (EMG + IMU + PPG)
   - User adaptation algorithms

3. **Proprietary Datasets**
   - Your collected EMG + pose data
   - User-specific calibrations
   - Domain-specific gestures

**Bottom Line:**
- Short-term: MANO accelerates development and validation. Your IP is in EMG→θ, not hand model.
- Long-term: Plan proprietary model, but don't block on it now. Focus on proving EMG→θ works first.

**Action Items:**
1. Check MANO commercial licensing terms
2. Use MANO for MVP and validation
3. Plan custom model roadmap (12+ months)
4. Focus IP protection on EMG pipeline, not hand representation

The hand model is infrastructure; the EMG→θ system is your IP.

## Future Improvements (Based on Experimental Results)

### Short-term (Easy wins)
1. **Remove regularization loss** - Actually hurts: 9.71mm → 8.49mm without it
   ```python
   λ_reg = 0.0  # Currently 0.01, drop to 0.0
   ```

2. **Add per-joint weighting** - Trust proximal joints more than fingertips
   ```python
   # Fingertips have 40-80% higher error, weight them less
   joint_weights = [2.0 if joint in proximal else 1.0 for joint in joints]
   ```

3. **Reduce iterations if speed needed** - 10 instead of 15 (acceptable trade-off)

### Medium-term
4. **Improve wrist alignment** - Use palm center as anchor instead of wrist
5. **Test on public datasets** - FreiHAND, HO-3D benchmarks
6. **Failure mode visualization** - Show the 16 Adam failures (0.3%)

### Long-term
7. **Stereo camera or learned depth** - Resolve fingertip ambiguity (11-14mm errors)
8. **Hard joint limit constraints** - Project onto feasible set instead of soft penalties
9. **Alternative detector** - Replace MediaPipe for extreme poses/occlusion
10. **Proprietary hand model** - Custom representation optimized for EMG (see Strategic Considerations above)

## Development History

### Session 11 - Transformer Training Progress & Gradient Fix (v2.5)
- **Date**: December 2, 2024
- **Critical Bug Fix**: Fixed `RuntimeError: element 0 of tensors does not require grad` at epoch 6
  - **Root cause**: When NaN was detected, penalty tensors were created with `torch.tensor(100.0, ...)` which don't require gradients
  - **Fix**: Changed to `pred_theta.sum() * 0.0 + penalty_value` to ensure tensors require gradients
  - **Fixed locations**: 
    - NaN joint loss penalty: `torch.tensor(100.0, ...)` → `pred_theta.sum() * 0.0 + 100.0`
    - Zero joint loss: `torch.tensor(0.0, ...)` → `pred_theta.sum() * 0.0`
    - Zero vertex loss: `torch.tensor(0.0, ...)` → `pred_theta.sum() * 0.0`
    - NaN total loss penalty: `torch.tensor(1000.0, ...)` → `pred_theta.sum() * 0.0 + 1000.0`
  - **Why this works**: `pred_theta.sum() * 0.0 + value` creates a tensor that requires gradients (from pred_theta) but evaluates to the desired value
- **Training Progress (transformer_v1)**:
  - **Epochs 1-4 completed**: Training stable, no NaN issues (fixes working)
  - **Loss breakdown**:
    - Total loss: ~2.43 (train), ~2.42 (val) - very stable, minimal decrease
    - Param loss: ~0.28
    - **Joint loss: ~0.114** → **~11.4mm MPJPE** (promising! close to target 12-15mm)
    - Reg loss: ~0.07
  - **Observations**:
    - Joint loss of 0.114 is actually good (~11.4mm MPJPE, within target range)
    - Total loss not decreasing much (2.4353 → 2.4304 train, 2.4235 → 2.4232 val)
    - Similar pattern to resnet_v2 (early plateau), but joint loss is much better
    - Training stable (no NaN), learning rate 5e-5 seems appropriate
    - Best model saved at epoch 3 (val_loss: 2.4232)
  - **Analysis**:
    - Joint loss contribution: 0.114 × 20.0 = 2.28 out of total 2.43 (dominates as intended)
    - Model is learning joint positions (joint loss is reasonable)
    - But total loss plateauing suggests model might need more epochs or LR adjustment
    - Early days (epoch 4) - need to monitor if loss starts decreasing or stays flat
  - **Next steps**: Continue training, monitor loss trend. If still plateauing at epoch 10-15, consider LR increase or architecture adjustment.
- **Evaluation Results (After 5 Epochs - Early Training)**:
  - **MPJPE: 47.41mm** (similar to resnet_v2's 47.44mm after 25 epochs)
  - **Median: 47.42mm, 95th percentile: 50.52mm**
  - **Parameter errors**: θ (pose) = 3.5266, β (shape) = 0.0000 (model not predicting beta)
  - **Per-joint errors**: Wrist highest (~95mm), fingertips high (~55-60mm), proximal joints better (~20-35mm)
  - **Context**: Only 5 epochs completed (vs resnet_v2's 25 epochs) - model may still improve with more training
  - **Key findings**:
    - Training joint loss (~0.114 = 11.4mm) doesn't match evaluation MPJPE (47.41mm)
    - Discrepancy suggests model learning relative joint positions but not global pose
    - Wrist error highest (~95mm) indicates global pose misalignment
    - Similar performance to resnet_v2 at this early stage (but resnet_v2 plateaued, transformer might improve)
  - **Root cause hypothesis**: Model learning finger articulation (relative positions) but not global hand pose (wrist position/orientation) - may improve with more training
  - **Recommendations**: 
    - Continue training to 20-30 epochs to see if transformer improves beyond resnet_v2 plateau
    - Verify evaluation uses PA-MPJPE (after alignment)
    - Monitor if loss starts decreasing more significantly
    - If still plateauing at 20-30 epochs, then investigate loss function/data issues

### Session 12 - Transformer v2 Results: Critical Finding - No Improvement (v2.5)
- **Date**: December 4, 2024
- **Experiment**: `freihand_transformer_v2` - Increased learning rate and loss weights to address plateauing
- **Hyperparameter Changes from v1**:
  - Learning rate: 5e-5 → 7.5e-5 (increased to escape local minimum)
  - lambda_joint: 20.0 → 25.0 (more emphasis on joint positions)
  - gradient_clip: 1.0 → 1.5 (allow larger gradients with higher LR)
- **Training Progress (Epochs 1-6)**:
  - Total loss: 3.0071 → 3.0019 (train), 2.9934 → 2.9930 (val) - **plateauing after epoch 1**
  - Joint loss: ~0.114 → **~11.4mm MPJPE** (good, but not improving)
  - Param loss: ~0.29 (high, not decreasing)
  - Reg loss: ~0.07 (stable)
  - Learning rate: 7.5e-5 → 7.43e-5 (cosine scheduler decaying too fast)
  - Best model: epoch 3 (val_loss: 2.9930)
- **Evaluation Results (After 6 Epochs)**:
  - **MPJPE: 47.40mm** (same as resnet_v2: 47.44mm) - **NO IMPROVEMENT**
  - Median: 47.41mm, 95th percentile: 50.50mm
  - θ error: 3.5382 ± 0.1643 (very high)
  - β error: 0.0000 (expected, predict_beta=False)
  - **Critical Finding**: Transformer architecture is NOT helping - identical performance to ResNet
- **Root Cause Analysis**:
  - **Loss Discrepancy Identified**: Training joint loss (0.114) suggests ~11.4mm MPJPE, but evaluation shows 47.40mm
    - Training uses raw MSE: `loss_joint = MSE(pred_joints, gt_joints)` (no alignment, assumes GT is in same coordinate frame)
    - Evaluation uses PA-MPJPE: `compute_mpjpe(pred_joints, gt_joints, align=True)` (with Procrustes alignment)
    - Model learns local joint positions (relative finger articulation) but fails at global pose alignment (wrist position/orientation)
    - The 4× discrepancy (11.4mm vs 47.40mm) indicates global pose misalignment is the main issue
  - **Learning Rate Issues**:
    - LR still too low (7.5e-5) - loss decreasing too slowly (~0.0001-0.0006 per epoch)
    - Cosine scheduler decaying from start (7.5e-5 → 7.43e-5 by epoch 5)
    - Model stuck in same local minimum as resnet_v2 (47.40mm vs 47.44mm)
  - **Architecture Not Helping**: Despite transformer's attention mechanisms, model performs identically to ResNet
    - Suggests hyperparameters are wrong, not architecture limitations
    - Need significantly higher learning rate and better schedule to escape local minimum
- **Key Insights**:
  - Transformer architecture alone doesn't solve the problem - hyperparameters are critical
  - Loss function discrepancy reveals model is learning local structure but not global pose
  - Both ResNet and Transformer stuck at ~47mm, suggesting fundamental training issue
  - Need to address learning rate schedule and potentially loss function alignment
- **Next Steps Needed**:
  - Increase learning rate significantly (7.5e-5 → 1.5e-4 or 2e-4)
  - Fix learning rate schedule: Constant LR for first 20 epochs, then cosine decay
  - Consider learning rate warmup: Start at 5e-5, ramp to 1.5e-4 over 10 epochs
  - Investigate loss function: Should training loss also use alignment? Or is global pose a separate problem?
  - Monitor: If still plateauing after hyperparameter fixes, may need to reconsider architecture or data augmentation

### Session 10 - Loss Function Deep Dive & Training Speed Clarification (v2.5)
- **Date**: December 2, 2024
- **Discussions**:
  - **Loss Function Comparison**: Detailed explanation of MANOLoss vs SimpleMANOLoss vs IK Loss
    - **MANOLoss (Training)**: `L = λ_param * ||θ_pred - θ_gt||² + λ_joint * ||J_pred - J_gt||² + λ_reg * ||θ||²`
      - Has parameter loss (learns MANO space)
      - Has joint position loss (prevents collapse)
      - No alignment needed (GT is aligned)
      - No temporal smoothness (static images)
      - Current weights: λ_param=0.5, λ_joint=20.0 (dominates), λ_reg=0.001
    - **IK Loss (pose_fitter)**: `L = λ_joint * ||sRJ(θ) + t - X||² + λ_bone * ||D(sRJ) - D(X)||² + λ_smooth * ||θ - θ_{t-1}||² + λ_reg * ||θ||²`
      - No parameter loss (only joint positions matter)
      - Uses Umeyama alignment (coordinate mismatch)
      - Has bone direction loss (monocular depth ambiguity)
      - Has temporal smoothness (video tracking)
      - Current weights: λ_joint=1.0, λ_bone=0.5, λ_smooth=0.1, λ_reg=0.01
    - **Key Insight**: Training learns to predict θ that minimizes joint error (similar to IK's objective), but through supervised learning rather than per-frame optimization
    - **Analogy**: IK = "Solve this specific equation" (optimization), Training = "Learn to solve equations" (supervised learning)
  - **Umeyama Alignment Derivation**: Closed-form solution for similarity transform (scale, rotation, translation)
    - Problem: `min_{s,R,t} ||sRY + t1ᵀ - X||²_F`
    - Solution: Center points → SVD of cross-covariance → rotation R = VUᵀ → scale s = tr(Σ)/||Ỹ||² → translation t = μₓ - sRμᵧ
    - Why it works: Removes global pose ambiguity, focuses on articulation, standard in hand pose estimation
    - Used in IK for alignment (MediaPipe ≠ MANO coordinates), not needed in training (GT is aligned)
  - **Training vs Inference Speed**: Clarified why transformer FPS appears low during training
    - **Training**: ~4-5 it/s (includes forward + backward + gradients + weight updates) → ~2x slower than inference
    - **Inference**: ~20-40 FPS on GPU (forward pass only) → Real-time capable ✓
    - **Why transformer is slower**: Self-attention O(n²) complexity, sequential processing, more parameters (~30-35M vs ~25M)
    - **ResNet inference**: ~50-100 FPS on GPU → Also real-time capable ✓
    - **Bottom line**: Both architectures are fast enough for real-time use on GPU (30 FPS video input)
    - **If too slow**: Use GPU, reduce transformer layers (4→2), use FP16 quantization, fall back to ResNet
- **Status**: ✅ Concepts clarified, training ready to proceed

### Session 9 - Transformer Training: NaN Issue & Stability Fixes (v2.5)
- **Date**: December 2, 2024
- **Changes**:
  - Started training transformer_v1 (HandFormer-style architecture)
  - Encountered NaN loss issue starting at epoch 6
  - Root cause: Extreme MANO parameters (pred_theta/pred_beta) causing NaN in MANO forward pass
  - Fixes applied:
    - Parameter clipping: Clamp pred_theta to [-3.14, 3.14] and pred_beta to [-3.0, 3.0] before MANO forward pass
    - NaN detection: Check for NaN/inf in loss computation and MANO output, skip bad batches with warnings
    - Reduced learning rate: Changed from 1e-4 to 5e-5 for more stable training
    - Loss safeguards: Replace NaN losses with large finite penalty (1000.0) to prevent training crash
  - Added diagnostic logging: Gradient norm tracking and loss component breakdown (param, joint, reg)
  - Fixed positional encoding: Made dynamic based on actual feature map size (49 patches from 7×7, not hardcoded 196)
  - Training observations: Epochs 1-5 showed stable loss (~2.42-2.43) but not decreasing, indicating learning rate may need further tuning
  - Updated ResNet weights loading: Changed from deprecated `pretrained=True` to `weights=models.ResNet50_Weights.IMAGENET1K_V1`
- **Restart Decision**: Recommended to restart from epoch 0 (not epoch 5 checkpoint) because:
  - Only 5 epochs lost (~35 minutes)
  - Loss wasn't decreasing anyway (stuck at 2.42)
  - Learning rate was wrong (1e-4 → 5e-5)
  - Clean start ensures no corrupted weights
  - All fixes applied from beginning
- **Status**: ✅ NaN fixes applied, ready to restart training from epoch 0

### Session 8 - Training Pipeline: Model Versions & Loss Function Fix (v2.5)
- **Date**: December 2, 2024
- **What We Did**:
  - Created complete training pipeline for image → MANO θ prediction
  - Implemented ResNet-50 + MLP architecture on FreiHAND dataset (130K images)
  - Trained first model version (resnet_v1) on Google Colab A100 GPU
  - Implemented comprehensive evaluation with PA-MPJPE (Procrustes Aligned Mean Per-Joint Position Error)
  - Added diagnostic code to inspect model predictions (parameter distributions, variance analysis)
  - Fixed model collapse issue and created resnet_v2 with corrected loss function
- **Model Versions**:
  - **resnet_v1**: Parameter-only loss (SimpleMANOLoss)
    - Result: 47mm PA-MPJPE, model collapse detected
    - Issue: Model learned to predict mean θ values (lowest parameter loss)
    - All predictions collapsed to same pose → model didn't learn diverse hand shapes
    - Theta variance: std 0.1605 (very low), beta: all zeros
    - Loss plateaued at ~0.1997-0.2002 (not decreasing)
  - **resnet_v2**: Proper MANO loss (MANOLoss with joint position loss)
    - Fix: Switched to MANOLoss (parameter + joint position loss)
    - Forces model to match actual joint positions, not just parameter values
    - Result: 47.44mm PA-MPJPE, loss plateaued at epoch 22-25 (train: 1.4107, val: 1.4088)
    - Issue: Loss plateauing suggests architecture limitations or learning rate too conservative
    - Conclusion: ResNet may plateau around 30-40mm, insufficient for target 15-20mm
- **transformer_v1**: Switched to Transformer architecture (HandFormer-style)
    - Architecture: ResNet-50 backbone + Transformer encoder (4 layers, 8 heads, d_model=256)
    - Hyperparameters: lambda_joint=20.0, lambda_param=0.5, predict_beta=False, gradient_clip=1.0
    - Training: 100 epochs, weight_decay=1e-4
    - Expected: 12-15mm MPJPE (based on HandFormer achieving 12.33mm on FreiHAND)
    - Status: Ready for training
- **Root Cause of Model Collapse**:
  - **Problem**: SimpleMANOLoss (parameter-only)
    - Only penalizes parameter differences: `L = ||θ_pred - θ_gt||²`
    - Model learns to predict mean θ values (lowest parameter loss)
    - Doesn't care if joint positions match → collapse to mean predictions
  - **Solution**: MANOLoss (parameter + joint position)
    - Penalizes BOTH parameters AND joint positions: `L = λ_param * ||θ_pred - θ_gt||² + λ_joint * ||J_pred - J_gt||²`
    - Forces model to match actual joint positions, not just parameter values
    - Model must learn diverse poses to minimize joint position errors
- **Fixes Applied**:
  - Changed `use_mano_loss: False` → `True` in training config
  - Added detailed loss function explanation comments in notebook
  - Removed all emojis from notebook
  - Fixed CUDA tensor issues in diagnostic plotting code
  - Fixed section numbering (Sections 15-18 properly ordered)
  - Simplified experiment management: folder-based system (EXPERIMENT_NAME must match existing Drive folder)
- **Why Umeyama Alignment Was Needed**:
  - Raw MPJPE measures absolute 3D position differences (including global pose/scale/rotation)
  - PA-MPJPE removes global differences to measure ONLY articulation accuracy
  - Standard practice in hand pose estimation (used in FreiHAND, HO-3D papers)
  - Before alignment: 705mm (model predicting wrong global pose)
  - After alignment: 47mm (model learning some structure, but still poor)
  - The 47mm error indicates model is learning hand shape but needs better training
- **Key Learnings**:
  - Loss function design is critical - parameter-only loss causes collapse
  - Joint position loss is essential for learning meaningful hand poses
  - Diagnostic code is crucial for detecting training issues early
  - PA-MPJPE is standard evaluation metric (must use alignment for fair comparison)
- **Experiment Management**:
  - Simplified to folder-based system: EXPERIMENT_NAME must match existing folder in Google Drive
  - To create new experiment: Create empty folder in Drive, set EXPERIMENT_NAME to match
  - To resume training: Set EXPERIMENT_NAME to existing folder, checkpoints auto-loaded
  - All sections (10, 12, 14-16) use same EXPERIMENT_NAME consistently
- **Next Steps**: 
  - resnet_v2 completed: Loss plateaued at 47.44mm PA-MPJPE
  - Switched to Transformer architecture (transformer_v1) for better accuracy potential
  - Transformer target: 12-15mm MPJPE (based on HandFormer achieving 12.33mm)
  - **Important consideration**: For EMG→pose goals, pose diversity/classifiability may matter more than absolute MPJPE accuracy

### Session 7 - Training Notebook Finalization & Colab Prep (v2.5)
- **Date**: December 1, 2024
- **Changes**:
  - Simplified notebook for public repo: removed all private repo/token handling code
  - Updated `.gitignore` to allow `src/training/*.ipynb` files to be committed
  - Verified all dependencies and imports for Colab compatibility
  - Added missing packages: scipy (for sparse matrices), matplotlib (for plotting)
  - Cleaned up comments to match coding style (lowercase, practical)
  - Removed duplicate/unnecessary cells for cleaner structure
  - Streamlined GPU check and Drive mount cells for easier debugging
  - Added comprehensive evaluation plots (Section 16): per-joint errors, error distributions, parameter comparisons, heatmaps, cumulative distributions
  - Verified all code is self-contained and ready for Colab
- **Notebook Status**:
  - ✅ Self-contained: all code inline (no external file dependencies)
  - ✅ Simple setup: clones public repo, mounts Drive, trains model
  - ✅ Dependencies verified: torch, scipy, matplotlib, albumentations, etc.
  - ✅ GPU optimized: A100 configuration (mixed precision, batch 64, 8 workers)
  - ✅ Easy debugging: clear cell structure, simple error messages
  - ✅ Production ready: automatic checkpoint saving to Google Drive
- **Ready for Training**:
  - Upload notebook to Colab or open from GitHub
  - Enable GPU: Runtime → Change runtime type → GPU (A100/T4/L4)
  - Run cells in order - everything is automated
  - Checkpoints automatically saved to `MyDrive/asha_checkpoints/`
- **Status**: ✅ Ready to train on Colab

### Session 6 - Image → MANO θ Training Pipeline (v2.5)
- **Date**: December 2024
- **Changes**:
  - Created complete self-contained training pipeline for image → MANO θ prediction
  - Implemented FreiHAND dataset loader (32K poses × 4 views = 130K images)
  - Built two architectures: ResNet + MLP (baseline) and Transformer (HandFormer-style)
  - Multi-term loss functions (parameter L2, joint position, regularization)
  - Complete Colab notebook (`train_colab.ipynb`) with all code inline
  - Google Drive integration: automatic dataset copy and checkpoint saving
  - Optimized for A100 GPU: mixed precision FP16, batch size 64, 8 workers
  - TensorBoard logging, evaluation metrics, checkpoint management
  - Documentation: training README (architecture notes consolidated)
- **Purpose**: Improve pose estimation quality, especially on complex poses (crossed fingers, occlusions)
- **Expected**: ~12-15mm MPJPE (competitive with HandFormer's 12.33mm) vs current 9.71mm MediaPipe + IK
- **Notebook Features**:
  - Self-contained: all code inline (models, losses, dataset loader, MANO layer) - no external files needed
  - Simple: clones public repo, mounts Drive, trains model
  - Debuggable: clear cell structure, simple error messages
  - Production-ready: automatic checkpoint saving to Drive
- **Next**: Train on Colab, integrate best model into `main_v2.py` to replace MediaPipe + IK
- **Status**: ✅ Training pipeline complete, ready for Colab training

### Session 5 - Midterm Report & Experimental Validation
- **Date**: November 25, 2024
- **Changes**:
  - Ran all 4 validation experiments (5,330 frames)
  - Generated publication-quality figures
  - Compiled results into MIDTERM_REPORT_FINAL.tex
  - Key findings: bone direction critical, regularization harmful, Adam optimal
  - Validated competitive performance (9.71mm vs HandFormer 10.92mm)
  - Cleaned up codebase, consolidated documentation
- **Status**: ✅ v1 Fully Validated

### Session 4 - Data Collection UI & Roadmap Update (v2 refinement)
- **Date**: October 9, 2024
- **Changes**:
  - Simplified GUI: removed EMG plot (caused lag), focus on data collection
  - Added structured session protocol selector (5 session types with target durations)
  - Fixed 3D hand orientation (palm/wrist now match camera view correctly)
  - Updated CLAUDE.md with v3 (training) and v4 (inference) roadmap
  - Cleared test data, ready for production data collection
- **Status**: ✅ v2 Ready for Production Data Collection

### Session 3 - EMG Integration & Synchronized Recording (v2)
- **Date**: October 8, 2024
- **Changes**:
  - Implemented MindRove WiFi board interface (8ch @ 500Hz)
  - EMG signal processing pipeline (notch + bandpass filters)
  - HDF5 synchronized recording (dual-rate: EMG @ 500Hz, pose @ 25fps)
  - 3-panel GUI with PyQtGraph real-time EMG plotting
  - Gesture labeling interface with recording controls
  - Comprehensive test suite for v2 components
  - Full v2 documentation in CLAUDE.md
- **Files Created**: emg_utils.py, data_recorder.py, main_v2.py, test_main_v2.py
- **Status**: ✅ v2 Complete

### Session 2 - Full IK + LBS Implementation & Refactor + Documentation
- **Date**: October 7-8, 2024
- **Changes**: Implemented full MANO articulation (IK + LBS), flattened directory structure, renamed files for clarity, comprehensive CLAUDE.md update with v2-v4 roadmap
- **Cost**: $28.55
- **API Time**: 3h 32m 38s
- **Wall Time**: 19h 14m 16s
- **Code Changes**: +3313 lines, -918 lines

### Session 1 - Initial MediaPipe Baseline (v0)
- **Date**: October 5, 2024
- **Changes**: Initial implementation with MediaPipe hand tracking and 3D visualization
- **Cost**: ~$15.48
- **Code Changes**: Initial codebase

## Documentation Structure

### Essential Documentation (Keep These)

**Root Level:**
- `README.md` - Main project documentation, setup instructions, quick start
- `CLAUDE.md` - This file (complete guide for Claude Code and catching up)

**Docs Folder:**
- `docs/REFERENCES.md` - Bibliography and paper citations
- `docs/reports/` - Academic reports (PDFs)
  - `CIS_6800_Final_Project_Proposal.pdf` - Original proposal
  - `CIS_6800_Final_Project_Midterm_Report.pdf` - Complete midterm with experiments
- `docs/figures/` - All figures (experiments + system screenshots)
- `docs/references/` - PDF copies of key papers (FreiHAND, HO-3D)

**Archived (docs/archive/):**
- Session notes and temporary files (not tracked in git)
- Intermediate LaTeX versions
- Work-in-progress documentation

### Quick Reference

**Key results to remember:**
- **9.71mm mean error** @ 25fps (competitive with HandFormer: 10.92mm)
- **99.7% convergence** with Adam (only 16 failures in 5,330 frames)
- **Bone direction loss critical** (+28% error without)
- **Regularization harmful** (8.49mm without vs 9.71mm with)
- **Fingertips hardest** (11-14mm error due to depth ambiguity)

**Best configuration:**
```python
Alignment: Umeyama scaled (6% better than rigid)
Optimizer: Adam (99.7% convergence)
Loss: Position + bone direction + temporal smoothness
Remove: Regularization (it hurts)
```

**Future directions:**
1. Remove regularization (easy win)
2. Per-joint weighting (trust proximal joints)
3. Stereo depth or learned depth (resolve fingertip ambiguity)
4. Hard joint constraints (anatomical feasibility)

**Next phase (v2.5 → v3):**
1. **v2.5 (current)**: Train image → θ model on FreiHAND
   - Train ResNet baseline on Colab (~15-20mm target)
   - Train Transformer if ResNet shows promise (~12-15mm target)
   - Integrate best model into `main_v2.py` to replace MediaPipe + IK
   - Expected: Better handling of complex poses, occlusions

2. **v3**: Train EMG → θ regression model
   - Use improved θ estimates from v2.5 for better training data
   - LSTM or 1D CNN + LSTM architecture (based on emg2pose)
   - Window size: 100-200ms @ 500Hz
   - Target: Real-time EMG-based hand tracking (v4)

## References

**Key Papers (Detailed in Research Background Section Above):**
- **emg2pose** (Meta/FAIR, 2024) - Large-scale EMG hand pose benchmark, validates our approach, architecture guidance for v3
- **NeuroPose** (Penn State, 2021) - EMG wearables for hand tracking, U-Net architecture reference
- **HandFormer** (Jiao et al., 2024) - Transformer-based pose estimation (12.33mm on FreiHAND), architecture we're using for transformer_v1
- **Drosakis & Argyros** (Petra, 2023) - MANO optimization from 2D keypoints, validates our IK approach
- **Tu et al.** (IEEE TPAMI, 2022) - Temporal consistency in video, justifies our temporal smoothness loss
- **MediaPipe** (Google, 2020) - Real-time hand detection, our baseline for v0-v2
- **MANO** (Romero et al., SIGGRAPH Asia 2017) - Parametric hand model, standard representation we use

**Additional Papers (See REFERENCES.md for Details):**
- **Guo et al.** (IEEE TCSVT, 2022) - CNN+GCN+attention, skeleton-aware features (relevant for future architectures)
- **Jiang et al.** (CVPR, 2023) - A2J-transformer for 3D hand pose (transformer approach)
- **Gao et al.** (Neurocomputing, 2021) - TIKNet transformer-based IK (transformer for MANO parameters)
- **Kalshetti & Chaudhuri** (IEEE TPAMI, 2025) - Differentiable rendering + ICP optimization (optimization approach)
- **Cai et al.** (ECCV 2018, TPAMI 2020) - Weak supervision from monocular RGB (relevant for low-cost GT generation)
- **Spurr et al.** (ICCV, 2021) - PECLR self-supervised learning (14.5% improvement on FreiHAND)
- **HandDiff** (Cheng et al., CVPR 2024) - Diffusion models for hand pose (modern approach)

**Datasets (Detailed Above):**
- **FreiHAND** (Zimmermann et al., ECCV 2019) - Primary dataset, 130K images, MANO GT, standard benchmark
- **HO-3D** (Hampali et al., CVPR 2020) - Secondary dataset, 103K frames, occlusions, robustness testing
- **DexYCB** (Chao et al., CVPR 2021) - Best for pose diversity, 582K frames, grasping interactions
- **DexCanvas, HOGraspNet, SHOWMe, AMASS, HUMBI, InterHand2.6M** - Additional datasets for future consideration

**Full bibliography:** See `docs/REFERENCES.md` for complete list with download links and detailed annotations

**Paper PDFs:** See `docs/references/` for PDF copies of key papers

---

**Last Updated:** December 2, 2024
**Project Status:** v1 validated, v2 complete, v2.5 (image → θ training) - resnet_v1 completed (model collapse), resnet_v2 completed (loss plateau at 47.44mm), transformer_v1 completed (47.41mm, no improvement), transformer_v2 completed (47.40mm, still no improvement - **CRITICAL ISSUE**)
**Current Phase:** Completed 6 epochs of transformer_v2 training. **Critical finding**: Transformer architecture is NOT improving over ResNet (47.40mm vs 47.44mm). Root cause identified: Loss discrepancy (training uses raw MSE, evaluation uses PA-MPJPE), learning rate too low and decaying too fast, model stuck in same local minimum. Need to significantly increase learning rate and fix schedule before continuing.
**Model Versions:**
- **resnet_v1**: Parameter-only loss (SimpleMANOLoss) - model collapse, 47mm PA-MPJPE
- **resnet_v2**: Proper MANO loss (MANOLoss with joint position) - loss plateaued at 47.44mm PA-MPJPE
- **transformer_v1**: Transformer architecture (HandFormer-style) - training completed 5 epochs, evaluation: 47.41mm MPJPE (similar to resnet_v2)
- **transformer_v2**: Increased LR (7.5e-5) and lambda_joint (25.0) - training completed 6 epochs, evaluation: 47.40mm MPJPE (**NO IMPROVEMENT**)
**What We're Trying:**
- Train transformer to achieve 12-15mm MPJPE → **Current (transformer_v2, epoch 6): 47.40mm MPJPE (stuck, no improvement over ResNet)**
- **Critical Next Steps**:
  - Increase learning rate significantly (7.5e-5 → 1.5e-4 or 2e-4)
  - Fix learning rate schedule: Constant LR for first 20 epochs, then cosine decay
  - Consider learning rate warmup: Start at 5e-5, ramp to 1.5e-4 over 10 epochs
  - Investigate loss function alignment: Training uses raw MSE, evaluation uses PA-MPJPE (4× discrepancy)
  - If still plateauing after hyperparameter fixes, reconsider architecture or data augmentation
- Eventually train EMG→pose model using improved predictions as ground truth
- Real-time camera-free hand tracking from EMG signals
**What's Working:**
- Training pipeline complete and tested
- NaN loss fixes applied (parameter clipping, NaN detection, reduced LR)
- Path reorganization complete (unified Drive structure)
- Loss function properly configured (MANOLoss with λ_joint=20.0)
- Transformer architecture implemented (HandFormer-style)
**What's Not Working / Issues:**
- ~~Training encountered NaN loss at epoch 6~~ → **Fixed with gradient fix (detached tensor issue)**
- Loss plateauing early: Training loss ~2.43, evaluation MPJPE 47.41mm (after only 5 epochs)
- **Key observation**: Model learning relative joint positions (~11.4mm training loss) but not global pose (47.41mm evaluation MPJPE)
- Wrist error ~95mm (highest) suggests global pose misalignment
- **Note**: Only 5 epochs completed - transformer may still improve with more training (resnet_v2 plateaued at epoch 25)
- Need to continue training to 20-30 epochs to determine if transformer improves beyond resnet_v2's plateau
**What We're Trying:**
- Transformer architecture for better accuracy (12-15mm target vs ResNet 30-40mm plateau)
- Different hyperparameters (reduced LR 5e-5, gradient clipping 1.0, λ_joint=20.0)
- Path-based experiment management (simple folder system)
- Parameter clipping and NaN safeguards for training stability
