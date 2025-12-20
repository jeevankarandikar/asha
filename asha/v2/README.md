# v2: Image → MANO θ (Deep Learning Training)

**Status**: ⏸️ PAUSED (Limited Colab Credits, Plateaued at 47mm)

**Approach**: Train deep learning models to predict MANO pose parameters from RGB hand images.

## Overview

v2 attempted to train ResNet and Transformer models on FreiHAND dataset (130K images) to predict 45 MANO θ parameters from single RGB images. **Both approaches plateaued at ~47mm MPJPE** despite multiple architectural iterations.

**Key Finding**: Image→θ training hit fundamental bottleneck. Led to pivot towards EMG-based approaches (v4, v5) which achieved better results (11.95mm).

## Architecture Attempts

### resnet_v1 (Model Collapse)
- **Architecture**: ResNet18 backbone + FC(512→45)
- **Loss**: SimpleMANOLoss (parameter-only, no joint supervision)
- **Result**: 47mm MPJPE, **model collapsed** to mean predictions
- **Issue**: Insufficient supervision, model learned to predict average pose

### resnet_v2 (Plateaued)
- **Architecture**: ResNet18 backbone + FC(512→45)
- **Loss**: MANOLoss (λ_param=1.0, λ_joint=25.0, λ_reg=0.001)
- **Training**: 50 epochs, lr=1e-4, batch_size=64
- **Result**: 47.44mm MPJPE @ epoch 22, **plateaued**
- **Issue**: Loss imbalance (joint loss dominated, suppressed parameter learning)

### transformer_v1 (Loss Imbalance)
- **Architecture**: HandFormer-style (4 layers, 8 heads, patch_size=16)
- **Loss**: MANOLoss (λ_param=0.5, λ_joint=25.0)
- **Training**: 50 epochs, lr=5e-5, batch_size=64
- **Result**: 47.41mm MPJPE, **same plateau**
- **Issue**: lambda_joint=25 too high, suppressed pose diversity

### transformer_v2 (No Improvement)
- **Changes**: Increased LR to 7.5e-5, kept λ_joint=25.0
- **Result**: 47.40mm MPJPE, **no improvement**
- **Conclusion**: LR not the issue, fundamental problem with loss balance

### transformer_v3 (Model Collapse)
- **Changes**: Reduced λ_joint to 5.0, λ_param=0.5, added angular loss + PCA prior
- **Training**: Stopped at epoch 7 after detecting collapse
- **Results**: 47.46mm MPJPE @ epoch 5, θ variance=0.0002 (2500x too low!)
- **Issue**: λ_joint=5.0 still too high, model predicted identical poses for all inputs
- **Diagnostics**: Entropy=0.0 bits, gesture F1=0.3333 (only 1 type), prediction range [-0.488, 1.361]

### transformer_v3.1 (Planned, Never Trained)
- **Proposed Changes**:
  - λ_param=1.0, λ_joint=1.0 (equal weights)
  - λ_angular=2.0, λ_pca=0.5 (stronger priors)
- **Target**: <20mm MPJPE
- **Status**: Never trained (pivoted to EMG instead)

## Root Cause Analysis

**Why All Models Plateaued at 47mm:**

1. **Loss Imbalance**: Joint loss (63 dims) dominated parameter loss (45 dims)
   - Even λ_joint=5.0 caused collapse to mean predictions
   - Model found "safe" strategy: minimize joint error with repetitive poses

2. **Global Pose Misalignment**:
   - Training used raw MSE on θ parameters
   - Evaluation used PA-MPJPE (Procrustes-aligned)
   - Mismatch between training objective and eval metric

3. **Limited Diversity Metrics**:
   - Early versions had no diversity tracking
   - Model could predict repetitive poses without penalty
   - Added diversity metrics (v3) caught collapse faster, but didn't fix root cause

4. **Dataset Mismatch**:
   - FreiHAND = egocentric (first-person, top-down view)
   - Deployment = allocentric (third-person, frontal webcam)
   - Viewpoint domain shift not addressed

## Datasets

**FreiHAND** (Primary):
- 130K RGB images with MANO ground truth
- Egocentric viewpoint (wrist-mounted camera)
- Standard benchmark for hand pose estimation

**Planned (Never Used)**:
- HO-3D: 103K frames with occlusions
- DexYCB: 582K frames with grasping poses
- Multi-dataset curriculum for viewpoint diversity

## Training Infrastructure

**Google Colab**:
- GPU: T4, V100, or A100 (depending on availability)
- Training time: ~3-6 hours per 50 epochs
- **Issue**: Limited Colab credits prevented extensive experimentation

**Notebook**: `train_colab.ipynb`
- Self-contained training pipeline
- Inline dataset loading, model definition, training loop
- Automatic checkpointing to Google Drive
- **Status**: Preserved as-is (user requested no .ipynb modifications)

## Results Summary

| Model | MPJPE | Status | Key Issue |
|-------|-------|--------|-----------|
| resnet_v1 | 47mm | Collapsed | Parameter-only loss insufficient |
| resnet_v2 | 47.44mm | Plateaued | λ_joint=25 too high |
| transformer_v1 | 47.41mm | Plateaued | Loss imbalance |
| transformer_v2 | 47.40mm | Plateaued | LR not the issue |
| transformer_v3 | 47.46mm | Collapsed | λ_joint=5 still too high |
| transformer_v3.1 | - | Never trained | Pivoted to EMG |

**Comparison to v4/v5** (EMG-based):
- v2 (image→θ): 47mm with 130K training images
- v4 (EMG→θ): 34.92mm with 20 min EMG data (26% better)
- v5 (EMG→joints): 11.95mm with 20 min EMG data (**74% better!**)

## Why v2 Was Abandoned

1. **Plateaued Performance**: All architectures stuck at 47mm
2. **Limited Compute**: Colab credits insufficient for extensive hyperparameter search
3. **Better Alternative**: EMG approach (v4/v5) achieved superior results with minimal data
4. **Domain Mismatch**: FreiHAND (egocentric) ≠ deployment (allocentric)

**Pivot Decision (Dec 11, 2024)**: Focus on EMG→joints (v5) instead of debugging image→θ.

## Files

- **train_colab.ipynb**: Complete training pipeline (PRESERVED, no modifications per user request)
- Training configs and checkpoints stored in Google Drive (not in repo)

## Lessons Learned

1. **Loss Balance Critical**: Even small imbalance (5:1) causes model collapse
2. **Diversity Metrics Essential**: θ variance, entropy, gesture F1 catch collapse early
3. **Evaluation Alignment**: Training loss should match evaluation metric (PA-MPJPE vs MSE)
4. **Dataset Size ≠ Performance**: 130K images (v2) < 20 min EMG (v5) in accuracy
5. **Direct Supervision Better**: Joints (v5: 11.95mm) > parameters (v2: 47mm)

## Future Work (If Resumed)

**To break 47mm plateau (estimated effort: 2-4 weeks):**

1. **Fix Loss Balance**:
   - λ_param=1.0, λ_joint=1.0 (equal weights)
   - λ_angular=2.0, λ_pca=0.5 (stronger priors)
   - Add diversity losses (θ variance, entropy)

2. **Procrustes-Aligned Training**:
   - Apply Procrustes alignment during training
   - Match training objective to evaluation metric

3. **Multi-Dataset Training**:
   - FreiHAND (130K) + DexYCB (582K) + HO-3D (103K)
   - Curriculum: egocentric → mixed viewpoints
   - Target: Viewpoint-invariant representations

4. **Larger Models**:
   - HandFormer full (6 layers, 12 heads)
   - Vision Transformer (ViT-B/16)
   - Target: <15mm MPJPE on FreiHAND

**Estimated Target**: 10-15mm MPJPE (matching HandFormer 10.92mm)

**Recommendation**: Don't resume v2. EMG approach (v5) is superior and more aligned with project goals (prosthetics, camera-free tracking).

## Timeline

- **Dec 2-4, 2024**: resnet_v1, resnet_v2, transformer_v1, transformer_v2 trained and failed
- **Dec 11, 2024**: transformer_v3 collapsed at epoch 7
- **Dec 11, 2024**: Decision to pivot to EMG training (v4/v5)
- **Dec 13, 2024**: v4 and v5 trained successfully (34.92mm, 11.95mm)
- **Dec 14, 2024**: Codebase restructure (v2 archived to asha/v2/)
- **Status**: Permanently paused, EMG approach validated as superior

## Research Contribution

v2's failures provide valuable negative results:

1. **Parameter-only loss insufficient** for MANO pose prediction
2. **Joint-dominant loss causes collapse** even with modern architectures (Transformer)
3. **Direct joint prediction (v5) outperforms parametric (v2)** by 74%
4. **Small, high-quality EMG dataset (20 min) > large image dataset (130K)** for this task

**Key Insight**: For EMG→hand-pose, direct joint regression with limited supervised data beats large-scale image→parameter training. Pose manifold learning is harder than spatial coordinate prediction.
