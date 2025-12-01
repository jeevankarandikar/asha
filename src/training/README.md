# Training Pipeline for Image → MANO θ Prediction

This module contains the complete training pipeline for learning to predict MANO parameters directly from images, bypassing MediaPipe + IK optimization.

## Architecture Choices

Based on literature review:

### 1. **ResNet + MLP** (Start Here - Baseline)
- **Why**: Simple, fast, proven baseline
- **Performance**: Target ~15-20mm MPJPE on FreiHAND
- **Architecture**: ResNet-50/101 backbone (ImageNet pretrained) + MLP head: 2048 → 512 → 45 (θ) + 10 (β)
- **Training**: Faster, less memory
- **Use case**: Initial experiments, baseline comparison

### 2. **HandFormer-style Transformer** (Best Accuracy)
- **Why**: HandFormer (2024) achieves 12.33mm on FreiHAND
- **Architecture**: ResNet backbone + Transformer encoder + MLP head
- **Performance**: Target ~12-15mm MPJPE
- **Training**: Slower, more memory
- **Use case**: Final model for production

### 3. **Vision Transformer (ViT)** (Alternative)
- **Why**: Modern transformer architecture
- **Performance**: Similar to HandFormer
- **Training**: Requires more data
- **Use case**: If you have large datasets

## Datasets

### Primary: **FreiHAND**
- **Size**: 32K unique poses × 4 views = 130K images
- **Why**: 
  - MANO ground truth available
  - Standard benchmark
  - Good quality annotations (~5mm accuracy)
- **Use**: Primary training dataset

### Secondary: **HO-3D**
- **Size**: 103K frames
- **Why**:
  - More occlusion (hand-object interaction)
  - Challenging scenarios
  - Better for robustness
- **Use**: Fine-tuning, robustness testing (not implemented yet)

### Optional: **InterHand2.6M**
- **Size**: 2.6M frames
- **Why**: Large-scale, two-hand interactions
- **Use**: If you need maximum performance and have storage

## Training Strategy

1. **Phase 1**: Train on FreiHAND with ResNet
   - Quick baseline (~15-20mm)
   - Validate approach

2. **Phase 2**: Train Transformer on FreiHAND
   - Better accuracy (~12-15mm)
   - Compare with HandFormer

3. **Phase 3**: Fine-tune on HO-3D (future)
   - Improve robustness
   - Handle occlusions better

4. **Phase 4**: Fine-tune on your collected data (future)
   - Domain adaptation
   - Match your camera/lighting

## Loss Functions

1. **SimpleMANOLoss**: Parameter L2 + regularization
   - Fast, no MANO forward pass needed
   - Good for initial training

2. **MANOLoss**: Multi-term loss
   - Parameter loss (L2 on θ, β)
   - Joint position loss (MPJPE)
   - Vertex loss (optional)
   - Regularization
   - Requires MANO forward pass (slower but more accurate)

## Expected Performance

Based on literature:
- **MediaPipe + IK (current)**: ~16-17mm on FreiHAND, poor on complex poses
- **HandFormer (2024)**: 12.33mm on FreiHAND
- **Our ResNet**: Target 15-20mm (competitive baseline)
- **Our Transformer**: Target 12-15mm (close to HandFormer)

## Key Improvements Over Current System

1. **Handles occlusions**: Neural networks learn from diverse training data
2. **Better on complex poses**: Crossed fingers, self-occlusions
3. **More robust**: Less sensitive to MediaPipe detection failures
4. **Faster inference**: Single forward pass vs iterative IK optimization

## Usage (Google Colab)

### 1. Setup

Open `train_colab.ipynb` in Google Colab. The notebook will:
- Clone the repository from GitHub
- Install dependencies
- Mount Google Drive
- Copy dataset from Drive to Colab
- Set up all training code inline

### 2. Prepare Dataset

1. Upload FreiHAND dataset to Google Drive:
   - Zip your `freihand` folder from `~/datasets/freihand`
   - Upload to Google Drive (e.g., `MyDrive/datasets/freihand.zip`)
   - Or upload the entire folder

2. The notebook will automatically:
   - Mount Google Drive
   - Copy dataset to `/content/freihand/`
   - Verify dataset structure

### 3. Train Model

Run all cells in order. The notebook includes:
- Environment setup
- Dataset loading (130K images, 90/10 train/val split)
- Model creation (ResNet or Transformer)
- Training loop with TensorBoard logging
- Evaluation with detailed metrics
- Checkpoint saving to Google Drive

### 4. Export Results

Checkpoints and results are automatically saved to:
- Google Drive: `MyDrive/asha_checkpoints/`
- Local Colab: `/content/asha/checkpoints/` (backup)

Download from Drive or access via Colab file browser.

## Integration

After training, integrate the model into `main_v2.py`:

1. Load trained checkpoint
2. Replace MediaPipe + IK with model inference
3. Use as primary method, fallback to IK if confidence low

## Files

- `train_colab.ipynb`: Complete Colab notebook (all code inline, no external dependencies)
- `README.md`: This file
