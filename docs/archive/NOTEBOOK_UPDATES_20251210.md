# Notebook Updates Required

This document summarizes all updates needed in `train_colab.ipynb` to complete the synthesis requirements.

## Status Overview

- ✅ Config updated (lambda_joint=5.0)
- ✅ LR scheduler updated (cosine with restarts every 20 epochs)
- ✅ Enhanced losses import cell added
- ⚠️ Enhanced losses: Need to import from losses.py or update inline
- ⚠️ Augmentations: Ready to add to dataset transforms
- ⚠️ Diversity metrics: Need to add to evaluation section
- ⚠️ Multi-task head: Need to add to model architecture
- ⚠️ Dataset loader: Need to integrate curriculum mixing loader

## 1. Enhanced Loss Function

**Status**: Config updated (lambda_joint=5.0), import cell added

**Option A - Import from losses.py** (Recommended):
```python
# After cloning repo, add this cell:
import sys
sys.path.append('/content/asha/src')
from training.losses import MANOLoss as EnhancedMANOLoss, SimpleMANOLoss

# Then replace criterion initialization with:
criterion = EnhancedMANOLoss(
    lambda_param=config['lambda_param'],
    lambda_joint=config['lambda_joint'],  # Now 5.0
    lambda_angular=1.0,  # New: quaternion geodesic loss
    lambda_reg=config['lambda_reg'],
    lambda_pca=0.1,  # New: PCA prior
    mano_layer=mano_layer,
)
```

**Option B - Update inline MANOLoss class**:
See `src/training/losses.py` for complete implementation with:
- Quaternion geodesic angular loss
- Per-joint weights (proximal 2.0, distal 1.0, higher for pinky/thumb)
- Soft tanh clipping on θ ranges
- Gaussian PCA prior

## 2. Enhanced Data Augmentations

**Location**: `FreiHANDDataset.__init__` method, around line 894

**Replace**:
```python
A.Rotate(limit=15, p=0.5),
```

**With**:
```python
A.Rotate(limit=30, p=0.5),  # ±30° rotations (increased from 15°)
A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),  # Scaling 0.8-1.2
A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),  # CutMix-like occlusions (mimics DexYCB/HO-3D)
```

## 3. Cosine LR Scheduler with Restarts

**Status**: Scheduler updated to restart every 20 epochs

**Remaining**: Update training loop (around line 1650) to reset scheduler:
```python
# In training loop, after each epoch:
if epoch > 0 and epoch % 20 == 0:
    # Reset scheduler for restart
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    print(f"  LR scheduler restarted at epoch {epoch+1}")
```

## 4. Diversity Metrics (Evaluation Section)

**Location**: Evaluation section (around line 1966), after computing joint errors

**Add after** `joint_errors.append(batch_mpjpe)`:
```python
# Diversity metrics
# 1. θ variance (target >0.5 std)
theta_std = torch.std(pred_theta, dim=0).mean().item()
diversity_theta_variance = theta_std

# 2. Prediction entropy (>2.0 bits across batches)
theta_binned = torch.floor((pred_theta + np.pi) / (2 * np.pi / 10)).long()
entropy_per_batch = []
for b in range(pred_theta.shape[0]):
    hist = torch.bincount(theta_binned[b].flatten(), minlength=10)
    probs = hist.float() / hist.sum()
    probs = probs[probs > 0]
    entropy = -(probs * torch.log2(probs)).sum().item()
    entropy_per_batch.append(entropy)
diversity_entropy = np.mean(entropy_per_batch)

# 3. Gesture F1 (classify 5 discrete poses)
def classify_gesture(theta):
    """Classify gesture: fist, open, point, pinch, rest."""
    thumb_ext = theta[1:4].abs().mean()
    index_ext = theta[4:7].abs().mean()
    middle_ext = theta[7:10].abs().mean()
    if thumb_ext < 0.3 and index_ext < 0.3 and middle_ext < 0.3:
        return 0  # Fist
    elif thumb_ext > 0.8 and index_ext > 0.8 and middle_ext > 0.8:
        return 1  # Open
    elif index_ext > 0.8 and middle_ext < 0.3:
        return 2  # Point
    elif thumb_ext > 0.6 and index_ext > 0.6 and middle_ext < 0.5:
        return 3  # Pinch
    else:
        return 4  # Rest

pred_gestures = [classify_gesture(pred_theta[i]) for i in range(pred_theta.shape[0])]
gt_gestures = [classify_gesture(gt_theta[i]) for i in range(gt_theta.shape[0])]

from sklearn.metrics import f1_score
gesture_f1 = f1_score(gt_gestures, pred_gestures, average='weighted')

# Store diversity metrics
diversity_metrics = {
    'theta_variance': diversity_theta_variance,
    'entropy': diversity_entropy,
    'gesture_f1': gesture_f1,
}
```

**Add to results dictionary** (around line 2005):
```python
results['diversity'] = {
    'theta_variance': np.mean([m['theta_variance'] for m in diversity_metrics_list]),
    'entropy': np.mean([m['entropy'] for m in diversity_metrics_list]),
    'gesture_f1': np.mean([m['gesture_f1'] for m in diversity_metrics_list]),
}
```

## 5. Multi-Task: 2D Keypoint Heatmap Head

**Location**: Model architecture (around line 480-650)

**For ResNetMANO**, add after MLP head:
```python
# Auxiliary head for 2D keypoint heatmaps
self.heatmap_head = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, 21 * 64 * 64),  # 21 keypoints × 64×64 heatmaps
)
```

**For TransformerMANO**, add after MLP head:
```python
# Auxiliary head for 2D keypoint heatmaps
self.heatmap_head = nn.Sequential(
    nn.Linear(d_model, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 21 * 64 * 64),  # 21 keypoints × 64×64 heatmaps
)
```

**Update forward pass**:
```python
def forward(self, x):
    # ... existing forward code ...
    theta = self.head(features)
    heatmaps = self.heatmap_head(features).view(-1, 21, 64, 64)
    
    if self.predict_beta:
        beta = self.beta_head(features)
        return theta, beta, heatmaps
    else:
        return theta, heatmaps
```

**Update loss computation** (around line 1450):
```python
# Forward pass
if hasattr(model, 'predict_beta') and model.predict_beta:
    pred_theta, pred_beta, pred_heatmaps = model(images)
else:
    pred_theta, pred_heatmaps = model(images)
    pred_beta = torch.zeros_like(gt_beta)

# ... existing loss computation ...

# 2D heatmap loss (small weight 0.1)
if 'heatmaps_2d' in batch:
    gt_heatmaps = batch['heatmaps_2d'].to(device)
    loss_heatmap = F.mse_loss(pred_heatmaps, gt_heatmaps)
    total_loss += 0.1 * loss_heatmap
    losses['heatmap'] = loss_heatmap
```

**Generate 2D heatmaps in dataset** (update `FreiHANDDataset.__getitem__`):
```python
def create_heatmaps(joints_2d, heatmap_size=64):
    """Create Gaussian heatmaps for 2D keypoints."""
    heatmaps = np.zeros((21, heatmap_size, heatmap_size))
    for i, joint in enumerate(joints_2d):
        x, y = joint[0], joint[1]
        x_norm = (x + 1) / 2  # Normalize to [0, 1]
        y_norm = (y + 1) / 2
        x_pixel = int(x_norm * heatmap_size)
        y_pixel = int(y_norm * heatmap_size)
        xx, yy = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size))
        sigma = 2.0
        heatmap = np.exp(-((xx - x_pixel)**2 + (yy - y_pixel)**2) / (2 * sigma**2))
        heatmaps[i] = heatmap
    return heatmaps

# In __getitem__:
joints_2d_norm = joints_3d[:, :2]  # Use x, y coordinates
heatmaps_2d = create_heatmaps(joints_2d_norm, heatmap_size=64)

return {
    'image': image_tensor,
    'theta': theta,
    'beta': beta,
    'joints_3d': joints_3d,
    'heatmaps_2d': torch.from_numpy(heatmaps_2d),  # Add heatmaps
    'base_idx': base_idx,
    'view_idx': sample['view_idx'],
}
```

## 6. Curriculum Dataset Loader Integration

**Location**: Replace `create_dataloaders` function (around line 944)

**Option A - Use curriculum mixing loader**:
```python
# Import curriculum loader
import sys
sys.path.append('/content/asha/src')
from training.dataset_loader import create_dataloaders as create_curriculum_dataloaders

# Replace existing create_dataloaders call with:
train_loader, val_loader = create_curriculum_dataloaders(
    freihand_path=config['dataset_path'],
    ho3d_path=None,  # Set if HO-3D available
    dexycb_path=None,  # Set if DexYCB available
    batch_size=config['batch_size'],
    num_workers=config['num_workers'],
    epoch=epoch,  # Current epoch for curriculum mixing
    use_curriculum=True,
    ramp_epochs=20,
)
```

**Option B - Update existing function**:
Add epoch parameter and implement mixing logic (see `src/training/dataset_loader.py` for reference)

## Quick Integration Checklist

1. ✅ Config: `lambda_joint=5.0` (already done)
2. ✅ LR scheduler: Cosine with restarts (already done)
3. ⚠️ Import enhanced losses or update inline MANOLoss class
4. ⚠️ Update augmentations: rotations ±30°, scaling, CutMix
5. ⚠️ Add diversity metrics to evaluation
6. ⚠️ Add multi-task 2D heatmap head to model
7. ⚠️ Integrate curriculum dataset loader
8. ⚠️ Add scheduler reset logic in training loop

## Implementation Order

**Priority 1** (Critical for training):
1. Enhanced losses (import or update inline)
2. Enhanced augmentations
3. Scheduler reset logic

**Priority 2** (Improves evaluation):
4. Diversity metrics
5. Multi-task heatmap head

**Priority 3** (Dataset diversity):
6. Curriculum dataset loader (requires HO-3D/DexYCB datasets)

Most critical: Update MANOLoss to use enhanced version, add augmentations, and integrate curriculum dataset loader.
