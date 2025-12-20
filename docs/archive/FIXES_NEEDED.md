# Fixes Needed for train_colab.ipynb

## Current Issue

**Problem:** DataLoader workers are crashing during evaluation (Section 14) and diagnostics (Section 16) with error:
```
RuntimeError: DataLoader worker (pid(s) 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035) exited unexpectedly
```

**Root Cause:** The `val_loader` created in Section 7 uses `num_workers=8` (from config), which causes multiprocessing issues during evaluation.

## Required Fixes

### Fix 1: Section 14 (Evaluation) - Create Safe Evaluation Dataloader

**Location:** Section 14, right before `print("\n[Evaluating on validation set]")`

**What to add:**
```python
# recreate val_loader with 0 workers to avoid worker crashes
from torch.utils.data import DataLoader
eval_val_dataset = FreiHANDDataset(
    dataset_path=config['dataset_path'],
    split="val",
    use_augmentation=False,  # no augmentation for eval
    image_size=config['image_size'],
    use_all_views=config['use_all_views'],
)
eval_val_loader = DataLoader(
    eval_val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=0,  # single-threaded to avoid worker crashes
    pin_memory=False,
)
print(f"Created evaluation dataloader with {len(eval_val_dataset)} samples")

print("\n[Evaluating on validation set]")
with torch.no_grad():
    for batch in tqdm(eval_val_loader):  # <-- Change val_loader to eval_val_loader
        # ... rest of evaluation code
```

### Fix 2: Section 16 (Diagnostics) - Use Safe Dataloader

**Location:** Section 16, where `sample_batch = next(iter(val_loader))` appears

**What to change:**
1. Before `sample_batch = next(iter(val_loader))`, add:
```python
# create eval dataloader if not already created
if 'eval_val_loader' not in locals():
    from torch.utils.data import DataLoader
    eval_val_dataset = FreiHANDDataset(
        dataset_path=config['dataset_path'],
        split="val",
        use_augmentation=False,
        image_size=config['image_size'],
        use_all_views=config['use_all_views'],
    )
    eval_val_loader = DataLoader(
        eval_val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
    )
```

2. Change `sample_batch = next(iter(val_loader))` to `sample_batch = next(iter(eval_val_loader))`

### Fix 3: Section 15 (Evaluation Plots) - Ensure `results` Variable Exists

**Issue:** `NameError: name 'results' is not defined` when plotting evaluation results.

**Location:** Section 15 (Evaluation Results Visualization)

**What to check:**
- Ensure Section 14 (Evaluation) completes successfully and creates the `results` dictionary
- The `results` dictionary should contain:
  - `'per_joint_error_mm'`: list of per-joint errors
  - `'joint_error_mm'`: dict with 'mean', 'median', 'p95'
  - `'theta_error'`: dict with 'mean', 'std', 'min', 'max'
  - `'beta_error'`: dict with 'mean', 'std', 'min', 'max'
  - `'mpjpe_mm'`: float

**Fix:** Make sure Section 14 runs completely before Section 15. If `results` is not defined, re-run Section 14 first.

## Verification Steps

After applying fixes:

1. **Check Section 14:**
   - Search for `eval_val_loader` - should exist
   - Search for `num_workers=0` in evaluation dataloader
   - Search for `for batch in tqdm(eval_val_loader)` (not `val_loader`)

2. **Check Section 16:**
   - Search for `sample_batch = next(iter(eval_val_loader))` (not `val_loader`)
   - Should have `eval_val_loader` creation code before this line

3. **Test:**
   - Run Section 14 - should complete without worker crashes
   - Run Section 15 - should have `results` variable defined
   - Run Section 16 - should use `eval_val_loader` without crashes

## Current Training Status

- **Model:** ResNet-50 + MLP
- **Training:** Completed 25 epochs (interrupted at epoch 26)
- **Loss:** Train: 1.4107, Val: 1.4088 (still high, loss plateauing)
- **Checkpoint:** Saved at epoch 22 (val_loss: 1.4088)
- **Next Steps:** 
  - Fix evaluation dataloader issues
  - Complete evaluation to get MPJPE metrics
  - Consider hyperparameter tuning (lambda_joint=1.0, lr=5e-4) or Transformer architecture

## Key Files

- **Notebook:** `src/training/train_colab.ipynb`
- **Config:** Section 3 (Training Configuration)
- **Dataset Loader:** Section 6 (FreiHANDDataset class)
- **Evaluation:** Section 14
- **Plots:** Section 15 (Evaluation Results Visualization)
- **Diagnostics:** Section 16 (Model Prediction Diagnostics)

## Notes

- The training dataloader (`train_loader`) can keep `num_workers=8` - only evaluation needs 0 workers
- This is a common issue in Colab when using multiprocessing with DataLoaders
- Single-threaded evaluation is slower but stable and acceptable for evaluation runs

