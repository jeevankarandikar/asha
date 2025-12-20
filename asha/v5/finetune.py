"""
Fine-tune EMG→Joints model for electrode drift adaptation.

Quick calibration procedure:
1. Record 2-3 min of new data with current electrode placement
2. Fine-tune last layers only (freeze feature extraction)
3. Test with real-time inference

Usage:
    python -m asha.v5.finetune \
        --model models/v5/emg_joints_best.pth \
        --data data/v5/calibration_*.h5 \
        --epochs 15 \
        --output models/v5/emg_joints_finetuned.pth
"""

import os
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import EMGToJointsModel


# ============================================================================
# Model imported from .model (shared with train_colab.py)
# ============================================================================
# Model definition moved to asha/v5/model.py for reusability


# ============================================================================
# Dataset (same as train_joints_colab.py)
# ============================================================================

class EMGJointsDataset(Dataset):
    """Dataset for EMG→Joints fine-tuning."""

    def __init__(self, hdf5_paths, window_size=50):
        self.hdf5_paths = hdf5_paths
        self.window_size = window_size
        self.fs_emg = 500

        self.file_metadata = []
        self.sample_indices = []

        # Compute global EMG statistics FIRST (before normalization)
        self.emg_mean, self.emg_std = self._compute_global_statistics()

        self._build_index()

        print(f"EMGJointsDataset initialized:")
        print(f"  Files: {len(self.hdf5_paths)}")
        print(f"  Samples: {len(self.sample_indices)}")
        print(f"  Window: {window_size} samples ({window_size/self.fs_emg*1000:.0f}ms)")
        print(f"  Global EMG mean: {self.emg_mean}")
        print(f"  Global EMG std: {self.emg_std}")

    def _compute_global_statistics(self):
        """Compute global EMG mean/std across ALL files for consistent normalization."""
        print("Computing global EMG statistics across all files...")
        all_emg = []

        for path in self.hdf5_paths:
            try:
                with h5py.File(path, 'r') as f:
                    emg = f['emg/filtered'][:]
                    all_emg.append(emg)
            except Exception as e:
                print(f"  ERROR loading {path} for statistics: {e}")
                continue

        # Concatenate all EMG data
        all_emg = np.concatenate(all_emg, axis=0)  # [total_samples, 8]

        # Compute global statistics (per-channel)
        emg_mean = all_emg.mean(axis=0, keepdims=True)  # [1, 8]
        emg_std = all_emg.std(axis=0, keepdims=True) + 1e-6  # [1, 8]

        print(f"  Loaded {all_emg.shape[0]:,} total EMG samples from {len(self.hdf5_paths)} files")
        print(f"  Global statistics computed (shape: {emg_mean.shape})")

        return emg_mean, emg_std

    def _build_index(self):
        for file_idx, path in enumerate(self.hdf5_paths):
            path_obj = Path(path)
            try:
                with h5py.File(path, 'r') as f:
                    emg = f['emg/filtered'][:]
                    emg_ts = f['emg/timestamps'][:]
                    joints_3d = f['pose/joints_3d'][:]  # [N, 21, 3]
                    pose_ts = f['pose/timestamps'][:]
                    ik_error = f['pose/ik_error'][:]
                    mp_conf = f['pose/mp_confidence'][:]

                    # Filter low-quality frames
                    valid_mask = (ik_error < 15.0) & (mp_conf > 0.5)
                    joints_3d = joints_3d[valid_mask]
                    pose_ts = pose_ts[valid_mask]

                    print(f"  Loaded {path_obj.name}: {len(emg)} EMG samples, {len(joints_3d)} pose samples ({valid_mask.sum()}/{len(valid_mask)} valid)")

                    if len(joints_3d) == 0:
                        print(f"  WARNING: No valid pose samples in {path_obj.name}, skipping")
                        continue

                    # Interpolate joint data to EMG rate
                    joints_interp = np.zeros((len(emg_ts), 21, 3), dtype=np.float32)
                    for joint_idx in range(21):
                        for coord_idx in range(3):
                            joints_interp[:, joint_idx, coord_idx] = np.interp(
                                emg_ts, pose_ts, joints_3d[:, joint_idx, coord_idx]
                            )

                    # Flatten joints: [N_emg, 21, 3] → [N_emg, 63]
                    joints_flat = joints_interp.reshape(len(emg_ts), -1)

                    # Normalize EMG using GLOBAL statistics (not per-file!)
                    emg = (emg - self.emg_mean) / self.emg_std

                    # Store metadata
                    self.file_metadata.append({
                        'emg': emg,
                        'joints': joints_flat,
                    })

                    # Create sliding windows (50% overlap)
                    num_frames = len(emg)
                    stride = self.window_size // 2
                    for start in range(0, num_frames - self.window_size + 1, stride):
                        self.sample_indices.append((file_idx, start))

            except Exception as e:
                print(f"  ERROR loading {path}: {e}")
                continue

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        file_idx, start = self.sample_indices[idx]
        metadata = self.file_metadata[file_idx]

        end = start + self.window_size
        emg_window = metadata['emg'][start:end]  # [50, 8]
        joints_window = metadata['joints'][start:end]  # [50, 63]

        # Target is last frame
        target_joints = joints_window[-1]  # [63]

        # Transpose EMG: [50, 8] → [8, 50]
        emg_tensor = torch.from_numpy(emg_window.T).float()
        joints_tensor = torch.from_numpy(target_joints).float()

        return emg_tensor, joints_tensor


# ============================================================================
# Loss Function
# ============================================================================

def mpjpe_loss(pred_joints, gt_joints):
    """Mean Per-Joint Position Error (MPJPE)."""
    pred = pred_joints.view(-1, 21, 3)
    gt = gt_joints.view(-1, 21, 3)
    dist = torch.norm(pred - gt, dim=2)  # [batch, 21]
    mpjpe = dist.mean() * 1000  # Convert to mm
    return mpjpe


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for emg, joints_gt in tqdm(dataloader, desc="Fine-tuning", leave=False):
        emg = emg.to(device)
        joints_gt = joints_gt.to(device)

        # Forward
        joints_pred = model(emg)
        loss = mpjpe_loss(joints_pred, joints_gt)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for emg, joints_gt in tqdm(dataloader, desc="Evaluating", leave=False):
            emg = emg.to(device)
            joints_gt = joints_gt.to(device)

            joints_pred = model(emg)
            loss = mpjpe_loss(joints_pred, joints_gt)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ============================================================================
# Main Fine-tuning Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune EMG→Joints model")
    parser.add_argument("--model", type=str, required=True, help="Path to pre-trained model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path pattern to calibration data (e.g., data/*.h5)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lower than training)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--unfreeze-lstm", action="store_true", help="Unfreeze last LSTM layer")
    parser.add_argument("--output", type=str, default="models/v5/emg_joints_finetuned.pth", help="Output model path")
    args = parser.parse_args()

    print("="*60)
    print("EMG→Joints Fine-tuning (Electrode Drift Adaptation)")
    print("="*60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Device: {device} ({gpu_name})")
    else:
        print(f"✅ Device: {device}")

    # Load pre-trained model
    print(f"\nLoading pre-trained model: {args.model}")
    model = EMGToJointsModel(window_size=50, input_channels=8).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get original training performance
    original_mpjpe = checkpoint.get('val_mpjpe', 'unknown')
    original_epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Original model: Epoch {original_epoch}, Val MPJPE: {original_mpjpe}mm")

    # Freeze feature extractor
    model.freeze_feature_extractor()
    if args.unfreeze_lstm:
        model.unfreeze_last_lstm_layer()

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Load calibration data
    print(f"\nLoading calibration data: {args.data}")
    hdf5_files = sorted(glob(args.data))
    print(f"Found {len(hdf5_files)} HDF5 files")

    if len(hdf5_files) == 0:
        print(f"ERROR: No HDF5 files found matching pattern: {args.data}")
        return

    # Create dataset
    full_dataset = EMGJointsDataset(hdf5_files, window_size=50)

    # Split 80/20 (use less validation data since calibration dataset is small)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer (low learning rate for fine-tuning)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Fine-tuning
    print(f"\nStarting fine-tuning ({args.epochs} epochs, lr={args.lr})...")
    best_val_mpjpe = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        # LR schedule
        scheduler.step(val_loss)

        # Print
        print(f"  Train MPJPE: {train_loss:.2f}mm")
        print(f"  Val MPJPE: {val_loss:.2f}mm")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if val_loss < best_val_mpjpe:
            best_val_mpjpe = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mpjpe': val_loss,
                'train_mpjpe': train_loss,
                'original_model': args.model,
                'original_mpjpe': original_mpjpe,
                'finetuned': True,
                'emg_mean': full_dataset.emg_mean,  # CRITICAL: Save global statistics
                'emg_std': full_dataset.emg_std,
            }
            torch.save(checkpoint, args.output)
            print(f"  ✅ Saved fine-tuned model (MPJPE: {val_loss:.2f}mm)")

    # Plot training curves
    output_dir = Path(args.output).parent
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train MPJPE')
    plt.plot(val_losses, label='Val MPJPE')
    plt.axhline(y=original_mpjpe, color='r', linestyle='--', label=f'Original ({original_mpjpe:.2f}mm)')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (mm)')
    plt.title('Fine-tuning Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "finetuning_curves.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'finetuning_curves.png'}")

    print(f"\n{'='*60}")
    print(f"Fine-tuning complete!")
    print(f"Original model: {original_mpjpe:.2f}mm (epoch {original_epoch})")
    print(f"Fine-tuned model: {best_val_mpjpe:.2f}mm (epoch {epoch})")
    improvement = original_mpjpe - best_val_mpjpe if isinstance(original_mpjpe, (int, float)) else 0
    print(f"Improvement: {improvement:.2f}mm")
    print(f"Model saved: {args.output}")
    print(f"{'='*60}")

    print("\n📋 Next steps:")
    print(f"1. Test fine-tuned model:")
    print(f"   python src/programs/inference_joints.py --model {args.output}")
    print(f"2. If still stuck, record more diverse calibration data")
    print(f"3. Try --unfreeze-lstm flag for stronger adaptation")


if __name__ == "__main__":
    main()
