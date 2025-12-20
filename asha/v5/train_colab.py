"""
Train EMG→Joints model on Google Colab (direct joint prediction).

Simplified approach - predicts MediaPipe joints directly instead of MANO parameters.

Setup:
1. Upload to Google Drive:
   - MyDrive/Colab Notebooks/asha/data_v2/*.h5 (5 recording files)

2. Run in Colab:
   !python train_joints_colab.py

Output: 21 joints × 3 coords = 63 values (vs 45 MANO params)
Target: MPJPE < 15mm (better than v2's 34.92mm)
"""

import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# Model (EMGToJointsModel - inline for Colab compatibility)
# ============================================================================

class EMGToJointsModel(nn.Module):
    """EMG→Joints model (v5 architecture)."""

    def __init__(self, window_size=50, input_channels=8):
        super().__init__()

        # Conv1D feature extraction
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)

        # LSTM temporal modeling
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.3)

        # Output head (21 joints × 3 coords = 63)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 63)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [batch, 8, time] EMG input

        Returns:
            joints: [batch, 63] flattened joint coords
        """
        # x: [batch, 8, time]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.transpose(1, 2)  # [batch, time, channels]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last timestep

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        joints = self.fc2(x)  # [batch, 63]

        return joints

    def freeze_feature_extractor(self):
        """Freeze Conv1D and LSTM layers, only train FC layers."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False
        print("✅ Froze Conv1D + LSTM layers (only training FC layers)")

    def unfreeze_last_lstm_layer(self):
        """Optionally unfreeze last LSTM layer for more adaptation."""
        # Unfreeze last LSTM layer (layer 1)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh_l1' in name or 'weight_ih_l1' in name or 'bias_hh_l1' in name or 'bias_ih_l1' in name:
                param.requires_grad = True
        print("✅ Unfroze last LSTM layer for adaptation")


# ============================================================================
# Dataset
# ============================================================================

class EMGJointsDataset(Dataset):
    """Dataset for EMG→Joints training."""

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
                    # joints_3d: [N_pose, 21, 3] → interpolate to [N_emg, 21, 3]
                    joints_interp = np.zeros((len(emg_ts), 21, 3), dtype=np.float32)
                    for joint_idx in range(21):
                        for coord_idx in range(3):
                            joints_interp[:, joint_idx, coord_idx] = np.interp(
                                emg_ts, pose_ts, joints_3d[:, joint_idx, coord_idx]
                            )

                    # Flatten joints for storage: [N_emg, 21, 3] → [N_emg, 63]
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
    """
    Mean Per-Joint Position Error (MPJPE).

    Args:
        pred_joints: [batch, 63] predicted joint coords
        gt_joints: [batch, 63] ground truth joint coords

    Returns:
        loss: scalar MPJPE in millimeters
    """
    # Reshape to [batch, 21, 3]
    pred = pred_joints.view(-1, 21, 3)
    gt = gt_joints.view(-1, 21, 3)

    # Compute L2 distance per joint
    dist = torch.norm(pred - gt, dim=2)  # [batch, 21]

    # Mean over joints and batch
    mpjpe = dist.mean() * 1000  # Convert to mm

    return mpjpe


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mpjpe = 0

    for emg, joints_gt in tqdm(dataloader, desc="Training", leave=False):
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
        total_mpjpe += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_mpjpe = total_mpjpe / len(dataloader)

    return avg_loss, avg_mpjpe


def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_mpjpe = 0

    with torch.no_grad():
        for emg, joints_gt in tqdm(dataloader, desc="Evaluating", leave=False):
            emg = emg.to(device)
            joints_gt = joints_gt.to(device)

            joints_pred = model(emg)
            loss = mpjpe_loss(joints_pred, joints_gt)

            total_loss += loss.item()
            total_mpjpe += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_mpjpe = total_mpjpe / len(dataloader)

    return avg_loss, avg_mpjpe


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("="*60)
    print("EMG→Joints Training (Direct Joint Prediction - Colab)")
    print("="*60)

    # Mount Google Drive
    try:
        from google.colab import drive

        # Check if already mounted
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
            print("✅ Google Drive mounted")
        else:
            print("✅ Google Drive already mounted")

        # Colab paths (use strings, not Path objects)
        data_dir = "/content/drive/MyDrive/Colab Notebooks/asha/data_v2"
        checkpoint_dir = "/content/drive/MyDrive/Colab Notebooks/asha/checkpoints_joints"

    except ImportError:
        # Local paths (fallback)
        print("⚠️  Not running in Colab, using local paths")
        data_dir = "data/v4/emg_recordings"
        checkpoint_dir = "models/v5"

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Device: {device} ({gpu_name})")
    else:
        print(f"✅ Device: {device}")

    # Load data
    print("\nLoading dataset...")
    hdf5_files = sorted(glob(f"{data_dir}/*.h5"))
    print(f"Found {len(hdf5_files)} HDF5 files")

    if len(hdf5_files) == 0:
        print(f"ERROR: No HDF5 files found in {data_dir}")
        return

    # Create dataset
    full_dataset = EMGJointsDataset(hdf5_files, window_size=50)

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Model
    print("\nInitializing model...")
    model = EMGToJointsModel(window_size=50, input_channels=8).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Training
    num_epochs = 100
    best_val_mpjpe = float('inf')

    train_losses = []
    val_losses = []
    train_mpjpes = []
    val_mpjpes = []

    print(f"\nStarting training ({num_epochs} epochs)...")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss, train_mpjpe = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_mpjpes.append(train_mpjpe)

        # Evaluate
        val_loss, val_mpjpe = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_mpjpes.append(val_mpjpe)

        # LR schedule
        scheduler.step(val_loss)

        # Print
        print(f"  Train - Loss: {train_loss:.6f}, MPJPE: {train_mpjpe:.2f}mm")
        print(f"  Val   - Loss: {val_loss:.6f}, MPJPE: {val_mpjpe:.2f}mm")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mpjpe': val_mpjpe,
                'train_mpjpe': train_mpjpe,
                'emg_mean': full_dataset.emg_mean,  # CRITICAL: Save global statistics
                'emg_std': full_dataset.emg_std,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/emg_joints_best.pth")
            print(f"  ✅ Saved best model (MPJPE: {val_mpjpe:.2f}mm)")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mpjpe': val_mpjpe,
                'emg_mean': full_dataset.emg_mean,  # CRITICAL: Save global statistics
                'emg_std': full_dataset.emg_std,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/emg_joints_epoch{epoch}.pth")

    # Plot training curves
    print("\nPlotting training curves...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_mpjpes, label='Train')
    plt.plot(val_mpjpes, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (mm)')
    plt.title('Joint Prediction Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{checkpoint_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {checkpoint_dir}/training_curves.png")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val MPJPE: {best_val_mpjpe:.2f}mm (epoch {epoch})")
    print(f"Model saved: {checkpoint_dir}/emg_joints_best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
