"""
Train EMG→MANO model on Google Colab (v2 simplified data).

Self-contained script - all dependencies embedded.

Setup:
1. Upload to Google Drive:
   - MyDrive/Colab Notebooks/asha/data_v2/*.h5 (5 recording files)
   - MyDrive/Colab Notebooks/asha/MANO_RIGHT_numpy.pkl

2. Run in Colab:
   !python train_v2_colab.py

Expected: Val MPJPE < 10mm after 100 epochs (~1-2 hours on A100 GPU)
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# MANO Layer (CustomMANOLayer - embedded from src/model_utils/mano_model.py)
# ============================================================================

class CustomMANOLayer:
    """Custom MANO model loader (numpy-based, Python 3.11 compatible)."""

    def __init__(self, mano_path: str, side: str = "right", device: str = "cpu"):
        """Initialize MANO model from converted numpy pkl file."""
        pkl_path = Path(mano_path)

        if not pkl_path.exists():
            raise FileNotFoundError(f"MANO model not found: {pkl_path}")

        # Load pkl
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.device = device
        self.side = side.upper()

        # Convert to tensors
        self.shapedirs = torch.from_numpy(data['shapedirs'].astype(np.float32)).to(device)
        self.posedirs = torch.from_numpy(data['posedirs'].astype(np.float32)).to(device)
        self.v_template = torch.from_numpy(data['v_template'].astype(np.float32)).to(device)
        self.J_regressor = torch.from_numpy(data['J_regressor'].toarray().astype(np.float32)).to(device)
        self.weights = torch.from_numpy(data['weights'].astype(np.float32)).to(device)
        self.kintree_table = data['kintree_table']
        self.faces = torch.from_numpy(data['f'].astype(np.int64)).to(device)

    def __call__(self, pose_params: torch.Tensor, betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: MANO params → 3D hand mesh.

        Args:
            pose_params: [batch, 45] hand pose (15 joints × 3 axis-angle)
            betas: [batch, 10] shape parameters

        Returns:
            vertices: [batch, 778, 3] hand mesh vertices
            joints: [batch, 21, 3] 3D joint positions (16 skeleton + 5 fingertips)
        """
        batch_size = pose_params.shape[0]

        # Apply shape blend shapes
        v_shaped = self.v_template.unsqueeze(0) + torch.einsum('bl,mkl->bmk', betas, self.shapedirs)

        # Compute skeleton joints from shaped vertices
        J = torch.einsum('bik,ji->bjk', v_shaped, self.J_regressor)

        # Convert pose to rotation matrices (simplified - uses Rodriguez formula)
        pose_matrices = self._batch_rodrigues(pose_params.view(-1, 3)).view(batch_size, -1, 3, 3)

        # Apply pose blend shapes (use all 15 joints, not just 14)
        pose_feature = (pose_matrices - torch.eye(3, device=self.device)).view(batch_size, -1)
        v_posed = v_shaped + torch.einsum('bl,mkl->bmk', pose_feature, self.posedirs)

        # Linear blend skinning (LBS)
        T = self._global_rigid_transformation(pose_matrices, J)
        vertices = self._skinning(v_posed, self.weights, T)

        # Compute 21 output joints (16 skeleton + 5 fingertips)
        joints_skeleton = torch.einsum('bik,ji->bjk', vertices, self.J_regressor)

        # Fingertip indices (vertices at fingertips)
        fingertip_indices = [745, 317, 444, 556, 673]  # thumb, index, middle, ring, pinky
        fingertips = vertices[:, fingertip_indices, :]

        joints = torch.cat([joints_skeleton, fingertips], dim=1)  # [batch, 21, 3]

        return vertices, joints

    def _batch_rodrigues(self, rot_vecs: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrices (Rodriguez formula)."""
        batch_size = rot_vecs.shape[0]
        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # Outer product
        outer = torch.einsum('bi,bj->bij', rot_dir, rot_dir)

        # Skew-symmetric matrix
        K = torch.zeros((batch_size, 3, 3), device=self.device)
        K[:, 0, 1] = -rot_dir[:, 2]
        K[:, 0, 2] = rot_dir[:, 1]
        K[:, 1, 0] = rot_dir[:, 2]
        K[:, 1, 2] = -rot_dir[:, 0]
        K[:, 2, 0] = -rot_dir[:, 1]
        K[:, 2, 1] = rot_dir[:, 0]

        # Rodriguez formula: R = I + sin(θ)K + (1-cos(θ))K²
        eye = torch.eye(3, device=self.device).unsqueeze(0)
        rot_mat = eye + torch.sin(angle).unsqueeze(1) * K + (1 - torch.cos(angle).unsqueeze(1)) * torch.bmm(K, K)

        return rot_mat

    def _global_rigid_transformation(self, pose_matrices: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """Compute global transformation matrices for each joint."""
        batch_size = pose_matrices.shape[0]
        num_joints = pose_matrices.shape[1]  # 15 rotation matrices

        # Initialize results
        results = []

        # Root joint (wrist)
        results.append(
            self._make_transform_matrix(pose_matrices[:, 0], joints[:, 0])
        )

        # Apply kinematic chain (only for joints we have rotation matrices for)
        for i in range(1, num_joints):
            parent = self.kintree_table[0, i]
            results.append(
                torch.matmul(
                    results[parent],
                    self._make_transform_matrix(
                        pose_matrices[:, i],
                        joints[:, i] - joints[:, parent]
                    )
                )
            )

        # Stack results
        results = torch.stack(results, dim=1)  # [batch, num_joints, 4, 4]

        return results

    def _make_transform_matrix(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Create 4×4 transformation matrix from rotation and translation."""
        batch_size = R.shape[0]
        T = torch.zeros((batch_size, 4, 4), device=self.device)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        T[:, 3, 3] = 1
        return T

    def _skinning(self, vertices: torch.Tensor, weights: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
        """Apply linear blend skinning."""
        batch_size = vertices.shape[0]
        num_verts = vertices.shape[1]
        num_joints = transforms.shape[1]  # Number of transformation matrices

        # Homogeneous coordinates
        v_homo = torch.cat([vertices, torch.ones(batch_size, num_verts, 1, device=self.device)], dim=2)

        # Weighted blend
        v_transformed = torch.zeros_like(v_homo)
        for i in range(num_joints):
            # Apply transformation matrix to vertices: [batch, 4, 4] @ [batch, num_verts, 4]
            transformed = torch.einsum('bij,bvj->bvi', transforms[:, i], v_homo)
            v_transformed += transformed * weights[:, i].unsqueeze(0).unsqueeze(2)

        return v_transformed[:, :, :3]


# ============================================================================
# Model (SimpleEMGModel - imported from .model)
# ============================================================================

from .model import SimpleEMGModel


# ============================================================================
# Dataset (EMGDataset - from train_v2.py)
# ============================================================================

class EMGDataset(Dataset):
    """Dataset for v2 EMG training."""

    def __init__(self, hdf5_paths, window_size=50):
        self.hdf5_paths = hdf5_paths
        self.window_size = window_size
        self.fs_emg = 500

        self.file_metadata = []
        self.sample_indices = []

        # Compute global EMG statistics FIRST (before normalization)
        self.emg_mean, self.emg_std = self._compute_global_statistics()

        self._build_index()

        print(f"EMGDataset initialized:")
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
                    theta = f['pose/mano_theta'][:]
                    pose_ts = f['pose/timestamps'][:]
                    ik_error = f['pose/ik_error'][:]
                    mp_conf = f['pose/mp_confidence'][:]

                    # Filter low-quality frames
                    valid_mask = (ik_error < 15.0) & (mp_conf > 0.5)
                    theta = theta[valid_mask]
                    pose_ts = pose_ts[valid_mask]

                    print(f"  Loaded {path_obj.name}: {len(emg)} EMG samples, {len(theta)} pose samples ({valid_mask.sum()}/{len(valid_mask)} valid)")

                    if len(theta) == 0:
                        print(f"  WARNING: No valid pose samples in {path_obj.name}, skipping")
                        continue

                    # Interpolate pose data to EMG rate
                    theta_interp = np.zeros((len(emg_ts), 45), dtype=np.float32)
                    for i in range(45):
                        theta_interp[:, i] = np.interp(emg_ts, pose_ts, theta[:, i])

                    # Normalize EMG using GLOBAL statistics (not per-file!)
                    emg = (emg - self.emg_mean) / self.emg_std

                    # Store metadata
                    self.file_metadata.append({
                        'emg': emg,
                        'theta': theta_interp,
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
        emg_window = metadata['emg'][start:end]
        theta_window = metadata['theta'][start:end]

        target_theta = theta_window[-1]

        emg_tensor = torch.from_numpy(emg_window.T).float()
        target_tensor = torch.from_numpy(target_theta).float()

        return emg_tensor, target_tensor


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, mano_layer, optimizer, device):
    model.train()
    total_loss = 0
    total_theta_loss = 0
    total_joints_loss = 0

    for emg, target_theta in tqdm(dataloader, desc="Training", leave=False):
        emg = emg.to(device)
        target_theta = target_theta.to(device)

        # Forward
        pred_theta = model(emg)

        # Compute joints via MANO
        betas = torch.zeros(pred_theta.shape[0], 10, device=device)
        _, pred_joints = mano_layer(pred_theta, betas)
        _, target_joints = mano_layer(target_theta, betas)

        # Loss: θ + joints
        theta_loss = F.mse_loss(pred_theta, target_theta)
        joints_loss = F.mse_loss(pred_joints, target_joints)
        loss = theta_loss + joints_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_theta_loss += theta_loss.item()
        total_joints_loss += joints_loss.item()

    return (total_loss / len(dataloader),
            total_theta_loss / len(dataloader),
            total_joints_loss / len(dataloader))


def validate(model, dataloader, mano_layer, device):
    model.eval()
    total_loss = 0
    total_mpjpe = 0

    with torch.no_grad():
        for emg, target_theta in dataloader:
            emg = emg.to(device)
            target_theta = target_theta.to(device)

            pred_theta = model(emg)

            betas = torch.zeros(pred_theta.shape[0], 10, device=device)
            _, pred_joints = mano_layer(pred_theta, betas)
            _, target_joints = mano_layer(target_theta, betas)

            theta_loss = F.mse_loss(pred_theta, target_theta)
            joints_loss = F.mse_loss(pred_joints, target_joints)
            loss = theta_loss + joints_loss

            mpjpe = torch.norm(pred_joints - target_joints, dim=-1).mean()

            total_loss += loss.item()
            total_mpjpe += mpjpe.item() * 1000  # Convert to mm

    return total_loss / len(dataloader), total_mpjpe / len(dataloader)


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("EMG→MANO Training (v2 Simplified Data - Colab)")
    print("="*60)

    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")

        # Colab paths
        data_dir = "/content/drive/MyDrive/Colab Notebooks/asha/data_v2"
        mano_path = "/content/drive/MyDrive/Colab Notebooks/asha/MANO_RIGHT_numpy.pkl"
        output_dir = "/content/drive/MyDrive/Colab Notebooks/asha/checkpoints_v2"

    except ImportError:
        # Local paths (fallback)
        print("⚠️  Not running in Colab, using local paths")
        data_dir = "data/v4/emg_recordings"
        mano_path = "models/mano/MANO_RIGHT_numpy.pkl"
        output_dir = "models/v4"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Device: {device} ({gpu_name})")
    else:
        print(f"✅ Device: {device}")

    # Load data files
    data_files = sorted(glob(f"{data_dir}/*.h5"))
    if len(data_files) == 0:
        print(f"❌ ERROR: No data files found in {data_dir}")
        print("   Upload your 5 .h5 files to Google Drive:")
        print("   MyDrive/Colab Notebooks/asha/data_v2/")
        sys.exit(1)

    print(f"\nData files: {len(data_files)}")
    for f in data_files:
        print(f"  - {Path(f).name}")

    # Check MANO
    if not Path(mano_path).exists():
        print(f"❌ ERROR: MANO model not found: {mano_path}")
        print("   Upload MANO_RIGHT_numpy.pkl to Google Drive:")
        print("   MyDrive/Colab Notebooks/asha/")
        sys.exit(1)

    print(f"\n✅ MANO model: {mano_path}")

    # Hyperparameters
    epochs = 100
    batch_size = 64  # Optimized for A100 GPU
    lr = 1e-3
    window_size = 50

    print(f"\nHyperparameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Window size: {window_size} samples (100ms)")
    print("="*60 + "\n")

    # Load MANO
    print("Loading MANO model...")
    mano_layer = CustomMANOLayer(mano_path, side='right', device=str(device))
    print("✅ MANO loaded")

    # Create model
    print("\nCreating model...")
    model = SimpleEMGModel(window_size=window_size, input_channels=8).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created: {trainable_params:,} trainable params")

    # Load dataset
    print("\nLoading dataset...")
    dataset = EMGDataset(data_files, window_size=window_size)

    if len(dataset) == 0:
        print("❌ ERROR: No valid samples in dataset!")
        sys.exit(1)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"✅ Dataset split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    best_mpjpe = float('inf')
    train_losses = []
    val_losses = []
    val_mpjpes = []

    print(f"\n{'='*60}")
    print("Starting training...")
    print("="*60 + "\n")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Train
        train_loss, train_theta_loss, train_joints_loss = train_epoch(
            model, train_loader, mano_layer, optimizer, device
        )

        # Validate
        val_loss, val_mpjpe = validate(model, val_loader, mano_layer, device)

        # Step scheduler
        scheduler.step()

        # Log
        print(f"  Train Loss: {train_loss:.6f} (θ: {train_theta_loss:.4f}, joints: {train_joints_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val MPJPE:  {val_mpjpe:.2f}mm")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mpjpes.append(val_mpjpe)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mpjpe = val_mpjpe
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mpjpe': val_mpjpe,
                'emg_mean': full_dataset.emg_mean,  # CRITICAL: Save global statistics
                'emg_std': full_dataset.emg_std,
            }, f"{output_dir}/emg_model_v2_best.pth")
            print(f"  ✅ Saved best model (val_loss={val_loss:.6f}, MPJPE={val_mpjpe:.2f}mm)")

        # Checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'emg_mean': full_dataset.emg_mean,  # CRITICAL: Save global statistics
                'emg_std': full_dataset.emg_std,
            }, f"{output_dir}/emg_model_v2_epoch{epoch}.pth")
            print(f"  💾 Checkpoint saved: epoch {epoch}")

        print()

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_mpjpes, label='Val MPJPE', color='green')
    plt.axhline(y=10, color='r', linestyle='--', label='Target (10mm)')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (mm)')
    plt.title('Validation MPJPE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves_v2.png", dpi=150)
    print(f"✅ Saved training curves: {output_dir}/training_curves_v2.png")

    # Save raw loss data for later plotting
    np.savez(f"{output_dir}/training_history_v2.npz",
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             val_mpjpes=np.array(val_mpjpes),
             epochs=np.arange(1, epochs + 1))
    print(f"✅ Saved loss data: {output_dir}/training_history_v2.npz")

    # Final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print("="*60)
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val MPJPE: {best_mpjpe:.2f}mm")
    print(f"Checkpoint: {output_dir}/emg_model_v2_best.pth")

    if best_mpjpe < 10:
        print("✅ TARGET ACHIEVED! MPJPE < 10mm")
    elif best_mpjpe < 20:
        print("✅ GOOD! MPJPE < 20mm")
    else:
        print("⚠️  NEEDS IMPROVEMENT. MPJPE >= 20mm")

    print("="*60)


if __name__ == "__main__":
    main()