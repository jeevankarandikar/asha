"""
Transfer Learning - Mode 2: emg2pose → MANO

Train ChannelAdapter + AngleConverter for MindRove → emg2pose → MANO pipeline.

Usage:
    python transfer/training/train_mode2.py --data transfer/data/*.h5 --epochs 100
"""

import sys
from pathlib import Path

# Add paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "transfer" / "emg2pose"))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import MANO
from model_utils.mano_model import CustomMANOLayer

# Import emg2pose (will need to clone repo first)
try:
    from emg2pose.lightning import Emg2PoseModule
    from emg2pose.kinematics import load_default_hand_model
except ImportError:
    print("ERROR: emg2pose not found!")
    print("Please clone emg2pose repo:")
    print("  git clone https://github.com/facebookresearch/emg2pose.git transfer/emg2pose")
    sys.exit(1)


# ============================================================================
# Components
# ============================================================================

class ChannelAdapter(nn.Module):
    """
    Adapt MindRove 8ch → emg2pose 16ch.

    Uses Conv1D with anatomically-motivated initialization.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [batch, 8, time]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x  # [batch, 16, time]


class FrequencyUpsampler(nn.Module):
    """Upsample from 500Hz to 2kHz (4x) via linear interpolation."""

    def forward(self, x):
        # x: [batch, channels, time]
        return F.interpolate(x, scale_factor=4, mode='linear', align_corners=True)


class AngleConverter(nn.Module):
    """
    Convert emg2pose 20 angles → MANO 45 θ parameters.

    Learns mapping between two hand models.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(20, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 45)
        )

    def forward(self, angles):
        # angles: [batch, 20]
        return self.mlp(angles)  # [batch, 45]


class TransferModel(nn.Module):
    """
    Full Mode 2 pipeline:
    EMG → ChannelAdapter → FreqUpsampler → emg2pose → AngleConverter → MANO
    """

    def __init__(self, emg2pose_model, mano_layer, device='cpu'):
        super().__init__()
        self.device = device

        # Trainable components
        self.channel_adapter = ChannelAdapter()
        self.freq_upsampler = FrequencyUpsampler()
        self.angle_converter = AngleConverter()

        # Frozen components
        self.emg2pose = emg2pose_model
        self.mano_layer = mano_layer

        # Freeze emg2pose
        for param in self.emg2pose.parameters():
            param.requires_grad = False

        # Note: MANO layer is not a nn.Module, just holds fixed tensors

    def forward(self, emg):
        """
        Args:
            emg: [batch, 8, time] @ 500Hz

        Returns:
            pred_theta: [batch, 45] MANO parameters
            pred_joints: [batch, 21, 3] MANO joints
        """
        # Step 1: Adapt channels
        x = self.channel_adapter(emg)  # [batch, 16, time]

        # Step 2: Upsample frequency
        x = self.freq_upsampler(x)  # [batch, 16, time*4] @ 2kHz

        # Step 3: emg2pose inference (frozen)
        batch_size = x.shape[0]
        time_len = x.shape[2]

        # Prepare batch dict for emg2pose
        batch_dict = {
            'emg': x,  # [batch, 16, time]
            'joint_angles': torch.zeros(batch_size, 20, time_len, device=self.device),
            'no_ik_failure': torch.ones(batch_size, time_len, dtype=torch.bool, device=self.device)
        }

        with torch.no_grad():
            # emg2pose forward returns: (pred, state, info)
            pred_angles, _, _ = self.emg2pose.model(batch_dict, provide_initial_pos=True)
            # pred_angles: [batch, 20, time]

        # Extract last timepoint
        angles_last = pred_angles[:, :, -1]  # [batch, 20]

        # Step 4: Convert to MANO parameters
        pred_theta = self.angle_converter(angles_last)  # [batch, 45]

        # Step 5: MANO forward pass
        # MANO expects pose: [batch, 45], betas: [batch, 10]
        betas = torch.zeros(batch_size, 10, device=self.device)
        vertices, pred_joints = self.mano_layer(pred_theta, betas)
        # vertices: [batch, 778, 3], pred_joints: [batch, 21, 3]

        return pred_theta, pred_joints


# ============================================================================
# Dataset
# ============================================================================

class MindRoveDataset(Dataset):
    """
    Dataset loader for Mode 2 training.

    Loads:
    - EMG (8ch @ 500Hz) as input
    - MANO θ (45D) from IK as ground truth
    - Joints (21, 3) for evaluation
    """

    def __init__(self, hdf5_paths, window_size_sec=2.0):
        self.hdf5_paths = hdf5_paths
        self.fs_emg = 500
        self.window_size = int(window_size_sec * self.fs_emg)  # 1000 samples

        # Load data
        self.file_metadata = []
        self.sample_indices = []
        self._build_index()

        print(f"MindRoveDataset initialized:")
        print(f"  Files: {len(self.hdf5_paths)}")
        print(f"  Samples: {len(self.sample_indices)}")
        print(f"  Window: {window_size_sec}sec ({self.window_size} samples)")

    def _build_index(self):
        for file_idx, path in enumerate(self.hdf5_paths):
            path_obj = Path(path)  # Convert to Path for .name attribute
            try:
                with h5py.File(path, 'r') as f:
                    # Load data
                    emg = f['emg/filtered'][:]  # [N, 8]
                    emg_ts = f['emg/timestamps'][:]

                    theta = f['pose/mano_theta'][:]  # [M, 45]
                    joints = f['pose/joints_3d'][:]  # [M, 21, 3]
                    pose_ts = f['pose/timestamps'][:]
                    ik_error = f['pose/ik_error'][:]
                    mp_conf = f['pose/mp_confidence'][:]

                    # Filter low-quality frames
                    valid_mask = (ik_error < 15.0) & (mp_conf > 0.5)
                    theta = theta[valid_mask]
                    joints = joints[valid_mask]
                    pose_ts = pose_ts[valid_mask]

                    print(f"  Loaded {path_obj.name}: {len(emg)} EMG samples, {len(theta)} pose samples ({valid_mask.sum()}/{len(valid_mask)} valid)")

                    if len(theta) == 0:
                        print(f"  WARNING: No valid pose samples in {path_obj.name}, skipping")
                        continue

                    # Interpolate pose data to EMG rate
                    theta_interp = np.zeros((len(emg_ts), 45), dtype=np.float32)
                    joints_interp = np.zeros((len(emg_ts), 21, 3), dtype=np.float32)

                    for i in range(45):
                        theta_interp[:, i] = np.interp(emg_ts, pose_ts, theta[:, i])

                    for i in range(21):
                        for j in range(3):
                            joints_interp[:, i, j] = np.interp(emg_ts, pose_ts, joints[:, i, j])

                    # Normalize EMG
                    emg_mean = emg.mean(axis=0)
                    emg_std = emg.std(axis=0) + 1e-8
                    emg_norm = (emg - emg_mean) / emg_std

                    self.file_metadata.append({
                        'emg': emg_norm,
                        'theta': theta_interp,
                        'joints': joints_interp,
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
        meta = self.file_metadata[file_idx]

        end = start + self.window_size

        # Extract window
        emg = meta['emg'][start:end]  # [1000, 8]
        target_theta = meta['theta'][end - 1]  # [45] - last frame
        target_joints = meta['joints'][end - 1]  # [21, 3] - last frame

        # Transpose EMG for Conv1D: [8, 1000]
        emg = torch.from_numpy(emg).float().transpose(0, 1)
        target_theta = torch.from_numpy(target_theta).float()
        target_joints = torch.from_numpy(target_joints).float()

        return {
            'emg': emg,
            'target_theta': target_theta,
            'target_joints': target_joints
        }


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_loss_theta = 0.0
    total_loss_joints = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        emg = batch['emg'].to(device)
        target_theta = batch['target_theta'].to(device)
        target_joints = batch['target_joints'].to(device)

        # Forward
        pred_theta, pred_joints = model(emg)

        # Loss
        loss_theta = F.mse_loss(pred_theta, target_theta)
        loss_joints = F.mse_loss(pred_joints, target_joints)
        loss = loss_theta + loss_joints

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_theta += loss_theta.item()
        total_loss_joints += loss_joints.item()

    n = len(dataloader)
    return total_loss / n, total_loss_theta / n, total_loss_joints / n


def validate(model, dataloader, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_loss_theta = 0.0
    total_loss_joints = 0.0
    mpjpes = []

    with torch.no_grad():
        for batch in dataloader:
            emg = batch['emg'].to(device)
            target_theta = batch['target_theta'].to(device)
            target_joints = batch['target_joints'].to(device)

            # Forward
            pred_theta, pred_joints = model(emg)

            # Loss
            loss_theta = F.mse_loss(pred_theta, target_theta)
            loss_joints = F.mse_loss(pred_joints, target_joints)
            loss = loss_theta + loss_joints

            total_loss += loss.item()
            total_loss_theta += loss_theta.item()
            total_loss_joints += loss_joints.item()

            # MPJPE (mm)
            mpjpe = torch.norm(pred_joints - target_joints, dim=-1).mean(dim=-1)
            mpjpes.extend(mpjpe.cpu().numpy() * 1000)

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'loss_theta': total_loss_theta / n,
        'loss_joints': total_loss_joints / n,
        'mpjpe_mean': np.mean(mpjpes),
        'mpjpe_median': np.median(mpjpes),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Mode 2: emg2pose → MANO")
    parser.add_argument('--data', nargs='+', required=True, help='HDF5 data files (glob pattern)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='models/emg2pose/tracking_vemg2pose.ckpt',
                        help='emg2pose checkpoint path')
    parser.add_argument('--mano-path', type=str, default='models/mano/MANO_RIGHT_numpy.pkl',
                        help='MANO model path')
    parser.add_argument('--output-dir', type=str, default='transfer/checkpoints',
                        help='Output directory for checkpoints')
    args = parser.parse_args()

    # Setup device (MPS for M3 Mac)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✅ Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"⚠️  MPS not available, using CPU")

    # Expand glob patterns
    from glob import glob
    data_files = []
    for pattern in args.data:
        data_files.extend(glob(pattern))

    if len(data_files) == 0:
        print(f"ERROR: No data files found matching: {args.data}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Transfer Learning - Mode 2: emg2pose → MANO")
    print(f"{'='*60}")
    print(f"Data files: {len(data_files)}")
    for f in data_files:
        print(f"  - {f}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    # Load emg2pose model
    print("Loading emg2pose model...")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: emg2pose checkpoint not found: {checkpoint_path}")
        print(f"Download from: https://dl.fbaipublicfiles.com/emg2pose/tracking_vemg2pose.ckpt")
        sys.exit(1)

    emg2pose_model = Emg2PoseModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=str(device),
        weights_only=False  # Checkpoint contains OmegaConf objects
    )
    emg2pose_model.eval()
    print(f"✅ Loaded emg2pose model from {checkpoint_path}")

    # Load MANO model
    print("Loading MANO model...")
    mano_path = Path(args.mano_path)
    if not mano_path.exists():
        print(f"ERROR: MANO model not found: {mano_path}")
        print(f"Expected: models/MANO_RIGHT_numpy.pkl")
        sys.exit(1)

    mano_layer = CustomMANOLayer(
        mano_root=str(mano_path.parent),
        side='right',
        device=str(device)
    )
    print(f"✅ Loaded MANO model from {mano_path}")

    # Create model
    print("\nCreating transfer model...")
    model = TransferModel(emg2pose_model, mano_layer, device=str(device)).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created:")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Total params: {total_params:,}")
    print(f"   Frozen params: {total_params - trainable_params:,} (emg2pose + MANO)")

    # Load dataset
    print("\nLoading dataset...")
    dataset = MindRoveDataset(data_files, window_size_sec=2.0)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer + Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []
    val_mpjpes = []
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_loss_theta, train_loss_joints = train_epoch(
            model, train_loader, optimizer, device
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_mpjpes.append(val_metrics['mpjpe_mean'])

        print(f"  Train Loss: {train_loss:.6f} (θ: {train_loss_theta:.6f}, joints: {train_loss_joints:.6f})")
        print(f"  Val Loss:   {val_metrics['loss']:.6f} (θ: {val_metrics['loss_theta']:.6f}, joints: {val_metrics['loss_joints']:.6f})")
        print(f"  Val MPJPE:  {val_metrics['mpjpe_mean']:.2f}mm (median: {val_metrics['mpjpe_median']:.2f}mm)")
        print(f"  LR:         {current_lr:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_mpjpe': val_metrics['mpjpe_mean'],
            }
            torch.save(checkpoint, output_dir / 'transfer_mode2_best.pth')
            print(f"  ✅ Saved best model (val_loss={val_metrics['loss']:.6f}, MPJPE={val_metrics['mpjpe_mean']:.2f}mm)")

        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), output_dir / f'transfer_mode2_epoch{epoch+1}.pth')
            print(f"  💾 Saved checkpoint: epoch {epoch+1}")

        print()

    # Final results
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val MPJPE: {min(val_mpjpes):.2f}mm")
    print(f"Checkpoint: {output_dir / 'transfer_mode2_best.pth'}")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_mpjpes, label='Val MPJPE', linewidth=2, color='orange')
    plt.axhline(y=min(val_mpjpes), color='r', linestyle='--', label=f'Best: {min(val_mpjpes):.2f}mm')
    plt.axhline(y=15, color='g', linestyle='--', label='Target: 15mm')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (mm)')
    plt.title('Validation MPJPE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    print(f"✅ Saved training curves: {output_dir / 'training_curves.png'}")

    # Success criteria
    final_mpjpe = val_mpjpes[-1]
    print(f"\n{'='*60}")
    if final_mpjpe < 15.0:
        print(f"✅ SUCCESS! MPJPE ({final_mpjpe:.2f}mm) < 15mm target")
        print(f"   Ready for deployment!")
    elif final_mpjpe < 20.0:
        print(f"⚠️  GOOD but not great. MPJPE ({final_mpjpe:.2f}mm) in 15-20mm range")
        print(f"   Consider:")
        print(f"   - More training data (15-30 min total)")
        print(f"   - Longer training (200 epochs)")
    else:
        print(f"❌ NEEDS IMPROVEMENT. MPJPE ({final_mpjpe:.2f}mm) >= 20mm")
        print(f"   Recommendations:")
        print(f"   - Check EMG signal quality (electrode placement)")
        print(f"   - Collect more diverse data (varied gestures, speeds)")
        print(f"   - Increase model capacity (wider AngleConverter)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
