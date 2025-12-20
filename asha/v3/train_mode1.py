"""
Transfer Learning - Mode 1: Pure emg2pose (EMG → landmarks)

Train ChannelAdapter only for MindRove → emg2pose → landmarks pipeline.
Simpler than Mode 2, no MANO conversion needed.

Usage:
    python transfer/training/train_mode1.py --data transfer/data/*.h5 --epochs 100
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

# Import emg2pose
try:
    from emg2pose.lightning import Emg2PoseModule
    from emg2pose.kinematics import load_default_hand_model, forward_kinematics
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

    Uses Conv1D with spatial interpolation initialization.
    Assumes 8 channels evenly spaced around forearm.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Initialize with spatial interpolation
        self._init_spatial_interpolation()

    def _init_spatial_interpolation(self):
        """
        Initialize to create interpolated channels between adjacent electrodes.

        Strategy: 8 channels evenly spaced around forearm (circular)
        Create 16 output channels by interpolating:
        - Original channels (weights ~0.7)
        - Adjacent channels (weights ~0.15 each)
        """
        with torch.no_grad():
            # First conv: 8 → 32
            self.conv1.weight.zero_()

            for out_ch in range(32):
                # Map output channel to input channel region
                # 32 outputs / 8 inputs = 4 outputs per input
                in_ch = out_ch // 4

                # Main channel weight (center tap)
                self.conv1.weight[out_ch, in_ch, 2] = 0.7

                # Neighbor channels (circular wrap-around)
                prev_ch = (in_ch - 1) % 8
                next_ch = (in_ch + 1) % 8

                # Add spatial interpolation from neighbors
                self.conv1.weight[out_ch, prev_ch, 2] = 0.15
                self.conv1.weight[out_ch, next_ch, 2] = 0.15

            # Second conv: 32 → 16 (simple grouping)
            self.conv2.weight.zero_()
            for i in range(16):
                # Average 2 adjacent channels
                self.conv2.weight[i, i*2, 2] = 0.5
                self.conv2.weight[i, i*2+1, 2] = 0.5

            print("  ✅ ChannelAdapter initialized with spatial interpolation")

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


# ============================================================================
# Model
# ============================================================================

class TransferModelMode1(nn.Module):
    """
    Mode 1: Pure emg2pose pipeline (no MANO conversion).

    MindRove EMG (8ch @ 500Hz)
      ↓
    ChannelAdapter: 8→16ch [TRAINABLE]
      ↓
    FreqUpsampler: 500Hz→2kHz [FIXED]
      ↓
    emg2pose: 16ch@2kHz → 20 angles [FROZEN]
      ↓
    emg2pose FK: 20 angles → 21 landmarks [FROZEN, differentiable]
      ↓
    Output: 21 3D landmarks
    """

    def __init__(self, emg2pose_model, hand_model, device='cpu'):
        super().__init__()
        self.device = device

        # Trainable components
        self.channel_adapter = ChannelAdapter()
        self.freq_upsampler = FrequencyUpsampler()

        # Frozen components
        self.emg2pose = emg2pose_model

        # Move hand_model tensors to device (it's a NamedTuple)
        self.hand_model = self._move_hand_model_to_device(hand_model, device)

        # Freeze emg2pose
        for param in self.emg2pose.parameters():
            param.requires_grad = False

    def _move_hand_model_to_device(self, hand_model, device):
        """Move all tensors in HandModel NamedTuple to device."""
        from emg2pose.UmeTrack.lib.common.hand import HandModel

        # Convert device string to torch.device
        torch_device = torch.device(device)

        # Create new HandModel with tensors on correct device
        return HandModel(
            joint_rotation_axes=hand_model.joint_rotation_axes.to(torch_device),
            joint_rest_positions=hand_model.joint_rest_positions.to(torch_device),
            joint_frame_index=hand_model.joint_frame_index.to(torch_device),
            joint_parent=hand_model.joint_parent.to(torch_device),
            joint_first_child=hand_model.joint_first_child.to(torch_device),
            joint_next_sibling=hand_model.joint_next_sibling.to(torch_device),
            landmark_rest_positions=hand_model.landmark_rest_positions.to(torch_device),
            landmark_rest_bone_weights=hand_model.landmark_rest_bone_weights.to(torch_device),
            landmark_rest_bone_indices=hand_model.landmark_rest_bone_indices.to(torch_device),
            hand_scale=hand_model.hand_scale.to(torch_device) if hand_model.hand_scale is not None else None,
            mesh_vertices=hand_model.mesh_vertices.to(torch_device) if hand_model.mesh_vertices is not None else None,
            mesh_triangles=hand_model.mesh_triangles.to(torch_device) if hand_model.mesh_triangles is not None else None,
            dense_bone_weights=hand_model.dense_bone_weights.to(torch_device) if hand_model.dense_bone_weights is not None else None,
            joint_limits=hand_model.joint_limits.to(torch_device) if hand_model.joint_limits is not None else None,
        )

        print(f"✅ Mode 1 model created:")
        print(f"   Trainable: ChannelAdapter only (~5K params)")
        print(f"   Frozen: emg2pose (~68M params)")

    def forward(self, emg):
        """
        Args:
            emg: [batch, 8, time] @ 500Hz

        Returns:
            pred_landmarks: [batch, 21, 3] 3D hand landmarks
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
            'emg': x,
            'joint_angles': torch.zeros(batch_size, 20, time_len, device=self.device),
            'no_ik_failure': torch.ones(batch_size, time_len, dtype=torch.bool, device=self.device)
        }

        # NOTE: Don't use torch.no_grad() here!
        # emg2pose weights are frozen (requires_grad=False)
        # But we need gradients to flow through emg2pose operations to ChannelAdapter
        pred_angles, _, _ = self.emg2pose.model(batch_dict, provide_initial_pos=False)
        # pred_angles: [batch, 20, time]

        # Step 4: Forward kinematics (differentiable!)
        # pred_angles: [batch, 20, time] → landmarks: [batch, time, 21, 3]
        pred_landmarks = forward_kinematics(pred_angles, self.hand_model)

        # Extract last timepoint
        pred_landmarks = pred_landmarks[:, -1, :, :]  # [batch, 21, 3]

        return pred_landmarks


# ============================================================================
# Dataset
# ============================================================================

class MindRoveDatasetMode1(Dataset):
    """
    Dataset for Mode 1 training.

    Loads:
    - EMG (8ch @ 500Hz)
    - MediaPipe landmarks (21, 3) as ground truth
    """

    def __init__(self, hdf5_paths, window_size_sec=2.0):
        self.hdf5_paths = hdf5_paths
        self.fs_emg = 500
        self.window_size = int(window_size_sec * self.fs_emg)  # 1000 samples

        # Load data
        self.file_metadata = []
        self.sample_indices = []
        self._build_index()

        print(f"MindRoveDataset (Mode 1) initialized:")
        print(f"  Files: {len(self.hdf5_paths)}")
        print(f"  Samples: {len(self.sample_indices)}")
        print(f"  Window: {window_size_sec}sec ({self.window_size} samples)")

    def _build_index(self):
        for file_idx, path in enumerate(self.hdf5_paths):
            path_obj = Path(path)
            try:
                with h5py.File(path, 'r') as f:
                    # Load data
                    emg = f['emg/filtered'][:]  # [N, 8]
                    emg_ts = f['emg/timestamps'][:]

                    joints = f['pose/joints_3d'][:]  # [M, 21, 3] - MediaPipe landmarks
                    pose_ts = f['pose/timestamps'][:]
                    ik_error = f['pose/ik_error'][:]
                    mp_conf = f['pose/mp_confidence'][:]

                    # Filter low-quality frames
                    valid_mask = (ik_error < 15.0) & (mp_conf > 0.5)
                    joints = joints[valid_mask]
                    pose_ts = pose_ts[valid_mask]

                    print(f"  Loaded {path_obj.name}: {len(emg)} EMG samples, {len(joints)} pose samples ({valid_mask.sum()}/{len(valid_mask)} valid)")

                    if len(joints) == 0:
                        print(f"  WARNING: No valid pose samples in {path_obj.name}, skipping")
                        continue

                    # Interpolate pose data to EMG rate
                    joints_interp = np.zeros((len(emg_ts), 21, 3), dtype=np.float32)

                    for i in range(21):
                        for j in range(3):
                            joints_interp[:, i, j] = np.interp(emg_ts, pose_ts, joints[:, i, j])

                    # Normalize EMG (per-channel z-score)
                    emg_mean = emg.mean(axis=0, keepdims=True)
                    emg_std = emg.std(axis=0, keepdims=True) + 1e-6
                    emg = (emg - emg_mean) / emg_std

                    # Store metadata
                    self.file_metadata.append({
                        'emg': emg,
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
        metadata = self.file_metadata[file_idx]

        # Extract window
        end = start + self.window_size
        emg_window = metadata['emg'][start:end]  # [1000, 8]
        joints_window = metadata['joints'][start:end]  # [1000, 21, 3]

        # Get target (last frame of window)
        target_joints = joints_window[-1]  # [21, 3]

        # Convert to tensors and transpose EMG to [8, 1000]
        emg_tensor = torch.from_numpy(emg_window.T).float()
        target_tensor = torch.from_numpy(target_joints).float()

        return emg_tensor, target_tensor


# ============================================================================
# EMG Signal Priors
# ============================================================================

def amplitude_prior_loss(adapted_emg, target_mean=0.0, target_std=1.0):
    """
    Encourage adapted EMG to have realistic amplitude distribution.

    Args:
        adapted_emg: [batch, 16, time] - ChannelAdapter output
        target_mean, target_std: Expected EMG statistics (after normalization)
    """
    batch_mean = adapted_emg.mean(dim=(0, 2))  # [16]
    batch_std = adapted_emg.std(dim=(0, 2))    # [16]

    mean_loss = F.mse_loss(batch_mean, torch.zeros_like(batch_mean) + target_mean)
    std_loss = F.mse_loss(batch_std, torch.ones_like(batch_std) * target_std)

    return mean_loss + std_loss


def temporal_smoothness_loss(adapted_emg):
    """
    Encourage temporal coherence (adjacent samples should be similar).

    EMG signals are smooth - sudden jumps are typically noise.
    """
    # Compute differences between adjacent timepoints
    diff = adapted_emg[:, :, 1:] - adapted_emg[:, :, :-1]

    # Penalize large changes
    return torch.mean(diff ** 2)


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, lambda_priors=0.1):
    model.train()
    total_loss = 0
    total_landmark_loss = 0
    total_prior_loss = 0

    for emg, target_joints in tqdm(dataloader, desc="Training", leave=False):
        emg = emg.to(device)
        target_joints = target_joints.to(device)

        # Forward pass with intermediate adapted EMG
        adapted_emg = model.channel_adapter(emg)  # [batch, 16, time]

        # Full forward pass
        pred_joints = model(emg)

        # Main loss: Landmark accuracy
        landmark_loss = F.mse_loss(pred_joints, target_joints)

        # Prior losses: Encourage EMG-like signals
        amp_loss = amplitude_prior_loss(adapted_emg)
        smooth_loss = temporal_smoothness_loss(adapted_emg)
        prior_loss = amp_loss + 0.5 * smooth_loss

        # Total loss
        loss = landmark_loss + lambda_priors * prior_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_landmark_loss += landmark_loss.item()
        total_prior_loss += prior_loss.item()

    return (total_loss / len(dataloader),
            total_landmark_loss / len(dataloader),
            total_prior_loss / len(dataloader))


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_mpjpe = 0

    with torch.no_grad():
        for emg, target_joints in dataloader:
            emg = emg.to(device)
            target_joints = target_joints.to(device)

            # Forward
            pred_joints = model(emg)

            # Loss
            loss = F.mse_loss(pred_joints, target_joints)

            # MPJPE (Mean Per-Joint Position Error)
            mpjpe = torch.norm(pred_joints - target_joints, dim=-1).mean()

            total_loss += loss.item()
            total_mpjpe += mpjpe.item() * 1000  # Convert to mm

    avg_loss = total_loss / len(dataloader)
    avg_mpjpe = total_mpjpe / len(dataloader)

    return avg_loss, avg_mpjpe


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning - Mode 1')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                        help='HDF5 data files (supports glob patterns)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default='models/emg2pose/tracking_vemg2pose.ckpt',
                        help='emg2pose checkpoint path')
    parser.add_argument('--output-dir', type=str, default='transfer/checkpoints',
                        help='Output directory for checkpoints')
    args = parser.parse_args()

    # Setup device
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
    print(f"Transfer Learning - Mode 1: Pure emg2pose")
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
        weights_only=False
    )
    emg2pose_model.eval()
    print(f"✅ Loaded emg2pose model from {checkpoint_path}")

    # Load hand model (for FK)
    print("Loading hand model...")
    hand_model = load_default_hand_model()
    print(f"✅ Loaded emg2pose hand model")

    # Create model
    print("\nCreating transfer model...")
    model = TransferModelMode1(emg2pose_model, hand_model, device=str(device)).to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Total params: {total_params:,}")
    print(f"   Frozen params: {total_params - trainable_params:,}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = MindRoveDatasetMode1(data_files, window_size_sec=2.0)

    if len(dataset) == 0:
        print("ERROR: No valid samples in dataset!")
        sys.exit(1)

    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_mpjpe = float('inf')
    train_losses = []
    val_losses = []
    val_mpjpes = []

    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train (with EMG priors)
        train_loss, train_landmark_loss, train_prior_loss = train_epoch(
            model, train_loader, optimizer, device, lambda_priors=0.1
        )

        # Validate
        val_loss, val_mpjpe = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Log
        print(f"  Train Loss: {train_loss:.6f} (landmark: {train_landmark_loss:.3f}, prior: {train_prior_loss:.3f})")
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
            }, output_dir / 'transfer_mode1_best.pth')
            print(f"  ✅ Saved best model (val_loss={val_loss:.6f}, MPJPE={val_mpjpe:.2f}mm)")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'transfer_mode1_epoch{epoch}.pth')
            print(f"  💾 Saved checkpoint: epoch {epoch}")

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
    plt.plot(val_mpjpes, label='Val MPJPE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MPJPE (mm)')
    plt.title('Validation MPJPE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_mode1.png', dpi=150)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val MPJPE: {best_mpjpe:.2f}mm")
    print(f"Checkpoint: {output_dir / 'transfer_mode1_best.pth'}")
    print(f"✅ Saved training curves: {output_dir / 'training_curves_mode1.png'}")

    # Success criteria
    print(f"\n{'='*60}")
    if best_mpjpe < 20:
        print(f"✅ SUCCESS! MPJPE ({best_mpjpe:.2f}mm) < 20mm target")
        print(f"   Ready for deployment!")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT. MPJPE ({best_mpjpe:.2f}mm) >= 20mm")
        print(f"   Recommendations:")
        print(f"   - Collect more diverse data (15-20 min target)")
        print(f"   - Check EMG signal quality")
        print(f"   - Try initializing ChannelAdapter better")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
