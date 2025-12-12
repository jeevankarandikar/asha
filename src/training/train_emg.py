"""
EMG → MANO θ Training Script

Train TDS+LSTM model to predict MANO pose parameters from EMG signals.
Uses IK pseudo-labels from recorded data.

Usage:
    python src/training/train_emg.py --data data/session1_subj001_20251211_211056.h5
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal

# Try to import EMG model if available
try:
    from training.emg_model import EMGPoseModel
    HAS_EMG_MODEL = True
except ImportError:
    HAS_EMG_MODEL = False
    print("[warning] emg_model.py not found, using simplified model")


class EMGDataset(Dataset):
    """Dataset for EMG → θ regression"""

    def __init__(self, emg_data, theta_data, window_size=50, stride=10):
        """
        Args:
            emg_data: [N, 8] EMG samples (filtered, normalized)
            theta_data: [M, 45] MANO pose parameters (IK labels)
            window_size: Number of EMG samples per window (50 = 100ms @ 500Hz)
            stride: Stride for sliding window
        """
        self.emg = torch.FloatTensor(emg_data)
        self.theta = torch.FloatTensor(theta_data)
        self.window_size = window_size
        self.stride = stride

        # Compute valid windows
        self.num_windows = (len(self.emg) - window_size) // stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # Get EMG window
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        emg_window = self.emg[start_idx:end_idx]  # [window_size, 8]

        # Get corresponding theta (use last timestamp in window)
        # Map EMG index to theta index (different sampling rates)
        theta_idx = min(int(end_idx * len(self.theta) / len(self.emg)), len(self.theta) - 1)
        theta = self.theta[theta_idx]  # [45]

        return emg_window, theta


class SimpleEMGModel(nn.Module):
    """Simplified EMG→θ model (fallback if emg_model.py not available)"""

    def __init__(self, input_channels=8, hidden_size=256, output_dim=45):
        super().__init__()

        # Temporal convolutions (simplified TDS)
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)

        # LSTM
        self.lstm = nn.LSTM(256, hidden_size, num_layers=2, batch_first=True, dropout=0.2)

        # Output head
        self.fc = nn.Linear(hidden_size, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]

        # Convolutions
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # LSTM
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep

        # Output
        theta = self.fc(x)

        return theta


def load_hdf5_data(filepath):
    """Load EMG and pose data from HDF5 file"""
    print(f"\nLoading data from: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # Load EMG data
        emg_raw = f['emg/raw'][:]  # [N, 8]
        emg_filtered = f['emg/filtered'][:]  # [N, 8]
        emg_timestamps = f['emg/timestamps'][:]

        # Load pose data (IK labels)
        theta = f['pose/mano_theta'][:]  # [M, 45]
        pose_timestamps = f['pose/timestamps'][:]
        ik_error = f['pose/ik_error'][:]

        print(f"  EMG: {len(emg_filtered)} samples @ {len(emg_filtered)/emg_timestamps[-1]:.1f}Hz")
        print(f"  Pose: {len(theta)} samples @ {len(theta)/pose_timestamps[-1]:.1f}Hz")

    return emg_filtered, theta


def preprocess_emg(emg_data):
    """Normalize EMG data"""
    # Z-score normalization per channel
    emg_normalized = np.zeros_like(emg_data)
    for ch in range(emg_data.shape[1]):
        mean = np.mean(emg_data[:, ch])
        std = np.std(emg_data[:, ch])
        if std > 0:
            emg_normalized[:, ch] = (emg_data[:, ch] - mean) / std
        else:
            emg_normalized[:, ch] = emg_data[:, ch] - mean

    return emg_normalized


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for emg_window, theta_gt in dataloader:
        emg_window = emg_window.to(device)
        theta_gt = theta_gt.to(device)

        # Forward
        theta_pred = model(emg_window)
        loss = criterion(theta_pred, theta_gt)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for emg_window, theta_gt in dataloader:
            emg_window = emg_window.to(device)
            theta_gt = theta_gt.to(device)

            theta_pred = model(emg_window)
            loss = criterion(theta_pred, theta_gt)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--window-size', type=int, default=50, help='EMG window size (samples)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    args = parser.parse_args()

    print("=" * 60)
    print("EMG → MANO θ Training")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    emg_data, theta_data = load_hdf5_data(args.data)

    # Preprocess
    print("\nPreprocessing...")
    emg_data = preprocess_emg(emg_data)

    # Create dataset
    full_dataset = EMGDataset(emg_data, theta_data, window_size=args.window_size)
    print(f"  Total windows: {len(full_dataset)}")

    # Train/val split
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {len(train_dataset)} windows")
    print(f"  Val: {len(val_dataset)} windows")

    # Model
    if HAS_EMG_MODEL:
        print("\nUsing EMGPoseModel (TDS+LSTM)")
        model = EMGPoseModel(
            input_channels=8,
            output_dim=45,
            tds_channels=[64, 128, 256, 512],
            lstm_hidden=256,
            lstm_layers=2
        ).to(device)
    else:
        print("\nUsing SimpleEMGModel (fallback)")
        model = SimpleEMGModel(
            input_channels=8,
            hidden_size=256,
            output_dim=45
        ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:3d}/{args.epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'emg_model_best.pth')
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: emg_model_best.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()
