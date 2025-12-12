"""
Simple test script to verify EMG data flow and model inference.
No GUI - just print values to terminal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import torch
import torch.nn as nn

from utils.emg_utils import MindroveInterface


class SimpleEMGModel(nn.Module):
    """Same as inference.py"""
    def __init__(self, input_channels=8, hidden_size=256, output_dim=45):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(256, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        theta = self.fc(x)
        return theta


def main():
    print("=" * 60)
    print("EMG Data Flow Test")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    model = SimpleEMGModel(input_channels=8, hidden_size=256, output_dim=45)
    model.load_state_dict(torch.load('emg_model_best.pth', map_location='cpu'))
    model.eval()
    print("  ✓ Model loaded")

    # Connect to MindRove
    print("\n[2/4] Connecting to MindRove...")
    mindrove = MindroveInterface()
    if not mindrove.connect():
        print("  ✗ Failed to connect")
        return
    print("  ✓ Connected @ 500Hz")

    # Calibration phase
    print("\n[3/4] Calibrating (2 seconds)...")
    calibration_data = []
    start_time = time.time()

    while time.time() - start_time < 2.0:
        timestamps, emg_data = mindrove.get_data(num_samples=50)
        if len(emg_data) > 0:
            calibration_data.append(emg_data)
            print(f"  Calibration: {len(calibration_data)} batches, last shape: {emg_data.shape}")
        time.sleep(0.01)

    # Compute normalization stats
    all_calib = np.vstack(calibration_data)
    emg_mean = np.mean(all_calib, axis=0)
    emg_std = np.std(all_calib, axis=0) + 1e-6
    print(f"  ✓ Calibrated on {len(all_calib)} samples")
    print(f"    Mean: {emg_mean[:3]}")
    print(f"    Std: {emg_std[:3]}")

    # Inference loop
    print("\n[4/4] Running inference (press Ctrl+C to stop)...")
    print("=" * 60)
    window_size = 50
    emg_buffer = []

    try:
        iteration = 0
        while True:
            timestamps, emg_data = mindrove.get_data(num_samples=50)

            if len(emg_data) == 0:
                time.sleep(0.01)
                continue

            # Process each sample
            for sample in emg_data:
                # Normalize
                sample_norm = (sample - emg_mean) / emg_std
                emg_buffer.append(sample_norm)

                # Keep only last window_size samples
                if len(emg_buffer) > window_size:
                    emg_buffer.pop(0)

                # Run inference when buffer is full
                if len(emg_buffer) == window_size:
                    # Prepare input
                    emg_window = torch.FloatTensor(np.array(emg_buffer)).unsqueeze(0)

                    # Inference
                    with torch.no_grad():
                        theta_pred = model(emg_window)
                        theta_pred = theta_pred.squeeze(0).numpy()

                    # Print every 50 iterations (~1 second)
                    iteration += 1
                    if iteration % 50 == 0:
                        print(f"Iter {iteration:4d}: EMG=[{sample[0]:+.2f}, {sample[1]:+.2f}, ...], θ range=[{theta_pred.min():.2f}, {theta_pred.max():.2f}]")

            time.sleep(0.001)  # Small delay

    except KeyboardInterrupt:
        print("\n\n[info] Stopping...")

    finally:
        mindrove.disconnect()
        print("[info] Disconnected")
        print("=" * 60)


if __name__ == '__main__':
    main()
