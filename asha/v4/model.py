"""
SimpleEMGModel: EMG → MANO θ (v4)

Direct prediction of MANO pose parameters from EMG signals.
Architecture: Conv1D + LSTM → 45 MANO θ parameters
"""

import torch
import torch.nn as nn


class SimpleEMGModel(nn.Module):
    """EMG→MANO θ model (v4 architecture)."""

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

        # Output head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 45)  # MANO θ (45 pose params)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [batch, 8, time] EMG input
            
        Returns:
            theta: [batch, 45] MANO pose parameters
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
        theta = self.fc2(x)

        return theta
