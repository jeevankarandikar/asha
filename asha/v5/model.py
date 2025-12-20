"""
EMGToJointsModel: EMG → MediaPipe Joints (v5)

Direct prediction of 3D joint positions from EMG signals.
Architecture: Conv1D + LSTM → 63 joint coordinates (21 joints × 3)
"""

import torch
import torch.nn as nn


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
