"""
emg → mano θ model with tds conv featurizer and lstm decoder.

architecture based on emg2pose (meta/fair, 2024):
  - tds conv featurizer: downsample 500hz emg to 50hz
  - 2-layer lstm decoder: hidden 512, velocity prediction mode
  - preprocessing: bandpass 20-450hz, notch 60hz, per-channel normalization
  - optional imu fusion for v5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from typing import Optional, Tuple


class EMGPreprocessor:
    """
    emg signal preprocessing pipeline.
    
    steps:
      1. bandpass filter: 20-450hz (remove dc and high-frequency noise)
      2. notch filter: 60hz (power line noise)
      3. per-channel normalization: subtract baseline, divide by mvc
    """
    
    def __init__(
        self,
        sample_rate: int = 500,
        bandpass_low: float = 20.0,
        bandpass_high: float = 450.0,
        notch_freq: float = 60.0,
    ):
        """initialize emg preprocessor."""
        self.sample_rate = sample_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        
        # design filters
        # bandpass filter (butterworth, order 2)
        self.bandpass_b, self.bandpass_a = signal.butter(
            2, [bandpass_low, bandpass_high], btype='band', fs=sample_rate
        )
        
        # notch filter (60hz)
        self.notch_b, self.notch_a = signal.iirnotch(
            notch_freq, Q=30, fs=sample_rate
        )
        
        # per-channel statistics (updated during training)
        self.baseline = None  # [8] per-channel baseline
        self.mvc = None  # [8] per-channel maximum voluntary contraction
    
    def update_statistics(self, emg_data: np.ndarray):
        """update per-channel baseline and mvc statistics."""
        # baseline: mean of first 10% of data (resting state)
        baseline_samples = int(0.1 * len(emg_data))
        self.baseline = np.mean(emg_data[:baseline_samples], axis=0)  # [8]
        
        # mvc: 95th percentile (maximum voluntary contraction)
        self.mvc = np.percentile(np.abs(emg_data), 95, axis=0)  # [8]
        self.mvc = np.maximum(self.mvc, 1e-6)  # avoid division by zero
    
    def preprocess(self, emg_data: np.ndarray) -> np.ndarray:
        """preprocess emg signal. returns [N, 8] preprocessed emg."""
        emg = emg_data.copy()
        
        # apply bandpass filter (per channel)
        for ch in range(emg.shape[1]):
            emg[:, ch] = signal.filtfilt(self.bandpass_b, self.bandpass_a, emg[:, ch])
        
        # apply notch filter (per channel)
        for ch in range(emg.shape[1]):
            emg[:, ch] = signal.filtfilt(self.notch_b, self.notch_a, emg[:, ch])
        
        # per-channel normalization
        if self.baseline is not None and self.mvc is not None:
            emg = emg - self.baseline  # subtract baseline
            emg = emg / self.mvc  # divide by mvc
        
        return emg


class TDSBlock(nn.Module):
    """time-depth separable (tds) block for emg feature extraction."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        """
        Initialize TDS block.
        
        Args:
            in_channels: input channels
            out_channels: output channels
            kernel_size: convolution kernel size
            stride: convolution stride
            dilation: dilation rate
        """
        super().__init__()
        
        # Depthwise convolution (temporal)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,  # Depthwise: each channel processed separately
            padding=(kernel_size - 1) * dilation // 2,
        )
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass. returns [B, C_out, T_out] output tensor."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TDSFeaturizer(nn.Module):
    """
    tds convolutional featurizer for emg signals.
    
    architecture (per emg2pose):
      - initial convs: kernels 17/5/17, strides 5/2/4
      - 4 tds blocks: channels 16, kernel widths 9/9/5/5
      - downsample 500hz → 50hz (10× reduction)
    """
    
    def __init__(self, in_channels: int = 8, out_channels: int = 16):
        """initialize tds featurizer."""
        super().__init__()
        
        # initial convolutions (downsample 500hz → 50hz)
        # total reduction: 5 × 2 = 10× (500hz / 10 = 50hz)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=17, stride=5, padding=8)  # 5× reduction
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2)  # 2× reduction (total: 10×)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        
        # removed conv3 to achieve 10× reduction (was causing 20× reduction)
        # original had stride=2 here, giving 5 × 2 × 2 = 20× (25hz output)
        
        # 4 tds blocks
        self.tds1 = TDSBlock(16, out_channels, kernel_size=9)
        self.tds2 = TDSBlock(out_channels, out_channels, kernel_size=9)
        self.tds3 = TDSBlock(out_channels, out_channels, kernel_size=5)
        self.tds4 = TDSBlock(out_channels, out_channels, kernel_size=5)
        
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, T] EMG input (e.g., [B, 8, 500] for 1s @ 500Hz)
            
        Returns:
            features: [B, C_out, T_out] downsampled features (e.g., [B, 16, 50] for 1s @ 50Hz)
        """
        # Initial convolutions (10× reduction: 500Hz → 50Hz)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # tds blocks
        x = self.tds1(x)
        x = self.tds2(x)
        x = self.tds3(x)
        x = self.tds4(x)
        
        return x


class EMG2PoseModel(nn.Module):
    """
    complete emg → mano θ model.
    
    architecture:
      - tds featurizer: emg → features
      - 2-layer lstm decoder: features → Δθ (velocity)
      - integration: Δθ → θ (absolute)
      - optional imu fusion (for v5)
    """
    
    def __init__(
        self,
        emg_channels: int = 8,
        hidden_size: int = 512,
        output_size: int = 45,  # mano θ (45 params)
        num_lstm_layers: int = 2,
        use_velocity: bool = True,  # predict Δθ instead of absolute θ
        use_imu: bool = False,
        imu_dim: int = 18,  # 2 imus × 9-dof (accel + gyro + mag)
    ):
        """initialize emg2pose model."""
        super().__init__()
        
        self.use_velocity = use_velocity
        self.use_imu = use_imu
        
        # tds featurizer
        self.featurizer = TDSFeaturizer(in_channels=emg_channels, out_channels=16)
        
        # lstm input size: tds output channels
        lstm_input_size = self.featurizer.out_channels
        
        # imu fusion (if enabled)
        if use_imu:
            # linear layer to fuse imu features
            self.imu_fusion = nn.Linear(imu_dim, 16)
            lstm_input_size += 16  # concatenate imu features
        
        # 2-layer lstm decoder
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0,
        )
        
        # output head: lstm hidden → mano θ
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size),
        )
        
        # preprocessor
        self.preprocessor = EMGPreprocessor()
    
    def forward(
        self,
        emg: torch.Tensor,
        imu: Optional[torch.Tensor] = None,
        prev_theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """forward pass. returns [B, 45] mano θ parameters."""
        B, T, C = emg.shape
        
        # preprocess emg (convert to numpy, process, convert back)
        # note: in practice, preprocessing should be done offline or in dataloader
        # for now, assume input is already preprocessed
        
        # reshape for conv: [B, T, C] → [B, C, T]
        emg_conv = emg.transpose(1, 2)  # [B, C, T]
        
        # tds featurizer
        features = self.featurizer(emg_conv)  # [B, 16, T_out]
        
        # Reshape for LSTM: [B, C, T] → [B, T, C]
        features_lstm = features.transpose(1, 2)  # [B, T_out, 16]
        
        # IMU fusion (if enabled)
        if self.use_imu and imu is not None:
            # Process IMU through fusion layer
            imu_features = self.imu_fusion(imu)  # [B, T, 16]
            
            # Concatenate EMG and IMU features
            # Interpolate IMU to match EMG feature length if needed
            if imu_features.shape[1] != features_lstm.shape[1]:
                imu_features = F.interpolate(
                    imu_features.transpose(1, 2),
                    size=features_lstm.shape[1],
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
            
            features_lstm = torch.cat([features_lstm, imu_features], dim=2)  # [B, T_out, 32]
        
        # LSTM decoder
        lstm_out, _ = self.lstm(features_lstm)  # [B, T_out, hidden_size]
        
        # use last timestep output
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_size]
        
        # output head
        if self.use_velocity:
            # predict velocity (Δθ)
            delta_theta = self.output_head(last_hidden)  # [B, 45]
            
            # integrate: θ_t = θ_{t-1} + Δθ
            if prev_theta is not None:
                theta = prev_theta + delta_theta
            else:
                # first frame: use velocity as absolute (or initialize to zero)
                theta = delta_theta
        else:
            # predict absolute θ directly
            theta = self.output_head(last_hidden)  # [B, 45]
        
        return theta


def create_emg_model(
    emg_channels: int = 8,
    hidden_size: int = 512,
    output_size: int = 45,
    use_velocity: bool = True,
    use_imu: bool = False,
) -> EMG2PoseModel:
    """factory function to create emg2pose model."""
    return EMG2PoseModel(
        emg_channels=emg_channels,
        hidden_size=hidden_size,
        output_size=output_size,
        use_velocity=use_velocity,
        use_imu=use_imu,
    )
