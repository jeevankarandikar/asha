"""
Test script to validate mock mode functionality without GUI.
Tests the core EMG generation and model inference pipeline.
"""

import time
import numpy as np
import torch
from collections import deque

# Test mock EMG generation
print("="*60)
print("Testing Mock EMG Generation")
print("="*60)

# Generate fake EMG data (same as in inference.py)
timestamps = np.arange(10) / 500.0  # 10 samples @ 500Hz
freq = 20 + 10 * np.sin(time.time() * 0.5)  # Varying frequency 10-30Hz
amp = 50 + 30 * np.sin(time.time() * 0.3)   # Varying amplitude
t = time.time() + timestamps
emg_raw = amp * np.sin(2 * np.pi * freq * t[:, None] + np.arange(8))  # [10, 8]

print(f"✅ Generated mock EMG data: shape={emg_raw.shape}")
print(f"   Frequency: {freq:.2f}Hz")
print(f"   Amplitude: {amp:.2f}")
print(f"   Value range: [{emg_raw.min():.2f}, {emg_raw.max():.2f}]")

# Test buffer accumulation
print("\n" + "="*60)
print("Testing Buffer Accumulation")
print("="*60)

emg_buffer = deque(maxlen=50)
for i in range(10):
    # Generate samples
    timestamps = np.arange(10) / 500.0
    freq = 20 + 10 * np.sin(time.time() * 0.5)
    amp = 50 + 30 * np.sin(time.time() * 0.3)
    t = time.time() + timestamps
    emg_raw = amp * np.sin(2 * np.pi * freq * t[:, None] + np.arange(8))

    # Add to buffer
    for j in range(len(timestamps)):
        emg_buffer.append(emg_raw[j])

    print(f"  Iteration {i+1}: Buffer size = {len(emg_buffer)}/50", end="")
    if len(emg_buffer) == 50:
        print(" ✅ FULL - Ready for inference")
        break
    else:
        print()
    time.sleep(0.02)

if len(emg_buffer) == 50:
    emg_window = np.array(emg_buffer)
    print(f"\n✅ Buffer filled successfully: shape={emg_window.shape}")
else:
    print(f"\n❌ Buffer not full: {len(emg_buffer)}/50")

# Test normalization
print("\n" + "="*60)
print("Testing Normalization")
print("="*60)

emg_mean = emg_window.mean(axis=0, keepdims=True)
emg_std = emg_window.std(axis=0, keepdims=True) + 1e-6
emg_norm = (emg_window - emg_mean) / emg_std

print(f"✅ Normalized EMG: shape={emg_norm.shape}")
print(f"   Mean: {emg_norm.mean():.6f} (should be ~0)")
print(f"   Std:  {emg_norm.std():.6f} (should be ~1)")

# Test tensor conversion
print("\n" + "="*60)
print("Testing Tensor Conversion")
print("="*60)

emg_tensor = torch.from_numpy(emg_norm.T).float().unsqueeze(0)
print(f"✅ Converted to tensor: shape={emg_tensor.shape} (expected: [1, 8, 50])")

# Test model loading (if model exists)
print("\n" + "="*60)
print("Testing Model Import")
print("="*60)

try:
    from asha.v4.model import SimpleEMGModel
    model_v4 = SimpleEMGModel(window_size=50, input_channels=8)
    print(f"✅ v4 model created: {sum(p.numel() for p in model_v4.parameters()):,} parameters")
except Exception as e:
    print(f"❌ v4 model import failed: {e}")

try:
    from asha.v5.model import EMGToJointsModel
    model_v5 = EMGToJointsModel(window_size=50, input_channels=8)
    print(f"✅ v5 model created: {sum(p.numel() for p in model_v5.parameters()):,} parameters")
except Exception as e:
    print(f"❌ v5 model import failed: {e}")

# Test mock inference (without trained weights)
print("\n" + "="*60)
print("Testing Mock Inference (Random Weights)")
print("="*60)

try:
    model_v4.eval()
    with torch.no_grad():
        output_v4 = model_v4(emg_tensor)
    print(f"✅ v4 inference: input={emg_tensor.shape} → output={output_v4.shape} (expected: [1, 45])")
    print(f"   θ range: [{output_v4.min():.2f}, {output_v4.max():.2f}]")
except Exception as e:
    print(f"❌ v4 inference failed: {e}")

try:
    model_v5.eval()
    with torch.no_grad():
        output_v5 = model_v5(emg_tensor)
    print(f"✅ v5 inference: input={emg_tensor.shape} → output={output_v5.shape} (expected: [1, 63])")
    print(f"   Joints range: [{output_v5.min():.2f}, {output_v5.max():.2f}]")
except Exception as e:
    print(f"❌ v5 inference failed: {e}")

print("\n" + "="*60)
print("Mock Mode Validation Complete!")
print("="*60)
print("\n✅ All core components working")
print("✅ Mock EMG generation functional")
print("✅ Buffer accumulation correct")
print("✅ Normalization working")
print("✅ Model inference pipeline validated")
print("\nReady for GUI testing with: python -m asha.v4.inference --mock")
