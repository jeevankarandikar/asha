"""
Real-time EMG → Joints Inference with Integrated Calibration

Features:
- Built-in calibration mode for electrode drift adaptation
- Auto fine-tuning after calibration
- Seamless transition to inference

Usage:
    # Regular inference
    python -m asha.v5.inference --model models/v5/emg_joints_best.pth

    # Calibration mode (records → fine-tunes → inference)
    python -m asha.v5.inference --model models/v5/emg_joints_best.pth --calibrate
"""

import argparse
import sys
import threading
import time
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import pyrender
import trimesh
from tqdm import tqdm

from asha.core.emg_utils import MindroveInterface, filter_emg
from asha.core.tracker import get_landmarks, get_landmarks_world, draw_landmarks_on_frame
from asha.core.pose_fitter import mano_from_landmarks
from asha.shared.record import VideoThread as RecordVideoThread, EMGThread as RecordEMGThread
from .model import EMGToJointsModel


# ============================================================================
# Model imported from .model (shared with train_colab.py)
# ============================================================================
# Model definition moved to asha/v5/model.py for reusability


# ============================================================================
# Calibration Dataset
# ============================================================================

class CalibrationDataset(Dataset):
    """Quick dataset for fine-tuning on calibration data."""

    def __init__(self, hdf5_path, window_size=50):
        self.window_size = window_size

        with h5py.File(hdf5_path, 'r') as f:
            emg = f['emg/filtered'][:]
            emg_ts = f['emg/timestamps'][:]
            joints_3d = f['pose/joints_3d'][:]
            pose_ts = f['pose/timestamps'][:]
            ik_error = f['pose/ik_error'][:]
            mp_conf = f['pose/mp_confidence'][:]

        # Filter low-quality frames
        valid_mask = (ik_error < 15.0) & (mp_conf > 0.5)
        joints_3d = joints_3d[valid_mask]
        pose_ts = pose_ts[valid_mask]

        print(f"  Loaded: {len(emg)} EMG samples, {len(joints_3d)} pose samples ({valid_mask.sum()}/{len(valid_mask)} valid)")

        # Interpolate joints to EMG rate
        joints_interp = np.zeros((len(emg_ts), 21, 3), dtype=np.float32)
        for joint_idx in range(21):
            for coord_idx in range(3):
                joints_interp[:, joint_idx, coord_idx] = np.interp(
                    emg_ts, pose_ts, joints_3d[:, joint_idx, coord_idx]
                )

        joints_flat = joints_interp.reshape(len(emg_ts), -1)

        # Normalize EMG
        emg_mean = emg.mean(axis=0, keepdims=True)
        emg_std = emg.std(axis=0, keepdims=True) + 1e-6
        emg = (emg - emg_mean) / emg_std

        # Create sliding windows
        self.samples = []
        stride = window_size // 2
        for start in range(0, len(emg) - window_size + 1, stride):
            end = start + window_size
            emg_window = emg[start:end]  # [50, 8]
            target_joints = joints_flat[end - 1]  # [63]
            self.samples.append((emg_window, target_joints))

        print(f"  Created {len(self.samples)} training windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        emg_window, target_joints = self.samples[idx]
        emg_tensor = torch.from_numpy(emg_window.T).float()  # [8, 50]
        joints_tensor = torch.from_numpy(target_joints).float()  # [63]
        return emg_tensor, joints_tensor


# ============================================================================
# Calibration Recording
# ============================================================================

class CalibrationRecorder:
    """Records EMG + camera IK labels to HDF5."""

    def __init__(self, output_path: str, target_duration: int = 180):
        self.output_path = output_path
        self.target_duration = target_duration
        self.start_time = time.time()

        # Buffers
        self.emg_timestamps = []
        self.emg_filtered = []
        self.pose_joints = []
        self.pose_timestamps = []
        self.pose_ik_error = []
        self.pose_mp_conf = []

        print(f"📹 Recording to: {output_path}")
        print(f"⏱️  Target duration: {target_duration}s")

    def record_emg(self, timestamp: float, emg: np.ndarray):
        """Record EMG sample."""
        self.emg_timestamps.append(timestamp)
        self.emg_filtered.append(emg)

    def record_pose(self, timestamp: float, joints: np.ndarray, ik_error: float, mp_conf: float):
        """Record pose label."""
        self.pose_timestamps.append(timestamp)
        self.pose_joints.append(joints)
        self.pose_ik_error.append(ik_error)
        self.pose_mp_conf.append(mp_conf)

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        """Check if target duration reached."""
        return self.get_elapsed() >= self.target_duration

    def save(self):
        """Save to HDF5."""
        print(f"\n💾 Saving calibration data...")

        with h5py.File(self.output_path, 'w') as f:
            # EMG data
            emg_group = f.create_group('emg')
            emg_group.create_dataset('timestamps', data=np.array(self.emg_timestamps))
            emg_group.create_dataset('filtered', data=np.array(self.emg_filtered))

            # Pose data
            pose_group = f.create_group('pose')
            pose_group.create_dataset('timestamps', data=np.array(self.pose_timestamps))
            pose_group.create_dataset('joints_3d', data=np.array(self.pose_joints))
            pose_group.create_dataset('ik_error', data=np.array(self.pose_ik_error))
            pose_group.create_dataset('mp_confidence', data=np.array(self.pose_mp_conf))

        print(f"✅ Saved: {len(self.emg_timestamps)} EMG samples, {len(self.pose_timestamps)} pose labels")


# ============================================================================
# Calibration Threads
# ============================================================================

class CalibrationVideoThread(QtCore.QObject):
    """Wrapper for RecordVideoThread that records calibration data."""

    frame_signal = QtCore.pyqtSignal(object, object, float, float, object)  # frame, verts, ik_error, mp_conf, joints

    def __init__(self, recorder: CalibrationRecorder):
        super().__init__()
        self.recorder = recorder
        self._calib_frame_count = 0
        # Create wrapped thread
        self._thread = RecordVideoThread()
        self._thread.frame_signal.connect(self._handle_and_record)

    def _handle_and_record(self, frame, verts, ik_error, mp_confidence, theta, joints):
        """Handle frame from wrapped thread and record to calibration recorder."""
        self._calib_frame_count += 1

        # Record pose if valid
        if joints is not None:
            timestamp = self.recorder.get_elapsed()
            self.recorder.record_pose(timestamp, joints, ik_error, mp_confidence)

        # Debug output every 100 frames
        if self._calib_frame_count % 100 == 0:
            print(f"[VIDEO] Frame {self._calib_frame_count}, Hand detected: {joints is not None}, Joints recorded: {joints is not None}")

        # Re-emit for GUI (omit theta since calibration doesn't need it)
        self.frame_signal.emit(frame, verts, ik_error, mp_confidence, joints)

    def start(self):
        self._thread.start()

    def stop(self):
        self._thread.stop()

    def wait(self):
        self._thread.wait()


class CalibrationEMGThread(QtCore.QObject):
    """Wrapper for RecordEMGThread that records calibration data."""

    emg_status_signal = QtCore.pyqtSignal(bool)  # Connection status for GUI

    def __init__(self, recorder: CalibrationRecorder):
        super().__init__()
        self.recorder = recorder
        # Create wrapped thread
        self._thread = RecordEMGThread(enable_emg=True, enable_imu=False)
        self._thread.emg_signal.connect(self._handle_and_record)

    def _handle_and_record(self, timestamps, emg_raw, emg_filtered, imu_data, connected):
        """Handle EMG data from wrapped thread and record to calibration recorder."""
        # Emit connection status
        self.emg_status_signal.emit(connected)

        # Record EMG if connected and have data
        if connected and timestamps.size > 0:
            for i in range(len(timestamps)):
                timestamp = self.recorder.get_elapsed()
                self.recorder.record_emg(timestamp, emg_filtered[i])

    def start(self):
        self._thread.start()

    def stop(self):
        self._thread.stop()

    def wait(self):
        self._thread.wait()


# ============================================================================
# Inference Thread (same as before)
# ============================================================================

class EMGThread(QtCore.QThread):
    """Background thread for EMG acquisition and model inference."""

    prediction_signal = QtCore.pyqtSignal(object, float, bool)

    def __init__(self, model: nn.Module, device: torch.device, emg_mean=None, emg_std=None):
        super().__init__()
        self.model = model
        self.device = device
        self.emg_mean = emg_mean
        self.emg_std = emg_std
        self._running = threading.Event()
        self._running.set()

        self.emg_buffer = deque(maxlen=50)
        self.joints_history = deque(maxlen=10)
        self.joints_smoothed = None

    def run(self):
        interface = MindroveInterface()
        if not interface.connect():
            print("❌ Failed to connect to MindRove")
            return

        print("✅ MindRove connected @ 500Hz")

        while self._running.is_set():
            timestamps, emg_raw = interface.get_data(num_samples=10)

            if timestamps.size > 0:
                emg_filtered = filter_emg(emg_raw)

                for i in range(len(timestamps)):
                    self.emg_buffer.append(emg_filtered[i])

                    if len(self.emg_buffer) == 50:
                        emg_window = np.array(self.emg_buffer)

                        # Normalize using GLOBAL statistics (same as training!)
                        if self.emg_mean is not None and self.emg_std is not None:
                            # Use global statistics from training
                            emg_norm = (emg_window - self.emg_mean) / self.emg_std
                        else:
                            # Fallback to per-window (old behavior, causes stuck predictions)
                            emg_mean = emg_window.mean(axis=0, keepdims=True)
                            emg_std = emg_window.std(axis=0, keepdims=True) + 1e-6
                            emg_norm = (emg_window - emg_mean) / emg_std

                        # Inference
                        emg_tensor = torch.from_numpy(emg_norm.T).float().unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            joints_pred = self.model(emg_tensor).cpu().numpy()[0]

                        # Confidence
                        confidence = self._compute_confidence(emg_norm, joints_pred)

                        # Smoothing
                        joints_smoothed = self._smooth_joints(joints_pred)
                        self.joints_smoothed = joints_smoothed
                        self.joints_history.append(joints_pred)

                        self.prediction_signal.emit(joints_smoothed, confidence, True)

            time.sleep(0.01)

        interface.disconnect()

    def _compute_confidence(self, emg_norm, joints_pred):
        emg_strength = np.sqrt(np.mean(emg_norm ** 2))
        strength_score = min(emg_strength / 1.5, 1.0)

        joints_mag = np.linalg.norm(joints_pred)
        mag_score = 1.0 if joints_mag < 2.0 else 0.0

        if len(self.joints_history) > 0:
            delta = np.linalg.norm(joints_pred - self.joints_history[-1])
            change_score = 1.0 if delta < 0.05 else 0.0
        else:
            change_score = 1.0

        return (strength_score + mag_score + change_score) / 3.0

    def _smooth_joints(self, joints_pred, alpha=0.3):
        if self.joints_smoothed is None:
            return joints_pred
        return alpha * joints_pred + (1 - alpha) * self.joints_smoothed

    def stop(self):
        self._running.clear()


# ============================================================================
# Fine-tuning Function
# ============================================================================

def finetune_model(model: nn.Module, hdf5_path: str, device: torch.device,
                  epochs: int = 15, lr: float = 1e-5) -> float:
    """
    Fine-tune model on calibration data.

    Returns:
        best_val_mpjpe: Best validation MPJPE achieved
    """
    print(f"\n{'='*60}")
    print("🔧 Fine-tuning model on calibration data")
    print(f"{'='*60}\n")

    # Freeze feature extractor
    model.freeze_feature_extractor()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Load dataset
    print("\nLoading calibration data...")
    dataset = CalibrationDataset(hdf5_path, window_size=50)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples\n")

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    # Training loop
    best_val_mpjpe = float('inf')
    print(f"Fine-tuning for {epochs} epochs (lr={lr})...\n")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for emg, joints_gt in train_loader:
            emg, joints_gt = emg.to(device), joints_gt.to(device)

            joints_pred = model(emg)

            # MPJPE loss
            pred = joints_pred.view(-1, 21, 3)
            gt = joints_gt.view(-1, 21, 3)
            loss = torch.norm(pred - gt, dim=2).mean() * 1000

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emg, joints_gt in val_loader:
                emg, joints_gt = emg.to(device), joints_gt.to(device)
                joints_pred = model(emg)

                pred = joints_pred.view(-1, 21, 3)
                gt = joints_gt.view(-1, 21, 3)
                loss = torch.norm(pred - gt, dim=2).mean() * 1000

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{epochs} - Train: {train_loss:.2f}mm, Val: {val_loss:.2f}mm")

        if val_loss < best_val_mpjpe:
            best_val_mpjpe = val_loss

    print(f"\n✅ Fine-tuning complete! Best Val MPJPE: {best_val_mpjpe:.2f}mm\n")
    return best_val_mpjpe


# ============================================================================
# Unified GUI (Calibration + Inference)
# ============================================================================

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]


class UnifiedGUI(QtWidgets.QWidget):
    """GUI that handles both calibration and inference modes."""

    def __init__(self, model_path: str, calibrate: bool = False):
        super().__init__()
        self.setWindowTitle("EMG → Joints (Calibration + Inference)")
        self.setGeometry(100, 100, 1600, 700)

        self.model_path = model_path
        self.calibrate_mode = calibrate
        self.device = torch.device("cpu")

        # Load model
        print(f"\nLoading model: {model_path}")
        self.model = EMGToJointsModel(window_size=50, input_channels=8).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load GLOBAL EMG normalization statistics (CRITICAL fix for stuck predictions!)
        self.emg_mean = checkpoint.get('emg_mean', None)
        self.emg_std = checkpoint.get('emg_std', None)

        if self.emg_mean is None or self.emg_std is None:
            print("⚠️  WARNING: Checkpoint missing global EMG statistics!")
            print("   Model trained with old code. Using fallback per-window normalization.")
            print("   Retrain model with updated train_joints_colab.py for best results.")
            self.emg_mean = None
            self.emg_std = None
        else:
            print(f"✅ Loaded global EMG statistics: mean={self.emg_mean.shape}, std={self.emg_std.shape}")

        self.original_mpjpe = checkpoint.get('val_mpjpe', 'unknown')
        print(f"✅ Loaded: Epoch {checkpoint.get('epoch', '?')}, MPJPE {self.original_mpjpe}mm")

        # State
        self.current_joints = None
        self.current_verts = None
        self.confidence = 0.0
        self.calibration_file = None

        # Setup UI
        self._setup_ui()
        self._setup_renderer()

        # Start appropriate mode
        if self.calibrate_mode:
            self._start_calibration_mode()
        else:
            self._start_inference_mode()

    def _setup_ui(self):
        """Setup UI (handles both modes)."""
        layout = QtWidgets.QHBoxLayout()

        # Left: Visualization (camera OR 3D joints)
        self.viz_label = QtWidgets.QLabel()
        self.viz_label.setMinimumSize(800, 600)
        self.viz_label.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        self.viz_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.viz_label)

        # Right: Info/Control panel
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setAlignment(QtCore.Qt.AlignTop)

        # Title
        self.title_label = QtWidgets.QLabel("EMG → Joints Inference")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff00; padding: 10px;")
        right_layout.addWidget(self.title_label)

        # Status
        self.status_label = QtWidgets.QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("font-size: 12px; color: #ffaa00; padding: 5px;")
        right_layout.addWidget(self.status_label)

        # Progress (for calibration)
        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("font-size: 12px; color: #ffffff; padding: 5px;")
        right_layout.addWidget(self.progress_label)

        # Instructions
        self.instructions_label = QtWidgets.QLabel("")
        self.instructions_label.setStyleSheet("font-size: 11px; color: #888; padding: 10px;")
        self.instructions_label.setWordWrap(True)
        right_layout.addWidget(self.instructions_label)

        right_layout.addStretch()
        layout.addLayout(right_layout)

        self.setLayout(layout)

    def _setup_renderer(self):
        """Setup pyrender scene."""
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.1, 0.1, 0.1, 1.0])

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=camera_pose)

        self.renderer = pyrender.OffscreenRenderer(800, 600)
        self.mesh_nodes = []

    # ========================================================================
    # Calibration Mode
    # ========================================================================

    def _start_calibration_mode(self):
        """Start calibration recording."""
        print("\n" + "="*60)
        print("📹 CALIBRATION MODE")
        print("="*60)
        print("Instructions:")
        print("1. Perform varied gestures for 2-3 minutes")
        print("2. Model will auto fine-tune after recording")
        print("3. Inference will start automatically\n")

        self.title_label.setText("🔧 Calibration Mode")
        self.status_label.setText("Status: Recording...")
        self.status_label.setStyleSheet("font-size: 12px; color: #ff6600; padding: 5px;")
        self.instructions_label.setText(
            "Perform these gestures:\n\n"
            "• Open hand\n"
            "• Close fist\n"
            "• Pinch\n"
            "• Point\n"
            "• Hold each 2-3 seconds\n"
            "• Smooth transitions\n\n"
            "Recording will stop automatically..."
        )

        # Create temp file for calibration data
        self.calibration_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False).name

        # Setup recorder
        self.recorder = CalibrationRecorder(self.calibration_file, target_duration=180)

        # Start threads
        self.calib_video_thread = CalibrationVideoThread(self.recorder)
        self.calib_video_thread.frame_signal.connect(self._handle_calib_frame)
        self.calib_video_thread.start()

        self.calib_emg_thread = CalibrationEMGThread(self.recorder)
        self.calib_emg_thread.emg_status_signal.connect(self._handle_emg_status)
        self.calib_emg_thread.start()

        # Timer to check completion
        self.calib_timer = QtCore.QTimer()
        self.calib_timer.timeout.connect(self._check_calibration_complete)
        self.calib_timer.start(100)

    def _handle_calib_frame(self, frame, verts, ik_error, mp_conf, joints):
        """Handle calibration frame."""
        # Show camera feed
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qt_img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self.viz_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))

        # Update progress
        elapsed = self.recorder.get_elapsed()
        target = self.recorder.target_duration
        progress_pct = elapsed / target * 100
        self.progress_label.setText(f"⏱️  Progress: {elapsed:.1f}s / {target}s ({progress_pct:.0f}%)")

        # Debug output every 5 seconds
        if int(elapsed) % 5 == 0 and elapsed > 0:
            print(f"[CALIBRATION] {elapsed:.0f}s / {target}s ({progress_pct:.0f}%) - Pose samples: {len(self.recorder.pose_timestamps)}, EMG samples: {len(self.recorder.emg_timestamps)}")

    def _handle_emg_status(self, connected):
        """Handle EMG connection status."""
        if connected:
            self.status_label.setText("Status: Recording (EMG Connected ✓)")
            self.status_label.setStyleSheet("font-size: 12px; color: #00ff00; padding: 5px;")

    def _check_calibration_complete(self):
        """Check if calibration recording is complete."""
        if self.recorder.is_complete():
            self.calib_timer.stop()
            self._finish_calibration()

    def _finish_calibration(self):
        """Finish calibration and start fine-tuning."""
        print("\n" + "="*60)
        print("✅ Calibration recording complete!")
        print("="*60)

        # Stop threads
        self.calib_video_thread.stop()
        self.calib_emg_thread.stop()
        self.calib_video_thread.wait()
        self.calib_emg_thread.wait()

        # Save data
        self.recorder.save()

        # Update UI
        self.status_label.setText("Status: Fine-tuning model...")
        self.status_label.setStyleSheet("font-size: 12px; color: #ffaa00; padding: 5px;")
        self.instructions_label.setText(
            "Fine-tuning in progress...\n\n"
            "This may take 2-5 minutes.\n"
            "The model is adapting to your\n"
            "current electrode placement."
        )

        # Process events to update UI
        QtWidgets.QApplication.processEvents()

        # Fine-tune (blocking)
        final_mpjpe = finetune_model(
            self.model, self.calibration_file, self.device,
            epochs=15, lr=1e-5
        )

        print(f"\n✅ Fine-tuning complete!")
        print(f"   Original: {self.original_mpjpe}mm")
        print(f"   Fine-tuned: {final_mpjpe:.2f}mm\n")

        # Switch to inference mode
        self._start_inference_mode()

    # ========================================================================
    # Inference Mode
    # ========================================================================

    def _start_inference_mode(self):
        """Start inference mode."""
        print("\n" + "="*60)
        print("🚀 INFERENCE MODE")
        print("="*60)

        self.title_label.setText("✅ Inference Mode (Calibrated)")
        self.status_label.setText("Status: Starting...")
        self.instructions_label.setText(
            "Real-time inference active!\n\n"
            "• Perform gestures\n"
            "• Watch 3D hand update\n"
            "• Confidence score shows quality\n"
            "• Green = good, Yellow = uncertain"
        )

        # Start inference thread
        self.emg_thread = EMGThread(self.model, self.device, self.emg_mean, self.emg_std)
        self.emg_thread.prediction_signal.connect(self._handle_prediction)
        self.emg_thread.start()

        # Render timer
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self._render_joints)
        self.render_timer.start(33)  # 30fps

    def _handle_prediction(self, joints, confidence, connected):
        """Handle inference prediction."""
        self.current_joints = joints.reshape(21, 3)
        self.confidence = confidence

        # Update status
        if connected:
            self.status_label.setText(f"Status: Connected ✓ | Confidence: {confidence:.1%}")
            if confidence > 0.7:
                color = "#00ff00"
            elif confidence > 0.4:
                color = "#ffff00"
            else:
                color = "#ff6600"
            self.status_label.setStyleSheet(f"font-size: 12px; color: {color}; padding: 5px;")

    def _render_joints(self):
        """Render 3D hand joints."""
        if self.current_joints is None or self.confidence < 0.3:
            return

        # Clear previous
        for node in self.mesh_nodes:
            if node in self.scene.nodes:
                self.scene.remove_node(node)
        self.mesh_nodes.clear()

        # Draw joints
        for joint_pos in self.current_joints:
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
            sphere.visual.vertex_colors = [100, 200, 255, 255]
            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=True)
            pose = np.eye(4)
            pose[:3, 3] = joint_pos
            node = self.scene.add(mesh, pose=pose)
            self.mesh_nodes.append(node)

        # Draw bones
        for (start_idx, end_idx) in HAND_CONNECTIONS:
            start = self.current_joints[start_idx]
            end = self.current_joints[end_idx]
            direction = end - start
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            cylinder = trimesh.creation.cylinder(radius=0.003, height=length)
            cylinder.visual.vertex_colors = [200, 200, 200, 255]
            mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=True)

            midpoint = (start + end) / 2
            direction_normalized = direction / length
            z_axis = np.array([0, 0, 1])

            if not (np.allclose(direction_normalized, z_axis) or np.allclose(direction_normalized, -z_axis)):
                v = np.cross(z_axis, direction_normalized)
                c = np.dot(z_axis, direction_normalized)
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-10))
            else:
                rotation = np.eye(3)

            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = midpoint
            node = self.scene.add(mesh, pose=pose)
            self.mesh_nodes.append(node)

        # Render
        try:
            color, _ = self.renderer.render(self.scene)
            h, w, ch = color.shape
            qt_img = QtGui.QImage(color.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            self.viz_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        except:
            pass

    def closeEvent(self, event):
        """Cleanup."""
        if hasattr(self, 'emg_thread'):
            self.emg_thread.stop()
            self.emg_thread.wait()
        if hasattr(self, 'calib_video_thread'):
            self.calib_video_thread.stop()
            self.calib_video_thread.stop()
        self.renderer.delete()
        event.accept()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EMG→Joints with integrated calibration")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration before inference')
    args = parser.parse_args()

    print("=" * 60)
    print("EMG → Joints Inference (with Calibration)")
    print("=" * 60)
    print(f"Model: {args.model}")
    if args.calibrate:
        print("Mode: CALIBRATION → FINE-TUNING → INFERENCE")
    else:
        print("Mode: INFERENCE ONLY")
    print("=" * 60)

    app = QtWidgets.QApplication(sys.argv)
    gui = UnifiedGUI(args.model, calibrate=args.calibrate)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
