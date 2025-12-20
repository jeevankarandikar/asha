"""
Real-time EMG → MANO Inference (v2 Model)

Loads the v2 trained model (34.92mm MPJPE, 20 min training data) and visualizes
real-time hand pose from EMG signals. Camera-free tracking!

Differences from old inference.py:
- Uses v2 model architecture (matches train_v2_colab.py exactly)
- Trained on 20 min diverse data (vs v3's 3 min)
- Different hyperparameters (window_size=50, different conv layers)

Usage:
    python -m asha.v4.inference --model models/v4/emg_model_v2_best.pth

Download model from Colab:
    /content/drive/MyDrive/Colab Notebooks/asha/checkpoints_v2/emg_model_v2_best.pth
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PyQt5 import QtWidgets, QtCore, QtGui
import pyrender
import trimesh

from asha.core.emg_utils import MindroveInterface, filter_emg
from asha.core.mano_model import CustomMANOLayer, load_mano_layer
from .model import SimpleEMGModel


# ============================================================================
# Model Architecture (MUST match train_v2_colab.py EXACTLY)
# ============================================================================

class SimpleEMGModel(nn.Module):
    """
    EMG→MANO model architecture (v2).

    Architecture:
    - Conv1D (8→64→128) + BatchNorm + ReLU + MaxPool
    - LSTM (128→256, 2 layers, dropout=0.3)
    - FC (256→128→45 MANO params)

    Training: 100 epochs, A100 GPU, batch_size=64
    Performance: Val loss 0.078480, MPJPE 34.92mm
    """

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
        x = self.pool(x)  # Downsample
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # Downsample

        x = x.transpose(1, 2)  # [batch, time, channels]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last timestep

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        theta = self.fc2(x)

        return theta


# ============================================================================
# Confidence & Smoothing
# ============================================================================

def compute_confidence(emg_filtered: np.ndarray, theta_pred: np.ndarray,
                      theta_history: deque) -> float:
    """
    Compute 3-factor confidence score for prediction quality.

    Factors:
    1. EMG signal strength (RMS) - reject weak signals
    2. θ magnitude - reject extreme values
    3. Rate of change - reject jittery motion

    Args:
        emg_filtered: [50, 8] filtered EMG window
        theta_pred: [45] predicted MANO params
        theta_history: deque of recent θ predictions

    Returns:
        confidence: 0.0-1.0 score
    """
    # 1. EMG signal strength (RMS)
    emg_strength = np.sqrt(np.mean(emg_filtered ** 2))
    strength_score = min(emg_strength / 0.5, 1.0)  # Normalize to [0, 1]

    # 2. θ magnitude (reject extreme values)
    theta_mag = np.linalg.norm(theta_pred)
    # Typical θ range: [-2, 2], so magnitude should be < 5
    mag_score = 1.0 if theta_mag < 5.0 else 0.0

    # 3. Rate of change (reject jittery motion)
    if len(theta_history) > 0:
        delta = np.linalg.norm(theta_pred - theta_history[-1])
        # Allow up to 1.0 radian change per frame (30fps)
        change_score = 1.0 if delta < 1.0 else 0.0
    else:
        change_score = 1.0

    # Combined score (average of 3 factors)
    confidence = (strength_score + mag_score + change_score) / 3.0

    return confidence


def smooth_theta(theta_pred: np.ndarray, theta_smoothed: np.ndarray,
                alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential moving average for temporal smoothing.

    Args:
        theta_pred: [45] current prediction
        theta_smoothed: [45] previous smoothed value (or None)
        alpha: smoothing factor (0=no update, 1=no smoothing)

    Returns:
        theta_new: [45] smoothed prediction
    """
    if theta_smoothed is None:
        return theta_pred

    return alpha * theta_pred + (1 - alpha) * theta_smoothed


# ============================================================================
# EMG Processing Thread
# ============================================================================

class EMGThread(QtCore.QThread):
    """
    Background thread for EMG acquisition and model inference.

    Pipeline:
    1. Acquire EMG from MindRove @ 500Hz
    2. Filter (notch + bandpass)
    3. Buffer 50 samples (100ms window)
    4. Run model inference
    5. Compute confidence score
    6. Apply temporal smoothing
    7. Emit result to GUI
    """

    # Signal: (theta, confidence, connected)
    prediction_signal = QtCore.pyqtSignal(object, float, bool)

    def __init__(self, model: nn.Module, device: torch.device,
                 emg_mean=None, emg_std=None, parent: QtCore.QObject = None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.emg_mean = emg_mean
        self.emg_std = emg_std
        self._running = threading.Event()
        self._running.set()

        # Buffers
        self.emg_buffer = deque(maxlen=50)  # 50 samples @ 500Hz = 100ms
        self.theta_history = deque(maxlen=10)  # Last 10 predictions
        self.theta_smoothed = None

    def run(self):
        """Main loop: acquire EMG, run inference, emit predictions."""
        interface = MindroveInterface()
        if not interface.connect():
            print("❌ Failed to connect to MindRove")
            self._running.clear()
            return

        print("✅ MindRove connected @ 500Hz")
        print("🚀 Starting real-time inference...")

        while self._running.is_set():
            # Get EMG data (fetch available samples)
            timestamps, emg_raw = interface.get_data(num_samples=10)

            if timestamps.size > 0:
                # Filter EMG
                emg_filtered = filter_emg(emg_raw)

                # Add to buffer (per-sample)
                for i in range(len(timestamps)):
                    self.emg_buffer.append(emg_filtered[i])

                    # Run inference when buffer full
                    if len(self.emg_buffer) == 50:
                        emg_window = np.array(self.emg_buffer)  # [50, 8]

                        # Normalize using GLOBAL statistics (same as training!)
                        if self.emg_mean is not None and self.emg_std is not None:
                            # Use global statistics from training
                            emg_norm = (emg_window - self.emg_mean) / self.emg_std
                        else:
                            # Fallback to per-window (old behavior, causes stuck predictions)
                            emg_mean = emg_window.mean(axis=0, keepdims=True)
                            emg_std = emg_window.std(axis=0, keepdims=True) + 1e-6
                            emg_norm = (emg_window - emg_mean) / emg_std

                        # Prepare input: [1, 8, 50]
                        emg_tensor = torch.from_numpy(emg_norm.T).float().unsqueeze(0).to(self.device)

                        # Model inference
                        with torch.no_grad():
                            theta_pred = self.model(emg_tensor).cpu().numpy()[0]  # [45]

                        # Compute confidence (use normalized EMG, not raw µV values)
                        confidence = compute_confidence(emg_norm, theta_pred, self.theta_history)

                        # DEBUG: Print diagnostics every 10 frames (~0.33 seconds)
                        if len(self.theta_history) % 10 == 0:
                            print(f"\n[DEBUG] Frame {len(self.theta_history)}")
                            print(f"  θ range: [{theta_pred.min():.3f}, {theta_pred.max():.3f}]")
                            print(f"  θ magnitude: {np.linalg.norm(theta_pred):.3f}")
                            print(f"  EMG raw: {np.sqrt(np.mean(emg_window**2)):.1f} µV")
                            print(f"  EMG norm: {np.sqrt(np.mean(emg_norm**2)):.3f} (z-score)")
                            if len(self.theta_history) > 0:
                                print(f"  Δθ: {np.linalg.norm(theta_pred - self.theta_history[-1]):.3f}")
                            print(f"  Confidence: {confidence:.1%}")

                        # Temporal smoothing
                        theta_smoothed = smooth_theta(theta_pred, self.theta_smoothed, alpha=0.3)
                        self.theta_smoothed = theta_smoothed

                        # Update history
                        self.theta_history.append(theta_pred)

                        # Emit to GUI
                        self.prediction_signal.emit(theta_smoothed, confidence, True)

            time.sleep(0.01)  # 10ms sleep (check ~100 times/sec)

        interface.disconnect()
        print("✅ MindRove disconnected")

    def stop(self):
        """Stop the thread."""
        self._running.clear()
        self.wait()


# ============================================================================
# Main GUI
# ============================================================================

class InferenceGUI(QtWidgets.QMainWindow):
    """
    Main GUI for real-time EMG→MANO inference.

    Features:
    - 3D hand mesh visualization (pyrender)
    - Real-time @ 30fps
    - Confidence score display
    - Model info (MPJPE, training data)
    - Status indicator
    """

    def __init__(self, model_path: str):
        super().__init__()

        self.setWindowTitle("EMG → MANO Inference (v2 Model)")
        self.setGeometry(100, 100, 1000, 600)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # Load model
        print(f"\nLoading model: {model_path}")
        self.model = SimpleEMGModel(window_size=50, input_channels=8).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            val_mpjpe = checkpoint.get('val_mpjpe', 'N/A')
            epoch = checkpoint.get('epoch', 'N/A')
            print(f"✅ Loaded checkpoint: epoch {epoch}, MPJPE {val_mpjpe}mm")

            # Load GLOBAL EMG normalization statistics (CRITICAL fix for stuck predictions!)
            self.emg_mean = checkpoint.get('emg_mean', None)
            self.emg_std = checkpoint.get('emg_std', None)

            if self.emg_mean is None or self.emg_std is None:
                print("⚠️  WARNING: Checkpoint missing global EMG statistics!")
                print("   Model trained with old code. Using fallback per-window normalization.")
                print("   Retrain model with updated train_v2_colab.py for best results.")
                self.emg_mean = None
                self.emg_std = None
            else:
                print(f"✅ Loaded global EMG statistics: mean={self.emg_mean.shape}, std={self.emg_std.shape}")

            self.model_info = f"v2 Model (20 min training, MPJPE: {val_mpjpe:.2f}mm)" if isinstance(val_mpjpe, float) else "v2 Model"
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights")
            self.model_info = "v2 Model (20 min training, MPJPE: 34.92mm)"
            self.emg_mean = None
            self.emg_std = None

        self.model.eval()

        # Load MANO
        print("\nLoading MANO model...")
        mano_root = "models/mano"  # Directory path, not file path
        self.mano_layer = CustomMANOLayer(mano_root, side='right', device=str(self.device))
        print("✅ MANO loaded")

        # State
        self.current_theta = None
        self.confidence = 0.0
        self.connected = False

        # Setup UI
        self._setup_ui()

        # Setup pyrender (dark mode, reuse scene)
        self._setup_renderer()

        # Start EMG thread
        self.emg_thread = EMGThread(self.model, self.device, self.emg_mean, self.emg_std)
        self.emg_thread.prediction_signal.connect(self._handle_prediction)
        self.emg_thread.start()

        # Render timer (30fps)
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self._render)
        self.render_timer.start(33)  # ~30fps

    def _setup_ui(self):
        """Setup UI components."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QHBoxLayout(central_widget)

        # Left: 3D visualization
        self.render_label = QtWidgets.QLabel()
        self.render_label.setFixedSize(800, 600)
        self.render_label.setStyleSheet("border: 2px solid #333; background-color: #1a1a1a;")
        layout.addWidget(self.render_label)

        # Right: Info panel
        info_layout = QtWidgets.QVBoxLayout()

        # Model info
        self.model_label = QtWidgets.QLabel(self.model_info)
        self.model_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff00; padding: 10px;")
        self.model_label.setWordWrap(True)
        info_layout.addWidget(self.model_label)

        # Status
        self.status_label = QtWidgets.QLabel("Status: Connecting...")
        self.status_label.setStyleSheet("font-size: 12px; color: #ffff00; padding: 5px;")
        info_layout.addWidget(self.status_label)

        # Confidence
        self.confidence_label = QtWidgets.QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("font-size: 12px; color: #888; padding: 5px;")
        info_layout.addWidget(self.confidence_label)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Instructions:\n\n"
            "• Perform gestures with EMG electrodes\n"
            "• Watch 3D hand mesh update in real-time\n"
            "• Confidence score shows prediction quality\n"
            "• Green = good, Yellow = uncertain, Red = rejected"
        )
        instructions.setStyleSheet("font-size: 11px; color: #aaa; padding: 10px;")
        instructions.setWordWrap(True)
        info_layout.addWidget(instructions)

        info_layout.addStretch()
        layout.addLayout(info_layout)

    def _setup_renderer(self):
        """Setup pyrender scene (dark mode, reuse for efficiency)."""
        # Scene
        self.scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.1, 0.1, 0.1, 1.0])

        # Camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.scene.add(camera, pose=camera_pose)

        # Light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=camera_pose)

        # Renderer
        self.renderer = pyrender.OffscreenRenderer(800, 600)

        # Hand mesh node (will be updated)
        self.hand_mesh_node = None

    def _handle_prediction(self, theta: np.ndarray, confidence: float, connected: bool):
        """Handle prediction from EMG thread."""
        self.current_theta = theta
        self.confidence = confidence
        self.connected = connected

        # Update status
        if connected:
            self.status_label.setText("Status: Connected ✓")
            self.status_label.setStyleSheet("font-size: 12px; color: #00ff00; padding: 5px;")
        else:
            self.status_label.setText("Status: Not Connected")
            self.status_label.setStyleSheet("font-size: 12px; color: #ff0000; padding: 5px;")

        # Update confidence
        if confidence > 0.7:
            color = "#00ff00"  # Green
        elif confidence > 0.4:
            color = "#ffff00"  # Yellow
        else:
            color = "#ff6600"  # Orange

        self.confidence_label.setText(f"Confidence: {confidence:.1%}")
        self.confidence_label.setStyleSheet(f"font-size: 12px; color: {color}; padding: 5px;")

    def _render(self):
        """Render 3D hand mesh (called by timer @ 30fps)."""
        if self.current_theta is None:
            return

        # Reject low confidence predictions
        if self.confidence < 0.3:
            return

        # MANO forward pass
        theta_tensor = torch.from_numpy(self.current_theta).float().unsqueeze(0).to(self.device)
        betas = torch.zeros(1, 10, device=self.device)

        with torch.no_grad():
            vertices, joints = self.mano_layer(theta_tensor, betas)

        verts = vertices[0].cpu().numpy()

        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=self.mano_layer.faces.cpu().numpy())
        mesh.visual.vertex_colors = [200, 200, 200, 255]  # Light gray

        # Update scene
        if self.hand_mesh_node is not None:
            self.scene.remove_node(self.hand_mesh_node)

        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        self.hand_mesh_node = self.scene.add(mesh_node)

        # Render
        color, _ = self.renderer.render(self.scene)

        # Convert to Qt
        h, w, ch = color.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(color.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.render_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Clean up on close."""
        self.emg_thread.stop()
        self.renderer.delete()
        event.accept()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-time EMG→MANO inference (v2 model)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/v4/emg_model_v2_best.pth",
        help="Path to trained model checkpoint"
    )
    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("\nDownload from Colab:")
        print("  /content/drive/MyDrive/Colab Notebooks/asha/checkpoints_v2/emg_model_v2_best.pth")
        print("\nPlace in:")
        print(f"  {args.model}")
        sys.exit(1)

    print("="*60)
    print("EMG → MANO Inference (v2 Model)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Training: 20 min data, 100 epochs, A100 GPU")
    print(f"Performance: MPJPE 34.92mm")
    print("="*60 + "\n")

    app = QtWidgets.QApplication(sys.argv)
    gui = InferenceGUI(args.model)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
