"""
Real-time EMG → MANO Inference

Load trained EMG model and visualize real-time hand pose from EMG signals.
No camera needed - pure EMG-based tracking!

Usage:
    python src/programs/emg_inference.py --model emg_model_best.pth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from PyQt5 import QtWidgets, QtCore, QtGui
import pyrender
import trimesh

from utils.emg_utils import MindroveInterface, filter_emg


class SimpleEMGModel(nn.Module):
    """Same architecture as training script"""

    def __init__(self, input_channels=8, hidden_size=256, output_dim=45):
        super().__init__()

        # Temporal convolutions
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


class EMGThread(QtCore.QThread):
    """Background thread for EMG acquisition @ 500Hz"""

    # Signal: (emg_filtered [8,], theta [45,])
    emg_signal = QtCore.pyqtSignal(object, object)

    def __init__(self, model, window_size=50, parent=None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()

        self.model = model
        self.window_size = window_size
        self.emg_buffer = deque(maxlen=window_size)

        # Normalization stats (compute from first few seconds)
        self.emg_mean = None
        self.emg_std = None
        self.calibrating = True
        self.calibration_samples = []

    def run(self):
        """Main EMG processing loop"""
        print("[info] connecting to mindrove...")

        try:
            mindrove = MindroveInterface()
            if not mindrove.connect():
                raise Exception("Failed to connect to MindRove")
            print("[success] mindrove connected @ 500hz")

        except Exception as e:
            print(f"[error] failed to connect to mindrove: {e}")
            return

        batch_count = 0
        inference_count = 0

        while self._running.is_set():
            try:
                # Get EMG batch (timestamps, data)
                timestamps, emg_raw = mindrove.get_data(num_samples=50)

                if emg_raw is None or len(emg_raw) == 0:
                    continue

                # Check if we got valid data
                if emg_raw.shape[0] == 0 or emg_raw.shape[1] != 8:
                    continue

                # Filter EMG (function handles small buffers gracefully)
                try:
                    emg_filtered = filter_emg(emg_raw)
                except Exception as e:
                    # If filtering fails, use raw data as fallback
                    print(f"[warning] filter failed: {e}, using raw data")
                    emg_filtered = emg_raw

                # Process each sample
                for sample in emg_filtered:
                    # Calibration phase (first 2 seconds)
                    if self.calibrating:
                        self.calibration_samples.append(sample)
                        if len(self.calibration_samples) % 100 == 0:
                            print(f"[debug] calibration progress: {len(self.calibration_samples)}/1000 samples")
                        if len(self.calibration_samples) >= 1000:  # 2 seconds @ 500Hz
                            calib_data = np.array(self.calibration_samples)
                            self.emg_mean = np.mean(calib_data, axis=0)
                            self.emg_std = np.std(calib_data, axis=0) + 1e-6
                            self.calibrating = False
                            print("[info] calibration complete")
                            print(f"[info] EMG mean: {self.emg_mean[:3]}, std: {self.emg_std[:3]}")
                        continue

                    # Normalize
                    sample_normalized = (sample - self.emg_mean) / self.emg_std

                    # Add to buffer
                    self.emg_buffer.append(sample_normalized)

                    # When buffer is full, run inference
                    if len(self.emg_buffer) == self.window_size:
                        # Prepare input [1, window_size, 8]
                        emg_window = torch.FloatTensor(np.array(self.emg_buffer)).unsqueeze(0)

                        # Inference
                        with torch.no_grad():
                            theta_pred = self.model(emg_window)
                            theta_pred = theta_pred.squeeze(0).numpy()  # [45]

                        # Emit signal
                        self.emg_signal.emit(sample, theta_pred)

                        # Debug output every 50 predictions
                        inference_count += 1
                        if inference_count % 50 == 0:
                            print(f"[debug] inference {inference_count}: EMG=[{sample[0]:+.2f}, {sample[1]:+.2f}, ...], θ range=[{theta_pred.min():.2f}, {theta_pred.max():.2f}]")

                batch_count += 1

            except Exception as e:
                print(f"[error] emg thread: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Small delay to avoid overwhelming the system
            time.sleep(0.001)

        # Cleanup
        mindrove.disconnect()
        print("[info] emg thread stopped")

    def stop(self):
        self._running.clear()


class InferenceWindow(QtWidgets.QWidget):
    """Main window for EMG inference visualization"""

    def __init__(self, model, window_size=50):
        super().__init__()
        self.model = model
        self.window_size = window_size

        # Dark theme colors
        self.bg_color = '#1e1e1e'
        self.card_color = '#2d2d2d'
        self.text_color = '#e0e0e0'
        self.accent_color = '#4a9eff'

        # Apply dark theme
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.bg_color};
                color: {self.text_color};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QGroupBox {{
                background-color: {self.card_color};
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                color: {self.accent_color};
            }}
            QLabel {{
                color: {self.text_color};
            }}
        """)

        # Load MANO model
        print("[info] loading mano model...")
        from model_utils.mano_model import load_mano_layer
        import torch
        self.mano_layer = load_mano_layer('models', 'right', torch.device('cpu'))
        print("[success] mano model loaded")

        # Current state
        self.current_theta = np.zeros(45, dtype=np.float32)
        self.smoothed_theta = np.zeros(45, dtype=np.float32)  # Smoothed predictions
        self.vertices = None
        self.faces = None

        # Confidence filtering
        self.theta_history = deque(maxlen=10)  # Track last 10 predictions
        self.confidence_threshold = 0.5  # 0-1, higher = more strict

        # Get MANO faces
        from model_utils.pose_fitter import get_mano_faces
        self.faces = get_mano_faces()

        # Setup pyrender scene (create once, reuse every frame)
        self._setup_renderer()

        # Setup UI
        self.setup_ui()

        # Start EMG thread
        self.emg_thread = EMGThread(model, window_size)
        self.emg_thread.emg_signal.connect(self.on_emg_update)
        self.emg_thread.start()

    def _setup_renderer(self):
        """Setup pyrender scene and renderer (create once, reuse)"""
        # Create offscreen renderer (expensive, so do it once)
        self.renderer = pyrender.OffscreenRenderer(640, 640)

        # Create scene with dark background (matching record.py)
        self.scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0], ambient_light=[0.3, 0.3, 0.3])

        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1]
        ])
        self.scene.add(camera, pose=self.camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        self.scene.add(light, pose=self.camera_pose)

        # Mesh node (will be updated each frame)
        self.mesh_node = None

    def setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("EMG → MANO Inference (Camera-Free!)")
        self.setGeometry(100, 100, 900, 800)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Status bar at top
        status_widget = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_widget)

        status_label = QtWidgets.QLabel("Status: Calibrating...")
        status_label.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        status_layout.addWidget(status_label)
        status_layout.addStretch()
        self.status_label = status_label

        main_layout.addWidget(status_widget)

        # Center: 3D MANO visualization
        mesh_widget = QtWidgets.QGroupBox("3D Hand Model (EMG-Driven)")
        mesh_layout = QtWidgets.QVBoxLayout(mesh_widget)

        self.mesh_label = QtWidgets.QLabel()
        self.mesh_label.setMinimumSize(800, 700)
        self.mesh_label.setStyleSheet(f"background-color: {self.bg_color}; border: 1px solid #404040;")
        self.mesh_label.setAlignment(QtCore.Qt.AlignCenter)
        mesh_layout.addWidget(self.mesh_label)

        main_layout.addWidget(mesh_widget)

        # Render timer (30fps - now optimized!)
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self._render_to_label)
        self.render_timer.start(33)  # 30fps

    def on_emg_update(self, emg, theta):
        """Handle new EMG data and prediction with confidence filtering"""
        # Compute confidence
        confidence = self._compute_confidence(emg, theta)

        # Debug: print first few updates
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        if self._update_count <= 3:
            print(f"[debug] GUI received update {self._update_count}: theta shape={theta.shape}, confidence={confidence:.2f}")

        # Only update if confidence is high enough
        if confidence > self.confidence_threshold:
            # Exponential moving average for smoothing
            alpha = 0.3  # Smoothing factor (lower = smoother)
            if len(self.theta_history) == 0:
                self.smoothed_theta = theta
            else:
                self.smoothed_theta = alpha * theta + (1 - alpha) * self.smoothed_theta

            self.theta_history.append(theta)
            self.current_theta = self.smoothed_theta

            # Update status
            status_text = f"Status: Tracking ✓ (Confidence: {confidence:.0%})"
        else:
            # Low confidence - keep previous pose
            status_text = f"Status: Low Signal (Confidence: {confidence:.0%})"

        if not self.emg_thread.calibrating:
            self.status_label.setText(status_text)
            color = self.accent_color if confidence > self.confidence_threshold else "#ff9900"
            self.status_label.setStyleSheet(f"color: {color};")

        # Update MANO vertices
        theta_tensor = torch.FloatTensor(self.current_theta).unsqueeze(0)  # [1, 45]
        beta_tensor = torch.zeros(1, 10)  # [1, 10]

        with torch.no_grad():
            vertices, joints = self.mano_layer(theta_tensor, beta_tensor)
            self.vertices = vertices[0].numpy()  # [778, 3]

    def _compute_confidence(self, emg, theta):
        """Compute confidence score (0-1) based on signal quality and prediction"""
        confidence_factors = []

        # Factor 1: EMG signal strength (distance from zero)
        # If EMG is near zero, confidence should be low
        emg_magnitude = np.abs(emg).mean()
        if emg_magnitude > 1000:  # Strong signal
            confidence_factors.append(1.0)
        elif emg_magnitude > 100:  # Moderate signal
            confidence_factors.append(0.7)
        else:  # Weak/zero signal
            confidence_factors.append(0.2)

        # Factor 2: Theta magnitude (should be reasonable)
        # MANO θ values typically in [-3, 3], extreme values = low confidence
        theta_max = np.abs(theta).max()
        if theta_max < 4.0:  # Normal range
            confidence_factors.append(1.0)
        elif theta_max < 5.0:  # Borderline
            confidence_factors.append(0.5)
        else:  # Extreme values
            confidence_factors.append(0.1)

        # Factor 3: Rate of change (if we have history)
        if len(self.theta_history) > 0:
            delta = np.abs(theta - self.theta_history[-1]).max()
            if delta < 0.5:  # Small change
                confidence_factors.append(1.0)
            elif delta < 1.0:  # Moderate change
                confidence_factors.append(0.7)
            else:  # Large jump
                confidence_factors.append(0.3)

        # Combine factors (geometric mean)
        return np.prod(confidence_factors) ** (1.0 / len(confidence_factors))

    def _render_to_label(self):
        """Render 3D mesh to QLabel (optimized - reuses renderer)"""
        if self.vertices is None:
            if not hasattr(self, '_no_vertices_warned'):
                print("[debug] waiting for vertices...")
                self._no_vertices_warned = True
            return

        # Debug: first render
        if not hasattr(self, '_first_render_done'):
            print(f"[debug] first render! vertices shape: {self.vertices.shape}")
            self._first_render_done = True

        try:
            # Update mesh in scene (remove old, add new)
            if self.mesh_node is not None:
                self.scene.remove_node(self.mesh_node)

            # Create mesh
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            mesh.visual.vertex_colors = [200, 200, 200, 255]
            mesh_render = pyrender.Mesh.from_trimesh(mesh, smooth=False)

            # Add to scene
            self.mesh_node = self.scene.add(mesh_render)

            # Render (reuse existing renderer!)
            color, _ = self.renderer.render(self.scene)

            # Convert to QPixmap
            height, width, channel = color.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(color.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_img)
            self.mesh_label.setPixmap(pixmap.scaled(
                self.mesh_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            ))

        except Exception as e:
            print(f"[error] render: {e}")

    def closeEvent(self, event):
        """Cleanup on close"""
        print("[info] shutting down...")
        self.emg_thread.stop()
        self.emg_thread.wait()

        # Cleanup renderer
        if hasattr(self, 'renderer'):
            self.renderer.delete()
            print("[info] renderer cleaned up")

        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='emg_model_best.pth',
                        help='Path to trained EMG model')
    parser.add_argument('--window-size', type=int, default=50,
                        help='EMG window size (must match training)')
    args = parser.parse_args()

    print("=" * 60)
    print("EMG → MANO Real-Time Inference (Camera-Free!)")
    print("=" * 60)

    # Load model
    print(f"\n[info] loading model from: {args.model}")
    model = SimpleEMGModel(input_channels=8, hidden_size=256, output_dim=45)

    try:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("[success] model loaded")
    except FileNotFoundError:
        print(f"[error] model file not found: {args.model}")
        print("Train a model first: python src/training/train_emg.py")
        return

    model.eval()

    # Launch GUI
    app = QtWidgets.QApplication(sys.argv)
    window = InferenceWindow(model, window_size=args.window_size)
    window.show()

    print("\n[ready] EMG inference started!")
    print("[ready] Put on MindRove EMG sensors and move your hand")
    print("[ready] Calibrating for 2 seconds...")
    print("[ready] Press Ctrl+C or close window to exit\n")

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
