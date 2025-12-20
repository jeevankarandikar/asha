"""
Real-time EMG → Joint Angles Inference (Transfer Learning)

Load transfer learning model and visualize real-time hand pose from EMG signals.
No camera needed - pure EMG-based tracking using emg2pose architecture!

Usage:
    python transfer/programs/inference_transfer.py
    python transfer/programs/inference_transfer.py --no-emg  # Test without EMG hardware
"""

import sys
from pathlib import Path

# Add src/ to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
# Add transfer/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import threading
import time
from collections import deque

import numpy as np
import torch
from PyQt5 import QtWidgets, QtCore, QtGui
import pyrender
import trimesh

# Shared utilities from src/
from utils.emg_utils import MindroveInterface, filter_emg

# Transfer learning model
from training.transfer_modules import ChannelAdapter, FrequencyUpsampler
from training.load_emg2pose import load_pretrained_emg2pose


class EMGThread(QtCore.QThread):
    """Background thread for EMG acquisition + transfer model inference @ 500Hz"""

    # Signal: (emg_filtered [8,], joint_angles [20,])
    emg_signal = QtCore.pyqtSignal(object, object)

    def __init__(self, model_checkpoint: str, window_size=50, use_emg=True, parent=None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()

        self.use_emg = use_emg
        self.window_size = window_size
        self.emg_buffer = deque(maxlen=window_size)

        # Load transfer model
        print("[info] Loading transfer model...")
        self.channel_adapter = ChannelAdapter()
        self.freq_upsampler = FrequencyUpsampler()
        self.emg2pose_model = load_pretrained_emg2pose(
            checkpoint_path=model_checkpoint,
            device="cpu"  # Use CPU for real-time (or "mps" for Apple Silicon)
        )

        # Set to eval mode
        self.channel_adapter.eval()
        self.emg2pose_model.eval()
        print("[success] Transfer model loaded")

        # Normalization stats (compute from first few seconds)
        self.emg_mean = None
        self.emg_std = None
        self.calibrating = True
        self.calibration_samples = []

    def _run_inference(self) -> np.ndarray:
        """
        Run transfer model on current EMG window.

        Returns:
            joint_angles: [20] predicted joint angles
        """
        # Convert buffer to tensor: [1, window_size, 8]
        emg_window = np.array(list(self.emg_buffer))  # [window_size, 8]
        emg_tensor = torch.from_numpy(emg_window).float().unsqueeze(0)  # [1, 50, 8]

        with torch.no_grad():
            # Channel adapter: [1, 50, 8] → [1, 50, 16]
            emg_16ch = self.channel_adapter(emg_tensor)

            # Frequency upsampler: [1, 50, 16] → [1, 200, 16]
            emg_2khz = self.freq_upsampler(emg_16ch)

            # emg2pose: [1, 200, 16] → [1, time', 20]
            joint_angles_seq = self.emg2pose_model(emg_2khz)

            # Take last timestep: [1, 20]
            joint_angles = joint_angles_seq[:, -1, :].squeeze(0).cpu().numpy()

        return joint_angles

    def run(self):
        """Main EMG processing loop"""
        if self.use_emg:
            self._run_with_emg()
        else:
            self._run_without_emg()

    def _run_with_emg(self):
        """Run with real EMG hardware"""
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

                # Filter EMG
                try:
                    emg_filtered = filter_emg(emg_raw)
                except Exception as e:
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
                        # Run transfer model inference
                        joint_angles = self._run_inference()

                        # Emit signal
                        self.emg_signal.emit(sample, joint_angles)

                        # Debug output every 50 predictions
                        inference_count += 1
                        if inference_count % 50 == 0:
                            print(f"[debug] inference {inference_count}: EMG=[{sample[0]:+.2f}, {sample[1]:+.2f}, ...], angles range=[{joint_angles.min():.2f}, {joint_angles.max():.2f}]")

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

    def _run_without_emg(self):
        """Run without EMG (test mode with synthetic data)"""
        print("[info] running in test mode (no EMG hardware)")
        self.calibrating = False
        inference_count = 0

        while self._running.is_set():
            try:
                # Generate synthetic EMG data
                sample = np.random.randn(8) * 100

                # Add to buffer
                self.emg_buffer.append(sample)

                # When buffer is full, run inference
                if len(self.emg_buffer) == self.window_size:
                    # Run transfer model inference
                    joint_angles = self._run_inference()

                    # Emit signal
                    self.emg_signal.emit(sample, joint_angles)

                    # Debug output every 50 predictions
                    inference_count += 1
                    if inference_count % 50 == 0:
                        print(f"[debug] inference {inference_count}: angles range=[{joint_angles.min():.2f}, {joint_angles.max():.2f}]")

                # Simulate 500Hz
                time.sleep(1/500)

            except Exception as e:
                print(f"[error] test thread: {e}")
                import traceback
                traceback.print_exc()
                continue

    def stop(self):
        self._running.clear()


class InferenceWindow(QtWidgets.QWidget):
    """Main window for transfer learning inference visualization"""

    def __init__(self, model_checkpoint: str, window_size=50, use_emg=True):
        super().__init__()
        self.model_checkpoint = model_checkpoint
        self.window_size = window_size
        self.use_emg = use_emg

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

        # Load MANO model for visualization
        print("[info] loading mano model...")
        from model_utils.mano_model import load_mano_layer
        self.mano_layer = load_mano_layer('models/mano', 'right', torch.device('cpu'))
        print("[success] mano model loaded")

        # Current state
        self.current_angles = np.zeros(20, dtype=np.float32)
        self.smoothed_angles = np.zeros(20, dtype=np.float32)
        # For now, use zero MANO params for visualization
        # TODO: Convert 20 joint angles to MANO θ (45D) using learned IK layer
        self.current_theta = np.zeros(45, dtype=np.float32)
        self.vertices = None
        self.faces = None

        # Confidence filtering
        self.angles_history = deque(maxlen=10)
        self.confidence_threshold = 0.5

        # Get MANO faces
        from model_utils.pose_fitter import get_mano_faces
        self.faces = get_mano_faces()

        # Setup pyrender scene
        self._setup_renderer()

        # Setup UI
        self.setup_ui()

        # Start EMG thread
        self.emg_thread = EMGThread(model_checkpoint, window_size, use_emg)
        self.emg_thread.emg_signal.connect(self.on_emg_update)
        self.emg_thread.start()

    def _setup_renderer(self):
        """Setup pyrender scene and renderer"""
        self.renderer = pyrender.OffscreenRenderer(640, 640)

        # Create scene with dark background
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
        self.setWindowTitle("Transfer Learning EMG → Joint Angles Inference")
        self.setGeometry(100, 100, 900, 800)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)

        # Status bar at top
        status_widget = QtWidgets.QWidget()
        status_layout = QtWidgets.QHBoxLayout(status_widget)

        if self.use_emg:
            status_text = "Status: Calibrating..."
        else:
            status_text = "Status: Test Mode (No EMG)"

        status_label = QtWidgets.QLabel(status_text)
        status_label.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        status_layout.addWidget(status_label)
        status_layout.addStretch()
        self.status_label = status_label

        main_layout.addWidget(status_widget)

        # Center: 3D MANO visualization
        mesh_widget = QtWidgets.QGroupBox("3D Hand Model (Transfer Learning)")
        mesh_layout = QtWidgets.QVBoxLayout(mesh_widget)

        self.mesh_label = QtWidgets.QLabel()
        self.mesh_label.setMinimumSize(800, 700)
        self.mesh_label.setStyleSheet(f"background-color: {self.bg_color}; border: 1px solid #404040;")
        self.mesh_label.setAlignment(QtCore.Qt.AlignCenter)
        mesh_layout.addWidget(self.mesh_label)

        main_layout.addWidget(mesh_widget)

        # Render timer (30fps)
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self._render_to_label)
        self.render_timer.start(33)  # 30fps

    def on_emg_update(self, emg, joint_angles):
        """Handle new EMG data and joint angle predictions"""
        # Compute confidence
        confidence = self._compute_confidence(emg, joint_angles)

        # Debug: print first few updates
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        if self._update_count <= 3:
            print(f"[debug] GUI received update {self._update_count}: angles shape={joint_angles.shape}, confidence={confidence:.2f}")

        # Only update if confidence is high enough
        if confidence > self.confidence_threshold:
            # Exponential moving average for smoothing
            alpha = 0.3
            if len(self.angles_history) == 0:
                self.smoothed_angles = joint_angles
            else:
                self.smoothed_angles = alpha * joint_angles + (1 - alpha) * self.smoothed_angles

            self.angles_history.append(joint_angles)
            self.current_angles = self.smoothed_angles

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
        # TODO: Convert 20 joint angles to MANO θ (45D)
        # For now, use zero theta for visualization
        theta_tensor = torch.FloatTensor(self.current_theta).unsqueeze(0)  # [1, 45]
        beta_tensor = torch.zeros(1, 10)  # [1, 10]

        with torch.no_grad():
            vertices, joints = self.mano_layer(theta_tensor, beta_tensor)
            self.vertices = vertices[0].numpy()  # [778, 3]

    def _compute_confidence(self, emg, joint_angles):
        """Compute confidence score (0-1)"""
        confidence_factors = []

        # Factor 1: EMG signal strength
        emg_magnitude = np.abs(emg).mean()
        if emg_magnitude > 1000:
            confidence_factors.append(1.0)
        elif emg_magnitude > 100:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.2)

        # Factor 2: Joint angles magnitude (should be reasonable)
        angles_max = np.abs(joint_angles).max()
        if angles_max < 3.0:
            confidence_factors.append(1.0)
        elif angles_max < 5.0:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.1)

        # Factor 3: Rate of change
        if len(self.angles_history) > 0:
            delta = np.abs(joint_angles - self.angles_history[-1]).max()
            if delta < 0.5:
                confidence_factors.append(1.0)
            elif delta < 1.0:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)

        # Combine factors
        return np.prod(confidence_factors) ** (1.0 / len(confidence_factors))

    def _render_to_label(self):
        """Render 3D mesh to QLabel"""
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
            # Update mesh in scene
            if self.mesh_node is not None:
                self.scene.remove_node(self.mesh_node)

            # Create mesh
            mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            mesh.visual.vertex_colors = [200, 200, 200, 255]
            mesh_render = pyrender.Mesh.from_trimesh(mesh, smooth=False)

            # Add to scene
            self.mesh_node = self.scene.add(mesh_render)

            # Render
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
    parser = argparse.ArgumentParser(description="Transfer learning inference")
    parser.add_argument('--no-emg', action='store_true', help='Run without EMG hardware (test mode)')
    parser.add_argument('--model', type=str, default=None, help='Path to emg2pose checkpoint')
    parser.add_argument('--window-size', type=int, default=50, help='EMG window size')
    args = parser.parse_args()

    # Default checkpoint path
    if args.model is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.model = str(repo_root / "models" / "emg2pose" / "tracking_vemg2pose.ckpt")

    print("=" * 60)
    print("Transfer Learning EMG → Joint Angles Inference")
    print("=" * 60)

    # Launch GUI
    app = QtWidgets.QApplication(sys.argv)
    window = InferenceWindow(
        model_checkpoint=args.model,
        window_size=args.window_size,
        use_emg=(not args.no_emg)
    )
    window.show()

    if args.no_emg:
        print("\n[ready] Running in TEST MODE (no EMG hardware)")
        print("[ready] Close window to exit\n")
    else:
        print("\n[ready] EMG inference started!")
        print("[ready] Put on MindRove EMG sensors and move your hand")
        print("[ready] Calibrating for 2 seconds...")
        print("[ready] Press Ctrl+C or close window to exit\n")

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
