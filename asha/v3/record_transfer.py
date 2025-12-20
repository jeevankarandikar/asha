"""
Transfer learning recording system.

Records:
- EMG (8ch @ 500Hz)
- 20 joint angles from transfer model (emg2pose)
- MANO θ from MediaPipe + IK (for comparison/flexibility)
- MediaPipe landmarks

This gives us maximum flexibility during training - we can use either
joint angles or MANO params as ground truth.

Usage:
    python transfer/programs/record_transfer.py              # with EMG hardware
    python transfer/programs/record_transfer.py --no-emg     # camera-only testing
"""

import sys
from pathlib import Path

# Add src/ to path FIRST for shared utilities
src_path = str(Path(__file__).parent.parent.parent / "src")
sys.path.insert(0, src_path)

import argparse
import threading
from typing import Optional
import time

import numpy as np
import torch
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

import pyrender
import trimesh

# Shared utilities from src/
from model_utils.tracker import get_landmarks, get_landmarks_world, draw_landmarks_on_frame
from model_utils.pose_fitter import mano_from_landmarks, get_mano_faces
from utils.emg_utils import MindroveInterface, filter_emg

# Now add transfer/ to path for transfer-specific modules
transfer_path = str(Path(__file__).parent.parent)
sys.path.insert(0, transfer_path)

# No inference modules needed for recording!
# Just record EMG + camera data for training

# Import transfer's DataRecorder explicitly from its path
transfer_utils_path = transfer_path + "/utils"
sys.path.insert(0, transfer_utils_path)
from data_recorder import DataRecorder
sys.path.remove(transfer_utils_path)  # Clean up to avoid conflicts


class VideoThread(QtCore.QThread):
    """
    Background thread for webcam capture and hand tracking @ 25fps.

    Runs MediaPipe + MANO IK (same as src/programs/record.py).
    """

    # signal: (frame_bgr, verts, ik_error, mp_confidence, theta, joints, mp_landmarks)
    frame_signal = QtCore.pyqtSignal(object, object, float, float, object, object, object)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()
        self._cap_index = 0
        self._frame_count = 0

    def run(self):
        """Main loop: capture frames, run MediaPipe + MANO IK."""
        cap = cv2.VideoCapture(self._cap_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self._running.is_set():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                continue

            # MediaPipe detection (returns tuple or None)
            # NOTE: Pass BGR frame directly - MediaPipe functions handle RGB conversion internally
            landmarks_result = get_landmarks(frame)
            landmarks_world_result = get_landmarks_world(frame)

            # Debug: print every 30 frames (once per second)
            if self._frame_count % 30 == 0:
                print(f"[debug] Frame {self._frame_count}: landmarks={landmarks_result is not None}, world={landmarks_world_result is not None}")

            # Initialize pose data (None if no hand detected)
            verts = None
            theta = None
            joints = None
            ik_error = 0.0
            mp_confidence = 0.0
            landmarks = None
            landmarks_world = None

            if landmarks_world_result is not None:
                # Unpack tuple: (landmarks, confidence)
                landmarks_world, mp_confidence = landmarks_world_result

                # MANO IK fitting - Note: returns (verts, joints, theta, ik_error)
                verts, joints, theta, ik_error = mano_from_landmarks(landmarks_world)

            if landmarks_result is not None:
                # Also get image landmarks for drawing
                landmarks, _ = landmarks_result

            # Always emit signal (even if no hand detected) so camera feed shows
            self.frame_signal.emit(
                frame,
                verts,
                ik_error,
                mp_confidence,
                theta,
                joints,
                landmarks  # Use image landmarks for drawing (not world)
            )

            self._frame_count += 1
            time.sleep(1/30)  # ~30fps

        cap.release()

    def stop(self):
        """Stop the thread."""
        self._running.clear()
        self.wait()


class EMGRecordingThread(QtCore.QThread):
    """
    Background thread for EMG capture (NO INFERENCE).

    Simply records:
    1. MindRove EMG acquisition @ 500Hz
    2. Filters the signal
    3. Emits for saving

    NO inference during recording - that comes later after training!
    """

    # signal: (timestamp, emg_raw, emg_filtered)
    emg_signal = QtCore.pyqtSignal(float, object, object)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()
        print("✅ EMG recording thread initialized")

    def run(self):
        """Main loop: acquire and filter EMG @ 500Hz."""
        interface = MindroveInterface()
        if not interface.connect():
            print("❌ Failed to connect to MindRove")
            self._running.clear()
            return
        print("✅ MindRove connected @ 500Hz")

        while self._running.is_set():
            # Get EMG data (fetch available samples)
            timestamps, emg_raw = interface.get_data(num_samples=50)

            if timestamps.size > 0:
                # Filter EMG
                emg_filtered = filter_emg(emg_raw)

                # Emit signal for each sample
                for i in range(len(timestamps)):
                    self.emg_signal.emit(
                        timestamps[i],
                        emg_raw[i],
                        emg_filtered[i]
                    )

            time.sleep(0.01)  # 10ms sleep (check for data ~100 times/sec)

        interface.disconnect()
        print("✅ MindRove disconnected")

    def stop(self):
        """Stop the thread."""
        self._running.clear()
        self.wait()


class TransferRecordingGUI(QtWidgets.QMainWindow):
    """
    Main GUI for transfer learning recording.

    Displays:
    - Webcam feed with MediaPipe landmarks
    - 3D MANO mesh
    - Recording controls

    Saves:
    - EMG data (raw + filtered)
    - 20 joint angles from transfer model
    - MANO θ from IK
    - MediaPipe landmarks
    """

    def __init__(self, use_emg: bool = True):
        super().__init__()
        self.use_emg = use_emg

        self.setWindowTitle("Transfer Learning Recording")
        self.setGeometry(100, 100, 1400, 600)

        # Data recorder
        self.recorder: Optional[DataRecorder] = None
        self.recording = False

        # Session guidance
        self.session_active = False
        self.session_start_time = 0
        self.current_phase_idx = 0
        self.session_phases = [
            {"name": "Rest/Calibration", "duration": 30, "instructions": "Palm facing camera, hand relaxed"},
            {"name": "Gesture Practice", "duration": 150, "instructions": "Palm facing camera - NO wrist rotation! Gestures: Open, Fist, Pinch, Point, Thumbs up, Peace, OK sign - Hold each 2-3 sec, smooth transitions"},
            {"name": "Gesture Variations", "duration": 30, "instructions": "Same gestures but vary speed and strength"},
            {"name": "Cool Down", "duration": 30, "instructions": "Slow movements, back to rest"}
        ]

        # Setup UI
        self._setup_ui()

        # Session timer (updates every 100ms)
        self.session_timer = QtCore.QTimer()
        self.session_timer.timeout.connect(self._update_session_timer)
        self.session_timer.start(100)

        # Start threads
        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self._handle_video_frame)
        self.video_thread.start()

        if self.use_emg:
            self.emg_thread = EMGRecordingThread()
            self.emg_thread.emg_signal.connect(self._handle_emg_signal)
            self.emg_thread.start()

    def _setup_ui(self):
        """Setup UI components."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QHBoxLayout(central_widget)

        # Left: Webcam feed
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setFixedSize(640, 480)
        self.webcam_label.setStyleSheet("border: 2px solid black")
        layout.addWidget(self.webcam_label)

        # Right: Controls + session guidance
        right_panel = QtWidgets.QVBoxLayout()

        # Session info (large, prominent)
        self.session_info_label = QtWidgets.QLabel("Ready to Record - V2 (Simplified)\n\n⚠️ Palm facing camera ONLY - NO wrist rotation!\nTarget: 5 sessions × 4 min = 20 min total")
        self.session_info_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; background-color: #2d2d2d; color: #00ff00; border-radius: 5px;")
        self.session_info_label.setWordWrap(True)
        self.session_info_label.setMinimumHeight(100)
        right_panel.addWidget(self.session_info_label)

        # Phase timer (very large)
        self.phase_timer_label = QtWidgets.QLabel("--:--")
        self.phase_timer_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #00ff00; padding: 20px;")
        self.phase_timer_label.setAlignment(QtCore.Qt.AlignCenter)
        right_panel.addWidget(self.phase_timer_label)

        # Instructions (large, readable)
        self.instructions_label = QtWidgets.QLabel("Click 'Start Recording' to begin")
        self.instructions_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #1a1a1a; color: #ffffff; border-radius: 5px;")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setMinimumHeight(80)
        right_panel.addWidget(self.instructions_label)

        # Recording controls
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.record_button.clicked.connect(self._toggle_recording)
        right_panel.addWidget(self.record_button)

        # Stats (smaller)
        self.stats_label = QtWidgets.QLabel("Ready")
        self.stats_label.setStyleSheet("font-size: 11px; color: #888888;")
        right_panel.addWidget(self.stats_label)

        right_panel.addStretch()
        layout.addLayout(right_panel)

    def _handle_video_frame(self, frame, verts, ik_error, mp_confidence, theta, joints, landmarks):
        """Handle video frame from VideoThread."""
        # Draw landmarks on frame (if hand detected)
        if landmarks is not None:
            draw_landmarks_on_frame(frame, landmarks, mp_confidence)

        # Convert to Qt format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

        # Record pose data if recording
        if self.recording and self.recorder and theta is not None:
            self.recorder.record_pose(theta, joints, ik_error, mp_confidence)

    def _handle_emg_signal(self, timestamp, emg_raw, emg_filtered):
        """Handle EMG signal from EMGRecordingThread."""
        # Record EMG data if recording
        if self.recording and self.recorder:
            self.recorder.record_emg(
                emg_raw.reshape(1, -1),
                emg_filtered.reshape(1, -1)
            )
            # No inference during recording - just save raw EMG!

    def _toggle_recording(self):
        """Toggle recording on/off."""
        if not self.recording:
            # Start recording
            from datetime import datetime
            import time
            from pathlib import Path

            # Create transfer/data_v2 directory if it doesn't exist
            data_dir = Path(__file__).parent.parent / "data_v2"
            data_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(data_dir / f"transfer_session_{timestamp}.h5")

            self.recorder = DataRecorder(output_path, subject_id="transfer_test")
            self.recording = True
            self.session_active = True
            self.session_start_time = time.time()
            self.current_phase_idx = 0

            self.record_button.setText("Stop Recording")
            self.stats_label.setText(f"Recording to: {output_path}")
            print(f"📹 Started recording: {output_path}")

        else:
            # Stop recording
            if self.recorder:
                self.recorder.save()
                stats = self.recorder.get_stats()
                print(f"✅ Recording saved:")
                print(f"   EMG samples: {stats['emg_samples']}")
                print(f"   Pose samples: {stats['pose_samples']}")
                print(f"   Duration: {stats['duration_sec']:.1f}s")

            self.recording = False
            self.session_active = False
            self.recorder = None
            self.record_button.setText("Start Recording")
            self.stats_label.setText("Ready")
            self.session_info_label.setText("Ready to Record - V2 (Simplified)\n\n⚠️ Palm facing camera ONLY - NO wrist rotation!\nTarget: 5 sessions × 4 min = 20 min total")
            self.phase_timer_label.setText("--:--")
            self.instructions_label.setText("Click 'Start Recording' to begin")

    def _update_session_timer(self):
        """Update session timer and phase progression."""
        if not self.session_active:
            return

        import time
        elapsed = time.time() - self.session_start_time

        # Calculate total session time and current phase time
        phase_start_time = sum(p["duration"] for p in self.session_phases[:self.current_phase_idx])
        phase_elapsed = elapsed - phase_start_time

        # Check if we should advance to next phase
        current_phase = self.session_phases[self.current_phase_idx]
        if phase_elapsed >= current_phase["duration"]:
            self.current_phase_idx += 1

            # Check if session is complete
            if self.current_phase_idx >= len(self.session_phases):
                # Session complete - auto stop
                self._toggle_recording()
                return

            current_phase = self.session_phases[self.current_phase_idx]
            phase_start_time = sum(p["duration"] for p in self.session_phases[:self.current_phase_idx])
            phase_elapsed = elapsed - phase_start_time

        # Update UI
        phase_remaining = current_phase["duration"] - phase_elapsed
        mins = int(phase_remaining // 60)
        secs = int(phase_remaining % 60)

        # Phase timer (large countdown)
        self.phase_timer_label.setText(f"{mins:02d}:{secs:02d}")

        # Session info
        total_duration = sum(p["duration"] for p in self.session_phases)
        session_mins = int(elapsed // 60)
        session_secs = int(elapsed % 60)
        self.session_info_label.setText(
            f"Phase {self.current_phase_idx + 1}/5: {current_phase['name']}\n"
            f"Session time: {session_mins:02d}:{session_secs:02d} / {int(total_duration // 60):02d}:00"
        )

        # Instructions
        self.instructions_label.setText(f"💡 {current_phase['instructions']}")

    def closeEvent(self, event):
        """Clean up on close."""
        self.video_thread.stop()
        if self.use_emg:
            self.emg_thread.stop()
        event.accept()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Transfer learning recording - EMG + Camera")
    parser.add_argument("--no-emg", action="store_true", help="Camera-only mode (no EMG hardware)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    gui = TransferRecordingGUI(use_emg=(not args.no_emg))
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
