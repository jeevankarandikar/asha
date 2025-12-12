"""
real-time hand pose system - mano_v1 (full IK with LBS articulation).

features:
  - mediapipe hand landmark detection with world coordinates
  - full MANO model with linear blend skinning (LBS)
  - inverse kinematics (IK) optimization per frame
  - proper finger articulation with rodrigues rotations
  - forward kinematics through kinematic tree
  - pose blend shapes for realistic deformation
  - performance: ~40ms per frame (25fps)

capabilities:
  - fingers articulate correctly (bending, curling)
  - realistic hand deformation
  - pose optimization via IK (15 adam steps)

to run:
  python src/mano_v1.py

for baseline comparison:
  python src/mediapipe_v0.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
from typing import Optional

import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

import pyrender
import trimesh

from model_utils.tracker import get_landmarks, get_landmarks_world, draw_landmarks_on_frame
from model_utils.pose_fitter import mano_from_landmarks, get_mano_faces, debug_print_joint_order


class VideoThread(QtCore.QThread):
    """Background thread for webcam capture and hand tracking."""

    frame_signal = QtCore.pyqtSignal(object, object)  # (frame_bgr, verts)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()
        self._cap_index = 0

    def stop(self):
        """Signal the thread to stop."""
        self._running.clear()

    def run(self):
        """Main video capture loop."""
        cap = cv2.VideoCapture(self._cap_index)
        if not cap.isOpened():
            print(" Failed to open camera")
            self.frame_signal.emit(None, None)
            return

        print(" Camera opened successfully")

        try:
            while self._running.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # detect hand landmarks
                # prefer world coordinates (metric, more stable depth)
                # fall back to image coordinates (normalized 0-1) if world unavailable
                result = get_landmarks_world(frame)
                if result is None:
                    result = get_landmarks(frame)

                verts = None

                if result is not None:
                    landmarks, confidence = result
                    # draw landmarks on frame
                    draw_landmarks_on_frame(frame, landmarks, confidence)
                    # generate MANO mesh via IK fitting
                    verts, _, _, _ = mano_from_landmarks(landmarks)

                self.frame_signal.emit(frame, verts)
        finally:
            cap.release()
            print(" Camera released")


class RealtimeApp(QtWidgets.QWidget):
    """Main application window with dual-panel display."""

    def __init__(self):
        super().__init__()
        print(" initializing project asha (mano_v1 - full ik + lbs)...")

        self.setWindowTitle("project asha - mano_v1 (full ik + lbs)")
        self.setGeometry(100, 100, 1600, 600)

        # Setup GUI panels
        self._setup_ui()

        # Setup 3D rendering
        self._setup_3d_scene()

        # debug: uncomment to validate MANO joint mapping (run once, then comment out)
        # debug_print_joint_order()

        # Start video capture thread
        self.thread = VideoThread()
        self.thread.frame_signal.connect(self.update_frames)
        self.thread.start()

        print(" Application ready!")

    def _setup_ui(self):
        """Setup the dual-panel UI layout."""
        # Left panel: webcam feed
        self.label_video = QtWidgets.QLabel()
        self.label_video.setFixedSize(800, 600)
        self.label_video.setStyleSheet("background-color: black;")
        self.label_video.setAlignment(QtCore.Qt.AlignCenter)

        # Right panel: 3D mesh
        self.label_mesh = QtWidgets.QLabel()
        self.label_mesh.setFixedSize(800, 600)
        self.label_mesh.setStyleSheet("background-color: #111111;")
        self.label_mesh.setAlignment(QtCore.Qt.AlignCenter)

        # Horizontal layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.label_video)
        layout.addWidget(self.label_mesh)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

    def _setup_3d_scene(self):
        """Setup pyrender 3D scene with camera and lighting."""
        # Offscreen renderer
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=800,
            viewport_height=600
        )

        # Create scene
        self._scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        # Add camera
        self._camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self._camera_node = self._scene.add(
            self._camera,
            pose=self._get_camera_pose()
        )

        # Add lighting
        self._add_lights()

        # Initialize with neutral mesh
        self._faces = get_mano_faces()
        self._mesh_node = None
        self._init_neutral_mesh()

    def _get_camera_pose(self) -> np.ndarray:
        """Get camera pose matrix."""
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 0.6  # Move camera back from origin
        return pose

    def _add_lights(self):
        """Add directional lights to scene."""
        # Front light
        light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self._scene.add(light1, pose=np.array([
            [1, 0, 0, 0.2],
            [0, 1, 0, 0.2],
            [0, 0, 1, 1.0],
            [0, 0, 0, 1],
        ], dtype=np.float32))

        # Back light
        light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self._scene.add(light2, pose=np.array([
            [1, 0, 0, -0.2],
            [0, 1, 0, -0.2],
            [0, 0, 1, -1.0],
            [0, 0, 0, 1],
        ], dtype=np.float32))

    def _init_neutral_mesh(self):
        """Initialize scene with neutral MANO mesh."""
        neutral_verts = np.zeros((778, 3), dtype=np.float32)
        tri = trimesh.Trimesh(
            vertices=neutral_verts,
            faces=self._faces,
            process=False
        )
        mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
        self._mesh_node = self._scene.add(mesh)
        self._render_to_label()

    def closeEvent(self, event):
        """Handle application close."""
        print("🛑 Shutting down...")
        try:
            if self.thread.isRunning():
                self.thread.stop()
                self.thread.wait(1000)
        finally:
            try:
                self._renderer.delete()
            except Exception:
                pass
        return super().closeEvent(event)

    @QtCore.pyqtSlot(object, object)
    def update_frames(
        self,
        frame_bgr: Optional[np.ndarray],
        verts: Optional[np.ndarray]
    ):
        """
        Update both display panels.

        Args:
            frame_bgr: Webcam frame with landmarks drawn
            verts: MANO mesh vertices (778, 3)
        """
        # Update left panel: webcam feed
        if frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(
                rgb.data, w, h, 3 * w,
                QtGui.QImage.Format_RGB888
            )
            self.label_video.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))

        # Update right panel: 3D MANO mesh
        if verts is not None and verts.shape == (778, 3):
            self._update_mesh(verts)

        self._render_to_label()

    def _update_mesh(self, verts: np.ndarray):
        """Update 3D mesh with new vertices."""
        # Remove old mesh node
        try:
            self._scene.remove_node(self._mesh_node)
        except Exception:
            pass

        # Add new mesh with updated vertices
        tri = trimesh.Trimesh(
            vertices=verts,
            faces=self._faces,
            process=False
        )
        mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
        self._mesh_node = self._scene.add(mesh)

    def _render_to_label(self):
        """Render 3D scene to right panel."""
        try:
            color, _ = self._renderer.render(self._scene)
            h, w = color.shape[:2]
            qimg = QtGui.QImage(
                color.data, w, h, 3 * w,
                QtGui.QImage.Format_RGB888
            )
            self.label_mesh.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))
        except Exception:
            pass  # ignore render errors on cleanup


def main():
    """Main entry point."""
    print("=" * 60)
    print(" project asha - mano_v1 (full ik + lbs)")
    print("=" * 60)
    print()
    print(" features:")
    print("   - full linear blend skinning (lbs)")
    print("   - inverse kinematics optimization")
    print("   - finger articulation (bending/curling)")
    print("   - rodrigues rotations + forward kinematics")
    print("   - performance: ~40ms/frame (25fps)")
    print()
    print(" to run baseline (mediapipe only):")
    print("   python src/mediapipe_v0.py")
    print()

    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeApp()
    window.show()

    print("\n show your hand to the camera to start tracking")
    print("   watch fingers articulate in real-time!")
    print("   press ctrl+c or close window to exit\n")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
