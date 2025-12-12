"""
real-time hand pose system - main_v2 (mano ik + emg recording).

features:
  - mediapipe hand landmark detection with world coordinates
  - full mano ik with lbs articulation
  - mindrove 8-channel emg @ 500hz
  - synchronized recording (emg + mano θ)
  - minimal gui: webcam + mesh only (emg runs in background)
  - pure regression: continuous emg → θ mapping (no gesture labels)
  - performance: ~25fps camera, 500hz emg

to run:
  python src/main_v2.py              # with emg hardware
  python src/main_v2.py --no-emg     # camera-only mode for testing
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
from model_utils.pose_fitter import mano_from_landmarks, get_mano_faces
from utils.emg_utils import MindroveInterface, filter_emg
from utils.data_recorder import DataRecorder


class VideoThread(QtCore.QThread):
    """background thread for webcam capture and hand tracking @ 25fps."""

    # signal: (frame_bgr, verts, ik_error, mp_confidence, theta, joints)
    frame_signal = QtCore.pyqtSignal(object, object, float, float, object, object)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()
        self._cap_index = 0
        self._frame_count = 0

    def stop(self):
        """signal the thread to stop."""
        self._running.clear()

    def run(self):
        """main video capture loop."""
        cap = cv2.VideoCapture(self._cap_index)
        if not cap.isOpened():
            print("[error] failed to open camera")
            self.frame_signal.emit(None, None, 0.0, 0.0, None, None)
            return

        print("[success] camera opened successfully @ ~25fps")

        try:
            while self._running.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                self._frame_count += 1

                # detect hand landmarks (prefer world coordinates)
                result = get_landmarks_world(frame)
                if result is None:
                    result = get_landmarks(frame)

                verts = None
                ik_error = 0.0
                mp_confidence = 0.0
                theta = None
                joints = None

                if result is not None:
                    landmarks, mp_confidence = result

                    # draw landmarks on frame with confidence overlay
                    draw_landmarks_on_frame(frame, landmarks, mp_confidence)

                    # generate mano mesh via ik fitting
                    verts, joints, theta, ik_error = mano_from_landmarks(landmarks)

                    if self._frame_count % 100 == 0:  # every ~4 seconds
                        print(f"[debug] camera: frame {self._frame_count}, "
                              f"ik_error={ik_error:.4f}, mp_conf={mp_confidence:.2f}")

                self.frame_signal.emit(frame, verts, ik_error, mp_confidence, theta, joints)

        finally:
            cap.release()
            print(f"[info] camera released (processed {self._frame_count} frames)")


class EMGThread(QtCore.QThread):
    """background thread for emg acquisition @ 500hz with optional imu fusion."""

    # signal: (timestamps, emg_raw, emg_filtered, imu_data, connected)
    emg_signal = QtCore.pyqtSignal(object, object, object, object, bool)

    def __init__(self, parent: Optional[QtCore.QObject] = None, enable_emg: bool = True, enable_imu: bool = False):
        super().__init__(parent)
        self._running = threading.Event()
        self._running.set()
        self._enable_emg = enable_emg
        self._enable_imu = enable_imu
        self._mindrove = MindroveInterface() if enable_emg else None
        self._connected = False
        self._frame_count = 0
        # IMU data placeholder (for v5 Snaptic kit integration)
        self._imu_data = None  # Will be populated when IMU hardware is available

    def stop(self):
        """signal the thread to stop."""
        self._running.clear()

    def run(self):
        """main emg acquisition loop."""
        if not self._enable_emg:
            print("[info] emg disabled (testing mode)")
            # send periodic signals with dummy data so GUI still updates
            while self._running.is_set():
                dummy_timestamps = np.array([0.0])
                dummy_emg = np.zeros((1, 8), dtype=np.float32)
                dummy_imu = np.zeros((1, 18), dtype=np.float32) if self._enable_imu else None  # 2 IMUs × 9-DOF
                self.emg_signal.emit(dummy_timestamps, dummy_emg, dummy_emg, dummy_imu, False)
                threading.Event().wait(0.1)  # 10 Hz dummy updates
            return

        # attempt connection with retry
        print("[info] connecting to mindrove wifi board...")
        print("[info] make sure you are connected to mindrove wifi network")

        if not self._mindrove.connect():
            print("[error] failed to connect to mindrove board")
            print("[error] continuing without emg (camera-only mode)")
            # send periodic signals so GUI knows EMG is unavailable
            while self._running.is_set():
                dummy_timestamps = np.array([0.0])
                dummy_emg = np.zeros((1, 8), dtype=np.float32)
                dummy_imu = np.zeros((1, 18), dtype=np.float32) if self._enable_imu else None
                self.emg_signal.emit(dummy_timestamps, dummy_emg, dummy_emg, dummy_imu, False)
                threading.Event().wait(0.1)
            return

        self._connected = True
        print("[success] emg thread started @ 500hz")

        try:
            while self._running.is_set():
                # fetch data (non-blocking, get available samples)
                timestamps, emg_raw = self._mindrove.get_data(num_samples=50)

                if timestamps.size > 0:
                    # filter emg
                    emg_filtered = filter_emg(emg_raw)
                    
                    # IMU fusion stub (for v5 Snaptic kit)
                    # TODO: Integrate actual IMU hardware when available
                    # Expected format: [N, 18] where 18 = 2 IMUs × 9-DOF (accel + gyro + mag)
                    imu_data = None
                    if self._enable_imu:
                        # Placeholder: will be replaced with actual IMU data
                        # For now, create dummy data matching EMG timestamps
                        imu_data = np.zeros((len(timestamps), 18), dtype=np.float32)
                        # In v5: Replace with actual IMU readings from Snaptic kit
                        # imu_data = self._snaptic_interface.get_imu_data(timestamps)

                    # emit to gui (includes IMU data for fusion)
                    self.emg_signal.emit(timestamps, emg_raw, emg_filtered, imu_data, True)

                    self._frame_count += 1
                    if self._frame_count % 500 == 0:  # every second
                        print(f"[debug] emg: {self._frame_count} batches processed, "
                              f"latest batch: {emg_raw.shape[0]} samples")

                # small sleep to avoid busy-waiting
                threading.Event().wait(0.01)  # 10ms

        finally:
            if self._mindrove:
                self._mindrove.disconnect()
            print("[info] emg thread stopped")


class RealtimeApp(QtWidgets.QWidget):
    """
    main application window with structured data collection protocol.

    layout:
      row 1: [webcam feed 800x600] [3d mano mesh 800x600]
      row 2: [session type selector] [protocol description] [target duration]
      row 3: [record button] [emg status] [timer: elapsed/target] [quality]

    note: emg runs in background without visualization to avoid lag
    """

    def __init__(self, enable_emg: bool = True, enable_pseudo_labeling: bool = False, transformer_model_path: Optional[str] = None):
        super().__init__()
        print("[info] initializing project asha (main_v2 - emg + mano ik)...")

        self.setWindowTitle("project asha - main_v2 (emg + mano ik)")
        self.setGeometry(100, 100, 1600, 720)  # panels + protocol + controls

        # Pseudo-labeling mode: use transformer_v1 to generate GT θ for EMG recordings
        self.enable_pseudo_labeling = enable_pseudo_labeling
        self.transformer_model = None
        if enable_pseudo_labeling and transformer_model_path:
            try:
                import torch
                from pathlib import Path
                # Load transformer model for pseudo-labeling
                # TODO: Implement actual model loading based on transformer_v1 architecture
                print(f"[info] pseudo-labeling enabled: will use transformer model from {transformer_model_path}")
                print("[warning] transformer model loading not yet implemented - using IK θ for now")
                # self.transformer_model = torch.load(transformer_model_path, map_location='cpu')
                # self.transformer_model.eval()
            except Exception as e:
                print(f"[warning] failed to load transformer model: {e}")
                print("[info] falling back to IK-based θ generation")

        # setup gui
        self._setup_ui()

        # setup 3d rendering (for mano mesh)
        self._setup_3d_scene()

        # recorder state
        self.recorder: Optional[DataRecorder] = None
        self.recording = False
        self.record_start_time = 0.0
        self.emg_connected = False

        # start threads
        print("[info] starting camera thread...")
        self.video_thread = VideoThread()
        self.video_thread.frame_signal.connect(self.update_video_and_pose)
        self.video_thread.start()

        print("[info] starting emg thread...")
        self.emg_thread = EMGThread(enable_emg=enable_emg, enable_imu=False)  # IMU disabled for now (v5)
        self.emg_thread.emg_signal.connect(self.update_emg)
        self.emg_thread.start()

        print("[success] application ready!")

    def _setup_ui(self):
        """setup simple layout: [webcam+mesh] side by side, controls below."""
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # row 1: webcam + 3d mesh (side by side)
        panels_layout = QtWidgets.QHBoxLayout()
        panels_layout.setSpacing(0)

        # panel 1: webcam feed
        self.label_video = QtWidgets.QLabel()
        self.label_video.setFixedSize(800, 600)
        self.label_video.setStyleSheet("background-color: black;")
        self.label_video.setAlignment(QtCore.Qt.AlignCenter)
        panels_layout.addWidget(self.label_video)

        # panel 2: 3d mesh
        self.label_mesh = QtWidgets.QLabel()
        self.label_mesh.setFixedSize(800, 600)
        self.label_mesh.setStyleSheet("background-color: #111111;")
        self.label_mesh.setAlignment(QtCore.Qt.AlignCenter)
        panels_layout.addWidget(self.label_mesh)

        main_layout.addLayout(panels_layout)

        # row 2: session protocol selector
        protocol_layout = QtWidgets.QHBoxLayout()
        protocol_layout.setContentsMargins(10, 5, 10, 5)

        protocol_layout.addWidget(QtWidgets.QLabel("Session Type:"))
        self.session_combo = QtWidgets.QComboBox()
        self.session_protocols = {
            "Session 1: Basic Poses": {
                "duration": 180,  # 3 minutes
                "description": "Open, close, point, fist, rest"
            },
            "Session 2: Finger Articulation": {
                "duration": 240,  # 4 minutes
                "description": "Individual fingers, combinations"
            },
            "Session 3: Dynamic Movements": {
                "duration": 240,  # 4 minutes
                "description": "Smooth open/close cycles, continuous motion"
            },
            "Session 4: Functional Gestures": {
                "duration": 180,  # 3 minutes
                "description": "Grasp, pinch, thumbs up, OK sign"
            },
            "Session 5: Varied Practice": {
                "duration": 300,  # 5 minutes
                "description": "Mix of all movements, edge cases"
            }
        }
        self.session_combo.addItems(self.session_protocols.keys())
        self.session_combo.setFixedWidth(250)
        self.session_combo.currentTextChanged.connect(self.update_protocol_display)
        protocol_layout.addWidget(self.session_combo)

        protocol_layout.addSpacing(20)

        # protocol description
        self.label_protocol = QtWidgets.QLabel("Open, close, point, fist, rest")
        self.label_protocol.setStyleSheet("font-style: italic; color: #888;")
        protocol_layout.addWidget(self.label_protocol)

        protocol_layout.addStretch()

        # target duration
        self.label_target = QtWidgets.QLabel("Target: 3:00")
        self.label_target.setStyleSheet("font-size: 14px; font-weight: bold;")
        protocol_layout.addWidget(self.label_target)

        main_layout.addLayout(protocol_layout)

        # row 3: recording controls
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(10, 5, 10, 5)

        # record button
        self.btn_record = QtWidgets.QPushButton("⏺ Start Recording")
        self.btn_record.setFixedSize(150, 40)
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.btn_record.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.btn_record)

        controls_layout.addSpacing(20)

        # emg status indicator
        controls_layout.addWidget(QtWidgets.QLabel("EMG:"))
        self.label_emg_status = QtWidgets.QLabel("Connecting...")
        self.label_emg_status.setStyleSheet("font-family: monospace; color: yellow;")
        controls_layout.addWidget(self.label_emg_status)

        controls_layout.addStretch()

        # session timer
        self.label_timer = QtWidgets.QLabel("00:00 / 3:00")
        self.label_timer.setStyleSheet("font-size: 16px; font-weight: bold;")
        controls_layout.addWidget(self.label_timer)

        controls_layout.addSpacing(20)

        # quality indicators
        controls_layout.addWidget(QtWidgets.QLabel("IK Error:"))
        self.label_ik_error = QtWidgets.QLabel("0.000")
        self.label_ik_error.setStyleSheet("font-family: monospace;")
        controls_layout.addWidget(self.label_ik_error)

        controls_layout.addSpacing(10)

        controls_layout.addWidget(QtWidgets.QLabel("MP Conf:"))
        self.label_mp_conf = QtWidgets.QLabel("0.00")
        self.label_mp_conf.setStyleSheet("font-family: monospace;")
        controls_layout.addWidget(self.label_mp_conf)

        main_layout.addLayout(controls_layout)

        # timer for updating recording duration
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_timer_display)
        self.update_timer.start(100)  # update every 100ms

    def _setup_3d_scene(self):
        """setup pyrender 3d scene with camera and lighting (same as mano_v1)."""
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=800,
            viewport_height=600
        )

        self._scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        # add camera
        self._camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self._camera_node = self._scene.add(
            self._camera,
            pose=self._get_camera_pose()
        )

        # add lighting
        self._add_lights()

        # initialize with neutral mesh
        self._faces = get_mano_faces()
        self._mesh_node = None
        self._init_neutral_mesh()

    def _get_camera_pose(self) -> np.ndarray:
        """get camera pose matrix."""
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 0.6  # move camera back from origin
        return pose

    def _add_lights(self):
        """add directional lights to scene."""
        light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self._scene.add(light1, pose=np.array([
            [1, 0, 0, 0.2],
            [0, 1, 0, 0.2],
            [0, 0, 1, 1.0],
            [0, 0, 0, 1],
        ], dtype=np.float32))

        light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self._scene.add(light2, pose=np.array([
            [1, 0, 0, -0.2],
            [0, 1, 0, -0.2],
            [0, 0, 1, -1.0],
            [0, 0, 0, 1],
        ], dtype=np.float32))

    def _init_neutral_mesh(self):
        """initialize scene with neutral mano mesh (rotated to match camera view)."""
        neutral_verts = np.zeros((778, 3), dtype=np.float32)

        # apply same rotation as _update_mesh for consistency
        rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        rot_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        rot = rot_y @ rot_x
        neutral_verts = neutral_verts @ rot.T

        tri = trimesh.Trimesh(
            vertices=neutral_verts,
            faces=self._faces,
            process=False
        )
        mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
        self._mesh_node = self._scene.add(mesh)
        self._render_to_label()

    def toggle_recording(self):
        """toggle recording on/off."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """start recording session."""
        # warn if emg not connected
        if not self.emg_connected:
            reply = QtWidgets.QMessageBox.question(
                self, "EMG Not Connected",
                "EMG is not connected. Continue with camera-only recording?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                return

        # get subject id from user
        subject_id, ok = QtWidgets.QInputDialog.getText(
            self, "Subject ID", "Enter subject ID:"
        )
        if not ok or not subject_id:
            return

        # get session type and create descriptive filename
        session_type = self.session_combo.currentText()
        session_num = session_type.split(":")[0].replace("Session ", "")  # extract "1" from "Session 1: ..."

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/session{session_num}_subj{subject_id}_{timestamp}.h5"

        # initialize recorder with session metadata
        self.recorder = DataRecorder(output_path, subject_id, session_id=session_type)
        self.recording = True
        self.record_start_time = self.recorder.start_time

        # update ui
        self.btn_record.setText("⏹ Stop Recording")
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        emg_status = "with emg" if self.emg_connected else "camera-only"
        print(f"[success] recording started: {output_path} ({emg_status})")

    def stop_recording(self):
        """stop and save recording."""
        if not self.recorder:
            return

        # save recording
        self.recorder.save()

        # reset state
        self.recorder = None
        self.recording = False

        # update ui
        self.btn_record.setText("⏺ Start Recording")
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        print("[info] recording stopped")

    def update_protocol_display(self):
        """update protocol description and target duration when session type changes."""
        session_type = self.session_combo.currentText()
        protocol = self.session_protocols[session_type]

        # update description
        self.label_protocol.setText(protocol["description"])

        # update target duration
        target_mins = protocol["duration"] // 60
        target_secs = protocol["duration"] % 60
        self.label_target.setText(f"Target: {target_mins}:{target_secs:02d}")

    def update_timer_display(self):
        """update recording timer display with elapsed / target."""
        session_type = self.session_combo.currentText()
        target_duration = self.session_protocols[session_type]["duration"]
        target_mins = target_duration // 60
        target_secs = target_duration % 60

        if self.recording and self.recorder:
            elapsed = self.recorder._get_timestamp()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            # color changes based on progress
            if elapsed < target_duration:
                color = "#f44336"  # red (still recording)
            else:
                color = "#4CAF50"  # green (reached target)

            self.label_timer.setText(f"{minutes:02d}:{seconds:02d} / {target_mins}:{target_secs:02d}")
            self.label_timer.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {color};"
            )
        else:
            self.label_timer.setText(f"00:00 / {target_mins}:{target_secs:02d}")
            self.label_timer.setStyleSheet("font-size: 16px; font-weight: bold;")

    @QtCore.pyqtSlot(object, object, float, float, object, object)
    def update_video_and_pose(
        self,
        frame_bgr: Optional[np.ndarray],
        verts: Optional[np.ndarray],
        ik_error: float,
        mp_confidence: float,
        theta: Optional[np.ndarray],
        joints: Optional[np.ndarray]
    ):
        """update left+middle panels, record pose if active."""
        # update left panel: webcam feed
        if frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(
                rgb.data, w, h, 3 * w,
                QtGui.QImage.Format_RGB888
            )
            self.label_video.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))

        # update right panel: 3d mano mesh
        if verts is not None and verts.shape == (778, 3):
            self._update_mesh(verts)

        self._render_to_label()

        # update quality indicators
        self.label_ik_error.setText(f"{ik_error:.3f}")
        self.label_mp_conf.setText(f"{mp_confidence:.2f}")

        # record pose if active
        if self.recording and self.recorder and theta is not None and joints is not None:
            # Pseudo-labeling mode: use transformer_v1 to generate GT θ instead of IK
            if self.enable_pseudo_labeling and self.transformer_model is not None and frame_bgr is not None:
                try:
                    # Run transformer inference on frame to get better θ estimate
                    # TODO: Implement actual transformer inference
                    # For now, use IK θ (will be replaced when transformer model is loaded)
                    pseudo_theta = theta  # Placeholder
                    # pseudo_theta = self._run_transformer_inference(frame_bgr)
                    self.recorder.record_pose(pseudo_theta, joints, ik_error, mp_confidence)
                except Exception as e:
                    print(f"[warning] pseudo-labeling failed: {e}, using IK θ")
                    self.recorder.record_pose(theta, joints, ik_error, mp_confidence)
            else:
                # Standard mode: use IK θ
                self.recorder.record_pose(theta, joints, ik_error, mp_confidence)

    @QtCore.pyqtSlot(object, object, object, object, bool)
    def update_emg(
        self,
        timestamps: np.ndarray,
        emg_raw: np.ndarray,
        emg_filtered: np.ndarray,
        imu_data: Optional[np.ndarray],
        connected: bool
    ):
        """update emg status and record data if active (no visualization)."""
        self.emg_connected = connected

        # update status indicator
        if connected:
            self.label_emg_status.setText("Connected ✓")
            self.label_emg_status.setStyleSheet("font-family: monospace; color: #4CAF50;")
        else:
            self.label_emg_status.setText("Not Connected")
            self.label_emg_status.setStyleSheet("font-family: monospace; color: #f44336;")

        # record emg if active and connected
        if self.recording and self.recorder and connected:
            self.recorder.record_emg(emg_raw, emg_filtered, imu_data)

    def _update_mesh(self, verts: np.ndarray):
        """update 3d mesh with new vertices (rotated to match camera view)."""
        try:
            self._scene.remove_node(self._mesh_node)
        except Exception:
            pass

        # rotate hand to match camera orientation:
        # - 180° around X-axis (wrist at bottom)
        # - 180° around Y-axis (palm faces forward correctly)
        rot_x = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float32)

        rot_y = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=np.float32)

        # apply both rotations
        rot = rot_y @ rot_x
        verts_rotated = verts @ rot.T

        tri = trimesh.Trimesh(
            vertices=verts_rotated,
            faces=self._faces,
            process=False
        )
        mesh = pyrender.Mesh.from_trimesh(tri, smooth=False)
        self._mesh_node = self._scene.add(mesh)

    def _render_to_label(self):
        """render 3d scene to middle panel."""
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
    
    def _run_transformer_inference(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run transformer_v1 inference on frame to generate GT θ for pseudo-labeling.
        
        This replaces IK-based θ with transformer predictions for better EMG training data.
        
        Args:
            frame_bgr: [H, W, 3] BGR frame
            
        Returns:
            theta: [45] MANO θ parameters
        """
        if self.transformer_model is None:
            # Fallback: return None to use IK θ
            return None
        
        try:
            import torch
            from torchvision import transforms
            
            # Preprocess frame for transformer
            # TODO: Match preprocessing from train_colab.ipynb
            # For now, return None to use IK θ
            # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # preprocess = transforms.Compose([
            #     transforms.ToPILImage(),
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
            # frame_tensor = preprocess(frame_rgb).unsqueeze(0)
            # 
            # with torch.no_grad():
            #     theta_pred = self.transformer_model(frame_tensor)
            # return theta_pred[0].cpu().numpy()
            
            return None  # Placeholder until transformer model loading is implemented
        except Exception as e:
            print(f"[warning] transformer inference failed: {e}")
            return None

    def closeEvent(self, event):
        """handle application close."""
        print("[info] shutting down...")

        # stop recording if active
        if self.recording:
            self.stop_recording()

        # stop threads
        try:
            if self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait(1000)

            if self.emg_thread.isRunning():
                self.emg_thread.stop()
                self.emg_thread.wait(1000)
        finally:
            try:
                self._renderer.delete()
            except Exception:
                pass

        return super().closeEvent(event)


def main():
    """main entry point."""
    import argparse

    print("=" * 60)
    print("project asha - main_v2 (emg + mano ik)")
    print("=" * 60)
    print()
    print("features:")
    print("  - full linear blend skinning (lbs)")
    print("  - inverse kinematics optimization")
    print("  - mindrove 8-channel emg @ 500hz")
    print("  - synchronized recording (emg + mano θ)")
    print("  - minimal gui (webcam + mesh, emg in background)")
    print("  - pure regression (continuous emg → θ)")
    print()
    print("to run without emg (camera-only):")
    print("  python src/main_v2.py --no-emg")
    print()
    print("to enable pseudo-labeling (use transformer_v1 for GT θ):")
    print("  python src/main_v2.py --pseudo-label --transformer-model path/to/model.pth")
    print()

    # parse command line args
    parser = argparse.ArgumentParser(description="Project Asha v2 - EMG + Hand Tracking")
    parser.add_argument("--no-emg", action="store_true",
                       help="Run without EMG (camera-only mode for testing)")
    parser.add_argument("--pseudo-label", action="store_true",
                       help="Enable pseudo-labeling mode (use transformer_v1 for GT θ)")
    parser.add_argument("--transformer-model", type=str, default=None,
                       help="Path to transformer_v1 model checkpoint for pseudo-labeling")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = RealtimeApp(
        enable_emg=not args.no_emg,
        enable_pseudo_labeling=args.pseudo_label,
        transformer_model_path=args.transformer_model
    )
    window.show()

    print("\n[ready] show your hand to the camera to start tracking")
    if not args.no_emg:
        print("[ready] make sure mindrove wifi board is connected")
    print("[ready] press 'start recording' to collect emg → θ training data")
    print("[ready] press ctrl+c or close window to exit\n")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
