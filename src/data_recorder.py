"""
synchronized recording of emg signals and mano pose parameters to hdf5.

handles:
  - dual-rate data (emg @ 500hz, pose @ 25fps)
  - timestamp synchronization (time.perf_counter)
  - hdf5 dataset structure with metadata
  - gesture annotation
"""

import numpy as np
import h5py
import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path


class DataRecorder:
    """
    records synchronized emg + pose data to hdf5 file.

    data format:
      /emg/raw              [N, 8] float32       raw voltages
      /emg/filtered         [N, 8] float32       after notch+bandpass
      /emg/timestamps       [N] float64          seconds since session start

      /pose/mano_theta      [M, 45] float32      joint angles (ik output)
      /pose/joints_3d       [M, 21, 3] float32   mediapipe landmarks
      /pose/timestamps      [M] float64          seconds since session start
      /pose/ik_error        [M] float32          convergence quality
      /pose/mp_confidence   [M] float32          tracking quality

      /gestures/labels      [K] string           gesture names
      /gestures/start_times [K] float64          gesture start (sec)
      /gestures/end_times   [K] float64          gesture end (sec)

      /metadata/*           attributes           session metadata
    """

    def __init__(self, output_path: str, subject_id: str, session_id: Optional[str] = None):
        """
        initialize recorder.

        args:
          output_path: path to hdf5 file (e.g., "data/session_001.h5")
          subject_id: subject identifier
          session_id: optional session identifier (auto-generated if None)
        """
        self.output_path = Path(output_path)
        self.subject_id = subject_id
        self.session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")

        # ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # reference time for all timestamps (seconds)
        self.start_time = time.perf_counter()

        # emg data buffers
        self.emg_raw_buffer: List[np.ndarray] = []
        self.emg_filtered_buffer: List[np.ndarray] = []
        self.emg_timestamps_buffer: List[float] = []

        # pose data buffers
        self.theta_buffer: List[np.ndarray] = []
        self.joints_buffer: List[np.ndarray] = []
        self.pose_timestamps_buffer: List[float] = []
        self.ik_error_buffer: List[float] = []
        self.mp_confidence_buffer: List[float] = []

        # gesture annotation buffers
        self.gesture_labels: List[str] = []
        self.gesture_start_times: List[float] = []
        self.gesture_end_times: List[float] = []

        print(f"[info] data recorder initialized")
        print(f"  output: {self.output_path}")
        print(f"  subject: {self.subject_id}")
        print(f"  session: {self.session_id}")

    def _get_timestamp(self) -> float:
        """get current timestamp (seconds since session start)."""
        return time.perf_counter() - self.start_time

    def record_emg(self, emg_raw: np.ndarray, emg_filtered: np.ndarray):
        """
        buffer emg sample (called @ 500hz from EMGThread).

        args:
          emg_raw: [num_samples, 8] raw emg voltages
          emg_filtered: [num_samples, 8] filtered emg
        """
        timestamp = self._get_timestamp()

        # append to buffers
        self.emg_raw_buffer.append(emg_raw)
        self.emg_filtered_buffer.append(emg_filtered)

        # create per-sample timestamps
        num_samples = emg_raw.shape[0]
        sample_timestamps = timestamp + np.arange(num_samples) / 500.0  # assuming 500hz
        self.emg_timestamps_buffer.extend(sample_timestamps)

    def record_pose(
        self,
        theta: np.ndarray,
        joints: np.ndarray,
        ik_error: float,
        mp_confidence: float
    ):
        """
        buffer pose sample (called @ 25fps from VideoThread).

        args:
          theta: [45] mano pose parameters (axis-angle)
          joints: [21, 3] mediapipe joint positions
          ik_error: ik convergence error
          mp_confidence: mediapipe tracking confidence
        """
        timestamp = self._get_timestamp()

        # append to buffers
        self.theta_buffer.append(theta)
        self.joints_buffer.append(joints)
        self.pose_timestamps_buffer.append(timestamp)
        self.ik_error_buffer.append(ik_error)
        self.mp_confidence_buffer.append(mp_confidence)

    def add_gesture_label(self, label: str, start_time: Optional[float] = None,
                         end_time: Optional[float] = None):
        """
        annotate time range with gesture label.

        args:
          label: gesture name (e.g., "fist", "open", "pinch")
          start_time: start time in seconds (None = now)
          end_time: end time in seconds (None = now)
        """
        if start_time is None:
            start_time = self._get_timestamp()
        if end_time is None:
            end_time = self._get_timestamp()

        self.gesture_labels.append(label)
        self.gesture_start_times.append(start_time)
        self.gesture_end_times.append(end_time)

        print(f"[info] gesture labeled: {label} @ {start_time:.2f}s - {end_time:.2f}s")

    def get_stats(self) -> dict:
        """get current recording statistics."""
        emg_samples = sum(arr.shape[0] for arr in self.emg_raw_buffer)
        pose_samples = len(self.theta_buffer)
        duration = self._get_timestamp()

        return {
            'duration_sec': duration,
            'emg_samples': emg_samples,
            'pose_samples': pose_samples,
            'gestures': len(self.gesture_labels),
            'emg_rate_hz': emg_samples / duration if duration > 0 else 0,
            'pose_rate_hz': pose_samples / duration if duration > 0 else 0,
        }

    def save(self):
        """write all buffered data to hdf5 file."""
        print(f"[info] saving recording to {self.output_path}...")

        try:
            # concatenate buffers
            emg_raw = np.vstack(self.emg_raw_buffer) if self.emg_raw_buffer else np.array([]).reshape(0, 8)
            emg_filtered = np.vstack(self.emg_filtered_buffer) if self.emg_filtered_buffer else np.array([]).reshape(0, 8)
            emg_timestamps = np.array(self.emg_timestamps_buffer)

            theta = np.array(self.theta_buffer) if self.theta_buffer else np.array([]).reshape(0, 45)
            joints = np.array(self.joints_buffer) if self.joints_buffer else np.array([]).reshape(0, 21, 3)
            pose_timestamps = np.array(self.pose_timestamps_buffer)
            ik_error = np.array(self.ik_error_buffer)
            mp_confidence = np.array(self.mp_confidence_buffer)

            # write to hdf5
            with h5py.File(self.output_path, 'w') as f:
                # emg group
                emg_grp = f.create_group('emg')
                emg_grp.create_dataset('raw', data=emg_raw, compression='gzip')
                emg_grp.create_dataset('filtered', data=emg_filtered, compression='gzip')
                emg_grp.create_dataset('timestamps', data=emg_timestamps, compression='gzip')

                # pose group
                pose_grp = f.create_group('pose')
                pose_grp.create_dataset('mano_theta', data=theta, compression='gzip')
                pose_grp.create_dataset('joints_3d', data=joints, compression='gzip')
                pose_grp.create_dataset('timestamps', data=pose_timestamps, compression='gzip')
                pose_grp.create_dataset('ik_error', data=ik_error, compression='gzip')
                pose_grp.create_dataset('mp_confidence', data=mp_confidence, compression='gzip')

                # gestures group
                gestures_grp = f.create_group('gestures')
                # use variable-length string dtype for labels
                str_dtype = h5py.string_dtype(encoding='utf-8')
                gestures_grp.create_dataset('labels',
                    data=np.array(self.gesture_labels, dtype=object),
                    dtype=str_dtype
                )
                gestures_grp.create_dataset('start_times',
                    data=np.array(self.gesture_start_times)
                )
                gestures_grp.create_dataset('end_times',
                    data=np.array(self.gesture_end_times)
                )

                # metadata attributes
                f.attrs['subject_id'] = self.subject_id
                f.attrs['session_id'] = self.session_id
                f.attrs['date'] = datetime.now().strftime("%Y-%m-%d")
                f.attrs['time'] = datetime.now().strftime("%H:%M:%S")
                f.attrs['hand'] = 'right'  # TODO: make configurable
                f.attrs['emg_channels'] = 8
                f.attrs['emg_sampling_rate'] = 500
                f.attrs['camera_fps'] = 25
                f.attrs['duration_sec'] = self._get_timestamp()

            stats = self.get_stats()
            print(f"[success] recording saved!")
            print(f"  duration: {stats['duration_sec']:.1f}s")
            print(f"  emg: {stats['emg_samples']} samples @ {stats['emg_rate_hz']:.1f}hz")
            print(f"  pose: {stats['pose_samples']} samples @ {stats['pose_rate_hz']:.1f}hz")
            print(f"  gestures: {stats['gestures']}")
            print(f"  file size: {self.output_path.stat().st_size / 1024:.1f} KB")

        except Exception as e:
            print(f"[error] failed to save recording: {e}")
            raise
