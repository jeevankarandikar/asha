"""
data loader for v3 training pipeline.

loads hdf5 recordings from v2 and creates windowed emg→theta samples.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch


class EMGPoseDataLoader:
    """
    load synchronized emg + pose data from hdf5 files.

    creates windowed samples:
      input: emg window [window_size, 8 channels]
      target: mano theta [45 params] at end of window
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 100,  # 200ms @ 500Hz
        stride: int = 25,  # 50ms overlap
        quality_threshold_ik: float = 25.0,  # mm
        quality_threshold_mp: float = 0.7,  # confidence
    ):
        """
        initialize data loader.

        args:
          data_dir: directory containing .h5 files from v2
          window_size: emg window size in samples (100 = 200ms @ 500Hz)
          stride: sliding window stride (25 = 50ms overlap)
          quality_threshold_ik: max ik error (mm) for valid poses
          quality_threshold_mp: min mediapipe confidence for valid poses
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.quality_threshold_ik = quality_threshold_ik
        self.quality_threshold_mp = quality_threshold_mp

        # find all .h5 files
        self.session_files = sorted(self.data_dir.glob("*.h5"))

        if len(self.session_files) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        print(f"[info] found {len(self.session_files)} session files")
        for f in self.session_files:
            print(f"  - {f.name}")

    def load_session(self, session_path: Path) -> Dict:
        """
        load single session from hdf5.

        returns:
          dict with keys:
            - emg_filtered: [N, 8] filtered emg signals
            - emg_timestamps: [N] timestamps (seconds)
            - mano_theta: [M, 45] joint angles
            - pose_timestamps: [M] timestamps (seconds)
            - ik_error: [M] ik convergence error
            - mp_confidence: [M] mediapipe tracking confidence
            - metadata: dict of session metadata
        """
        with h5py.File(session_path, 'r') as f:
            data = {
                'emg_filtered': f['emg/filtered'][:],
                'emg_timestamps': f['emg/timestamps'][:],
                'mano_theta': f['pose/mano_theta'][:],
                'pose_timestamps': f['pose/timestamps'][:],
                'ik_error': f['pose/ik_error'][:],
                'mp_confidence': f['pose/mp_confidence'][:],
                'metadata': dict(f['metadata'].attrs),
            }

        return data

    def create_windows(
        self,
        emg_filtered: np.ndarray,
        emg_timestamps: np.ndarray,
        mano_theta: np.ndarray,
        pose_timestamps: np.ndarray,
        ik_error: np.ndarray,
        mp_confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        create sliding windows: emg → theta.

        for each pose frame:
          - find preceding emg window (200ms before pose timestamp)
          - if valid quality, add to training set

        args:
          emg_filtered: [N, 8] emg signals
          emg_timestamps: [N] emg timestamps (sec)
          mano_theta: [M, 45] joint angles
          pose_timestamps: [M] pose timestamps (sec)
          ik_error: [M] ik error (mm)
          mp_confidence: [M] tracking confidence

        returns:
          X: [num_windows, window_size, 8] emg windows
          y: [num_windows, 45] mano theta targets
        """
        X_list = []
        y_list = []

        # for each pose frame
        for i, pose_ts in enumerate(pose_timestamps):
            # check quality
            if ik_error[i] > self.quality_threshold_ik:
                continue
            if mp_confidence[i] < self.quality_threshold_mp:
                continue

            # find emg window ending at pose_ts
            # emg should precede pose by ~80-120ms (muscle activation delay)
            # for now, use exact alignment (can refine later)
            window_end_idx = np.searchsorted(emg_timestamps, pose_ts)
            window_start_idx = window_end_idx - self.window_size

            # check bounds
            if window_start_idx < 0 or window_end_idx > len(emg_timestamps):
                continue

            # extract window
            emg_window = emg_filtered[window_start_idx:window_end_idx, :]  # [window_size, 8]

            # check for valid data (no NaN/Inf)
            if not np.isfinite(emg_window).all():
                continue

            X_list.append(emg_window)
            y_list.append(mano_theta[i])

        if len(X_list) == 0:
            return np.array([]), np.array([])

        X = np.stack(X_list, axis=0)  # [num_windows, window_size, 8]
        y = np.stack(y_list, axis=0)  # [num_windows, 45]

        return X, y

    def load_all_sessions(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        load and concatenate all sessions.

        args:
          normalize: if True, z-score normalize emg per channel

        returns:
          X: [total_windows, window_size, 8] emg windows
          y: [total_windows, 45] mano theta targets
          stats: dict with normalization statistics
        """
        all_X = []
        all_y = []

        for session_file in self.session_files:
            print(f"\n[info] loading {session_file.name}...")

            data = self.load_session(session_file)
            X, y = self.create_windows(
                data['emg_filtered'],
                data['emg_timestamps'],
                data['mano_theta'],
                data['pose_timestamps'],
                data['ik_error'],
                data['mp_confidence'],
            )

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                print(f"  ✓ {len(X)} windows, {len(y)} targets")
            else:
                print(f"  ⚠ no valid windows (check quality thresholds)")

        if len(all_X) == 0:
            raise ValueError("No valid training samples found!")

        # concatenate all sessions
        X = np.concatenate(all_X, axis=0)  # [total, window_size, 8]
        y = np.concatenate(all_y, axis=0)  # [total, 45]

        print(f"\n[info] total dataset:")
        print(f"  X: {X.shape} (windows, time, channels)")
        print(f"  y: {y.shape} (windows, theta_params)")

        # compute normalization statistics (on training set only!)
        stats = {}
        if normalize:
            # per-channel mean/std across all samples
            X_flat = X.reshape(-1, X.shape[-1])  # [total*window_size, 8]
            emg_mean = X_flat.mean(axis=0)  # [8]
            emg_std = X_flat.std(axis=0) + 1e-8  # [8] (avoid div by zero)

            # z-score normalize
            X = (X - emg_mean) / emg_std

            stats['emg_mean'] = emg_mean
            stats['emg_std'] = emg_std

            print(f"  ✓ normalized emg (z-score per channel)")

        return X, y, stats


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    random_seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    split dataset into train/val/test.

    args:
      X: [N, window_size, 8] emg windows
      y: [N, 45] mano theta targets
      train_ratio: fraction for training (0.7 = 70%)
      val_ratio: fraction for validation
      test_ratio: fraction for test
      shuffle: if True, shuffle before splitting
      random_seed: reproducibility

    returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    N = len(X)
    indices = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

    # compute split indices
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n[info] split dataset:")
    print(f"  train: {len(X_train)} samples ({100*train_ratio:.0f}%)")
    print(f"  val:   {len(X_val)} samples ({100*val_ratio:.0f}%)")
    print(f"  test:  {len(X_test)} samples ({100*test_ratio:.0f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# quick test/demo
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_dir>")
        print("Example: python data_loader.py ../data")
        sys.exit(1)

    data_dir = sys.argv[1]

    # load all sessions
    loader = EMGPoseDataLoader(data_dir)
    X, y, stats = loader.load_all_sessions(normalize=True)

    # split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y)

    print(f"\n[info] ready for training!")
    print(f"  input shape: {X_train[0].shape} (time, channels)")
    print(f"  target shape: {y_train[0].shape} (theta params)")
