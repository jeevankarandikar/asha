"""
test suite for main_v2 (emg + mano ik system).

validates:
  - emg signal processing (filtering)
  - data recorder (hdf5 format)
  - emg utils (interface simulation)
  - integration tests
"""

import sys
import numpy as np
import h5py
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.emg_utils import filter_emg
from utils.data_recorder import DataRecorder


def test_emg_filtering():
    """test emg filtering pipeline (notch + bandpass)."""
    print("\n[test] emg filtering pipeline...")

    # generate synthetic emg signal: sum of sine waves + noise
    fs = 500.0  # 500hz sampling
    duration = 1.0  # 1 second
    num_samples = int(fs * duration)
    t = np.arange(num_samples) / fs

    # create 8 channels
    emg_data = np.zeros((num_samples, 8))

    for ch in range(8):
        # emg-like signal: multiple frequency components
        signal = (
            np.sin(2 * np.pi * 50 * t) * 0.5 +  # 50hz (muscle activity)
            np.sin(2 * np.pi * 60 * t) * 1.0 +  # 60hz (power line noise)
            np.sin(2 * np.pi * 120 * t) * 0.3 + # 120hz harmonic
            np.random.randn(num_samples) * 0.1  # noise
        )
        emg_data[:, ch] = signal

    # apply filtering
    filtered = filter_emg(emg_data, fs)

    # validation checks
    assert filtered.shape == emg_data.shape, "shape mismatch"
    assert not np.any(np.isnan(filtered)), "nan values in output"
    assert not np.any(np.isinf(filtered)), "inf values in output"

    # check that 60hz component is attenuated
    # simple check: variance should be reduced
    variance_raw = np.var(emg_data, axis=0).mean()
    variance_filtered = np.var(filtered, axis=0).mean()

    print(f"  variance raw: {variance_raw:.4f}")
    print(f"  variance filtered: {variance_filtered:.4f}")
    print(f"  reduction: {(1 - variance_filtered/variance_raw)*100:.1f}%")

    # notch filter should reduce variance (removing 60hz component)
    assert variance_filtered < variance_raw, "filtering should reduce variance"

    print("[pass] emg filtering pipeline")


def test_data_recorder():
    """test data recorder (hdf5 writing and format)."""
    print("\n[test] data recorder...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_session.h5"

        # create recorder
        recorder = DataRecorder(str(output_path), subject_id="test_subject")

        # simulate recording session
        # emg data @ 500hz
        for i in range(10):
            emg_raw = np.random.randn(50, 8).astype(np.float32)
            emg_filtered = np.random.randn(50, 8).astype(np.float32)
            recorder.record_emg(emg_raw, emg_filtered)

        # pose data @ 25fps
        for i in range(5):
            theta = np.random.randn(45).astype(np.float32)
            joints = np.random.randn(21, 3).astype(np.float32)
            ik_error = np.random.rand()
            mp_confidence = np.random.rand()
            recorder.record_pose(theta, joints, ik_error, mp_confidence)

        # gesture labels
        recorder.add_gesture_label("fist", 0.0, 1.0)
        recorder.add_gesture_label("open", 1.0, 2.0)

        # save to file
        recorder.save()

        # verify file exists
        assert output_path.exists(), "hdf5 file not created"

        # verify file contents
        with h5py.File(output_path, 'r') as f:
            # check groups exist
            assert 'emg' in f, "missing emg group"
            assert 'pose' in f, "missing pose group"
            assert 'gestures' in f, "missing gestures group"

            # check emg datasets
            assert 'raw' in f['emg'], "missing emg/raw"
            assert 'filtered' in f['emg'], "missing emg/filtered"
            assert 'timestamps' in f['emg'], "missing emg/timestamps"

            emg_raw = f['emg/raw'][:]
            assert emg_raw.shape == (500, 8), f"unexpected emg shape: {emg_raw.shape}"

            # check pose datasets
            assert 'mano_theta' in f['pose'], "missing pose/mano_theta"
            assert 'joints_3d' in f['pose'], "missing pose/joints_3d"
            assert 'timestamps' in f['pose'], "missing pose/timestamps"

            theta = f['pose/mano_theta'][:]
            assert theta.shape == (5, 45), f"unexpected theta shape: {theta.shape}"

            joints = f['pose/joints_3d'][:]
            assert joints.shape == (5, 21, 3), f"unexpected joints shape: {joints.shape}"

            # check gestures
            labels = f['gestures/labels'][:]
            assert len(labels) == 2, f"unexpected gesture count: {len(labels)}"
            assert labels[0].decode('utf-8') == 'fist', "incorrect gesture label"

            # check metadata
            assert 'subject_id' in f.attrs, "missing subject_id metadata"
            assert f.attrs['subject_id'] == 'test_subject', "incorrect subject_id"

        print(f"  file size: {output_path.stat().st_size / 1024:.1f} KB")
        print("[pass] data recorder")


def test_data_recorder_empty():
    """test data recorder with no data (edge case)."""
    print("\n[test] data recorder (empty)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_empty.h5"

        recorder = DataRecorder(str(output_path), subject_id="test_empty")
        recorder.save()  # save without recording any data

        assert output_path.exists(), "hdf5 file not created"

        with h5py.File(output_path, 'r') as f:
            emg_raw = f['emg/raw'][:]
            assert emg_raw.shape == (0, 8), "unexpected shape for empty emg"

            theta = f['pose/mano_theta'][:]
            assert theta.shape == (0, 45), "unexpected shape for empty theta"

        print("[pass] data recorder (empty)")


def test_data_recorder_stats():
    """test data recorder statistics calculation."""
    print("\n[test] data recorder stats...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_stats.h5"

        recorder = DataRecorder(str(output_path), subject_id="test_stats")

        # record some data
        for i in range(5):
            emg_raw = np.random.randn(100, 8).astype(np.float32)
            emg_filtered = np.random.randn(100, 8).astype(np.float32)
            recorder.record_emg(emg_raw, emg_filtered)

        for i in range(3):
            theta = np.random.randn(45).astype(np.float32)
            joints = np.random.randn(21, 3).astype(np.float32)
            recorder.record_pose(theta, joints, 0.1, 0.9)

        # get stats
        stats = recorder.get_stats()

        assert stats['emg_samples'] == 500, f"unexpected emg count: {stats['emg_samples']}"
        assert stats['pose_samples'] == 3, f"unexpected pose count: {stats['pose_samples']}"
        assert stats['duration_sec'] > 0, "duration should be positive"
        assert stats['emg_rate_hz'] > 0, "emg rate should be positive"

        print(f"  emg samples: {stats['emg_samples']}")
        print(f"  pose samples: {stats['pose_samples']}")
        print(f"  duration: {stats['duration_sec']:.3f}s")
        print(f"  emg rate: {stats['emg_rate_hz']:.1f}hz")
        print(f"  pose rate: {stats['pose_rate_hz']:.1f}hz")
        print("[pass] data recorder stats")


def test_data_recorder_with_imu():
    """test data recorder with IMU data."""
    print("\n[test] data recorder with IMU...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_imu.h5"

        recorder = DataRecorder(str(output_path), subject_id="test_imu")

        # record emg + imu data @ 500hz
        for i in range(10):
            emg_raw = np.random.randn(50, 8).astype(np.float32)
            emg_filtered = np.random.randn(50, 8).astype(np.float32)
            imu_data = np.random.randn(50, 18).astype(np.float32)  # 2 sensors × 9-DOF
            recorder.record_emg(emg_raw, emg_filtered, imu_data)

        # record pose data @ 25fps
        for i in range(5):
            theta = np.random.randn(45).astype(np.float32)
            joints = np.random.randn(21, 3).astype(np.float32)
            recorder.record_pose(theta, joints, 0.1, 0.9)

        # save to file
        recorder.save()

        # verify file contents
        with h5py.File(output_path, 'r') as f:
            # check imu group exists
            assert 'imu' in f, "missing imu group"
            assert 'data' in f['imu'], "missing imu/data"
            assert 'timestamps' in f['imu'], "missing imu/timestamps"

            # verify shapes
            imu_data = f['imu/data'][:]
            assert imu_data.shape == (500, 18), f"unexpected imu shape: {imu_data.shape}"

            imu_timestamps = f['imu/timestamps'][:]
            assert len(imu_timestamps) == 500, f"unexpected timestamp count: {len(imu_timestamps)}"

        # check stats
        stats = recorder.get_stats()
        assert stats['imu_samples'] == 500, f"unexpected imu count: {stats['imu_samples']}"
        assert stats['imu_rate_hz'] > 0, "imu rate should be positive"

        print(f"  imu samples: {stats['imu_samples']}")
        print(f"  imu rate: {stats['imu_rate_hz']:.1f}hz")
        print("[pass] data recorder with IMU")


def run_all_tests():
    """run all tests."""
    print("=" * 60)
    print("main_v2 test suite")
    print("=" * 60)

    tests = [
        test_emg_filtering,
        test_data_recorder,
        test_data_recorder_empty,
        test_data_recorder_stats,
        test_data_recorder_with_imu,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"[fail] {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[error] {test_fn.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
