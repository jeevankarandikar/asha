"""
tests mano model loading and basic functionality. run after setup to verify.
"""

import sys
from pathlib import Path

# add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent))

import torch
from mano_utils import get_torch_device, mano_from_landmarks, get_mano_faces
import numpy as np


def test_device():
    """Test PyTorch device detection."""
    print("\n" + "=" * 60)
    print(" Testing PyTorch Device")
    print("=" * 60)

    device = get_torch_device()
    print(f"ok: Device: {device}")

    if device.type == "mps":
        print("ok - Apple Silicon MPS acceleration available!")
    else:
        print("warning:  Running on CPU (MPS not available)")

    # Additional checks
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")


def test_mano_loading():
    """Test MANO model loading."""
    print("\n" + "=" * 60)
    print(" Testing MANO Model Loading")
    print("=" * 60)

    try:
        # Test getting face indices
        print("Loading MANO faces...")
        faces = get_mano_faces()
        print(f"ok: Faces shape: {faces.shape}")
        print(f"ok: Number of triangles: {len(faces)}")

        # Test mesh generation
        print("\nGenerating neutral MANO mesh...")
        dummy_landmarks = np.zeros((21, 3), dtype=np.float32)
        verts, joints = mano_from_landmarks(dummy_landmarks)

        print(f"ok: Vertices shape: {verts.shape}")
        print(f"ok: Joints shape: {joints.shape}")
        print(f"ok: Vertices dtype: {verts.dtype}")
        print(f"ok: Joints dtype: {joints.dtype}")

        # Verify shapes
        assert verts.shape == (778, 3), f"Expected (778, 3), got {verts.shape}"
        assert joints.shape == (21, 3), f"Expected (21, 3), got {joints.shape}"

        print("\nok - MANO model loaded successfully!")
        return True

    except FileNotFoundError as e:
        print(f"\nerror: MANO loading failed: {e}")
        print("\nhint: Make sure you've run the conversion script:")
        print("   1. source mano_convert_env/bin/activate")
        print("   2. python realtime_asha/src/convert_mano_to_numpy.py")
        return False
    except Exception as e:
        print(f"\nerror: MANO loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mano_inference():
    """Test MANO inference performance."""
    print("\n" + "=" * 60)
    print(" Testing MANO Inference Performance")
    print("=" * 60)

    import time

    dummy_landmarks = np.zeros((21, 3), dtype=np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        mano_from_landmarks(dummy_landmarks)

    # Benchmark
    print("Running benchmark (100 iterations)...")
    start = time.time()
    for _ in range(100):
        verts, joints = mano_from_landmarks(dummy_landmarks)
    elapsed = time.time() - start

    avg_time = elapsed / 100 * 1000  # ms
    fps = 100 / elapsed

    print(f"ok: Average time per inference: {avg_time:.2f} ms")
    print(f"ok: Potential FPS: {fps:.1f}")

    if fps >= 30:
        print("ok - Performance suitable for real-time (≥30 FPS)")
    else:
        print("warning:  Performance may be below real-time (<30 FPS)")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" Project Asha — MANO Test Suite")
    print("=" * 70)

    # Test 1: Device
    test_device()

    # Test 2: MANO loading
    success = test_mano_loading()
    if not success:
        print("\nerror: MANO loading failed. Please check:")
        print("   1. MANO models are in realtime_asha/mano_models/")
        print("   2. Models have been converted to *_CONVERTED.pkl format")
        print("   3. Run conversion script if needed")
        return 1

    # Test 3: Performance
    test_mano_inference()

    print("\n" + "=" * 70)
    print("ok - All tests passed! Ready to run the application.")
    print("=" * 70)
    print("\nRun the application with:")
    print("  python realtime_asha/src/realtime_mano.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
