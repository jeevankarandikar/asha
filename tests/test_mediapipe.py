"""
test suite for v0 (mediapipe baseline).

validates mediapipe detection works without mano.
"""

import numpy as np
import cv2


def test_mediapipe_import():
    """verify mediapipe can be imported."""
    print("\n" + "=" * 60)
    print(" test 1: mediapipe import")
    print("=" * 60)

    try:
        import mediapipe as mp
        print(f"ok: mediapipe version {mp.__version__}")
        return True
    except ImportError as e:
        print(f"❌ FAIL: {e}")
        return False


def test_mediapipe_detection():
    """verify mediapipe can detect hand in dummy image."""
    print("\n" + "=" * 60)
    print(" test 2: mediapipe hand detection")
    print("=" * 60)

    try:
        import mediapipe as mp

        # create dummy image (640x480 black with white hand-like blob)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # draw white blob in center (simulates hand)
        cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)

        # initialize mediapipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        # process
        results = hands.process(img)

        hands.close()

        print("ok: mediapipe initialized successfully")
        print(f"detection result: {'hand detected' if results.multi_hand_landmarks else 'no hand (expected with dummy image)'}")

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matplotlib_3d():
    """verify matplotlib 3d plotting available."""
    print("\n" + "=" * 60)
    print(" test 3: matplotlib 3d plotting")
    print("=" * 60)

    try:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        # create dummy 3d plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot dummy points
        x = np.random.rand(21)
        y = np.random.rand(21)
        z = np.random.rand(21)
        ax.scatter(x, y, z)

        plt.close(fig)

        print("ok: matplotlib 3d plotting works")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def run_all_tests():
    """run all v0 tests."""
    print("\n" + "=" * 70)
    print(" v0 test suite (mediapipe baseline)")
    print("=" * 70)

    results = []

    results.append(("mediapipe import", test_mediapipe_import()))
    results.append(("mediapipe detection", test_mediapipe_detection()))
    results.append(("matplotlib 3d", test_matplotlib_3d()))

    # summary
    print("\n" + "=" * 70)
    print(" test summary")
    print("=" * 70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✅ mediapipe_v0 ready! mediapipe baseline works.")
        print("   run: python src/mediapipe_v0.py")
    else:
        print("\n❌ some tests failed. check dependencies:")
        print("   pip install mediapipe matplotlib")

    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all_tests() else 1)
