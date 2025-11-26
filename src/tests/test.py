"""
master test runner for all versions.

runs test suites for mediapipe_v0 and mano_v1.
reports overall status across all versions.
"""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_version_test(version: str) -> bool:
    """run test for specific version."""
    print("\n" + "=" * 70)
    print(f" testing {version}")
    print("=" * 70)

    if version == "mediapipe_v0":
        from test_mediapipe_v0 import run_all_tests
        return run_all_tests()
    elif version == "mano_v1":
        from test_mano_v1 import run_all_tests
        return run_all_tests()
    else:
        print(f"❌ unknown version: {version}")
        return False


def main():
    """run all version tests."""
    print("\n" + "=" * 70)
    print(" project asha - complete test suite")
    print("=" * 70)
    print()
    print("testing progression: mediapipe_v0 → mano_v1")
    print()

    results = {}

    # test each version
    for version in ["mediapipe_v0", "mano_v1"]:
        try:
            results[version] = run_version_test(version)
        except Exception as e:
            print(f"\n❌ {version} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[version] = False

    # final summary
    print("\n" + "=" * 70)
    print(" final summary")
    print("=" * 70)

    for version in ["mediapipe_v0", "mano_v1"]:
        status = "✅ PASS" if results.get(version) else "❌ FAIL"
        desc = {
            "mediapipe_v0": "mediapipe baseline (scatter plot)",
            "mano_v1": "full ik + lbs (articulation)"
        }
        print(f"  {status}: {version} - {desc[version]}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 all versions working!")
        print("\nrun:")
        print("  python src/mediapipe_v0.py  # mediapipe only")
        print("  python src/mano_v1.py       # full articulation")
    else:
        failed = [v for v, passed in results.items() if not passed]
        print(f"\n❌ some versions failed: {', '.join(failed)}")
        print("\ncheck individual test output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
