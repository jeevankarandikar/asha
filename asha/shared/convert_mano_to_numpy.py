"""
converts legacy mano .pkl files (with chumpy objects) to pure numpy.

run in python 3.10 environment with chumpy installed:
  source mano_convert_env/bin/activate
  python src/convert_mano_to_numpy.py

this script:
  - checks if numpy versions already exist (skips if so)
  - converts chumpy objects to numpy arrays
  - saves as MANO_{side}_numpy.pkl (keeps originals intact)
"""

import pickle
import numpy as np
import os
from pathlib import Path


def recursive_to_numpy(obj):
    """recursively convert chumpy objects to numpy arrays."""
    try:
        import chumpy
        has_chumpy = True
    except ImportError:
        has_chumpy = False
        print("  warning: chumpy not installed. conversion may fail.")

    if isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [recursive_to_numpy(v) for v in obj]
        return type(obj)(converted)
    elif has_chumpy and 'chumpy' in str(type(obj)):
        return np.array(obj)
    elif hasattr(obj, 'r'):
        try:
            return np.asarray(obj.r)
        except Exception:
            pass
    elif hasattr(obj, '__array__'):
        try:
            return np.asarray(obj)
        except Exception:
            pass

    return obj


def convert_file(input_path):
    """convert single mano pkl file from chumpy to numpy format."""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f" ✗ file not found: {input_path}")
        return False

    # determine output path (MANO_RIGHT.pkl → MANO_RIGHT_numpy.pkl)
    output_path = input_path.with_name(
        input_path.stem + "_numpy" + input_path.suffix
    )

    # check if already converted
    if output_path.exists():
        print(f"\n ✓ {output_path.name} already exists (skipping)")
        return True

    print(f"\n converting {input_path.name}...")

    # load original pickle
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print(f"   ✓ loaded original file")
    except Exception as e:
        print(f"   ✗ failed to load: {e}")
        return False

    # convert to numpy
    try:
        data_np = recursive_to_numpy(data)
        print(f"   ✓ converted chumpy → numpy")
    except Exception as e:
        print(f"   ✗ conversion failed: {e}")
        return False

    # save converted pickle
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(data_np, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✓ saved: {output_path.name}")
    except Exception as e:
        print(f"   ✗ failed to save: {e}")
        return False

    print(f" ✓ successfully converted {input_path.name} → {output_path.name}")
    return True


def main():
    """convert mano model files."""
    # determine models directory
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir.parent / "models" / "mano"

    if not models_dir.exists():
        print(f"✗ models/mano directory not found: {models_dir}")
        print("  create it and place MANO_RIGHT.pkl and MANO_LEFT.pkl inside")
        print(f"  download from: https://mano.is.tue.mpg.de/download.php")
        return

    print("=" * 60)
    print("mano model converter (chumpy → numpy)")
    print("=" * 60)
    print(f"models/mano directory: {models_dir}")

    # convert both hands
    files_to_convert = [
        models_dir / "MANO_RIGHT.pkl",
        models_dir / "MANO_LEFT.pkl"
    ]

    success_count = 0
    for filepath in files_to_convert:
        if convert_file(filepath):
            success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(files_to_convert):
        print(f"✓ all {success_count} files ready!")
        print("\nnext steps:")
        print("  1. deactivate this environment: deactivate")
        print("  2. activate runtime environment: source asha_env/bin/activate")
        print("  3. run the application: python src/mano_v1.py")
    else:
        print(f"✗ converted {success_count}/{len(files_to_convert)} files")
        print("  check errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
