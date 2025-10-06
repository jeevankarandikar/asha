"""
converts legacy mano pkl files (with chumpy objects) to pure numpy. run in python 3.10 env.
"""

import pickle
import numpy as np
import os
from pathlib import Path


def recursive_to_numpy(obj):
    """recursively convert chumpy objects to numpy arrays."""
    # Try to import chumpy if available
    try:
        import chumpy
        has_chumpy = True
    except ImportError:
        has_chumpy = False
        print("  Warning: chumpy not installed. This may fail on chumpy objects.")

    if isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [recursive_to_numpy(v) for v in obj]
        return type(obj)(converted)
    elif has_chumpy and 'chumpy' in str(type(obj)):
        # Convert chumpy object to numpy
        return np.array(obj)
    elif hasattr(obj, 'r'):
        # Alternative chumpy detection
        try:
            return np.asarray(obj.r)
        except Exception:
            pass
    elif hasattr(obj, '__array__'):
        # Any array-like object
        try:
            return np.asarray(obj)
        except Exception:
            pass

    return obj


def convert_file(input_path, output_suffix="_CONVERTED"):
    """convert single mano pkl file from chumpy to numpy format."""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f" File not found: {input_path}")
        return False

    print(f"\n Converting {input_path.name}...")

    # Load original pickle
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print(f"   ok: Loaded original file")
    except Exception as e:
        print(f"    Failed to load: {e}")
        return False

    # Convert to numpy
    try:
        data_np = recursive_to_numpy(data)
        print(f"   ok: Converted chumpy objects to numpy")
    except Exception as e:
        print(f"    Conversion failed: {e}")
        return False

    # Backup original
    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
    try:
        if not backup_path.exists():
            import shutil
            shutil.copy2(input_path, backup_path)
            print(f"   ok: Backed up to {backup_path.name}")
        else:
            print(f"     Backup already exists: {backup_path.name}")
    except Exception as e:
        print(f"     Backup failed: {e}")

    # Save as converted pickle
    pkl_output = input_path.with_name(
        input_path.stem + output_suffix + input_path.suffix
    )
    try:
        with open(pkl_output, 'wb') as f:
            pickle.dump(data_np, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ok: Saved pickle: {pkl_output.name}")
    except Exception as e:
        print(f"    Failed to save pickle: {e}")
        return False

    # Save as .npz
    npz_output = pkl_output.with_suffix('.npz')
    try:
        np.savez(npz_output, **data_np)
        print(f"   ok: Saved npz: {npz_output.name}")
    except Exception as e:
        print(f"     Failed to save npz: {e}")

    print(f" Successfully converted {input_path.name}")
    return True


def main():
    # Determine MANO models directory
    script_dir = Path(__file__).resolve().parent
    mano_dir = script_dir.parent / "mano_models"

    if not mano_dir.exists():
        print(f" MANO models directory not found: {mano_dir}")
        print("   Please create it and place MANO_RIGHT.pkl and MANO_LEFT.pkl inside.")
        return

    print("=" * 60)
    print(" MANO Model Converter (chumpy → NumPy)")
    print("=" * 60)
    print(f" MANO directory: {mano_dir}")

    # Convert both hands
    files_to_convert = [
        mano_dir / "MANO_RIGHT.pkl",
        mano_dir / "MANO_LEFT.pkl"
    ]

    success_count = 0
    for filepath in files_to_convert:
        if convert_file(filepath):
            success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(files_to_convert):
        print(f" All {success_count} files converted successfully!")
        print("\n Next steps:")
        print("   1. Deactivate this environment: deactivate")
        print("   2. Activate runtime environment: source asha_env/bin/activate")
        print("   3. Run the application: python realtime_asha/src/realtime_mano.py")
    else:
        print(f"  Converted {success_count}/{len(files_to_convert)} files")
        print("   Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
