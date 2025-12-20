"""Quick script to check HDF5 data format."""
import h5py
from pathlib import Path

# Check v4 data
print("="*60)
print("Checking v4 data format")
print("="*60)

v4_files = sorted(Path('data/v4/emg_recordings').glob('*.h5'))
print(f"Found {len(v4_files)} files")

if v4_files:
    with h5py.File(v4_files[0], 'r') as f:
        print(f"\nFile: {v4_files[0].name}")
        print("Keys:", list(f.keys()))
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")

# Check v5 data
print("\n" + "="*60)
print("Checking v5 data format")
print("="*60)

v5_files = sorted(Path('data/v5').glob('*.h5'))
print(f"Found {len(v5_files)} files")

if v5_files:
    with h5py.File(v5_files[0], 'r') as f:
        print(f"\nFile: {v5_files[0].name}")
        print("Keys:", list(f.keys()))
        for key in f.keys():
            print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
