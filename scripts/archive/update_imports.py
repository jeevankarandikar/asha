#!/usr/bin/env python3
"""
Update imports after reorganizing src/ directory structure.

Old structure:
  src/*.py

New structure:
  src/model_utils/  (mano_model, tracker, pose_fitter)
  src/programs/     (mediapipe_v0, mano_v1, main_v2)
  src/experiments/  (pose_fitter_experimental, run_experiments, freihand_loader)
  src/tests/        (test_*)
  src/utils/        (emg_utils, data_recorder, data_loader, visualize_experiments, etc.)
"""

import re
from pathlib import Path

# Define the import mappings
IMPORT_MAPPINGS = {
    # Model utils
    'from mano_model import': 'from model_utils.mano_model import',
    'import mano_model': 'import model_utils.mano_model as mano_model',
    'from tracker import': 'from model_utils.tracker import',
    'import tracker': 'import model_utils.tracker as tracker',
    'from pose_fitter import': 'from model_utils.pose_fitter import',
    'import pose_fitter': 'import model_utils.pose_fitter as pose_fitter',

    # Experiments
    'from pose_fitter_experimental import': 'from experiments.pose_fitter_experimental import',
    'import pose_fitter_experimental': 'import experiments.pose_fitter_experimental as pose_fitter_experimental',
    'from run_experiments import': 'from experiments.run_experiments import',
    'from freihand_loader import': 'from experiments.freihand_loader import',

    # Utils
    'from emg_utils import': 'from utils.emg_utils import',
    'from data_recorder import': 'from utils.data_recorder import',
    'from data_loader import': 'from utils.data_loader import',
    'from visualize_experiments import': 'from utils.visualize_experiments import',
    'from extract_results import': 'from utils.extract_results import',
    'from generate_plots import': 'from utils.generate_plots import',
}

def update_file(file_path: Path):
    """Update imports in a single file."""
    if not file_path.exists():
        return

    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Apply all mappings
    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = content.replace(old_import, new_import)

    # Only write if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✓ Updated: {file_path.relative_to(Path.cwd())}")
        return True
    return False

def main():
    src_dir = Path(__file__).parent.parent / "src"

    print("Updating imports in reorganized src/ directory...")
    print(f"Root: {src_dir}\n")

    updated_count = 0

    # Update all Python files in the new structure
    for subdir in ['model_utils', 'programs', 'experiments', 'tests', 'utils']:
        subdir_path = src_dir / subdir
        if not subdir_path.exists():
            continue

        print(f"[{subdir}/]")
        for py_file in subdir_path.glob("*.py"):
            if update_file(py_file):
                updated_count += 1
        print()

    print(f"\n{'='*60}")
    print(f"✓ Updated {updated_count} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
