12/14/2025

     Project Asha: Complete Codebase Restructuring Plan

     Overview

     Restructure from monolithic src/ to version-based organization with pip-installable package. This enables cleaner imports, better 
     reproducibility, and maps to the project's evolution timeline.

     Goals

     1. ✅ Remove src/ and transfer/ folder structures
     2. ✅ Create version-based folders (v0-v5) as first-class modules
     3. ✅ Shared utilities in asha/core/ package
     4. ✅ Data and models organized by version
     5. ✅ Make project pip-installable (pip install -e .)
     6. ✅ Fix all import paths throughout codebase
     7. ✅ Rename data/v2_recordings → data/v4/emg_recordings

     Target Directory Structure

     asha/
     ├── setup.py                    # NEW: Pip installation
     ├── asha/                       # NEW: Core package (pip installable)
     │   ├── __init__.py
     │   ├── core/                   # Shared utilities
     │   │   ├── __init__.py
     │   │   ├── mano_model.py      # From src/model_utils/
     │   │   ├── tracker.py
     │   │   ├── pose_fitter.py
     │   │   ├── emg_utils.py       # From src/utils/
     │   │   ├── data_recorder.py
     │   │   └── visualization.py
     │   │
     │   ├── v0/                     # MediaPipe baseline
     │   │   ├── __init__.py
     │   │   ├── README.md
     │   │   └── mediapipe_demo.py  # From src/programs/mediapipe.py
     │   │
     │   ├── v1/                     # MANO IK
     │   │   ├── __init__.py
     │   │   ├── README.md
     │   │   └── mano.py            # From src/programs/mano.py
     │   │
     │   ├── v2/                     # Image → θ training (PAUSED)
     │   │   ├── __init__.py
     │   │   ├── README.md
     │   │   └── train_colab.ipynb  # From src/training/train_colab.ipynb
     │   │
     │   ├── v3/                     # Transfer learning (ABANDONED)
     │   │   ├── __init__.py
     │   │   ├── README.md          # Mark as ABANDONED
     │   │   └── (archive files from transfer/)
     │   │
     │   ├── v4/                     # EMG → MANO θ (SimpleEMGModel)
     │   │   ├── __init__.py
     │   │   ├── README.md
     │   │   ├── model.py           # NEW: Extract SimpleEMGModel
     │   │   ├── train_colab.py     # From src/training/train_v2_colab.py
     │   │   └── inference.py       # From src/programs/inference_v2.py
     │   │
     │   ├── v5/                     # EMG → Joints (EMGToJointsModel)
     │   │   ├── __init__.py
     │   │   ├── README.md
     │   │   ├── model.py           # NEW: Extract EMGToJointsModel
     │   │   ├── train_colab.py     # From src/training/train_joints_colab.py
     │   │   ├── inference.py       # From src/programs/inference_joints.py
     │   │   └── finetune.py        # From src/training/finetune_joints.py
     │   │
     │   └── shared/
     │       ├── __init__.py
     │       ├── record.py          # From src/programs/record.py
     │       └── convert_mano_to_numpy.py
     │
     ├── experiments/                # From src/experiments/
     │   ├── __init__.py
     │   └── pose_fitter_experimental.py
     │
     ├── tests/                      # From src/tests/
     │   └── (all test files)
     │
     ├── scripts/                    # From root level
     │   ├── setup_runtime_env.sh
     │   ├── setup_convert_env.sh
     │   └── check_data.py
     │
     ├── data/                       # Reorganized by version
     │   ├── v0/
     │   ├── v1/
     │   ├── v2/                    # Image training data (FreiHAND)
     │   ├── v4/                    # NEW: From data/v2_recordings
     │   │   └── emg_recordings/
     │   ├── v5/                    # Calibration data
     │   └── transfer/              # Archived
     │
     ├── models/                     # Reorganized by version
     │   ├── mano/                  # MANO base models
     │   ├── v2/                    # Image training checkpoints
     │   ├── v3/                    # Transfer learning (emg2pose)
     │   ├── v4/                    # EMG → θ checkpoints
     │   │   ├── emg_model_best.pth
     │   │   └── emg_model_v2_best.pth
     │   └── v5/                    # EMG → Joints checkpoints
     │       └── emg_joints_best.pth
     │
     ├── docs/                       # Unchanged
     ├── results/                    # Unchanged
     └── environments/               # Unchanged (gitignored)

     File Migration Map

     Phase 1: Core Package Setup

     NEW Files:

     1. setup.py (root)
       - Create pip-installable package
       - Define dependencies
       - Set version to 0.5.0 (matching v5)
     2. asha/init.py
       - Package entry point
       - Define public API
     3. asha/core/init.py
       - Export core utilities
       - Example: from .mano_model import MANOModel

     Move Core Utilities:

     src/model_utils/mano_model.py    → asha/core/mano_model.py
     src/model_utils/tracker.py       → asha/core/tracker.py
     src/model_utils/pose_fitter.py   → asha/core/pose_fitter.py
     src/utils/emg_utils.py           → asha/core/emg_utils.py
     src/utils/data_recorder.py       → asha/core/data_recorder.py
     src/utils/visualize_*.py         → asha/core/visualization.py (consolidate)

     Phase 2: Version-Specific Code

     v0 (MediaPipe):

     src/programs/mediapipe.py → asha/v0/mediapipe_demo.py
     CREATE: asha/v0/README.md

     v1 (MANO IK):

     src/programs/mano.py → asha/v1/mano.py
     CREATE: asha/v1/README.md

     v2 (Image Training - PAUSED):

     src/training/train_colab.ipynb → asha/v2/train_colab.ipynb
     CREATE: asha/v2/README.md (mark as PAUSED)

     v3 (Transfer Learning - ABANDONED):

     transfer/training/train_mode1.py → asha/v3/train_mode1.py
     transfer/training/train_mode2.py → asha/v3/train_mode2.py
     transfer/programs/* → asha/v3/ (archive)
     CREATE: asha/v3/README.md (mark as ABANDONED, link to postmortem)

     v4 (EMG → MANO θ):

     src/training/train_v2_colab.py → asha/v4/train_colab.py
     src/programs/inference_v2.py → asha/v4/inference.py
     EXTRACT: SimpleEMGModel class → asha/v4/model.py (lines 41-130 from train_v2_colab.py)
     CREATE: asha/v4/README.md

     v5 (EMG → Joints):

     src/training/train_joints_colab.py → asha/v5/train_colab.py
     src/programs/inference_joints.py → asha/v5/inference.py
     src/training/finetune_joints.py → asha/v5/finetune.py
     EXTRACT: EMGToJointsModel class → asha/v5/model.py (lines 36-93 from train_joints_colab.py)
     CREATE: asha/v5/README.md

     Shared:

     src/programs/record.py → asha/shared/record.py
     src/utils/convert_mano_to_numpy.py → asha/shared/convert_mano_to_numpy.py

     Phase 3: Data & Models Reorganization

     Data:

     data/v2_recordings/* → data/v4/emg_recordings/*
     data/calibration_*.h5 → data/v5/
     transfer/data/ → data/v3/ (archive)

     Models:

     models/MANO_*.pkl → models/mano/
     models/emg_model_best.pth → models/v4/
     models/emg_model_v2_best.pth → models/v4/
     models/emg_joints_*.pth → models/v5/
     models/emg2pose/* → models/v3/emg2pose/

     Phase 4: Tests & Experiments

     src/experiments/* → experiments/ (remove src/ prefix)
     src/tests/* → tests/ (remove src/ prefix)

     Import Changes

     Before (Current):

     # Messy sys.path hacks
     import sys
     sys.path.insert(0, str(Path(__file__).parent.parent))
     from model_utils.tracker import get_landmarks
     from utils.emg_utils import MindroveInterface

     After (Clean Package Imports):

     # After pip install -e .
     from asha.core.tracker import get_landmarks
     from asha.core.emg_utils import MindroveInterface
     from asha.v5.model import EMGToJointsModel

     # Or relative imports within package
     from ..core.tracker import get_landmarks
     from .model import EMGToJointsModel

     Critical Files to Update

     Files with Import Statements (28 references across 9 files):

     1. asha/core/*.py (update internal imports)
     2. asha/v4/train_colab.py (import from asha.core, import .model)
     3. asha/v4/inference.py (import from asha.core, import .model)
     4. asha/v5/train_colab.py (import from asha.core, import .model)
     5. asha/v5/inference.py (import from asha.core, import .model)
     6. asha/v5/finetune.py (import from asha.core, import .model)
     7. asha/shared/record.py (import from asha.core)
     8. tests/*.py (update all test imports)
     9. experiments/*.py (update experiment imports)

     Files with Path References:

     1. data/v4/emg_recordings/README.md (update paths, was data/v2_recordings/README.md)
     2. transfer/README.md (update references to new structure)
     3. docs/TRANSFER_LEARNING_POSTMORTEM.md (update file paths)
     4. CLAUDE.md (complete rewrite of structure section)

     Implementation Steps

     Step 1: Create Package Structure (30 min)

     1. Create asha/ directory at root
     2. Create asha/__init__.py with package metadata
     3. Create asha/core/, asha/v0/, asha/v1/, asha/v2/, asha/v3/, asha/v4/, asha/v5/, asha/shared/
     4. Add __init__.py to each subdirectory
     5. Create setup.py at root with dependencies and package config

     Step 2: Move Core Utilities (1 hour)

     1. Move all files from src/model_utils/ → asha/core/
     2. Move EMG/data files from src/utils/ → asha/core/
     3. Update imports within asha/core/ files (relative imports)
     4. Create asha/core/__init__.py exporting main classes
     5. Test: pip install -e . and from asha.core import MANOModel

     Step 3: Migrate v4 (EMG → θ) (1 hour)

     1. Extract SimpleEMGModel class → asha/v4/model.py
     2. Move src/training/train_v2_colab.py → asha/v4/train_colab.py
     3. Move src/programs/inference_v2.py → asha/v4/inference.py
     4. Update imports in both files:
       - Change from src.model_utils import → from asha.core import
       - Add from .model import SimpleEMGModel
     5. Create asha/v4/README.md documenting v4 approach
     6. Test: Run inference to verify imports work

     Step 4: Migrate v5 (EMG → Joints) (1 hour)

     1. Extract EMGToJointsModel class → asha/v5/model.py
     2. Move src/training/train_joints_colab.py → asha/v5/train_colab.py
     3. Move src/programs/inference_joints.py → asha/v5/inference.py
     4. Move src/training/finetune_joints.py → asha/v5/finetune.py
     5. Update imports in all three files
     6. Create asha/v5/README.md documenting v5 approach
     7. Test: Run inference to verify imports work

     Step 5: Migrate v0, v1, v2, v3 (1 hour)

     1. Move src/programs/mediapipe.py → asha/v0/mediapipe_demo.py
     2. Move src/programs/mano.py → asha/v1/mano.py
     3. Move src/training/train_colab.ipynb → asha/v2/train_colab.ipynb
     4. Archive transfer learning files → asha/v3/
     5. Create README.md for each version
     6. Update imports in v0 and v1

     Step 6: Migrate Shared Programs (30 min)

     1. Move src/programs/record.py → asha/shared/record.py
     2. Move src/utils/convert_mano_to_numpy.py → asha/shared/convert_mano_to_numpy.py
     3. Update imports to use asha.core
     4. Test: Run record.py to verify it works

     Step 7: Reorganize Data & Models (30 min)

     1. Rename data/v2_recordings/ → data/v4/emg_recordings/
     2. Create data/v0/, data/v1/, data/v2/, data/v3/, data/v5/ directories
     3. Create models/v2/, models/v3/, models/v4/, models/v5/ directories
     4. Move existing model files to appropriate version folders
     5. Update path references in training/inference scripts

     Step 8: Move Tests & Experiments (30 min)

     1. Move src/experiments/* → experiments/
     2. Move src/tests/* → tests/
     3. Update all imports in test files
     4. Run test suite to verify

     Step 9: Update Documentation (30 min)

     1. Update data/v4/emg_recordings/README.md (was v2_recordings)
     2. Update transfer/README.md with new paths
     3. Update docs/TRANSFER_LEARNING_POSTMORTEM.md with new paths
     4. Rewrite structure section in CLAUDE.md
     5. Update root README.md if needed

     Step 10: Delete Old Directories (5 min)

     1. Delete src/ directory (after verifying all files moved)
     2. Delete empty transfer/ subdirectories (keep README)
     3. Update .gitignore if needed

     Step 11: Final Validation (30 min)

     1. Run pip install -e . from root
     2. Test v4 inference: python -m asha.v4.inference --model models/v4/emg_model_v2_best.pth
     3. Test v5 inference: python -m asha.v5.inference --model models/v5/emg_joints_best.pth
     4. Test recording: python -m asha.shared.record
     5. Run test suite: pytest tests/
     6. Verify all imports work

     Success Criteria

     - pip install -e . works without errors
     - All imports use from asha.core or from asha.v{X}
     - No sys.path hacks in any file
     - v4 inference runs successfully
     - v5 inference runs successfully
     - Recording program works
     - All tests pass
     - Documentation updated with new structure
     - Old src/ directory deleted
     - Data and models organized by version

     Estimated Time

     - Phase 1 (Package Setup): 30 min
     - Phase 2 (Core Utilities): 1 hour
     - Phase 3 (v4 Migration): 1 hour
     - Phase 4 (v5 Migration): 1 hour
     - Phase 5 (v0-v3 Migration): 1 hour
     - Phase 6 (Shared Programs): 30 min
     - Phase 7 (Data & Models): 30 min
     - Phase 8 (Tests & Experiments): 30 min
     - Phase 9 (Documentation): 30 min
     - Phase 10 (Cleanup): 5 min
     - Phase 11 (Validation): 30 min

     Total: ~7.5 hours

     Risk Mitigation

     1. Import Breakage: Test after each phase, fix immediately
     2. Path Errors: Use absolute imports from package, avoid relative when possible
     3. Colab Compatibility: Test notebooks can still pip install -e . from Drive
     4. Data Loss: Do NOT delete old directories until fully verified
     5. Checkpoint Incompatibility: Keep checkpoint names unchanged to avoid retraining

     Notes

     - Keep checkpoint filenames unchanged (emg_model_v2_best.pth stays as-is)
     - Colab notebooks will need !pip install -e /content/drive/MyDrive/.../asha
     - Tests can import like external users: from asha.v5.model import EMGToJointsModel
     - Each version README should document status, key results, and quick start
     - v3 README marks transfer learning as ABANDONED with link to postmortem