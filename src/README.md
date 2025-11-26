# Source Code Organization

The `src/` directory has been reorganized into logical modules for better maintainability.

## Directory Structure

```
src/
├── model_utils/          # Core MANO and pose estimation utilities
│   ├── mano_model.py           # Custom MANO layer implementation
│   ├── tracker.py              # MediaPipe hand tracking wrapper
│   └── pose_fitter.py          # IK optimization for pose fitting
│
├── programs/             # Main applications (runnable programs)
│   ├── mediapipe_v0.py         # v0: MediaPipe baseline (scatter plot)
│   ├── mano_v1.py              # v1: Full IK + LBS articulation
│   └── main_v2.py              # v2: Full system (CV + EMG recording)
│
├── experiments/          # CIS 6800 experiment code
│   ├── pose_fitter_experimental.py   # Experimental IK with configurable losses
│   ├── run_experiments.py            # Experiment runner (Exp 1-5)
│   └── freihand_loader.py            # FreiHAND dataset loader for Exp 5
│
├── tests/                # Test suites
│   ├── test.py                 # Main test runner
│   ├── test_mediapipe_v0.py    # v0 tests
│   ├── test_mano_v1.py         # v1 tests
│   └── test_main_v2.py         # v2 tests (EMG + recorder)
│
└── utils/                # Utilities and helpers
    ├── emg_utils.py            # MindRove interface + signal processing
    ├── data_recorder.py        # HDF5 synchronized recording
    ├── data_loader.py          # EMG → θ dataset loader (for v3 training)
    ├── visualize_experiments.py # Figure generation for paper
    ├── extract_results.py      # Metrics extraction
    ├── generate_plots.py       # Plot generation
    └── convert_mano_to_numpy.py # MANO model converter
```

## Module Purposes

### `model_utils/` - Core Pose Estimation

**Purpose**: Core utilities for MANO parametric model and pose estimation.

**Contents**:
- `mano_model.py`: Custom MANO layer (avoids chumpy dependency)
- `tracker.py`: MediaPipe Hands wrapper (landmark detection)
- `pose_fitter.py`: IK optimization (baseline implementation)

**Used by**: All programs, experiments, tests

---

### `programs/` - Runnable Applications

**Purpose**: Main applications you can run interactively.

**Contents**:
- `mediapipe_v0.py`: Baseline MediaPipe tracking (60fps, scatter plot)
- `mano_v1.py`: Full MANO IK optimization (25fps, 3D mesh)
- `main_v2.py`: Complete system with EMG recording

**How to run**:
```bash
source asha_env/bin/activate
python -m programs.mano_v1          # Run v1
python -m programs.main_v2          # Run v2 with EMG
```

---

### `experiments/` - CIS 6800 Experiments

**Purpose**: Experimental code for your Computer Vision final project.

**Contents**:
- `pose_fitter_experimental.py`: Configurable IK with experimental features
  - Joint limit constraints
  - Confidence weighting
  - Alternative alignment methods
  - Metrics collection
- `run_experiments.py`: Automated experiment runner (Exp 1-5)
- `freihand_loader.py`: FreiHAND dataset loader for benchmarking

**How to run**:
```bash
# Run all experiments
python -m experiments.run_experiments --video test.mp4

# Run specific experiments
python -m experiments.run_experiments --video test.mp4 --experiments 1,2,3

# Evaluate on FreiHAND
python -m experiments.freihand_loader --num-samples 100
```

---

### `tests/` - Test Suites

**Purpose**: Validation and testing of all components.

**Contents**:
- `test.py`: Main test runner (runs all version tests)
- `test_mediapipe_v0.py`: Validates v0 (tracking, visualization)
- `test_mano_v1.py`: Validates v1 (IK convergence, performance)
- `test_main_v2.py`: Validates v2 (EMG filtering, HDF5 recording)

**How to run**:
```bash
python -m tests.test              # Run all tests
python -m tests.test_mano_v1      # Run specific test
```

---

### `utils/` - Utilities and Helpers

**Purpose**: Reusable utilities that don't fit in other categories.

**Contents**:
- **EMG & Data**:
  - `emg_utils.py`: MindRove WiFi interface, signal processing
  - `data_recorder.py`: HDF5 synchronized recording
  - `data_loader.py`: Dataset loader for v3 training (future)

- **Visualization**:
  - `visualize_experiments.py`: Generate publication-quality figures
  - `extract_results.py`: Extract metrics from recordings
  - `generate_plots.py`: Plot generation utilities

- **Setup**:
  - `convert_mano_to_numpy.py`: Convert MANO models to numpy format

**How to use**:
```bash
# Generate figures
python -m utils.visualize_experiments --results results/experiments

# Extract metrics
python -m utils.extract_results --session data/session_001.h5
```

---

## Import Structure

After reorganization, imports follow this pattern:

```python
# From model_utils/
from model_utils.mano_model import CustomMANOLayer
from model_utils.tracker import MediaPipeHandTracker
from model_utils.pose_fitter import mano_from_landmarks

# From experiments/
from experiments.pose_fitter_experimental import ExperimentalIKFitter, ExperimentConfig
from experiments.freihand_loader import FreiHANDLoader

# From utils/
from utils.emg_utils import MindroveInterface, filter_emg
from utils.data_recorder import DataRecorder
from utils.visualize_experiments import plot_ik_error_histogram
```

---

## Migration Guide

If you have code that imports from the old structure:

**Old**:
```python
from tracker import MediaPipeHandTracker
from pose_fitter import mano_from_landmarks
from emg_utils import MindroveInterface
```

**New**:
```python
from model_utils.tracker import MediaPipeHandTracker
from model_utils.pose_fitter import mano_from_landmarks
from utils.emg_utils import MindroveInterface
```

All imports have been automatically updated in the reorganized files.

---

## Running Programs

### From Project Root

```bash
cd /path/to/asha
source asha_env/bin/activate

# Run programs
python src/programs/mano_v1.py
python src/programs/main_v2.py

# Run experiments
python src/experiments/run_experiments.py --video test.mp4

# Run tests
python src/tests/test.py
```

### As Python Modules

```bash
# Add src/ to PYTHONPATH
export PYTHONPATH="/path/to/asha/src:$PYTHONPATH"

# Run as modules
python -m programs.mano_v1
python -m experiments.run_experiments --video test.mp4
python -m tests.test
```

---

## Questions?

- **Adding new experiments?** → Put them in `experiments/`
- **Adding new utilities?** → Put them in `utils/`
- **Modifying core MANO/IK?** → Edit files in `model_utils/`
- **Creating new application?** → Add to `programs/`

For more details, see:
- Main README: `../README.md`
- Experiment guide: `../docs/EXPERIMENTS_GUIDE.md`
- CLAUDE.md: Development history and architecture
