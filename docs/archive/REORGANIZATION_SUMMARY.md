# Source Code Reorganization Summary

## What Changed

The `src/` directory has been reorganized from a flat structure into logical modules:

### Before (Flat Structure)
```
src/
├── mano_model.py
├── tracker.py
├── pose_fitter.py
├── mediapipe_v0.py
├── mano_v1.py
├── main_v2.py
├── pose_fitter_experimental.py
├── run_experiments.py
├── freihand_loader.py
├── test_*.py
├── emg_utils.py
├── data_recorder.py
└── ...
```

### After (Modular Structure)
```
src/
├── model_utils/          # Core MANO and pose estimation
│   ├── mano_model.py
│   ├── tracker.py
│   └── pose_fitter.py
│
├── programs/             # Runnable applications
│   ├── mediapipe_v0.py
│   ├── mano_v1.py
│   └── main_v2.py
│
├── experiments/          # CIS 6800 experiments
│   ├── pose_fitter_experimental.py
│   ├── run_experiments.py
│   └── freihand_loader.py
│
├── tests/                # Test suites
│   ├── test.py
│   ├── test_mediapipe_v0.py
│   ├── test_mano_v1.py
│   └── test_main_v2.py
│
└── utils/                # Utilities
    ├── emg_utils.py
    ├── data_recorder.py
    ├── data_loader.py
    ├── visualize_experiments.py
    ├── extract_results.py
    ├── generate_plots.py
    └── convert_mano_to_numpy.py
```

---

## Benefits

1. **Clear Organization**: Easy to find what you need
   - Core utilities in `model_utils/`
   - Applications in `programs/`
   - Experiments in `experiments/`

2. **Better Modularity**: Each directory has a clear purpose

3. **Easier Navigation**: No more scrolling through 20+ files

4. **Professional Structure**: Follows Python best practices

---

## How to Use

### Running Programs

**v1 (MANO IK tracking)**:
```bash
python src/programs/mano_v1.py
```

**v2 (EMG + recording)**:
```bash
python src/programs/main_v2.py
```

### Running Experiments

**Quick test** (100 frames):
```bash
python src/experiments/run_experiments.py --video test.mp4 --quick
```

**Full experiments**:
```bash
python src/experiments/run_experiments.py \
  --video validation.mp4 \
  --experiments 1,2,3,4 \
  --output-dir results/full_run
```

**FreiHAND evaluation**:
```bash
python src/experiments/freihand_loader.py \
  --dataset-path ~/datasets/freihand \
  --num-samples 100
```

### Generating Figures

```bash
python src/utils/visualize_experiments.py \
  --results results/full_run \
  --output docs/figures
```

### Running Tests

```bash
python src/tests/test.py              # All tests
python src/tests/test_mano_v1.py     # Specific test
```

---

## Import Changes

All imports have been automatically updated. If you write new code:

**Old style** (no longer works):
```python
from tracker import MediaPipeHandTracker
from pose_fitter import mano_from_landmarks
```

**New style** (use this):
```python
from model_utils.tracker import MediaPipeHandTracker
from model_utils.pose_fitter import mano_from_landmarks
from utils.emg_utils import MindroveInterface
from experiments.pose_fitter_experimental import ExperimentalIKFitter
```

---

## Experiments Overview

Your CIS 6800 final project experiments are in `experiments/`:

### Experiment 1: Loss Ablation
**Goal**: Which loss terms matter most?

```bash
python src/experiments/run_experiments.py --video test.mp4 --experiments 1
```

**Tests**:
- Baseline (all losses)
- No bone loss
- No temporal smoothing
- Position only

**Metric**: Mean IK error (mm)

---

### Experiment 2: Alignment Methods
**Goal**: Justify using Umeyama with scale

```bash
python src/experiments/run_experiments.py --video test.mp4 --experiments 2
```

**Tests**:
- Umeyama with scale (similarity transform)
- Umeyama rigid (no scale)
- Kabsch (pure rigid)

**Metric**: Mean IK error + scale factor

---

### Experiment 3: Optimizer Comparison
**Goal**: Show Adam is best trade-off

```bash
python src/experiments/run_experiments.py --video test.mp4 --experiments 3
```

**Tests**:
- Adam (adaptive learning)
- SGD (basic gradient descent)
- L-BFGS (second-order)

**Metrics**: Error + speed (ms/frame)

---

### Experiment 4: Per-Joint Error Analysis
**Goal**: Identify failure modes

```bash
python src/experiments/run_experiments.py --video test.mp4 --experiments 4
```

**Output**:
- Per-joint error distribution
- Worst 5 joints identified
- Bar chart for paper

---

### Experiment 5: FreiHAND Evaluation
**Goal**: Benchmark against SOTA

```bash
# First download FreiHAND
./scripts/download_datasets.sh freihand

# Then evaluate
python src/experiments/freihand_loader.py \
  --dataset-path ~/datasets/freihand \
  --num-samples 1000
```

**Comparison**:
```
HandFormer (2024):  10.92mm (STEREO), 12.33mm (FreiHAND)
Drosakis (2023):    ~11-13mm
Your method:        ??? mm
```

---

## Dataset Information

### FreiHAND
- **Size**: 130K training images, 3,960 test images
- **Ground truth**: Multi-view triangulation (~5mm accuracy)
- **Download**: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreiHand.en.html
- **Location**: `~/datasets/freihand/`
- **What it tests**: General hand pose accuracy

### HO-3D (Optional)
- **Size**: 77K training frames, 11K test frames
- **Ground truth**: RGB-D + object models
- **Download**: https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/
- **Location**: `~/datasets/ho3d/`
- **What it tests**: Robustness to occlusion (hands holding objects)

---

## Quick Reference

### File Locations

| File Type | Location | Example |
|-----------|----------|---------|
| Core models | `model_utils/` | `mano_model.py` |
| Applications | `programs/` | `mano_v1.py` |
| Experiments | `experiments/` | `run_experiments.py` |
| Tests | `tests/` | `test_mano_v1.py` |
| Utilities | `utils/` | `visualize_experiments.py` |

### Common Tasks

| Task | Command |
|------|---------|
| Run v1 | `python src/programs/mano_v1.py` |
| Run experiments | `python src/experiments/run_experiments.py --video test.mp4` |
| Generate figures | `python src/utils/visualize_experiments.py --results results` |
| Run tests | `python src/tests/test.py` |

---

## For Your Paper (Due Dec 8)

1. **Run all experiments**:
   ```bash
   python src/experiments/run_experiments.py \
     --video your_validation_video.mp4 \
     --experiments 1,2,3,4
   ```

2. **Generate figures**:
   ```bash
   python src/utils/visualize_experiments.py \
     --results results/experiments \
     --output docs/figures
   ```

3. **Evaluate on FreiHAND** (if downloaded):
   ```bash
   python src/experiments/freihand_loader.py --num-samples 1000
   ```

4. **Use the figures**:
   - `docs/figures/exp1_loss_ablation.png`
   - `docs/figures/exp2_alignment_methods.png`
   - `docs/figures/exp3_optimizer_comparison.png`
   - `docs/figures/exp4_per_joint_errors.png`
   - `docs/figures/experiment_summary_table.tex` (LaTeX table)

---

## Questions?

- **How do I add a new experiment?** → Add function to `experiments/run_experiments.py`
- **How do I modify IK?** → Edit `model_utils/pose_fitter.py`
- **How do I test new features?** → Use `experiments/pose_fitter_experimental.py`
- **Where are my results?** → Check `results/experiments/`
- **How do I create new figures?** → Add function to `utils/visualize_experiments.py`

For more details, see:
- `src/README.md` - Complete module documentation
- `docs/EXPERIMENTS_GUIDE.md` - How to run experiments
- `CLAUDE.md` - Full project history and architecture
