# CIS 6800 Experiments Guide

Quick guide for running experiments and generating figures for your final project.

## Setup

```bash
source asha_env/bin/activate
cd src
```

## What Was Created

### 1. **pose_fitter_experimental.py** - Experimental IK Framework
- Configurable loss terms (for ablation studies)
- Alternative alignment methods (Kabsch, weighted Procrustes)
- Joint limit constraints (anatomical feasibility)
- Confidence weighting (trust high-quality joints more)
- Improved temporal smoothing (velocity + acceleration)
- Comprehensive metrics collection

### 2. **run_experiments.py** - Experiment Runner
Runs all 5 experiments from your proposal:
- Exp 1: Loss ablation
- Exp 2: Alignment methods comparison
- Exp 3: Optimizer comparison
- Exp 4: Per-joint error analysis
- Exp 5: Public dataset evaluation (placeholder)

### 3. **visualize_experiments.py** - Figure Generator
Generates publication-quality figures:
- IK error histograms
- Per-joint error bar charts
- Loss ablation comparisons
- Alignment method comparisons
- Optimizer comparisons
- LaTeX summary table

## Quick Start

### Step 1: Record a Test Video (if you don't have one)

```bash
# Run v1 to test and record your hand movements
python src/mano_v1.py

# Or use your existing validation recording
```

### Step 2: Run Quick Test (100 frames)

```bash
python src/run_experiments.py \
  --video path/to/your/video.mp4 \
  --experiments 1,2,3 \
  --quick
```

This will:
- Process only 100 frames (fast test)
- Run experiments 1, 2, and 3
- Save results to `results/experiments/`

### Step 3: Run Full Experiments

```bash
# Run all experiments on full video
python src/run_experiments.py \
  --video path/to/your/validation_video.mp4 \
  --experiments 1,2,3,4 \
  --output-dir results/full_experiments
```

### Step 4: Generate Figures

```bash
python src/visualize_experiments.py \
  --results results/full_experiments \
  --output docs/figures
```

This generates:
- `exp1_loss_ablation.png`
- `exp2_alignment_methods.png`
- `exp3_optimizer_comparison.png`
- `exp4_per_joint_errors.png`
- `experiment_summary_table.tex` (for paper)

## Experiment Details

### Experiment 1: Loss Ablation

Tests which loss terms matter most:
- `baseline`: all losses (position + bone + temporal + reg)
- `no_bone`: without bone direction loss
- `no_temporal`: without temporal smoothness
- `no_reg`: without regularization
- `position_only`: only position loss

**Expected outcome:** Identify most important loss term for your paper.

### Experiment 2: Alignment Methods

Compares alignment approaches:
- `umeyama_scaled`: Similarity transform (with scale)
- `umeyama_rigid`: Rigid transform (no scale)
- `kabsch`: Pure Kabsch algorithm

**Expected outcome:** Justify choice of Umeyama alignment.

### Experiment 3: Optimizer Comparison

Compares optimization algorithms:
- `adam`: Adaptive moment estimation (default)
- `sgd`: Stochastic gradient descent
- `lbfgs`: Limited-memory BFGS (second-order)

**Expected outcome:** Show Adam is best trade-off (speed vs. accuracy).

### Experiment 4: Per-Joint Error Analysis

Analyzes error distribution across 21 joints.

**Expected outcome:** Identify which joints are hardest (fingertips? thumb? wrist?).

### Experiment 5: Public Dataset Evaluation

**Status:** Placeholder (requires dataset download).

To implement:
1. Download FreiHAND or HO-3D
2. Implement dataset loader
3. Run evaluation

## Advanced Usage

### Custom Experiment Configuration

```python
from pose_fitter_experimental import ExperimentalIKFitter, ExperimentConfig

# Create custom config
config = ExperimentConfig(
    use_bone_loss=True,
    lambda_bone=0.8,  # increase bone loss weight
    use_joint_limits=True,  # enable anatomical constraints
    lambda_limits=0.1,
    alignment_method="weighted_procrustes",  # confidence-weighted
    optimizer_type="adam",
    num_iterations=20,  # more iterations
)

# Run fitter
fitter = ExperimentalIKFitter(config)
verts, joints, theta, metrics = fitter.fit(landmarks, mediapipe_confidence)

# Check results
print(f"IK error: {metrics['ik_error_mm']:.2f} mm")
print(f"Per-joint errors: {metrics['per_joint_error_mm']}")
print(f"Converged: {metrics['converged']}")
```

### Test Improvements

Want to test a new idea? Easy!

```python
# Example: Add joint limit constraints
config_baseline = ExperimentConfig()
config_with_limits = ExperimentConfig(
    use_joint_limits=True,
    lambda_limits=0.1
)

# Compare
result_baseline = run_on_video("video.mp4", config_baseline)
result_with_limits = run_on_video("video.mp4", config_with_limits)

print(f"Baseline error: {result_baseline['mean_ik_error_mm']:.2f} mm")
print(f"With limits: {result_with_limits['mean_ik_error_mm']:.2f} mm")
```

## Tips for Paper

### What to Report

1. **Baseline Results** (from v1 validation):
   - Mean error: 10.8mm
   - Median: 8.8mm
   - 95th percentile: 21.5mm
   - Convergence: 96.8%

2. **Ablation Study** (Exp 1):
   - Which loss terms matter most?
   - Quantify impact of each term
   - Justify your multi-term loss design

3. **Design Choices** (Exp 2-3):
   - Why Umeyama over Kabsch?
   - Why Adam over SGD/L-BFGS?
   - Trade-offs (speed vs. accuracy)

4. **Failure Mode Analysis** (Exp 4):
   - Which joints have highest error?
   - Why? (depth ambiguity, occlusion, etc.)
   - How to improve?

### Figures for Paper

Use these in your report:
1. **Fig 1:** IK error histogram (from exp4 or baseline)
2. **Fig 2:** Per-joint error bar chart (exp4)
3. **Fig 3:** Loss ablation comparison (exp1)
4. **Fig 4:** Optimizer comparison (exp3)
5. **Table:** Experiment summary (LaTeX table generated)

### Writing Tips

- **Emphasize:** 10.8mm competitive with HandFormer (10.92mm)
- **Highlight:** Real-time performance (25fps) + interpretability
- **Discuss:** Trade-offs (single camera vs. mocap accuracy)
- **Future work:** Joint limits, shape optimization, public datasets

## Troubleshooting

### "No video found"
```bash
# Specify video path explicitly
python run_experiments.py --video path/to/video.mp4
```

### "MPS not available" warning
- Normal! Code uses CPU for SVD compatibility
- Will be fast enough for experiments

### Experiments taking too long
```bash
# Use --quick mode for testing
python run_experiments.py --video vid.mp4 --quick
```

### Import errors
```bash
# Make sure environment is activated
source asha_env/bin/activate

# Install missing packages
pip install scipy matplotlib seaborn
```

## Next Steps

### For Your Final Project (by Dec 8):

1. **This Week (Nov 24-30):**
   - [ ] Record validation video (5-10 min, diverse poses)
   - [ ] Run all experiments (1-4)
   - [ ] Generate figures
   - [ ] Start writing results section

2. **Next Week (Dec 1-7):**
   - [ ] Analyze results, identify best configuration
   - [ ] (Optional) Test improvements (joint limits, etc.)
   - [ ] Complete paper draft
   - [ ] Prepare final presentation

3. **Final Week (Dec 8):**
   - [ ] Finalize paper
   - [ ] Submit code + report

## Questions?

If you want to:
- **Add new loss terms** → modify `ExperimentalIKFitter.fit()` in `pose_fitter_experimental.py`
- **Test new alignment** → add method in `pose_fitter_experimental.py`, update `_get_alignment_fn()`
- **New experiment** → add function in `run_experiments.py`, follow pattern
- **New figure** → add function in `visualize_experiments.py`

Good luck with your final project! 🚀
