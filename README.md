# Project Asha

Camera-free hand pose estimation via EMG and inverse kinematics pseudo-labeling.

## Overview

Project Asha achieves real-time hand tracking using surface EMG (electromyography) muscle signals, eliminating the need for cameras or expensive motion capture systems. The system generates training labels through inverse kinematics optimization on webcam video (9.71mm accuracy), then trains EMG models that predict 3D hand pose at 14.92mm MPJPE while running at 30 fps. This approach democratizes EMG hand tracking research by removing the $100K+ mocap barrier.

## Key Results

- **IK Pseudo-Labeling**: 9.71mm MPJPE @ 25fps (beats HandFormer's 10.92mm stereo baseline)
- **EMG Tracking**: 14.92mm MPJPE @ 30fps (camera-free, competitive with vision SOTA)
- **Training Data**: 20 minutes of self-collected data (vs typical 370+ hours for EMG)
- **Hardware Cost**: ~$600 EMG armband (vs $100K+ mocap systems)

## Features

- Real-time camera-free hand pose estimation at 30fps
- Multi-term MANO inverse kinematics with bone direction and temporal smoothing
- Direct joint prediction architecture (Conv1D + LSTM, 1M parameters)
- Systematic validation through 7 controlled experiments
- Works through occlusions, darkness, and visual noise

## System Architecture

The system operates in two stages:

**Stage 1: Pseudo-Label Generation**
```
Webcam → MediaPipe (21 keypoints) → MANO IK Optimization → 9.71mm labels
```

**Stage 2: EMG Model Training**
```
EMG (8ch @ 500Hz) → Conv1D + LSTM → Direct Joints (63 values) → 14.92mm MPJPE
```

## Hardware

- **EMG Device**: MindRove 8-channel armband @ 500Hz sampling rate
- **Camera** (training only): Standard webcam @ 30fps
- **Training**: NVIDIA A100 GPU (1.5 hours for 100 epochs)
- **Inference**: Any modern CPU (<10ms latency)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/asha.git
cd asha

# Create virtual environment
python3.11 -m venv asha_env
source asha_env/bin/activate

# Install package
pip install -e .

# Download MANO models from https://mano.is.tue.mpg.de
# Place MANO_RIGHT.pkl in models/mano/

# Convert MANO to numpy format (one-time setup)
python -m asha.shared.convert_mano_to_numpy
```

## Quick Start

**Real-time EMG hand tracking** (camera-free):
```bash
python -m asha.v5.inference --model models/v5/emg_joints_best.pth
```

**MANO IK visualization** (pseudo-labeling system):
```bash
python -m asha.v1.mano
```

**Record training data** (EMG + video):
```bash
python -m asha.shared.record
```

## Repository Structure

```
asha/
├── asha/                  # Main package
│   ├── core/              # Shared utilities (MANO, MediaPipe, EMG)
│   ├── v1/                # MANO IK (9.71mm)
│   ├── v4/                # EMG→θ (34.92mm)
│   ├── v5/                # EMG→Joints (14.92mm, best)
│   └── shared/            # Recording utilities
├── tests/                 # Unit tests
├── experiments/           # IK validation experiments
└── docs/                  # Documentation and figures
```

## Training Details

**Model**: Conv1D (8→64→128 channels) + 2-layer LSTM (256 hidden units) + FC layer (63 outputs)

**Data**: 23,961 training windows (20 minutes, 5 sessions) with 8-channel EMG @ 500Hz

**Optimization**: Adam with learning rate schedule (1e-3 → 5e-4 @ epoch 78)

**Result**: 19.84mm (epoch 1) → 14.92mm (epoch 99), 24.8% improvement

## Experimental Validation

Seven systematic experiments validate the approach:

1. Loss ablation: Bone direction loss critical (+28% error without)
2. Alignment methods: Umeyama with scale estimation optimal (6% improvement)
3. Optimizer comparison: Adam achieves 99.7% convergence rate
4. Per-joint analysis: Fingertips most challenging due to depth ambiguity
5. Architecture comparison: Direct joints 2.3× better than parametric
6. Training convergence: Learning rate schedule provides 1.95mm improvement
7. Ground truth quality: Only 5mm gap between IK labels and EMG predictions

Dataset: 5,330 frames (IK validation), 23,961 windows (EMG training)

## References

**Key Papers**:
- MANO (Romero et al., SIGGRAPH Asia 2017): Parametric hand model
- HandFormer (Jiao et al., Pattern Recognition Letters 2024): Vision transformer baseline
- emg2pose (Meta FAIR, 2024): Large-scale EMG benchmark
- NeuroPose (Chen et al., UIST 2021): EMG with anatomical constraints

Full bibliography available in `docs/REFERENCES.md`

## Documentation

- `CLAUDE.md`: Complete project documentation and session history
- `docs/reports/`: LaTeX source and PDFs for proposal, midterm, and final reports
- `docs/TRANSFER_LEARNING_POSTMORTEM.md`: Analysis of failed transfer learning approach

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

MANO models courtesy of Max Planck Institute for Intelligent Systems. Compute provided by Google Colab Pro+.
