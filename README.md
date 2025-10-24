# Project Asha

Real-time 3D hand pose estimation via MANO parametric model optimization + EMG integration for camera-free tracking.

## Quick Start

```bash
git clone https://github.com/jeevankarandikar/asha.git
cd asha
./setup_runtime_env.sh
source asha_env/bin/activate

# Run systems
python src/mano_v1.py          # CV pipeline (MANO IK optimization)
python src/main_v2.py          # Full system (CV + EMG recording)
```

## Systems

- **mano_v1** - MANO IK optimization with multi-term loss (~25 fps, real-time)
- **main_v2** - EMG integration + synchronized data recording (CV pipeline + 8ch EMG @ 500Hz)

## Core Files

- **pose_fitter.py** - MANO IK optimization (joint position, bone direction, temporal, regularization)
- **mano_model.py** - Custom MANO layer (LBS implementation)
- **tracker.py** - MediaPipe hand tracking wrapper
- **emg_utils.py** - MindRove interface + signal processing
- **data_recorder.py** - HDF5 synchronized recording

## For CIS 6800 Proposal

See **[`docs/README.md`](docs/README.md)** for complete workflow:
- Extract metrics from v1 system
- Generate publication-quality plots
- Compile LaTeX proposal

## mano setup

### 1. download mano models

- get from https://mano.is.tue.mpg.de/download.php
- place MANO_RIGHT.pkl and MANO_LEFT.pkl in `models/`

### 2. convert models (python 3.10)

```bash
./setup_convert_env.sh
source mano_convert_env/bin/activate
python src/convert_mano_to_numpy.py
deactivate
```

### 3. test

```bash
source asha_env/bin/activate
python src/test.py
```

### 4. run

```bash
python src/mano_v1.py
```

## troubleshooting

- **camera won't open** - check permissions in system preferences
- **mps not available** - need apple silicon, falls back to cpu
- **mano loading fails** - run conversion script above
