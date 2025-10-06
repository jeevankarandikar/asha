#  Project Asha — Setup Guide

Complete setup instructions for the real-time MediaPipe → MANO hand pose system.

---

## prerequisites

- macos (apple silicon recommended for mps acceleration)
- python 3.10 (for mano conversion)
- python 3.11 (for runtime)
- webcam with permissions

install python:
```bash
brew install python@3.10 python@3.11
```

---

## setup overview

two separate environments:

1. conversion environment (python 3.10) - convert legacy mano .pkl files
2. runtime environment (python 3.11) - run the application with mps acceleration

---

## step 1: download mano models

mano models not included due to licensing.

1. visit https://mano.is.tue.mpg.de/download.php
2. register and download mano models (v1.2)
3. extract MANO_RIGHT.pkl and MANO_LEFT.pkl
4. place in project:

```bash
mkdir -p realtime_asha/mano_models
# copy MANO_RIGHT.pkl and MANO_LEFT.pkl here
```

verify:
```bash
ls -lh realtime_asha/mano_models/
# should show MANO_RIGHT.pkl and MANO_LEFT.pkl
```

---

## step 2: convert mano models (python 3.10)

legacy mano .pkl files contain chumpy objects that break in modern python/numpy. convert them to pure numpy format.

### 2.1 setup conversion environment

```bash
# Run the automated setup script
chmod +x setup_convert_env.sh
./setup_convert_env.sh
```

or manually:
```bash
python3.10 -m venv mano_convert_env
source mano_convert_env/bin/activate
pip install --upgrade pip
pip install numpy==1.23.5 scipy==1.10.1 chumpy==0.70
```

### 2.2 run conversion script

```bash
source mano_convert_env/bin/activate
python realtime_asha/src/convert_mano_to_numpy.py
```

expected output:
```
mano model converter (chumpy → numpy)
converting MANO_RIGHT.pkl...
  loaded original file
  converted chumpy objects to numpy
  backed up to MANO_RIGHT.pkl.bak
  saved pickle: MANO_RIGHT_CONVERTED.pkl
successfully converted
```

### 2.3 verify conversion

check converted files exist:
```bash
ls -lh realtime_asha/mano_models/
# Should now show:
#   MANO_RIGHT.pkl.bak           (backup)
#   MANO_RIGHT_CONVERTED.pkl     (converted)
#   MANO_RIGHT_CONVERTED.npz     (converted)
#   MANO_LEFT.pkl.bak
#   MANO_LEFT_CONVERTED.pkl
#   MANO_LEFT_CONVERTED.npz
```

### 2.4 deactivate

```bash
deactivate
```

conversion complete. won't need this environment again unless you get new mano models.

---

## step 3: setup runtime environment (python 3.11)

create the main environment for running the application.

### 3.1 automated setup

```bash
# Run the automated setup script
chmod +x setup_runtime_env.sh
./setup_runtime_env.sh
```

or manually:

```bash
# create virtual environment
python3.11 -m venv asha_env
source asha_env/bin/activate

# install pytorch with mps support for apple silicon
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# install core dependencies
pip install -r realtime_asha/requirements.txt
```

### 3.2 verify mps support

```bash
source asha_env/bin/activate
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

expected output:
```
MPS Available: True
```

if false, system falls back to cpu (slower but functional).

### 3.3 about mano loading

uses custom mano loader (mano_layer.py) that reads converted numpy-based mano files. avoids compatibility issues with manopth and chumpy in python 3.11.

---

## step 4: test mano loading

run test suite to verify everything works:

```bash
source asha_env/bin/activate
python realtime_asha/src/test_mano.py
```

expected output:
```
project asha - mano test suite

testing pytorch device
device: mps
apple silicon mps acceleration available

testing mano model loading
loading mano model (right hand) on device: mps
mano model loaded successfully
faces shape: (1538, 3)
vertices shape: (778, 3)
joints shape: (21, 3)

all tests passed. ready to run.
```

if tests fail, check:
- mano models in realtime_asha/mano_models/
- converted files (*_CONVERTED.pkl) exist
- manopth installed correctly

---

## step 5: run the application

```bash
source asha_env/bin/activate
python realtime_asha/src/realtime_mano.py
```

expected output:
```
project asha - real-time hand pose system
initializing...
loading mano model (right hand) on device: mps
mano model loaded successfully
application ready
camera opened successfully

show your hand to the camera to start tracking
press ctrl+c or close window to exit
```

### what you should see

- left panel: live webcam feed with green dots on hand landmarks
- right panel: 3d mano hand mesh (currently neutral pose)

### controls

- close window or ctrl+c: exit
- camera access: ensure terminal has webcam permissions (system preferences → security & privacy → camera)

---

## troubleshooting

### camera not opening

solution:
- grant terminal webcam permissions: system preferences → security & privacy → camera
- check camera not in use by another app
- try different camera index (edit _cap_index in realtime_mano.py)

### mano loading fails

solution:
- verify converted files exist in realtime_asha/mano_models/
- re-run conversion script
- check manopth installed: pip list | grep manopth

### mps not available

solution:
- ensure you're on apple silicon mac (m1/m2/m3)
- update macos to latest version
- system falls back to cpu (slower but functional)

### import errors

solution:
- ensure runtime environment activated: source asha_env/bin/activate
- reinstall dependencies: pip install -r realtime_asha/requirements.txt

### performance issues

solution:
- close other applications
- check cpu/gpu usage
- verify mps acceleration enabled
- run performance test: python realtime_asha/src/test_mano.py

---

## project structure

```
asha/
├── SETUP.md                          # this file
├── setup_convert_env.sh              # conversion env setup
├── setup_runtime_env.sh              # runtime env setup
├── mano_convert_env/                 # python 3.10 (conversion)
├── asha_env/                         # python 3.11 (runtime)
└── realtime_asha/
    ├── mano_models/
    │   ├── MANO_RIGHT.pkl            # original (from mano website)
    │   ├── MANO_LEFT.pkl             # original
    │   ├── MANO_RIGHT_CONVERTED.pkl  # converted (generated)
    │   └── MANO_LEFT_CONVERTED.pkl   # converted (generated)
    ├── src/
    │   ├── realtime_simple.py        # simple version (recommended)
    │   ├── realtime_mano.py          # mano version (research)
    │   ├── mediapipe_utils.py        # hand landmark detection
    │   ├── mano_layer.py             # custom mano loader
    │   ├── mano_utils.py             # mano model interface
    │   ├── convert_mano_to_numpy.py  # conversion utility
    │   └── test_mano.py              # test suite
    ├── requirements.txt              # python dependencies
    └── README.md                     # project overview
```

---

## quick start (summary)

condensed version for experienced users:

```bash
# 1. Get MANO models (download from MANO website)
mkdir -p realtime_asha/mano_models
# Place MANO_RIGHT.pkl and MANO_LEFT.pkl here

# 2. Convert MANO models (Python 3.10)
./setup_convert_env.sh
source mano_convert_env/bin/activate
python realtime_asha/src/convert_mano_to_numpy.py
deactivate

# 3. Setup runtime environment (Python 3.11)
./setup_runtime_env.sh

# 4. Test
source asha_env/bin/activate
python realtime_asha/src/test_mano.py

# 5. Run
python realtime_asha/src/realtime_mano.py
```

---

## current status & future work

### working features

- real-time hand landmark detection via mediapipe
- dual-panel gui (webcam + 3d visualization)
- mano model loading and mesh generation
- apple silicon mps acceleration
- offscreen 3d rendering with pyrender

### placeholder/todo

- landmark → mano mapping: currently shows neutral pose regardless of gesture
  - need learned mapping from mediapipe landmarks to mano pose parameters
  - current code in mano_utils.py:mano_from_landmarks() is placeholder
- emg integration: future phase will add mindrove 8-channel emg signals

### performance

on apple m3 pro:
- mediapipe: ~30-60 fps (hand detection)
- mano inference: ~100+ fps (neutral pose generation)
- overall system: ~25-35 fps (bottleneck is rendering)

---

## additional resources

- mano: https://mano.is.tue.mpg.de/
- mediapipe hands: https://google.github.io/mediapipe/solutions/hands.html
- manopth: https://github.com/hassony2/manopth
- pytorch mps: https://pytorch.org/docs/stable/notes/mps.html

---

## license & attribution

this project uses:
- mano models (requires registration and license agreement)
- mediapipe (apache license 2.0)
- pytorch (bsd-style license)
- manopth (gnu gpl v2)

mano models require separate licensing and cannot be redistributed. users must download directly from mano website.

---

setup complete.
