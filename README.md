# asha

real-time 3d hand tracking with mano parametric model

## quick start

```bash
git clone https://github.com/jeevankarandikar/asha.git
cd asha
./setup_runtime_env.sh
source asha_env/bin/activate
python src/mano_v1.py
```

## versions

- **mediapipe_v0** - mediapipe only (scatter plot, ~60 fps)
- **mano_v1** - full ik + lbs (articulation, ~25 fps)

## files

- **mediapipe_v0.py** - mediapipe baseline
- **mano_v1.py** - full ik + articulation
- **tracker.py** - hand tracking wrapper
- **mano_model.py** - custom mano loader with lbs
- **pose_fitter.py** - mano interface + ik optimization
- **test.py** - run all tests
- **test_mediapipe_v0.py, test_mano_v1.py** - version-specific tests

## requirements

- python 3.11
- webcam

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
