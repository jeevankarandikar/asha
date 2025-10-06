# asha

real-time 3d hand tracking

## quick start

```bash
git clone https://github.com/jeevankarandikar/asha.git
cd asha
./setup_runtime_env.sh
source asha_env/bin/activate
python realtime_asha/src/realtime_simple.py
```

## files

- **realtime_simple.py** - mediapipe only (recommended, 60 fps)
- **realtime_mano.py** - with mano parametric model (needs mano setup below)
- **mediapipe_utils.py** - hand tracking wrapper
- **mano_layer.py** - custom mano loader
- **mano_utils.py** - mano interface
- **test_mano.py** - tests

## requirements

- python 3.11
- webcam

## mano setup (optional)

only needed for realtime_mano.py

### 1. download mano models

- get from https://mano.is.tue.mpg.de/download.php
- place MANO_RIGHT.pkl and MANO_LEFT.pkl in `realtime_asha/mano_models/`

### 2. convert models (python 3.10)

```bash
./setup_convert_env.sh
source mano_convert_env/bin/activate
python realtime_asha/src/convert_mano_to_numpy.py
deactivate
```

### 3. test

```bash
source asha_env/bin/activate
python realtime_asha/src/test_mano.py
```

### 4. run

```bash
python realtime_asha/src/realtime_mano.py
```

## troubleshooting

- **camera won't open** - check permissions in system preferences
- **mps not available** - need apple silicon, falls back to cpu
- **mano loading fails** - run conversion script above
