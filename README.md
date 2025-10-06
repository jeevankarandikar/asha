# project asha

real-time 3d hand tracking using mediapipe. simple, fast, actually works.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/mediapipe-0.10-green.svg)](https://google.github.io/mediapipe/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## what it does

tracks your hand in 3d space using just a webcam. two views:
- left: camera feed with hand overlay
- right: 3d visualization that moves with your hand

uses google's mediapipe neural network. no inverse kinematics, no parametric models.

---

## features

- real-time 3d hand tracking (60 fps)
- actually moves with your hand
- simple codebase (~250 lines)
- no model files to download
- gpu accelerated on apple silicon

also includes mano version for research (neutral pose, needs ik).

---

## quick start

### simple version (recommended)

```bash
# clone
git clone https://github.com/yourusername/asha.git
cd asha

# setup
./setup_runtime_env.sh
source asha_env/bin/activate

# run
python realtime_asha/src/realtime_simple.py
```

done.

### mano version (research)

if you need parametric mesh (neutral pose only):

```bash
# requires mano models and conversion
# see SETUP.md for details

python realtime_asha/src/realtime_mano.py
```

---

## how it works

```
camera frame
    ↓
mediapipe neural network (google)
    ↓
21 3d keypoints (x, y, z in meters)
    ↓
plot in 3d space
    ↓
done
```

no inverse kinematics. mediapipe does the hard work.

**technical:**
- mediapipe uses cnn for hand detection
- trained on millions of hands
- estimates 3d from single camera
- outputs world landmarks (real-world coordinates)
- we visualize those coordinates

---

## project structure

```
asha/
├── realtime_asha/
│   └── src/
│       ├── realtime_simple.py         # simple version (recommended)
│       ├── realtime_mano.py           # mano version (research)
│       ├── mano_layer.py              # custom mano loader
│       ├── mano_utils.py              # mano interface
│       ├── mediapipe_utils.py         # mediapipe wrapper
│       └── test_mano.py               # tests
├── setup_runtime_env.sh               # environment setup
└── SETUP.md                           # detailed instructions
```

---

## why not mano?

mano requires inverse kinematics:
- mediapipe gives: joint positions
- mano needs: joint rotations

converting position to rotation is hard:
- non-linear
- multiple solutions
- needs optimization (slow) or neural network (training required)
- adds 100ms+ latency

our approach: skip mano, use mediapipe directly
- faster (60 fps vs 10 fps)
- simpler (250 lines vs 1000+)
- works now

trade-off: no parametric mesh. good enough for most use cases.

---

## when to use which version

### use simple version for:
- gesture recognition
- motion capture
- interactive applications
- demos/prototypes
- anything real-time

### use mano version for:
- hand mesh research
- accurate measurements
- hand-object interaction simulation
- publications requiring parametric model

---

## performance

```
component              time       fps
---------------------------------------
mediapipe detection    ~16ms      60
3d plotting            ~1ms       1000
---------------------------------------
total (simple)         ~16ms      60 fps

vs mano version:
mediapipe              ~16ms
ik solver              ~100ms     ← bottleneck
mano forward           ~0.5ms
rendering              ~30ms
---------------------------------------
total (mano)           ~146ms     7 fps
```

---

## requirements

- python 3.11
- webcam
- macos/linux/windows
- (apple silicon recommended for best performance)

dependencies:
- mediapipe
- opencv-python
- matplotlib
- pyqt5
- numpy

for mano version, also need:
- pytorch
- mano models (separate download)

---

## troubleshooting

**camera won't open:**
- check permissions (system preferences → privacy)
- close other apps using camera
- try different camera: edit `_cap_index` in code

**slow performance:**
- check if gpu acceleration is working
- close other gpu-intensive apps
- try lowering camera resolution

**mediapipe not detecting hand:**
- improve lighting
- move hand closer to camera
- adjust detection thresholds in code

---

## license

mit license - see LICENSE file

mano models require separate license from https://mano.is.tue.mpg.de/

---

## credits

- mediapipe by google
- mano hand model by max planck institute
- pytorch for mps acceleration

---

built for real-time hand tracking that actually works.
