#  Project Asha — Real-Time MediaPipe → MANO Hand Pose

Real-time 3D hand pose reconstruction system combining MediaPipe hand tracking with MANO parametric hand model.

## Features

-  Real-time hand landmark detection via MediaPipe
-  MANO 3D hand mesh generation
-  Dual-panel GUI (webcam + 3D visualization)
-  Apple Silicon MPS acceleration
-  Cross-platform (macOS, Linux, Windows)

## Quick Start

```bash
# Setup (one-time)
./setup_runtime_env.sh

# Activate environment
source asha_env/bin/activate

# Run application
python realtime_asha/src/realtime_mano.py
```

## Full Setup Instructions

** See [SETUP.md](../SETUP.md) for complete setup guide**, including:
- MANO model download and conversion
- Environment setup (Python 3.10 + 3.11)
- Troubleshooting
- Performance optimization

## Project Structure

```
realtime_asha/
├── mano_models/              # MANO model files (download separately)
│   ├── MANO_RIGHT.pkl
│   ├── MANO_LEFT.pkl
│   ├── MANO_RIGHT_CONVERTED.pkl  (generated)
│   └── MANO_LEFT_CONVERTED.pkl   (generated)
├── src/
│   ├── realtime_mano.py      # Main application
│   ├── mediapipe_utils.py    # Hand landmark detection
│   ├── mano_utils.py         # MANO model interface
│   ├── convert_mano_to_numpy.py  # MANO conversion utility
│   └── test_mano.py          # Test suite
├── requirements.txt
└── README.md
```

## Requirements

- **Python 3.11** (runtime)
- **Python 3.10** (MANO model conversion, one-time)
- **Webcam**
- **MANO models** ([download here](https://mano.is.tue.mpg.de/download.php))

## System Requirements

- macOS (Apple Silicon recommended for MPS acceleration)
- Linux (CPU/CUDA)
- Windows (CPU/CUDA)

## Performance

On Apple M3 Pro:
- MediaPipe: ~30-60 FPS
- MANO inference: ~100+ FPS
- Overall system: ~25-35 FPS

## Current Status

**Working:**
- Hand tracking and landmark detection
- 3D mesh visualization
- Real-time rendering

**TODO:**
- Implement landmark → MANO pose parameter mapping (currently shows neutral pose)
- EMG signal integration (future phase)

## License

This project uses MANO models which require separate licensing. Download models from the [MANO website](https://mano.is.tue.mpg.de/download.php).

## Links

- **Setup Guide**: [SETUP.md](../SETUP.md)
- **Developer Docs**: [CLAUDE.md](../CLAUDE.md)
- **MANO Models**: https://mano.is.tue.mpg.de/
- **MediaPipe**: https://google.github.io/mediapipe/solutions/hands.html
