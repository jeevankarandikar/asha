# v0: MediaPipe Baseline

**Status**: ✅ Complete (Reference Implementation)

**Approach**: Real-time hand tracking using Google MediaPipe Hands (v0.10+).

## Overview

v0 establishes the **baseline** for Project Asha - off-the-shelf camera-based hand tracking without EMG. Serves as ground truth for evaluating EMG-based approaches (v4, v5).

**Key Purpose**: Provide reference implementation and performance ceiling for camera-based systems.

## Architecture

**MediaPipe Hands**:
- Pre-trained CNN landmark detector (21 keypoints)
- World coordinates (3D positions in meters)
- Screen coordinates (2D normalized [0, 1])
- Confidence scores per landmark

**Output**: 21 hand landmarks (MCP, PIP, DIP, TIP for each finger + wrist)

## Performance

**Frame Rate**: 60 fps (webcam-limited, MediaPipe runs >100fps on CPU)
**Latency**: <10ms (end-to-end from camera to display)
**Accuracy**: MediaPipe's reported accuracy ~95% landmark detection in good lighting

**Strengths**:
- ✅ High frame rate, low latency
- ✅ Robust to hand orientation and occlusions
- ✅ No training required (pre-trained model)
- ✅ CPU-only (no GPU needed)

**Limitations**:
- ❌ Requires camera (not suitable for prosthetics or VR)
- ❌ Sensitive to lighting conditions
- ❌ Privacy concerns (camera-based)
- ❌ Limited depth accuracy (monocular camera)

## Files

- **mediapipe_demo.py**: Real-time hand tracking visualization with MediaPipe

## Quick Start

### Running

```bash
# Activate environment
source asha_env/bin/activate

# Run MediaPipe demo
python -m asha.v0.mediapipe_demo
```

**Controls**:
- ESC: Quit application
- Webcam displays with hand landmarks overlay
- 3D visualization shows hand pose in world coordinates

## Use Cases

1. **Baseline Comparison**: Compare EMG methods (v4, v5) against camera-based gold standard
2. **Data Collection**: Use MediaPipe landmarks as pseudo-labels for v1 IK
3. **Testing**: Verify camera and visualization pipeline
4. **Demonstration**: Show what's possible with camera-based tracking

## Technical Details

**MediaPipe Hands Configuration**:
- `model_complexity=1`: High-accuracy model (vs 0=lite, 2=full)
- `min_detection_confidence=0.3`: Lower threshold for detection (increased from 0.7 after debugging)
- `min_tracking_confidence=0.3`: Lower threshold for tracking (increased from 0.5 after debugging)
- `max_num_hands=1`: Track single hand only

**Coordinate Systems**:
- **Screen**: Normalized 2D [0, 1] × [0, 1] (x, y)
- **World**: 3D meters relative to wrist (x, y, z)

## Integration with v1

v0 → v1 pipeline:
1. MediaPipe detects 21 hand landmarks (v0)
2. MANO IK fits parametric hand model to landmarks (v1)
3. MANO LBS generates 778 mesh vertices (v1)
4. Result: Full 3D hand mesh from camera input

## Timeline

- **Oct 5, 2024**: v0 complete, baseline established
- **Oct 7, 2024**: Integrated with v1 MANO IK pipeline
- **Dec 13, 2024**: Fixed MediaPipe confidence thresholds (0.7→0.3, 0.5→0.3)
- **Dec 14, 2024**: Codebase restructure (v0 extracted to asha/v0/)

## Next Steps

v0 is feature-complete and requires no further development. Serves as reference for:
- Comparing EMG-based methods (v4, v5)
- Collecting training data via v1 IK pseudo-labels
- Demonstrating camera-based tracking capabilities

**Recommendation**: Use v0 as baseline, focus development on EMG methods (v5 priority).
