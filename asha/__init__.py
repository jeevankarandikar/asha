"""
Project Asha - EMG-to-Hand-Pose System

A real-time EMG-to-hand-pose system for prosthetic control and gesture recognition.
Predicts 3D hand poses from surface EMG muscle signals, enabling camera-free hand tracking.

Version History:
- v0: MediaPipe baseline (21 keypoints @ 60fps)
- v1: MANO IK + LBS (9.71mm MPJPE @ 25fps)
- v2: Image → θ training (ResNet/Transformer, 47mm - PAUSED)
- v3: Transfer learning (emg2pose adapter - ABANDONED)
- v4: EMG → MANO θ (SimpleEMGModel, 34.92mm MPJPE)
- v5: EMG → Joints (EMGToJointsModel, 11.95mm MPJPE - BEST!)
"""

__version__ = "0.5.0"
__author__ = "Jeevan Karandikar"
__license__ = "MIT"

# Public API exports (will be populated after migration)
__all__ = [
    "__version__",
]
