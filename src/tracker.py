"""
wraps google's mediapipe hand tracking to get 21 3d keypoints from camera frames.
"""

import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

# create one mediapipe instance (singleton pattern)
# reusing same instance is faster than creating new ones
_hands = mp_hands.Hands(
    static_image_mode=False,  # video mode (not single images)
    max_num_hands=1,  # only track one hand
    model_complexity=1,  # balance between speed and accuracy
    min_detection_confidence=0.7,  # threshold for detecting new hand
    min_tracking_confidence=0.5,  # threshold for tracking existing hand
)


def get_landmarks(frame_bgr: np.ndarray) -> np.ndarray | None:
    """find hand in image and return 21 keypoints with (x,y,z) coords, or none if no hand."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # mediapipe neural network expects rgb
    # opencv gives us bgr, so flip the channels
    rgb = frame_bgr[:, :, ::-1]
    results = _hands.process(rgb)

    # did we find a hand?
    if results.multi_hand_landmarks:
        # get first hand (we only track one anyway)
        lm = results.multi_hand_landmarks[0].landmark
        # convert to numpy array: 21 points, each with (x, y, z)
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    return None


def get_landmarks_world(frame_bgr: np.ndarray) -> np.ndarray | None:
    """
    find hand in image and return 21 keypoints in world coordinates (meters), or none if no hand.

    world coordinates are:
    - metric (meters, not normalized)
    - origin at wrist, right-handed coordinate system
    - more stable depth than image landmarks
    - preferred for 3d pose estimation / MANO fitting

    falls back to image coordinates (normalized 0-1) if world landmarks unavailable.
    includes validation to reject bad detections (span >2m or NaN values).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # mediapipe neural network expects rgb
    rgb = frame_bgr[:, :, ::-1]
    results = _hands.process(rgb)

    # prefer world landmarks (metric coordinates, better for 3d)
    if results.multi_hand_world_landmarks:
        lm = results.multi_hand_world_landmarks[0].landmark
        coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

        # sanity check: reject if scale is unrealistic
        # typical hand span is ~0.2m, reject if >2m (likely bad detection)
        span = coords.max(axis=0) - coords.min(axis=0)
        if np.any(span > 2.0) or np.any(np.isnan(coords)):
            # bad world landmarks, fall through to image coords
            pass
        else:
            return coords

    # fallback: use normalized image coordinates (old behavior)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    return None


def draw_landmarks_on_frame(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    """draw green dots on hand keypoints. modifies frame in-place."""
    if landmarks is None:
        return

    h, w = frame_bgr.shape[:2]

    for x, y, _ in landmarks:
        # convert normalized coords (0-1) to pixel coords
        cx = int(x * w)
        cy = int(y * h)

        # make sure point is inside frame
        if 0 <= cx < w and 0 <= cy < h:
            # draw green dot
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 200, 0), 1)  # darker border for visibility
