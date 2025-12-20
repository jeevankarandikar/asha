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
    min_detection_confidence=0.3,  # threshold for detecting new hand (lowered for better detection)
    min_tracking_confidence=0.3,  # threshold for tracking existing hand (lowered for stability)
)


def get_landmarks(frame_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    find hand in image and return 21 keypoints with (x,y,z) coords, or none if no hand.

    returns:
      (landmarks, confidence): landmarks [21, 3] and confidence score [0-1]
      or None if no hand detected
    """
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
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

        # get confidence score from multi_handedness
        confidence = 0.0
        if results.multi_handedness and len(results.multi_handedness) > 0:
            confidence = results.multi_handedness[0].classification[0].score

        return landmarks, confidence

    return None


def get_landmarks_world(frame_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    find hand in image and return 21 keypoints in world coordinates (meters), or none if no hand.

    world coordinates are:
    - metric (meters, not normalized)
    - origin at wrist, right-handed coordinate system
    - more stable depth than image landmarks
    - preferred for 3d pose estimation / MANO fitting

    falls back to image coordinates (normalized 0-1) if world landmarks unavailable.
    includes validation to reject bad detections (span >2m or NaN values).

    returns:
      (landmarks, confidence): landmarks [21, 3] and confidence score [0-1]
      or None if no hand detected
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    # mediapipe neural network expects rgb
    rgb = frame_bgr[:, :, ::-1]
    results = _hands.process(rgb)

    # get confidence score (same for both world and image landmarks)
    confidence = 0.0
    if results.multi_handedness and len(results.multi_handedness) > 0:
        confidence = results.multi_handedness[0].classification[0].score

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
            return coords, confidence

    # fallback: use normalized image coordinates (old behavior)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        landmarks = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return landmarks, confidence

    return None


def draw_landmarks_on_frame(frame_bgr: np.ndarray, landmarks: np.ndarray, confidence: float = 0.0) -> None:
    """
    draw green dots on hand keypoints with optional confidence overlay. modifies frame in-place.

    args:
      frame_bgr: image frame
      landmarks: [21, 3] hand keypoints
      confidence: optional confidence score to display (0-1)
    """
    if landmarks is None:
        return

    h, w = frame_bgr.shape[:2]

    # MediaPipe hand connections (bone structure)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm connections
    ]

    # Draw connections (bones) first so dots appear on top
    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1, _ = landmarks[start_idx]
        x2, y2, _ = landmarks[end_idx]

        # Convert to pixel coords
        cx1, cy1 = int(x1 * w), int(y1 * h)
        cx2, cy2 = int(x2 * w), int(y2 * h)

        # Check both points are in frame
        if (0 <= cx1 < w and 0 <= cy1 < h and
            0 <= cx2 < w and 0 <= cy2 < h):
            # Draw line (green, thicker than original)
            cv2.line(frame_bgr, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

    # Draw landmarks (dots) on top of connections
    for x, y, _ in landmarks:
        # convert normalized coords (0-1) to pixel coords
        cx = int(x * w)
        cy = int(y * h)

        # make sure point is inside frame
        if 0 <= cx < w and 0 <= cy < h:
            # draw green dot
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 200, 0), 1)  # darker border for visibility

    # draw confidence score in top-left corner
    if confidence > 0:
        text = f"MP Conf: {confidence:.2f}"
        cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
