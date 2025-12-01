#!/usr/bin/env python3
"""
extract performance metrics from v1 system.
runs mano IK tracking and collects real performance metrics.
press 'q' when done, metrics auto-saved to json.

usage:
    python src/extract_results.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import numpy as np
import cv2

from model_utils.tracker import get_landmarks_world, get_landmarks
from model_utils.pose_fitter import mano_from_landmarks

class MetricsCollector:
    """collect metrics during v1 operation."""
    def __init__(self):
        self.ik_errors = []
        self.confidences = []
        self.frame_times = []
        self.start_time = time.time()

    def add_frame(self, frame_time, ik_error=None, confidence=None):
        """add frame metrics."""
        self.frame_times.append(frame_time)
        if ik_error is not None:
            self.ik_errors.append(ik_error)
        if confidence is not None:
            self.confidences.append(confidence)

    def get_metrics(self):
        """compute final statistics."""
        if len(self.ik_errors) == 0:
            return None

        ik_errors = np.array(self.ik_errors)
        confidences = np.array(self.confidences)
        frame_times = np.array(self.frame_times)

        return {
            # ik error stats
            'ik_error_mean': float(np.mean(ik_errors)),
            'ik_error_median': float(np.median(ik_errors)),
            'ik_error_std': float(np.std(ik_errors)),
            'ik_error_95th': float(np.percentile(ik_errors, 95)),
            'ik_errors_all': ik_errors.tolist(),

            # confidence stats
            'confidence_mean': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'confidences_all': confidences.tolist(),

            # performance
            'fps_mean': float(1.0 / np.mean(frame_times)),
            'frame_time_mean_ms': float(np.mean(frame_times) * 1000),

            # quality filtering
            'total_frames': len(self.frame_times),
            'valid_poses': len(ik_errors),
            'detection_rate': float(len(ik_errors) / len(self.frame_times)),
            'high_confidence': int(np.sum(confidences > 0.7)),
            'low_ik_error': int(np.sum(ik_errors < 25)),
            'passes_both': int(np.sum((confidences > 0.7) & (ik_errors < 25))),
            'retention_rate': float(np.sum((confidences > 0.7) & (ik_errors < 25)) / len(ik_errors)),

            # session info
            'duration_sec': float(time.time() - self.start_time),
        }


def run_v1_with_metrics():
    """run v1 and collect metrics. press 'q' or close window to finish."""
    print("="*60)
    print("extracting results from v1 system")
    print("="*60)
    print("\ninstructions:")
    print("  1. position your hand in front of camera")
    print("  2. move through various poses (fist, open, point, etc.)")
    print("  3. press 'q' or close window when done")
    print("  4. metrics will be saved to docs/v1_metrics.json")
    print("\nstarting in 3 seconds...")
    time.sleep(3)

    collector = MetricsCollector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("error: cannot open camera")
        return None

    print("\ncamera opened")
    print("collecting metrics... (press 'q' to finish)\n")

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # get landmarks
            result = get_landmarks_world(frame)
            if result is None:
                result = get_landmarks(frame)

            ik_error = None
            confidence = None

            if result is not None:
                landmarks, confidence = result

                # draw landmarks
                for lm in landmarks:
                    x, y = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # run mano ik (returns: verts, joints, theta, ik_error)
                verts, joints, theta, ik_error = mano_from_landmarks(landmarks)

            # collect metrics
            frame_time = time.time() - frame_start
            collector.add_frame(frame_time, ik_error, confidence)

            # display info
            fps = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(frame, f"fps: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"frames: {len(collector.frame_times)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if ik_error is not None:
                cv2.putText(frame, f"ik error: {ik_error:.1f}mm", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('metrics collection (v1) - press q to finish', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return collector.get_metrics()


def save_metrics(metrics, filepath='docs/v1_metrics.json'):
    """save metrics to json."""
    Path(filepath).parent.mkdir(exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nmetrics saved to {filepath}")


def print_summary(metrics):
    """print metrics summary."""
    if metrics is None:
        return

    print("\n" + "="*60)
    print("metrics summary")
    print("="*60)
    print(f"\nik error:")
    print(f"   mean: {metrics['ik_error_mean']:.1f}mm")
    print(f"   median: {metrics['ik_error_median']:.1f}mm")
    print(f"   95th: {metrics['ik_error_95th']:.1f}mm")
    print(f"\nperformance:")
    print(f"   fps: {metrics['fps_mean']:.1f}")
    print(f"   frame time: {metrics['frame_time_mean_ms']:.1f}ms")
    print(f"\nquality:")
    print(f"   total frames: {metrics['total_frames']}")
    print(f"   valid poses: {metrics['valid_poses']}")
    print(f"   retention rate: {metrics['retention_rate']*100:.1f}%")
    print(f"\nsession: {metrics['duration_sec']:.0f}s")
    print("="*60)


if __name__ == "__main__":
    metrics = run_v1_with_metrics()

    if metrics:
        save_metrics(metrics)
        print_summary(metrics)
        print("\ndone! next step:")
        print("   python src/generate_plots.py")
    else:
        print("\nno metrics collected. check camera and hand detection.")
        sys.exit(1)
