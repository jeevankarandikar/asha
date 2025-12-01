"""
experiment runner for cis 6800 final project.

runs the 5 planned experiments from proposal:
  1. loss ablation
  2. alignment methods comparison
  3. optimizer comparison
  4. per-joint error analysis
  5. public dataset evaluation

usage:
  python run_experiments.py --video <path_to_video>
  python run_experiments.py --experiments 1,2,3  # run specific experiments
  python run_experiments.py --quick  # fast test mode
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from typing import Dict, List
import numpy as np
import cv2

from experiments.pose_fitter_experimental import ExperimentalIKFitter, ExperimentConfig
from model_utils.tracker import get_landmarks_world


# ============================================================
# experiment 1: loss ablation
# ============================================================

def experiment_1_loss_ablation(video_path: str, output_dir: Path) -> Dict:
    """
    test which loss terms matter most.

    configurations to test:
      - baseline: all losses enabled
      - no_bone: disable bone direction loss
      - no_temporal: disable temporal smoothness
      - no_reg: disable regularization
      - position_only: only position loss
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: LOSS ABLATION")
    print("=" * 60)

    configs = {
        "baseline": ExperimentConfig(
            use_position_loss=True,
            use_bone_loss=True,
            use_temporal_loss=True,
            use_regularization=True,
        ),
        "no_bone": ExperimentConfig(
            use_position_loss=True,
            use_bone_loss=False,
            use_temporal_loss=True,
            use_regularization=True,
        ),
        "no_temporal": ExperimentConfig(
            use_position_loss=True,
            use_bone_loss=True,
            use_temporal_loss=False,
            use_regularization=True,
        ),
        "no_reg": ExperimentConfig(
            use_position_loss=True,
            use_bone_loss=True,
            use_temporal_loss=True,
            use_regularization=False,
        ),
        "position_only": ExperimentConfig(
            use_position_loss=True,
            use_bone_loss=False,
            use_temporal_loss=False,
            use_regularization=False,
        ),
    }

    results = {}
    for name, config in configs.items():
        print(f"\n[{name}] Running...")
        result = run_on_video(video_path, config)
        results[name] = result
        print(f"  Mean IK error: {result['mean_ik_error_mm']:.2f} mm")
        print(f"  Median IK error: {result['median_ik_error_mm']:.2f} mm")
        print(f"  Converged: {result['convergence_rate']*100:.1f}%")

    # save results
    output_file = output_dir / "exp1_loss_ablation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    return results


# ============================================================
# experiment 2: alignment methods
# ============================================================

def experiment_2_alignment_methods(video_path: str, output_dir: Path) -> Dict:
    """
    compare different alignment methods:
      - umeyama (similarity transform with scale)
      - kabsch (rigid-only, no scale)
      - umeyama_rigid (umeyama without scale)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: ALIGNMENT METHODS")
    print("=" * 60)

    configs = {
        "umeyama_scaled": ExperimentConfig(
            alignment_method="umeyama",
            use_scale=True,
        ),
        "umeyama_rigid": ExperimentConfig(
            alignment_method="umeyama",
            use_scale=False,
        ),
        "kabsch": ExperimentConfig(
            alignment_method="kabsch",
            use_scale=False,
        ),
    }

    results = {}
    for name, config in configs.items():
        print(f"\n[{name}] Running...")
        result = run_on_video(video_path, config)
        results[name] = result
        print(f"  Mean IK error: {result['mean_ik_error_mm']:.2f} mm")
        print(f"  Mean scale: {result.get('mean_scale', 1.0):.3f}")

    # save results
    output_file = output_dir / "exp2_alignment_methods.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    return results


# ============================================================
# experiment 3: optimizer comparison
# ============================================================

def experiment_3_optimizer_comparison(video_path: str, output_dir: Path) -> Dict:
    """
    compare optimizers:
      - adam (default)
      - sgd
      - lbfgs
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: OPTIMIZER COMPARISON")
    print("=" * 60)

    configs = {
        "adam": ExperimentConfig(
            optimizer_type="adam",
            learning_rate=1e-2,
        ),
        "sgd": ExperimentConfig(
            optimizer_type="sgd",
            learning_rate=1e-2,
        ),
        "lbfgs": ExperimentConfig(
            optimizer_type="lbfgs",
            learning_rate=1.0,  # lbfgs typically uses lr=1
            num_iterations=10,  # lbfgs does line search
        ),
    }

    results = {}
    for name, config in configs.items():
        print(f"\n[{name}] Running...")
        result = run_on_video(video_path, config)
        results[name] = result
        print(f"  Mean IK error: {result['mean_ik_error_mm']:.2f} mm")
        print(f"  Mean time/frame: {result['mean_time_ms']:.1f} ms")

    # save results
    output_file = output_dir / "exp3_optimizer_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    return results


# ============================================================
# experiment 4: per-joint error analysis
# ============================================================

def experiment_4_per_joint_analysis(video_path: str, output_dir: Path) -> Dict:
    """
    analyze error distribution across 21 joints.

    identify:
      - which joints have highest error
      - failure modes (thumb vs. fingers)
      - confidence correlation
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: PER-JOINT ERROR ANALYSIS")
    print("=" * 60)

    config = ExperimentConfig()  # baseline config
    result = run_on_video(video_path, config, collect_per_joint=True)

    # analyze per-joint errors
    per_joint_errors = np.array(result['per_joint_errors'])  # [num_frames, 21]
    joint_names = [
        "wrist",
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    ]

    joint_stats = {}
    for i, name in enumerate(joint_names):
        errors = per_joint_errors[:, i]
        joint_stats[name] = {
            'mean_mm': float(errors.mean()),
            'std_mm': float(errors.std()),
            'median_mm': float(np.median(errors)),
            '95th_percentile_mm': float(np.percentile(errors, 95)),
        }

    result['joint_stats'] = joint_stats

    # identify worst joints
    mean_errors = [(name, stats['mean_mm']) for name, stats in joint_stats.items()]
    mean_errors.sort(key=lambda x: x[1], reverse=True)

    print("\nWorst 5 joints (by mean error):")
    for i, (name, error) in enumerate(mean_errors[:5]):
        print(f"  {i+1}. {name}: {error:.2f} mm")

    # save results
    output_file = output_dir / "exp4_per_joint_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    return result


# ============================================================
# experiment 5: public dataset evaluation (placeholder)
# ============================================================

def experiment_5_public_dataset(dataset_name: str, output_dir: Path) -> Dict:
    """
    evaluate on public datasets (freihand, ho-3d).

    NOTE: this requires downloading and preprocessing the datasets.
    placeholder for now.
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT 5: PUBLIC DATASET EVALUATION ({dataset_name})")
    print("=" * 60)

    print("\n⚠ This experiment requires public dataset setup:")
    print("  1. Download FreiHAND or HO-3D dataset")
    print("  2. Extract and preprocess")
    print("  3. Implement dataset loader")
    print("\nSkipping for now (placeholder).")

    return {"status": "not_implemented"}


# ============================================================
# video processing utilities
# ============================================================

def run_on_video(
    video_path: str,
    config: ExperimentConfig,
    max_frames: int = None,
    collect_per_joint: bool = False,
) -> Dict:
    """
    run ik fitting on video with given config.

    args:
      video_path: path to video file
      config: experiment configuration
      max_frames: limit number of frames (for quick tests)
      collect_per_joint: collect per-joint errors (expensive)

    returns:
      results dict with metrics
    """
    # load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"  Video: {Path(video_path).name}")
    print(f"  FPS: {fps:.1f}, Frames: {total_frames}")

    # initialize fitter
    fitter = ExperimentalIKFitter(config)

    # metrics collection
    ik_errors = []
    per_joint_errors = []
    scales = []
    times = []
    converged_count = 0

    # process frames
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # mediapipe detection
        result = get_landmarks_world(frame)
        if result is None:
            continue
        landmarks, confidence = result

        # ik fitting
        start_time = time.perf_counter()
        verts, joints, theta, metrics = fitter.fit(landmarks)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # collect metrics
        ik_errors.append(metrics['ik_error_mm'])
        times.append(elapsed_ms)
        if 'scale' in metrics:
            scales.append(metrics['scale'])
        if metrics['converged']:
            converged_count += 1

        if collect_per_joint:
            per_joint_errors.append(metrics['per_joint_error_mm'])

        # progress
        if (frame_idx + 1) % 50 == 0 or frame_idx == total_frames - 1:
            print(f"  Processed {frame_idx + 1}/{total_frames} frames...", end='\r')

    print()  # newline after progress

    cap.release()

    # compute statistics
    ik_errors = np.array(ik_errors)
    results = {
        'num_frames': len(ik_errors),
        'mean_ik_error_mm': float(ik_errors.mean()),
        'std_ik_error_mm': float(ik_errors.std()),
        'median_ik_error_mm': float(np.median(ik_errors)),
        'percentile_95_mm': float(np.percentile(ik_errors, 95)),
        'convergence_rate': converged_count / len(ik_errors),
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
    }

    if scales:
        results['mean_scale'] = float(np.mean(scales))
        results['std_scale'] = float(np.std(scales))

    if per_joint_errors:
        results['per_joint_errors'] = [e.tolist() for e in per_joint_errors]

    return results


# ============================================================
# main cli
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run CIS 6800 experiments")

    parser.add_argument(
        "--video",
        type=str,
        required=False,
        help="Path to input video file"
    )

    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3,4",
        help="Comma-separated experiment numbers (e.g., '1,2,3')"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/experiments",
        help="Output directory for results"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: process only 100 frames"
    )

    args = parser.parse_args()

    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # parse experiment numbers
    exp_nums = [int(x.strip()) for x in args.experiments.split(",")]

    print("\n" + "=" * 60)
    print("CIS 6800 FINAL PROJECT - EXPERIMENT RUNNER")
    print("3D Hand Pose Estimation via Multi-Term MANO Optimization")
    print("=" * 60)
    print(f"Experiments to run: {exp_nums}")
    print(f"Output directory: {output_dir}")
    if args.quick:
        print("Quick mode: 100 frames")

    # check video
    if args.video:
        video_path = args.video
        if not Path(video_path).exists():
            print(f"\n❌ Error: Video file not found: {video_path}")
            return
    else:
        # try to find test video
        test_videos = list(Path(".").glob("**/test*.mp4")) + list(Path(".").glob("**/validation*.mp4"))
        if test_videos:
            video_path = str(test_videos[0])
            print(f"\nUsing test video: {video_path}")
        else:
            print("\n❌ Error: No video specified and no test video found.")
            print("Usage: python run_experiments.py --video <path_to_video>")
            return

    max_frames = 100 if args.quick else None

    # run experiments
    all_results = {}

    if 1 in exp_nums:
        results = experiment_1_loss_ablation(video_path, output_dir)
        all_results['exp1'] = results

    if 2 in exp_nums:
        results = experiment_2_alignment_methods(video_path, output_dir)
        all_results['exp2'] = results

    if 3 in exp_nums:
        results = experiment_3_optimizer_comparison(video_path, output_dir)
        all_results['exp3'] = results

    if 4 in exp_nums:
        results = experiment_4_per_joint_analysis(video_path, output_dir)
        all_results['exp4'] = results

    if 5 in exp_nums:
        results = experiment_5_public_dataset("FreiHAND", output_dir)
        all_results['exp5'] = results

    # save combined results
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
