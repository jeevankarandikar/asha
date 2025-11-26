"""
FreiHAND dataset loader for Experiment 5.

Loads FreiHAND dataset and runs MANO IK evaluation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from tqdm import tqdm

# Default FreiHAND location
DEFAULT_FREIHAND_PATH = Path.home() / "datasets" / "freihand"


class FreiHANDLoader:
    """
    Loader for FreiHAND dataset.

    FreiHAND structure:
      freihand/
        ├── training/
        │   ├── rgb/              # 00000000.jpg - 00032559.jpg (130K total, 4 views each)
        │   ├── mask/             # segmentation masks
        │   └── ...
        ├── evaluation/
        │   └── rgb/              # evaluation images
        ├── training_xyz.json     # 3D joint positions [32560, 21, 3]
        ├── training_mano.json    # MANO parameters
        ├── training_K.json       # camera intrinsics
        └── ...
    """

    def __init__(self, dataset_path: str = None):
        """
        Initialize FreiHAND loader.

        Args:
            dataset_path: path to freihand root (defaults to ~/datasets/freihand)
        """
        self.root = Path(dataset_path) if dataset_path else DEFAULT_FREIHAND_PATH

        if not self.root.exists():
            raise ValueError(
                f"FreiHAND dataset not found at {self.root}\n"
                f"Please download it first:\n"
                f"  ./scripts/download_datasets.sh freihand\n"
                f"Or manually download from:\n"
                f"  https://lmb.informatik.uni-freiburg.de/resources/datasets/FreiHand.en.html"
            )

        # Load annotations
        self.xyz_path = self.root / "training_xyz.json"
        self.mano_path = self.root / "training_mano.json"
        self.K_path = self.root / "training_K.json"

        if not self.xyz_path.exists():
            raise ValueError(f"Annotations not found: {self.xyz_path}")

        print(f"[FreiHAND] Loading from: {self.root}")

        # Load 3D joint positions (ground truth)
        with open(self.xyz_path, 'r') as f:
            self.xyz = np.array(json.load(f))  # [32560, 21, 3] in meters

        # Load MANO parameters (optional, for reference)
        if self.mano_path.exists():
            with open(self.mano_path, 'r') as f:
                self.mano_params = json.load(f)
        else:
            self.mano_params = None

        # Load camera intrinsics
        if self.K_path.exists():
            with open(self.K_path, 'r') as f:
                self.K = np.array(json.load(f))  # [32560, 3, 3]
        else:
            self.K = None

        # Image directory
        self.rgb_dir = self.root / "training" / "rgb"

        print(f"  → Loaded {len(self.xyz)} samples")
        print(f"  → 3D joints: {self.xyz.shape} (samples, 21 joints, xyz)")

    def __len__(self):
        return len(self.xyz)

    def get_sample(self, idx: int) -> Dict:
        """
        Get a single sample from dataset.

        Args:
            idx: sample index (0-32559 for unique frames, × 4 for views)

        Returns:
            dict with:
                - image: [H, W, 3] RGB image
                - joints_3d: [21, 3] 3D joint positions (meters)
                - K: [3, 3] camera intrinsics (if available)
                - sample_id: int
        """
        # Each sample has 4 views (different camera angles)
        # idx 0-32559 are unique frames, each with 4 views
        base_idx = idx % 32560
        view_idx = idx // 32560  # 0-3

        # Load image (4 views per sample: 00000000-00000003 are views of sample 0)
        img_id = base_idx * 4 + view_idx
        img_path = self.rgb_dir / f"{img_id:08d}.jpg"

        if not img_path.exists():
            raise ValueError(f"Image not found: {img_path}")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ground truth 3D joints
        joints_3d = self.xyz[base_idx]  # [21, 3]

        sample = {
            'image': image,
            'joints_3d': joints_3d,
            'sample_id': idx,
            'base_id': base_idx,
            'view_id': view_idx,
        }

        if self.K is not None:
            sample['K'] = self.K[base_idx]

        return sample

    def get_subset(self, num_samples: int = 100, seed: int = 42) -> List[int]:
        """
        Get random subset of sample indices (for quick evaluation).

        Args:
            num_samples: number of samples to select
            seed: random seed

        Returns:
            list of sample indices
        """
        rng = np.random.default_rng(seed)
        # Only use first view (view_id=0) for simplicity
        max_samples = min(num_samples, 32560)
        indices = rng.choice(32560, size=max_samples, replace=False)
        return sorted(indices.tolist())


# ============================================================
# Evaluation utilities
# ============================================================

def evaluate_on_freihand(
    freihand_path: str = None,
    num_samples: int = 100,
    config = None,
    verbose: bool = True
) -> Dict:
    """
    Run MANO IK evaluation on FreiHAND dataset.

    Args:
        freihand_path: path to FreiHAND dataset
        num_samples: number of samples to evaluate (default: 100)
        config: ExperimentConfig (uses baseline if None)
        verbose: print progress

    Returns:
        results dict with metrics
    """
    from experiments.pose_fitter_experimental import ExperimentalIKFitter, ExperimentConfig
    from model_utils.tracker import get_landmarks_world

    # Load dataset
    loader = FreiHANDLoader(freihand_path)

    # Get subset
    subset_indices = loader.get_subset(num_samples)

    if verbose:
        print(f"\n[FreiHAND Evaluation]")
        print(f"  Samples: {num_samples}")

    # Initialize fitter
    config = config or ExperimentConfig()
    fitter = ExperimentalIKFitter(config)

    # Metrics
    detection_count = 0
    ik_errors = []
    per_joint_errors = []
    gt_errors = []  # error vs. FreiHAND ground truth

    # Process samples
    iterator = tqdm(subset_indices) if verbose else subset_indices

    for idx in iterator:
        sample = loader.get_sample(idx)
        image = sample['image']
        gt_joints_3d = sample['joints_3d']  # FreiHAND ground truth

        # MediaPipe detection
        result = get_landmarks_world(image)
        if result is None:
            continue

        mp_landmarks, mp_confidence = result
        detection_count += 1

        # MANO IK fitting
        verts, joints, theta, metrics = fitter.fit(mp_landmarks)

        # IK error (vs. MediaPipe)
        ik_errors.append(metrics['ik_error_mm'])
        per_joint_errors.append(metrics['per_joint_error_mm'])

        # Ground truth error (vs. FreiHAND)
        # Note: Need to align our predicted joints to FreiHAND ground truth
        # (different coordinate frames)
        # For now, skip this (would need careful alignment)

    if detection_count == 0:
        print("  ⚠ No hands detected by MediaPipe!")
        return {'error': 'no_detections'}

    # Compute statistics
    ik_errors = np.array(ik_errors)
    per_joint_errors = np.array(per_joint_errors)

    results = {
        'dataset': 'FreiHAND',
        'num_samples': num_samples,
        'num_detected': detection_count,
        'detection_rate': detection_count / num_samples,
        'mean_ik_error_mm': float(ik_errors.mean()),
        'std_ik_error_mm': float(ik_errors.std()),
        'median_ik_error_mm': float(np.median(ik_errors)),
        'percentile_95_mm': float(np.percentile(ik_errors, 95)),
        'per_joint_errors': per_joint_errors.tolist(),
    }

    if verbose:
        print(f"\nResults:")
        print(f"  Detection rate: {results['detection_rate']*100:.1f}%")
        print(f"  Mean IK error: {results['mean_ik_error_mm']:.2f} mm")
        print(f"  Median IK error: {results['median_ik_error_mm']:.2f} mm")
        print(f"  95th percentile: {results['percentile_95_mm']:.2f} mm")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FreiHAND dataset evaluation")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to FreiHAND dataset (default: ~/datasets/freihand)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/freihand_eval.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_on_freihand(
        freihand_path=args.dataset_path,
        num_samples=args.num_samples,
        verbose=True
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
