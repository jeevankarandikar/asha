"""
HO-3D dataset loader for Experiment 5.

Loads HO-3D dataset (hand-object interaction) and runs MANO IK evaluation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from tqdm import tqdm

# Default HO-3D location
DEFAULT_HO3D_PATH = Path.home() / "datasets" / "ho3d"


class HO3DLoader:
    """
    Loader for HO-3D dataset (hand-object 3D poses).

    HO-3D structure:
      ho3d/
        ├── train/
        │   ├── ABF10/           # Sequence folders
        │   │   ├── rgb/         # RGB images (00000.png, 00001.png, ...)
        │   │   ├── depth/       # Depth maps
        │   │   └── meta/        # Annotations (00000.pkl, 00001.pkl, ...)
        │   ├── BB10/
        │   └── ...              # 10 training sequences
        └── evaluation/
            └── ...              # Evaluation sequences
    """

    def __init__(self, dataset_path: str = None, split: str = "train"):
        """
        Initialize HO-3D loader.

        Args:
            dataset_path: path to ho3d root (defaults to ~/datasets/ho3d)
            split: 'train' or 'evaluation'
        """
        self.root = Path(dataset_path) if dataset_path else DEFAULT_HO3D_PATH
        self.split = split
        self.split_dir = self.root / split

        if not self.root.exists():
            raise ValueError(
                f"HO-3D dataset not found at {self.root}\n"
                f"Please download it first:\n"
                f"  kaggle datasets download -d marcmarais/ho3d-v3 -p ~/datasets/ho3d\n"
                f"Or manually download from:\n"
                f"  https://www.kaggle.com/datasets/marcmarais/ho3d-v3"
            )

        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        # Collect all sequences
        self.sequences = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])

        if not self.sequences:
            raise ValueError(f"No sequences found in {self.split_dir}")

        print(f"[HO-3D] Loading from: {self.root}")
        print(f"  Split: {split}")
        print(f"  Sequences: {len(self.sequences)}")

        # Build index of all frames
        self.frame_index = []  # (sequence_path, frame_number)
        for seq_dir in self.sequences:
            rgb_dir = seq_dir / "rgb"
            if not rgb_dir.exists():
                continue

            # Count frames in this sequence
            frames = sorted(rgb_dir.glob("*.jpg"))
            for frame_path in frames:
                frame_num = int(frame_path.stem)
                self.frame_index.append((seq_dir, frame_num))

        print(f"  → Total frames: {len(self.frame_index)}")

    def __len__(self):
        return len(self.frame_index)

    def get_sample(self, idx: int) -> Dict:
        """
        Get a single sample from dataset.

        Args:
            idx: sample index

        Returns:
            dict with:
                - image: [H, W, 3] RGB image
                - joints_3d: [21, 3] 3D joint positions (meters) if available
                - sample_id: int
                - sequence: str (sequence name)
        """
        if idx >= len(self.frame_index):
            raise IndexError(f"Index {idx} out of range (max: {len(self.frame_index)})")

        seq_dir, frame_num = self.frame_index[idx]
        seq_name = seq_dir.name

        # Load RGB image
        img_path = seq_dir / "rgb" / f"{frame_num:04d}.jpg"
        if not img_path.exists():
            raise ValueError(f"Image not found: {img_path}")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {
            'image': image,
            'sample_id': idx,
            'sequence': seq_name,
            'frame_number': frame_num,
        }

        # Try to load annotations (meta files)
        meta_path = seq_dir / "meta" / f"{frame_num:05d}.pkl"
        if meta_path.exists():
            try:
                import pickle
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)

                # HO-3D annotations contain hand joint positions
                if 'handJoints3D' in meta:
                    sample['joints_3d'] = np.array(meta['handJoints3D'])  # [21, 3]

                if 'camMat' in meta:
                    sample['K'] = np.array(meta['camMat'])  # [3, 3] camera intrinsics

            except Exception as e:
                print(f"Warning: Could not load meta for {seq_name}/{frame_num}: {e}")

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
        max_samples = min(num_samples, len(self.frame_index))
        indices = rng.choice(len(self.frame_index), size=max_samples, replace=False)
        return sorted(indices.tolist())


# ============================================================
# Evaluation utilities
# ============================================================

def evaluate_on_ho3d(
    ho3d_path: str = None,
    num_samples: int = 100,
    config = None,
    verbose: bool = True
) -> Dict:
    """
    Run MANO IK evaluation on HO-3D dataset.

    Args:
        ho3d_path: path to HO-3D dataset
        num_samples: number of samples to evaluate (default: 100)
        config: ExperimentConfig (uses baseline if None)
        verbose: print progress

    Returns:
        results dict with metrics
    """
    from experiments.pose_fitter_experimental import ExperimentalIKFitter, ExperimentConfig
    from model_utils.tracker import get_landmarks_world

    # Load dataset
    loader = HO3DLoader(ho3d_path)

    # Get subset
    subset_indices = loader.get_subset(num_samples)

    if verbose:
        print(f"\n[HO-3D Evaluation]")
        print(f"  Samples: {num_samples}")

    # Initialize fitter
    config = config or ExperimentConfig()
    fitter = ExperimentalIKFitter(config)

    # Metrics
    detection_count = 0
    ik_errors = []
    per_joint_errors = []

    # Process samples
    iterator = tqdm(subset_indices) if verbose else subset_indices

    for idx in iterator:
        sample = loader.get_sample(idx)
        image = sample['image']

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

        # Note: Ground truth comparison would require careful alignment
        # HO-3D ground truth is in camera coordinates, need transformation

    if detection_count == 0:
        print("  ⚠ No hands detected by MediaPipe!")
        return {'error': 'no_detections'}

    # Compute statistics
    ik_errors = np.array(ik_errors)
    per_joint_errors = np.array(per_joint_errors)

    results = {
        'dataset': 'HO-3D',
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

    parser = argparse.ArgumentParser(description="HO-3D dataset evaluation")

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to HO-3D dataset (default: ~/datasets/ho3d)"
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
        default="results/ho3d_eval.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_on_ho3d(
        ho3d_path=args.dataset_path,
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
