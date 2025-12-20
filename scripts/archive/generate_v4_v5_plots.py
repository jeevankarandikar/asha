"""
Generate evaluation and diagnostics plots for v4 and v5 EMG models.
Creates plots similar to resnet/transformer to complete the figure set.
"""

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add asha package to path
sys.path.insert(0, str(Path(__file__).parent))

from asha.v4.model import SimpleEMGModel
from asha.v5.model import EMGToJointsModel
from asha.core.mano_model import MANOModel

# Initialize MANO for v4 evaluation
mano = MANOModel(model_path='models/mano/MANO_RIGHT_numpy.npz')


def load_emg_data(data_dir, window_size=50):
    """Load all EMG recordings and create windows."""
    data_dir = Path(data_dir)
    h5_files = sorted(data_dir.glob('*.h5'))

    all_emg = []
    all_labels = []

    print(f"Loading data from {len(h5_files)} files...")

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            emg = f['emg_filtered'][:]  # [N, 8]

            # Check if we have theta (v4) or joints (v5)
            if 'pose_theta' in f:
                labels = f['pose_theta'][:]  # [M, 45]
            elif 'pose_joints' in f:
                labels = f['pose_joints'][:].reshape(-1, 63)  # [M, 21, 3] → [M, 63]
            else:
                print(f"  Skipping {h5_file.name} - no labels found")
                continue

            # Resample EMG to match pose framerate
            emg_rate = 500  # Hz
            pose_rate = 25  # Hz
            downsample = emg_rate // pose_rate  # 20

            # Create sliding windows
            for i in range(len(labels)):
                start_idx = i * downsample
                end_idx = start_idx + window_size

                if end_idx <= len(emg):
                    emg_window = emg[start_idx:end_idx]  # [50, 8]
                    all_emg.append(emg_window)
                    all_labels.append(labels[i])

        print(f"  Loaded {h5_file.name}: {len(all_emg)} windows so far")

    emg_array = np.array(all_emg)  # [N, 50, 8]
    labels_array = np.array(all_labels)  # [N, 45] or [N, 63]

    print(f"Total: {len(emg_array)} windows")
    return emg_array, labels_array


def compute_mpjpe(pred_joints, gt_joints):
    """Compute Mean Per-Joint Position Error in mm."""
    pred_joints = pred_joints.reshape(-1, 21, 3)
    gt_joints = gt_joints.reshape(-1, 21, 3)
    errors = np.linalg.norm(pred_joints - gt_joints, axis=2)  # [N, 21]
    return errors * 1000  # Convert to mm


def evaluate_v4(model_path, data_dir):
    """Evaluate v4 model (EMG → θ)."""
    print("\n" + "="*60)
    print("Evaluating v4 Model (EMG → MANO θ)")
    print("="*60)

    # Load model
    model = SimpleEMGModel(window_size=50, input_channels=8)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"   Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    # Load data
    emg_data, theta_labels = load_emg_data(data_dir)

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(emg_data))
    val_emg = emg_data[split_idx:]
    val_theta = theta_labels[split_idx:]

    print(f"\nValidation set: {len(val_emg)} samples")

    # Normalize EMG (global stats from training set)
    train_emg = emg_data[:split_idx]
    emg_mean = train_emg.mean(axis=(0, 1))
    emg_std = train_emg.std(axis=(0, 1)) + 1e-6
    val_emg_norm = (val_emg - emg_mean) / emg_std

    # Prepare input: [N, 50, 8] → [N, 8, 50]
    val_input = torch.from_numpy(val_emg_norm.transpose(0, 2, 1)).float()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred_theta = model(val_input).numpy()

    # Convert theta to joints via MANO forward pass
    print("Converting θ to joints via MANO...")
    pred_joints = []
    gt_joints = []

    for i in range(len(pred_theta)):
        # Predicted
        vertices_pred, joints_pred = mano.forward(pred_theta[i])
        pred_joints.append(joints_pred)

        # Ground truth
        vertices_gt, joints_gt = mano.forward(val_theta[i])
        gt_joints.append(joints_gt)

    pred_joints = np.array(pred_joints)  # [N, 21, 3]
    gt_joints = np.array(gt_joints)  # [N, 21, 3]

    # Compute MPJPE
    per_sample_errors = compute_mpjpe(pred_joints, gt_joints)  # [N, 21]
    mean_mpjpe = per_sample_errors.mean(axis=1)  # [N]

    results = {
        'mean_mpjpe': mean_mpjpe.mean(),
        'median_mpjpe': np.median(mean_mpjpe),
        'per_sample_errors': mean_mpjpe,
        'per_joint_errors': per_sample_errors.mean(axis=0),  # [21]
        'best_5': np.argsort(mean_mpjpe)[:5],
        'worst_5': np.argsort(mean_mpjpe)[-5:],
    }

    print(f"\n📊 Results:")
    print(f"   Mean MPJPE: {results['mean_mpjpe']:.2f} mm")
    print(f"   Median MPJPE: {results['median_mpjpe']:.2f} mm")
    print(f"   Best: {mean_mpjpe.min():.2f} mm")
    print(f"   Worst: {mean_mpjpe.max():.2f} mm")

    return results


def evaluate_v5(model_path, data_dir):
    """Evaluate v5 model (EMG → Joints)."""
    print("\n" + "="*60)
    print("Evaluating v5 Model (EMG → Joints)")
    print("="*60)

    # Load model
    model = EMGToJointsModel(window_size=50, input_channels=8)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"   Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    # Load data
    emg_data, joint_labels = load_emg_data(data_dir)

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(emg_data))
    val_emg = emg_data[split_idx:]
    val_joints = joint_labels[split_idx:]

    print(f"\nValidation set: {len(val_emg)} samples")

    # Normalize EMG
    train_emg = emg_data[:split_idx]
    emg_mean = train_emg.mean(axis=(0, 1))
    emg_std = train_emg.std(axis=(0, 1)) + 1e-6
    val_emg_norm = (val_emg - emg_mean) / emg_std

    # Prepare input: [N, 50, 8] → [N, 8, 50]
    val_input = torch.from_numpy(val_emg_norm.transpose(0, 2, 1)).float()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred_joints = model(val_input).numpy()

    # Compute MPJPE
    per_sample_errors = compute_mpjpe(pred_joints, val_joints)  # [N, 21]
    mean_mpjpe = per_sample_errors.mean(axis=1)  # [N]

    results = {
        'mean_mpjpe': mean_mpjpe.mean(),
        'median_mpjpe': np.median(mean_mpjpe),
        'per_sample_errors': mean_mpjpe,
        'per_joint_errors': per_sample_errors.mean(axis=0),  # [21]
        'best_5': np.argsort(mean_mpjpe)[:5],
        'worst_5': np.argsort(mean_mpjpe)[-5:],
    }

    print(f"\n📊 Results:")
    print(f"   Mean MPJPE: {results['mean_mpjpe']:.2f} mm")
    print(f"   Median MPJPE: {results['median_mpjpe']:.2f} mm")
    print(f"   Best: {mean_mpjpe.min():.2f} mm")
    print(f"   Worst: {mean_mpjpe.max():.2f} mm")

    return results


def plot_eval(v4_results, v5_results, output_dir):
    """Create eval comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Mean MPJPE comparison
    models = ['v4 (EMG→θ)', 'v5 (EMG→Joints)', 'IK Ground Truth']
    mpjpes = [v4_results['mean_mpjpe'], v5_results['mean_mpjpe'], 9.71]
    colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']

    bars = ax1.bar(models, mpjpes, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, mpjpes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.2f} mm', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Model Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(mpjpes) * 1.2)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Error distributions
    ax2.violinplot([v4_results['per_sample_errors'], v5_results['per_sample_errors']],
                   positions=[0, 1], showmeans=True, showmedians=True)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['v4 (EMG→θ)', 'v5 (EMG→Joints)'])
    ax2.set_ylabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution (Validation Set)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'v4_v5_eval.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'v4_v5_eval.png'}")
    plt.close()


def plot_diagnostics(v4_results, v5_results, output_dir):
    """Create diagnostics plot."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Per-joint error comparison
    ax1 = fig.add_subplot(gs[0, :])
    joint_names = ['Wrist', 'Thumb1', 'Thumb2', 'Thumb3', 'Thumb4',
                   'Index1', 'Index2', 'Index3', 'Index4',
                   'Middle1', 'Middle2', 'Middle3', 'Middle4',
                   'Ring1', 'Ring2', 'Ring3', 'Ring4',
                   'Pinky1', 'Pinky2', 'Pinky3', 'Pinky4']

    x = np.arange(21)
    width = 0.35

    bars1 = ax1.bar(x - width/2, v4_results['per_joint_errors'], width,
                    label='v4 (EMG→θ)', color='#FF6B6B', edgecolor='black')
    bars2 = ax1.bar(x + width/2, v5_results['per_joint_errors'], width,
                    label='v5 (EMG→Joints)', color='#4ECDC4', edgecolor='black')

    ax1.set_ylabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Joint Error Analysis', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: v4 error histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(v4_results['per_sample_errors'], bins=50, color='#FF6B6B',
             edgecolor='black', alpha=0.7)
    ax2.axvline(v4_results['mean_mpjpe'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {v4_results['mean_mpjpe']:.2f} mm")
    ax2.set_xlabel('MPJPE (mm)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('v4 Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: v5 error histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(v5_results['per_sample_errors'], bins=50, color='#4ECDC4',
             edgecolor='black', alpha=0.7)
    ax3.axvline(v5_results['mean_mpjpe'], color='blue', linestyle='--',
                linewidth=2, label=f"Mean: {v5_results['mean_mpjpe']:.2f} mm")
    ax3.set_xlabel('MPJPE (mm)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('v5 Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.savefig(output_dir / 'v4_v5_diagnostics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'v4_v5_diagnostics.png'}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("EMG Model Evaluation & Plot Generation")
    print("="*60)

    # Evaluate v4
    v4_results = evaluate_v4(
        model_path='models/v4/emg_model_v2_best.pth',
        data_dir='data/v4/emg_recordings'
    )

    # Evaluate v5
    v5_results = evaluate_v5(
        model_path='models/v5/emg_joints_best.pth',
        data_dir='data/v4/emg_recordings'  # v5 uses same EMG data, different labels
    )

    # Generate plots
    output_dir = Path('docs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)

    plot_eval(v4_results, v5_results, output_dir)
    plot_diagnostics(v4_results, v5_results, output_dir)

    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - docs/figures/v4_v5_eval.png")
    print(f"  - docs/figures/v4_v5_diagnostics.png")
    print(f"\nSummary:")
    print(f"  v4 (EMG→θ): {v4_results['mean_mpjpe']:.2f} mm MPJPE")
    print(f"  v5 (EMG→Joints): {v5_results['mean_mpjpe']:.2f} mm MPJPE")
    print(f"  Improvement: {v4_results['mean_mpjpe'] - v5_results['mean_mpjpe']:.2f} mm")
    print(f"  ({100*(v4_results['mean_mpjpe'] - v5_results['mean_mpjpe'])/v4_results['mean_mpjpe']:.1f}% better)")


if __name__ == '__main__':
    main()
