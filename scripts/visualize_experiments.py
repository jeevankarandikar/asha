#!/usr/bin/env python3
"""
Generate publication-quality visualizations for experimental results.
Creates figures for loss ablation, alignment comparison, optimizer comparison, and per-joint error.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['font.family'] = 'serif'


def load_experiment_results(exp_dir: Path):
    """Load all experiment JSON results."""
    results = {}
    result_files = {
        'exp1': exp_dir / 'exp1_loss_ablation.json',
        'exp2': exp_dir / 'exp2_alignment_methods.json',
        'exp3': exp_dir / 'exp3_optimizer_comparison.json',
        'exp4': exp_dir / 'exp4_per_joint_analysis.json',
    }

    for name, path in result_files.items():
        if path.exists():
            with open(path, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {path} not found")

    return results


def plot_loss_ablation(data, output_path: Path):
    """Create bar chart for loss ablation study."""
    # Data is flat, not nested under 'configurations'
    configs = data

    # Extract data
    names = []
    mean_errors = []
    median_errors = []
    colors = []

    config_order = ['baseline', 'no_bone', 'no_temporal', 'no_reg', 'position_only']
    config_labels = {
        'baseline': 'Baseline\n(All Losses)',
        'no_bone': 'No Bone\nDirection',
        'no_temporal': 'No Temporal\nSmoothing',
        'no_reg': 'No\nRegularization',
        'position_only': 'Position\nOnly'
    }

    for config_name in config_order:
        config = configs[config_name]
        names.append(config_labels[config_name])
        mean_errors.append(config['mean_ik_error_mm'])
        median_errors.append(config['median_ik_error_mm'])

        # Color: baseline blue, worse red, better green
        baseline_mean = configs['baseline']['mean_ik_error_mm']
        if config_name == 'baseline':
            colors.append('#3498db')  # Blue
        elif config['mean_ik_error_mm'] > baseline_mean:
            colors.append('#e74c3c')  # Red (worse)
        else:
            colors.append('#2ecc71')  # Green (better)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, mean_errors, width, label='Mean Error', color=colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, median_errors, width, label='Median Error', color=colors, alpha=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

    # Baseline reference line
    baseline_mean = configs['baseline']['mean_ik_error_mm']
    ax.axhline(y=baseline_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline Mean')

    ax.set_xlabel('Loss Configuration')
    ax.set_ylabel('IK Error (mm)')
    ax.set_title('Experiment 1: Loss Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_alignment_comparison(data, output_path: Path):
    """Create comparison chart for alignment methods."""
    # Data is flat, not nested under 'methods'
    methods = data

    # Extract data
    method_order = ['umeyama_scaled', 'umeyama_rigid', 'kabsch']
    method_labels = {
        'umeyama_scaled': 'Umeyama\n(Scaled)',
        'umeyama_rigid': 'Umeyama\n(Rigid)',
        'kabsch': 'Kabsch'
    }

    names = []
    mean_errors = []
    scales = []
    colors = ['#2ecc71', '#3498db', '#3498db']  # Scaled is best (green)

    for method_name in method_order:
        method = methods[method_name]
        names.append(method_labels[method_name])
        mean_errors.append(method['mean_ik_error_mm'])
        scales.append(method.get('mean_scale', 1.0))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Mean error comparison
    x = np.arange(len(names))
    bars = ax1.bar(x, mean_errors, color=colors, alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Alignment Method')
    ax1.set_ylabel('Mean IK Error (mm)')
    ax1.set_title('Error Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Scale factor comparison
    bars2 = ax2.bar(x, scales, color=colors, alpha=0.8)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No Scale (1.0)')
    ax2.set_xlabel('Alignment Method')
    ax2.set_ylabel('Estimated Scale Factor')
    ax2.set_title('Scale Estimation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.95, 1.02])

    fig.suptitle('Experiment 2: Alignment Method Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_optimizer_comparison(data, output_path: Path):
    """Create comparison chart for optimizer performance."""
    # Data is flat, not nested under 'optimizers'
    optimizers = data

    # Extract data
    opt_order = ['adam', 'sgd', 'lbfgs']
    opt_labels = {'adam': 'Adam', 'sgd': 'SGD', 'lbfgs': 'L-BFGS'}

    names = []
    mean_errors = []
    times = []
    conv_rates = []
    colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Adam green, SGD red, L-BFGS orange

    for opt_name in opt_order:
        opt = optimizers[opt_name]
        names.append(opt_labels[opt_name])
        mean_errors.append(opt['mean_ik_error_mm'])
        times.append(opt['mean_time_ms'])
        conv_rates.append(opt['convergence_rate'] * 100)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(names))

    # Plot 1: Mean error
    bars1 = ax1.bar(x, mean_errors, color=colors, alpha=0.8)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Mean IK Error (mm)')
    ax1.set_title('Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Time per frame
    bars2 = ax2.bar(x, times, color=colors, alpha=0.8)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    ax2.axhline(y=40, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='25fps target (40ms)')
    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Time per Frame (ms)')
    ax2.set_title('Speed')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Convergence rate
    bars3 = ax3.bar(x, conv_rates, color=colors, alpha=0.8)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('Optimizer')
    ax3.set_ylabel('Convergence Rate (%)')
    ax3.set_title('Reliability')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylim([90, 101])
    ax3.grid(axis='y', alpha=0.3)

    fig.suptitle('Experiment 3: Optimizer Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_per_joint_analysis(data, output_path: Path):
    """Create bar chart for per-joint error analysis."""
    # Use the worst 5 from terminal output
    # Based on experimental results:
    # 1. thumb_tip: 14.13 mm
    # 2. wrist: 13.13 mm
    # 3. index_tip: 11.85 mm
    # 4. thumb_ip: 11.41 mm
    # 5. pinky_tip: 11.32 mm

    joint_data = [
        ('Thumb Tip', 14.13),
        ('Wrist', 13.13),
        ('Index Tip', 11.85),
        ('Thumb IP', 11.41),
        ('Pinky Tip', 11.32)
    ]

    names = [name for name, _ in joint_data]
    errors = [error for _, error in joint_data]

    # Color gradient: red (worst) to orange
    colors = ['#e74c3c', '#e74c3c', '#e67e22', '#e67e22', '#f39c12']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    y = np.arange(len(names))
    bars = ax.barh(y, errors, color=colors, alpha=0.8)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.2f} mm',
                ha='left', va='center', fontsize=9)

    # Add mean line
    mean_error = data.get('mean_ik_error_mm', 9.71)
    ax.axvline(x=mean_error, color='gray', linestyle='--', linewidth=1.5,
               label=f'Overall Mean: {mean_error:.2f} mm')

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Mean Error (mm)')
    ax.set_title('Experiment 4: Per-Joint Error Analysis (Worst 5 Joints)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # Worst at top
    ax.set_xlim([0, 16])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(results, output_path: Path):
    """Create a summary comparison table."""
    # Create text-based table for LaTeX inclusion
    summary = []
    summary.append("% Experiment Summary Table - Auto-generated")
    summary.append("% Copy into LaTeX document")
    summary.append("")
    summary.append("\\begin{table}[h]")
    summary.append("\\centering")
    summary.append("\\small")
    summary.append("\\begin{tabular}{@{}llcc@{}}")
    summary.append("\\toprule")
    summary.append("\\textbf{Experiment} & \\textbf{Best Config} & \\textbf{Error (mm)} & \\textbf{Key Finding} \\\\")
    summary.append("\\midrule")

    # Exp 1: Loss ablation
    exp1 = results['exp1']
    best_config = min(exp1.items(), key=lambda x: x[1]['mean_ik_error_mm'])
    summary.append(f"Loss Ablation & {best_config[0]} & {best_config[1]['mean_ik_error_mm']:.2f} & Position-only best \\\\")

    # Exp 2: Alignment
    exp2 = results['exp2']
    best_method = min(exp2.items(), key=lambda x: x[1]['mean_ik_error_mm'])
    summary.append(f"Alignment & {best_method[0]} & {best_method[1]['mean_ik_error_mm']:.2f} & Scale helps 6\\% \\\\")

    # Exp 3: Optimizer
    exp3 = results['exp3']
    best_opt = min(exp3.items(), key=lambda x: x[1]['mean_ik_error_mm'])
    summary.append(f"Optimizer & {best_opt[0].upper()} & {best_opt[1]['mean_ik_error_mm']:.2f} & Adam most reliable \\\\")

    # Exp 4: Per-joint (use hardcoded worst)
    summary.append(f"Per-Joint & Worst: thumb\\_tip & 14.13 & Fingertips hardest \\\\")

    summary.append("\\bottomrule")
    summary.append("\\end{tabular}")
    summary.append("\\caption{Experimental results summary (5,330 frames).}")
    summary.append("\\label{tab:exp_summary}")
    summary.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))

    print(f"✓ Saved: {output_path}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    exp_dir = project_root / 'src' / 'results' / 'experiments'
    output_dir = project_root / 'docs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results...")
    results = load_experiment_results(exp_dir)

    if not results:
        print("Error: No experiment results found!")
        return

    print(f"\nGenerating visualizations in {output_dir}...\n")

    # Generate plots
    if 'exp1' in results:
        plot_loss_ablation(results['exp1'], output_dir / 'exp1_loss_ablation.png')

    if 'exp2' in results:
        plot_alignment_comparison(results['exp2'], output_dir / 'exp2_alignment.png')

    if 'exp3' in results:
        plot_optimizer_comparison(results['exp3'], output_dir / 'exp3_optimizer.png')

    if 'exp4' in results:
        plot_per_joint_analysis(results['exp4'], output_dir / 'exp4_per_joint.png')

    # Generate summary table
    if len(results) == 4:
        create_summary_table(results, output_dir / 'experiment_summary_table.tex')

    print(f"\n✓ All visualizations complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
