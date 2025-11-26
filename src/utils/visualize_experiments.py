"""
visualization tools for cis 6800 final project.

generates publication-quality figures for paper:
  - ik error histograms
  - per-joint error bar charts
  - loss ablation comparisons
  - alignment method comparisons
  - optimizer convergence plots
  - temporal consistency analysis

usage:
  python visualize_experiments.py --results results/experiments
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# publication style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


# ============================================================
# figure 1: ik error histogram
# ============================================================

def plot_ik_error_histogram(errors_mm: np.ndarray, output_path: Path):
    """
    plot ik error distribution with statistics.

    generates histogram with:
      - mean, median, 95th percentile markers
      - kde overlay
      - clean labels for paper
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # compute statistics
    mean_error = errors_mm.mean()
    median_error = np.median(errors_mm)
    p95_error = np.percentile(errors_mm, 95)

    # histogram with kde
    ax.hist(errors_mm, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # kde overlay
    from scipy import stats
    kde = stats.gaussian_kde(errors_mm)
    x_range = np.linspace(errors_mm.min(), errors_mm.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # vertical lines for statistics
    ax.axvline(mean_error, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.1f} mm')
    ax.axvline(median_error, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_error:.1f} mm')
    ax.axvline(p95_error, color='red', linestyle='--', linewidth=2, label=f'95th: {p95_error:.1f} mm')

    ax.set_xlabel('IK Error (mm)')
    ax.set_ylabel('Density')
    ax.set_title('IK Error Distribution (543 frames)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================
# figure 2: per-joint error bar chart
# ============================================================

def plot_per_joint_errors(joint_stats: Dict, output_path: Path):
    """
    bar chart showing mean error per joint with error bars.

    groups joints by finger for clarity.
    """
    joint_names = list(joint_stats.keys())
    mean_errors = [joint_stats[name]['mean_mm'] for name in joint_names]
    std_errors = [joint_stats[name]['std_mm'] for name in joint_names]

    # color by finger
    colors = []
    for name in joint_names:
        if 'wrist' in name:
            colors.append('gray')
        elif 'thumb' in name:
            colors.append('tab:red')
        elif 'index' in name:
            colors.append('tab:blue')
        elif 'middle' in name:
            colors.append('tab:green')
        elif 'ring' in name:
            colors.append('tab:orange')
        elif 'pinky' in name:
            colors.append('tab:purple')
        else:
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(len(joint_names))
    bars = ax.bar(x_pos, mean_errors, yerr=std_errors, color=colors, alpha=0.7, capsize=3, edgecolor='black')

    ax.set_xlabel('Joint')
    ax.set_ylabel('Mean Error (mm)')
    ax.set_title('Per-Joint Error Analysis')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc='gray', alpha=0.7, edgecolor='black', label='Wrist'),
        plt.Rectangle((0,0),1,1, fc='tab:red', alpha=0.7, edgecolor='black', label='Thumb'),
        plt.Rectangle((0,0),1,1, fc='tab:blue', alpha=0.7, edgecolor='black', label='Index'),
        plt.Rectangle((0,0),1,1, fc='tab:green', alpha=0.7, edgecolor='black', label='Middle'),
        plt.Rectangle((0,0),1,1, fc='tab:orange', alpha=0.7, edgecolor='black', label='Ring'),
        plt.Rectangle((0,0),1,1, fc='tab:purple', alpha=0.7, edgecolor='black', label='Pinky'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================
# figure 3: loss ablation comparison
# ============================================================

def plot_loss_ablation(exp1_results: Dict, output_path: Path):
    """
    bar chart comparing different loss configurations.

    shows mean ik error for each ablation.
    """
    configs = list(exp1_results.keys())
    mean_errors = [exp1_results[cfg]['mean_ik_error_mm'] for cfg in configs]
    std_errors = [exp1_results[cfg]['std_ik_error_mm'] for cfg in configs]

    # sort by error (best to worst)
    sorted_indices = np.argsort(mean_errors)
    configs = [configs[i] for i in sorted_indices]
    mean_errors = [mean_errors[i] for i in sorted_indices]
    std_errors = [std_errors[i] for i in sorted_indices]

    # color baseline differently
    colors = ['green' if cfg == 'baseline' else 'steelblue' for cfg in configs]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, mean_errors, yerr=std_errors, color=colors, alpha=0.7, capsize=3, edgecolor='black')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Mean IK Error (mm)')
    ax.set_title('Experiment 1: Loss Ablation Study')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # add values on top of bars
    for i, (bar, err) in enumerate(zip(bars, mean_errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_errors[i],
                f'{err:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================
# figure 4: alignment methods comparison
# ============================================================

def plot_alignment_comparison(exp2_results: Dict, output_path: Path):
    """
    grouped bar chart comparing alignment methods.

    shows mean error and mean scale for each method.
    """
    methods = list(exp2_results.keys())
    mean_errors = [exp2_results[m]['mean_ik_error_mm'] for m in methods]
    mean_scales = [exp2_results[m].get('mean_scale', 1.0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # subplot 1: mean ik error
    x_pos = np.arange(len(methods))
    bars1 = ax1.bar(x_pos, mean_errors, color='steelblue', alpha=0.7, edgecolor='black')

    ax1.set_xlabel('Alignment Method')
    ax1.set_ylabel('Mean IK Error (mm)')
    ax1.set_title('(a) IK Error Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # add values
    for bar, err in zip(bars1, mean_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.1f}', ha='center', va='bottom', fontsize=9)

    # subplot 2: mean scale
    bars2 = ax2.bar(x_pos, mean_scales, color='orange', alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Alignment Method')
    ax2.set_ylabel('Mean Scale Factor')
    ax2.set_title('(b) Estimated Scale')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=1, label='s=1 (no scale)')
    ax2.legend()

    # add values
    for bar, scale in zip(bars2, mean_scales):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{scale:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Experiment 2: Alignment Method Comparison')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================
# figure 5: optimizer comparison
# ============================================================

def plot_optimizer_comparison(exp3_results: Dict, output_path: Path):
    """
    grouped bar chart comparing optimizers.

    shows mean error and mean time per frame.
    """
    optimizers = list(exp3_results.keys())
    mean_errors = [exp3_results[opt]['mean_ik_error_mm'] for opt in optimizers]
    mean_times = [exp3_results[opt]['mean_time_ms'] for opt in optimizers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # subplot 1: mean ik error
    x_pos = np.arange(len(optimizers))
    bars1 = ax1.bar(x_pos, mean_errors, color='steelblue', alpha=0.7, edgecolor='black')

    ax1.set_xlabel('Optimizer')
    ax1.set_ylabel('Mean IK Error (mm)')
    ax1.set_title('(a) Accuracy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([opt.upper() for opt in optimizers])
    ax1.grid(axis='y', alpha=0.3)

    # add values
    for bar, err in zip(bars1, mean_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.1f}', ha='center', va='bottom', fontsize=9)

    # subplot 2: mean time
    bars2 = ax2.bar(x_pos, mean_times, color='orange', alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Optimizer')
    ax2.set_ylabel('Mean Time per Frame (ms)')
    ax2.set_title('(b) Speed')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([opt.upper() for opt in optimizers])
    ax2.grid(axis='y', alpha=0.3)

    # add values
    for bar, t in zip(bars2, mean_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Experiment 3: Optimizer Comparison')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================
# figure 6: summary comparison table
# ============================================================

def generate_summary_table(exp1_results: Dict, exp2_results: Dict, exp3_results: Dict, output_path: Path):
    """
    generate latex table summarizing all experiments.

    outputs .tex file ready to include in paper.
    """
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}lcc@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Configuration} & \textbf{Mean Error (mm)} & \textbf{Std (mm)} \\")
    lines.append(r"\midrule")

    # exp 1: loss ablation
    lines.append(r"\textit{Loss Ablation:} & & \\")
    for name, res in exp1_results.items():
        mean_err = res['mean_ik_error_mm']
        std_err = res['std_ik_error_mm']
        lines.append(f"  {name.replace('_', ' ')} & {mean_err:.2f} & {std_err:.2f} \\\\")

    lines.append(r"\midrule")

    # exp 2: alignment
    lines.append(r"\textit{Alignment Methods:} & & \\")
    for name, res in exp2_results.items():
        mean_err = res['mean_ik_error_mm']
        std_err = res['std_ik_error_mm']
        lines.append(f"  {name.replace('_', ' ')} & {mean_err:.2f} & {std_err:.2f} \\\\")

    lines.append(r"\midrule")

    # exp 3: optimizer
    lines.append(r"\textit{Optimizers:} & & \\")
    for name, res in exp3_results.items():
        mean_err = res['mean_ik_error_mm']
        std_err = res['std_ik_error_mm']
        lines.append(f"  {name.upper()} & {mean_err:.2f} & {std_err:.2f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Experiment Results Summary}")
    lines.append(r"\label{tab:exp_summary}")
    lines.append(r"\end{table}")

    # write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Saved LaTeX table: {output_path}")


# ============================================================
# main cli
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results directory (from run_experiments.py)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="docs/figures",
        help="Output directory for figures"
    )

    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING FIGURES FOR PAPER")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # load experiment results
    exp1_file = results_dir / "exp1_loss_ablation.json"
    exp2_file = results_dir / "exp2_alignment_methods.json"
    exp3_file = results_dir / "exp3_optimizer_comparison.json"
    exp4_file = results_dir / "exp4_per_joint_analysis.json"

    if exp1_file.exists():
        print("\n[Experiment 1] Loss Ablation")
        with open(exp1_file, 'r') as f:
            exp1_results = json.load(f)
        plot_loss_ablation(exp1_results, output_dir / "exp1_loss_ablation.png")

    if exp2_file.exists():
        print("\n[Experiment 2] Alignment Methods")
        with open(exp2_file, 'r') as f:
            exp2_results = json.load(f)
        plot_alignment_comparison(exp2_results, output_dir / "exp2_alignment_methods.png")

    if exp3_file.exists():
        print("\n[Experiment 3] Optimizer Comparison")
        with open(exp3_file, 'r') as f:
            exp3_results = json.load(f)
        plot_optimizer_comparison(exp3_results, output_dir / "exp3_optimizer_comparison.png")

    if exp4_file.exists():
        print("\n[Experiment 4] Per-Joint Analysis")
        with open(exp4_file, 'r') as f:
            exp4_results = json.load(f)
        if 'joint_stats' in exp4_results:
            plot_per_joint_errors(exp4_results['joint_stats'], output_dir / "exp4_per_joint_errors.png")

    # generate summary table
    if exp1_file.exists() and exp2_file.exists() and exp3_file.exists():
        print("\n[Summary] LaTeX Table")
        with open(exp1_file, 'r') as f:
            exp1_results = json.load(f)
        with open(exp2_file, 'r') as f:
            exp2_results = json.load(f)
        with open(exp3_file, 'r') as f:
            exp3_results = json.load(f)
        generate_summary_table(exp1_results, exp2_results, exp3_results, output_dir / "experiment_summary_table.tex")

    print("\n" + "=" * 60)
    print("✓ ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
