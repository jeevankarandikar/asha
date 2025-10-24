#!/usr/bin/env python3
"""
generate publication-quality plots from extracted results.
reads v1_metrics.json and creates png figures for papers/presentations.

usage:
    python src/generate_plots.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image

# publication settings
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10

os.makedirs('docs/figures', exist_ok=True)


def load_metrics():
    """load metrics from json."""
    with open('docs/v1_metrics.json', 'r') as f:
        return json.load(f)


def plot_ik_error_histogram(metrics):
    """figure: ik error distribution."""
    ik_errors = np.array(metrics['ik_errors_all'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ik_errors, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')

    # statistics lines
    ax.axvline(metrics['ik_error_mean'], color='red', linestyle='--',
               linewidth=2, label=f"mean: {metrics['ik_error_mean']:.1f}mm")
    ax.axvline(metrics['ik_error_median'], color='green', linestyle='--',
               linewidth=2, label=f"median: {metrics['ik_error_median']:.1f}mm")
    ax.axvline(metrics['ik_error_95th'], color='orange', linestyle='--',
               linewidth=2, label=f"95th percentile: {metrics['ik_error_95th']:.1f}mm")

    ax.set_xlabel('ik convergence error (mm)', fontweight='bold')
    ax.set_ylabel('frequency', fontweight='bold')
    ax.set_title('mano ik error distribution', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    # add some padding to x-axis
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 1, xlim[1] + 1)

    plt.tight_layout(pad=1.5)
    plt.savefig('docs/figures/ik_error_histogram.png', dpi=300, bbox_inches='tight')
    print("✓ ik_error_histogram.png")
    plt.close()


def plot_quality_filtering(metrics):
    """figure: quality filtering pipeline."""
    fig, ax = plt.subplots(figsize=(10, 5))

    stages = ['total\nframes', 'detected\nhand', 'confidence\n>0.7',
              'ik error\n<25mm', 'final\n(passes both)']
    counts = [
        metrics['total_frames'],
        metrics['valid_poses'],
        metrics['high_confidence'],
        metrics['low_ik_error'],
        metrics['passes_both']
    ]
    colors = ['#cccccc', '#9ecae1', '#6baed6', '#4292c6', '#2171b5']

    bars = ax.bar(range(len(stages)), counts, color=colors, edgecolor='black',
                   alpha=0.8, width=0.6)

    # add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel('frame count', fontweight='bold')
    ax.set_title(f'quality filtering pipeline ({metrics["retention_rate"]*100:.1f}% retention)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(alpha=0.3, axis='y')

    # add some vertical space
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout(pad=1.5)
    plt.savefig('docs/figures/quality_filtering.png', dpi=300, bbox_inches='tight')
    print("✓ quality_filtering.png")
    plt.close()


def plot_ui_progression():
    """figure: system progression with screenshots."""
    # paths to screenshots (user should place them here)
    screenshot_paths = [
        'docs/figures/screenshots/v0_mediapipe.png',
        'docs/figures/screenshots/v1_mano.png',
        'docs/figures/screenshots/v2_emg.png'
    ]

    labels = ['v0: mediapipe baseline', 'v1: mano ik + lbs', 'v2: emg integration']

    # check if screenshots exist
    missing = [p for p in screenshot_paths if not os.path.exists(p)]
    if missing:
        print(f"\n⚠️  screenshots missing: {', '.join(missing)}")
        print("   place screenshots in docs/figures/screenshots/")
        print("   - v0_mediapipe.png")
        print("   - v1_mano.png")
        print("   - v2_emg.png")
        return

    # load images
    images = [Image.open(p) for p in screenshot_paths]

    # create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
        ax.axis('off')

    plt.suptitle('system progression: v0 → v1 → v2', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(pad=2.0)
    plt.savefig('docs/figures/ui_progression.png', dpi=300, bbox_inches='tight')
    print("✓ ui_progression.png")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("generating plots")
    print("="*60 + "\n")

    try:
        metrics = load_metrics()
        plot_ik_error_histogram(metrics)
        plot_quality_filtering(metrics)
        plot_ui_progression()

        print("\n" + "="*60)
        print("all plots generated")
        print("="*60)
        print("\nlocation: docs/figures/")
        print("\nnext step:")
        print("   cd docs && pdflatex CIS6800_Proposal.tex")

    except FileNotFoundError:
        print("error: v1_metrics.json not found!")
        print("\nrun this first:")
        print("   python src/extract_results.py")
