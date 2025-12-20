"""
Simple comparison charts using known final results.
Run this quickly to get the essential figures for your presentation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path('docs/figures')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Creating Comparison Charts")
print("="*60)

# ============================================================
# Chart 1: v4 vs v5 vs IK Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

models = ['v4\n(EMG→θ)', 'v5\n(EMG→Joints)', 'IK Ground\nTruth']
mpjpe = [34.92, 14.92, 9.71]
colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']

bars = ax.bar(models, mpjpe, color=colors, edgecolor='black', linewidth=2, width=0.6)

# Add value labels
for bar, val in zip(bars, mpjpe):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{val:.2f} mm',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotation
ax.annotate('', xy=(1, 14.92), xytext=(0, 34.92),
            arrowprops=dict(arrowstyle='<->', color='green', lw=3))
ax.text(0.5, 25, '57% better\n(19.99mm)',
        ha='center', fontsize=13, color='green', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='green', linewidth=2))

ax.set_ylabel('MPJPE (mm)', fontsize=15, fontweight='bold')
ax.set_title('EMG Model Architecture Comparison', fontsize=17, fontweight='bold', pad=20)
ax.set_ylim(0, 42)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.savefig(output_dir / 'comparison_v4_vs_v5.png', dpi=300, bbox_inches='tight')
print(f"✅ Created: docs/figures/comparison_v4_vs_v5.png")
plt.close()

# ============================================================
# Chart 2: State-of-the-Art Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

methods = ['HandFormer\n(Stereo)', 'HandFormer\n(FreiHAND)', 'Ours\n(IK)', 'Ours\n(EMG v5)']
mpjpe_values = [10.92, 12.33, 9.71, 14.92]
colors_sota = ['#BDC3C7', '#95A5A6', '#2ECC71', '#3498DB']
fps_values = [5, 5, 25, 30]

bars = ax.bar(methods, mpjpe_values, color=colors_sota, edgecolor='black', linewidth=2, width=0.6)

# Add value labels with FPS
for bar, val, fps in zip(bars, mpjpe_values, fps_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.6,
            f'{val:.2f} mm\n@ {fps} fps',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('MPJPE (mm)', fontsize=15, fontweight='bold')
ax.set_title('State-of-the-Art Comparison', fontsize=17, fontweight='bold', pad=20)
ax.set_ylim(0, 19)

# Add threshold line
ax.axhline(y=15, color='red', linestyle='--', alpha=0.6, linewidth=2.5)
ax.text(3.6, 15.6, '15mm threshold', fontsize=11, color='red', ha='right', fontweight='bold')

# Add camera-free badge
ax.text(3, 2.5, '📷 Camera-Free', fontsize=13, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', edgecolor='blue',
                 alpha=0.8, linewidth=2))

ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.savefig(output_dir / 'comparison_state_of_art.png', dpi=300, bbox_inches='tight')
print(f"✅ Created: docs/figures/comparison_state_of_art.png")
plt.close()

# ============================================================
# Chart 3: All Versions Timeline (Bonus)
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

versions = ['v0\nMediaPipe\n(Baseline)', 'v1\nMANO IK\n(Pseudo-labels)',
            'v2.5\nImage→θ\n(Paused)', 'v4\nEMG→θ', 'v5\nEMG→Joints']
mpjpe_vals = [np.nan, 9.71, 47.0, 34.92, 14.92]
colors_timeline = ['#E74C3C', '#2ECC71', '#E67E22', '#FF6B6B', '#4ECDC4']
markers = ['x', 'o', 'x', 's', 's']

x_pos = np.arange(len(versions))

# Plot bars
bars = []
for i, (x, y, c, m) in enumerate(zip(x_pos, mpjpe_vals, colors_timeline, markers)):
    if not np.isnan(y):
        bar = ax.bar(x, y, color=c, edgecolor='black', linewidth=2, width=0.6, alpha=0.8)
        bars.append(bar)
        # Add marker on top
        ax.plot(x, y, marker=m, markersize=15, color='black', markerfacecolor=c,
               markeredgewidth=2, zorder=10)
        # Add value label
        ax.text(x, y + 2, f'{y:.1f} mm', ha='center', va='bottom',
               fontsize=12, fontweight='bold')

# Add status annotations
ax.text(0, 5, 'Keypoints\nonly', ha='center', fontsize=10, style='italic')
ax.text(2, 50, 'Plateaued\n@ 47mm', ha='center', fontsize=10, style='italic', color='red')

ax.set_xticks(x_pos)
ax.set_xticklabels(versions, fontsize=11, fontweight='bold')
ax.set_ylabel('MPJPE (mm)', fontsize=15, fontweight='bold')
ax.set_title('Project Evolution: Baseline → Camera-Free Tracking',
            fontsize=17, fontweight='bold', pad=20)
ax.set_ylim(0, 55)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)

# Add annotations
ax.annotate('', xy=(1, 9.71), xytext=(4, 14.92),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5, linestyle='--'))
ax.text(2.5, 5, 'IK labels enable\nEMG training', fontsize=11, ha='center',
       color='green', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2))

plt.tight_layout()
plt.savefig(output_dir / 'version_timeline.png', dpi=300, bbox_inches='tight')
print(f"✅ Created: docs/figures/version_timeline.png")
plt.close()

# ============================================================
# Summary
# ============================================================

print("\n" + "="*60)
print("✅ ALL COMPARISON CHARTS CREATED!")
print("="*60)
print("\nGenerated files:")
print("  1. docs/figures/comparison_v4_vs_v5.png")
print("  2. docs/figures/comparison_state_of_art.png")
print("  3. docs/figures/version_timeline.png (BONUS)")
print("\nUse these in your presentation:")
print("  - Slide 10: comparison_v4_vs_v5.png")
print("  - Slide 13: comparison_state_of_art.png")
print("  - Slide 15 (optional): version_timeline.png")
print("\nYou're ready for your presentation! 🎉")
print("="*60)
