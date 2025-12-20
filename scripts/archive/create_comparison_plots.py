"""
Quick script to generate comparison charts for final report.
Run this after organizing your training result plots.
"""

import matplotlib.pyplot as plt
import numpy as np

# Create output directory if needed
import os
os.makedirs('docs/results', exist_ok=True)

# ============================================================
# Figure 1: v4 vs v5 Architecture Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

models = ['v4\n(EMG→θ)', 'v5\n(EMG→Joints)', 'IK Ground\nTruth']
mpjpe = [34.92, 14.92, 9.71]
colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']

bars = ax.bar(models, mpjpe, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, mpjpe):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val:.2f} mm',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('MPJPE (mm)', fontsize=14, fontweight='bold')
ax.set_title('EMG Model Architecture Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 40)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add improvement annotation
ax.annotate('', xy=(1, 14.92), xytext=(0, 34.92),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(0.5, 25, '57% better\n(19.99mm)',
        ha='center', fontsize=11, color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))

plt.tight_layout()
plt.savefig('docs/results/comparison_v4_vs_v5.png', dpi=300, bbox_inches='tight')
print("✅ Created: docs/results/comparison_v4_vs_v5.png")
plt.close()

# ============================================================
# Figure 2: State-of-the-Art Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

methods = ['HandFormer\n(Stereo)', 'HandFormer\n(FreiHAND)', 'Ours\n(IK)', 'Ours\n(EMG v5)']
mpjpe_values = [10.92, 12.33, 9.71, 14.92]
colors_sota = ['#95A5A6', '#95A5A6', '#2ECC71', '#3498DB']
fps_values = [5, 5, 25, 30]  # Frame rates

bars = ax.bar(methods, mpjpe_values, color=colors_sota, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val, fps in zip(bars, mpjpe_values, fps_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{val:.2f} mm\n@ {fps} fps',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('MPJPE (mm)', fontsize=14, fontweight='bold')
ax.set_title('State-of-the-Art Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 18)

# Add threshold line
ax.axhline(y=15, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(3.5, 15.5, '15mm threshold', fontsize=10, color='red', ha='right')

# Add camera-free indicator
ax.text(3, 2, '📷 Camera-Free', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='blue', alpha=0.7))

ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('docs/results/comparison_state_of_art.png', dpi=300, bbox_inches='tight')
print("✅ Created: docs/results/comparison_state_of_art.png")
plt.close()

# ============================================================
# Summary
# ============================================================

print("\n" + "="*60)
print("COMPARISON PLOTS CREATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  1. docs/results/comparison_v4_vs_v5.png")
print("  2. docs/results/comparison_state_of_art.png")
print("\nUse these in:")
print("  - Slide 10 (Exp 5 - Architecture Comparison)")
print("  - Slide 13 (SOTA Comparison)")
print("\nNext steps:")
print("  1. Copy v4/v5 training plots from Google Drive → docs/results/")
print("  2. Extract IK experiment plots from midterm PDF → docs/results/")
print("  3. Insert all figures into presentation slides")
print("="*60)
