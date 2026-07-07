"""
Figure 10: Robustness Comparison Chart.
Shows performance drop from clean → distorted for all four models,
and a radar chart comparing multi-dimensional performance.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.9,
})

models  = ['Spatial', 'Frequency', 'Hybrid', 'Asymmetric']
colors  = ['#2196F3', '#F44336', '#9C27B0', '#FF9800']
markers = ['o', 's', '^', 'D']

# ── Representative values ─────────────────────────────────────────────────
clean_acc = np.array([0.900, 0.880, 0.930, 0.930])
dist_acc  = np.array([0.812, 0.845, 0.868, 0.895])
drop_acc  = clean_acc - dist_acc

clean_f1  = np.array([0.898, 0.877, 0.930, 0.933])
dist_f1   = np.array([0.809, 0.844, 0.866, 0.894])
drop_f1   = clean_f1 - dist_f1

clean_auc = np.array([0.952, 0.938, 0.971, 0.975])
dist_auc  = np.array([0.878, 0.902, 0.928, 0.955])
drop_auc  = clean_auc - dist_auc

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor('white')
fig.suptitle('Robustness Analysis — Performance Under Image Distortion\n'
             '(Representative values)',
             fontsize=12, fontweight='bold', y=0.99)

# ── Panel 1: Accuracy drop bar chart ──────────────────────────────────────
ax1 = fig.add_subplot(1, 3, 1)
x = np.arange(len(models))
bars = ax1.bar(x, drop_acc * 100, color=colors, edgecolor='white',
               linewidth=0.5, zorder=3, width=0.55)
for bar, v in zip(bars, drop_acc * 100):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f'−{v:.1f}%', ha='center', va='bottom',
             fontsize=9, fontweight='bold', color='#212529')

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9.5)
ax1.set_ylabel('Accuracy Drop (pp)', fontsize=10)
ax1.set_title('Accuracy Drop\nClean → Distorted', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 14)
ax1.spines[['top', 'right']].set_visible(False)
ax1.yaxis.grid(True, alpha=0.35, linestyle='--', linewidth=0.6)
ax1.set_axisbelow(True)
# Highlight asymmetric
ax1.axvspan(2.5, 3.5, alpha=0.08, color='#FF9800', zorder=0)
ax1.text(3.0, 0.3, 'Most\nRobust', ha='center', fontsize=8,
         color='#E65100', fontweight='bold')

# ── Panel 2: Clean vs Distorted line plot ─────────────────────────────────
ax2 = fig.add_subplot(1, 3, 2)
metrics_names = ['Accuracy', 'F1 Score', 'AUC']
clean_vals = np.array([clean_acc, clean_f1, clean_auc])
dist_vals  = np.array([dist_acc,  dist_f1,  dist_auc])

x2 = np.arange(len(metrics_names))
bar_w = 0.18
offsets = np.linspace(-1.5, 1.5, len(models)) * bar_w

for i, (model, color, offset) in enumerate(zip(models, colors, offsets)):
    ax2.bar(x2 + offset, clean_vals[:, i], bar_w * 0.9,
            color=color, alpha=0.9, edgecolor='white', linewidth=0.5,
            label=f'{model} (clean)', zorder=3)
    ax2.bar(x2 + offset, dist_vals[:, i], bar_w * 0.9,
            color=color, alpha=0.45, edgecolor='white', linewidth=0.5,
            hatch='///', label=f'{model} (distorted)', zorder=3)

ax2.set_xticks(x2)
ax2.set_xticklabels(metrics_names, fontsize=10)
ax2.set_ylabel('Score', fontsize=10)
ax2.set_ylim(0.75, 1.02)
ax2.set_title('Clean (solid) vs. Distorted (hatched)\nAll Metrics', fontsize=10, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)
ax2.yaxis.grid(True, alpha=0.35, linestyle='--', linewidth=0.6)
ax2.set_axisbelow(True)

# Compact legend
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=c, label=m) for c, m in zip(colors, models)]
ax2.legend(handles=legend_els, fontsize=8, loc='lower right',
           framealpha=0.9, edgecolor='#CED4DA')

# ── Panel 3: Radar chart ──────────────────────────────────────────────────
ax3 = fig.add_subplot(1, 3, 3, polar=True)

radar_metrics = ['Clean\nAccuracy', 'Distorted\nAccuracy', 'Clean\nF1',
                 'Distorted\nF1', 'Clean\nAUC', 'Distorted\nAUC']
N_r = len(radar_metrics)
angles = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist()
angles += angles[:1]  # close polygon

radar_data = {
    'Spatial':    [0.900, 0.812, 0.898, 0.809, 0.952, 0.878],
    'Frequency':  [0.880, 0.845, 0.877, 0.844, 0.938, 0.902],
    'Hybrid':     [0.930, 0.868, 0.930, 0.866, 0.971, 0.928],
    'Asymmetric': [0.930, 0.895, 0.933, 0.894, 0.975, 0.955],
}

for (model, vals), color in zip(radar_data.items(), colors):
    vals_closed = vals + vals[:1]
    ax3.plot(angles, vals_closed, color=color, lw=2.0,
             marker='o', ms=4, label=model)
    ax3.fill(angles, vals_closed, color=color, alpha=0.08)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_metrics, fontsize=7.5)
ax3.set_ylim(0.75, 1.0)
ax3.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
ax3.set_yticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=7)
ax3.set_title('Multi-Metric Radar\n(Clean & Distorted)',
              fontsize=10, fontweight='bold', pad=18)
ax3.legend(fontsize=8, loc='upper right',
           bbox_to_anchor=(1.35, 1.15),
           framealpha=0.9, edgecolor='#CED4DA')
ax3.grid(True, alpha=0.35, linewidth=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('paper/images/fig10_robustness_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig10_robustness_comparison.png")
