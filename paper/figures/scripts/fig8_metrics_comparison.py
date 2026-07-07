"""
Figure 8: Precision / Recall / F1 / Accuracy comparison bar chart.
Grouped bars for all four models on clean and distorted validation sets.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.9,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
})

models = ['Spatial', 'Frequency', 'Hybrid', 'Asymmetric']

# ── Representative metric values ──────────────────────────────────────────
# Clean validation set
clean = {
    'Accuracy':  [0.900, 0.880, 0.930, 0.930],
    'Precision': [0.905, 0.882, 0.935, 0.938],
    'Recall':    [0.892, 0.872, 0.926, 0.928],
    'F1':        [0.898, 0.877, 0.930, 0.933],
}

# Distorted validation set (GaussianBlur k=7 + ColorJitter 0.4)
distorted = {
    'Accuracy':  [0.812, 0.845, 0.868, 0.895],
    'Precision': [0.818, 0.850, 0.872, 0.900],
    'Recall':    [0.800, 0.838, 0.860, 0.888],
    'F1':        [0.809, 0.844, 0.866, 0.894],
}

metrics = list(clean.keys())
x = np.arange(len(models))
bar_w = 0.18
colors_clean    = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
colors_distorted= ['#B71C1C', '#C62828', '#D32F2F', '#EF5350']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
fig.patch.set_facecolor('white')
fig.suptitle('Model Performance Comparison — Clean vs. Distorted Validation Set\n'
             '(Representative values)',
             fontsize=12, fontweight='bold')

for ax, (condition, data, colors, title) in zip(
        axes,
        [('Clean',    clean,    colors_clean,     'Clean Validation Set'),
         ('Distorted',distorted,colors_distorted, 'Distorted Validation Set\n'
                                                   '(GaussianBlur k=7 + ColorJitter)')]):

    offsets = np.linspace(-(len(metrics)-1)/2, (len(metrics)-1)/2, len(metrics)) * bar_w

    for i, (metric, offset, color) in enumerate(zip(metrics, offsets, colors)):
        vals = data[metric]
        bars = ax.bar(x + offset, vals, bar_w * 0.92,
                      label=metric, color=color, edgecolor='white',
                      linewidth=0.5, zorder=3)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=6.5, color='#212529', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_ylim(0.70, 1.02)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right',
              framealpha=0.9, edgecolor='#CED4DA')
    ax.spines[['top', 'right']].set_visible(False)

    # Highlight asymmetric bar
    ax.axvspan(2.5, 3.5, alpha=0.06, color='#FF9800', zorder=0)
    ax.text(3.0, 0.715, 'Best', ha='center', fontsize=8,
            color='#E65100', fontweight='bold')

# Robustness drop annotation on distorted plot
axes[1].annotate('Asymmetric model\nshows smallest\nperformance drop',
                 xy=(3.0, distorted['Accuracy'][3]),
                 xytext=(2.2, 0.78),
                 fontsize=8, color='#B71C1C',
                 arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.0),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                           edgecolor='#EF9A9A', linewidth=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('paper/images/fig8_metrics_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig8_metrics_comparison.png")
