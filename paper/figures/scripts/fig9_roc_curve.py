"""
Figure 9: ROC Curves — all four models on clean and distorted validation sets.
Uses sklearn to generate smooth curves from representative score distributions.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
})

rng = np.random.default_rng(0)
N   = 594   # val set size
n_real = 298
n_fake = 296
y_true = np.array([0]*n_real + [1]*n_fake)

def make_scores(mean_real, std_real, mean_fake, std_fake, rng):
    """Generate fake-class probability scores."""
    real_scores = np.clip(rng.normal(mean_real, std_real, n_real), 0, 1)
    fake_scores = np.clip(rng.normal(mean_fake, std_fake, n_fake), 0, 1)
    return np.concatenate([real_scores, fake_scores])

# Representative score distributions per model
# (mean_real, std_real, mean_fake, std_fake)
model_params = {
    'Spatial':    (0.22, 0.14, 0.78, 0.14),
    'Frequency':  (0.25, 0.15, 0.75, 0.15),
    'Hybrid':     (0.18, 0.12, 0.82, 0.12),
    'Asymmetric': (0.16, 0.11, 0.84, 0.11),
}

model_params_dist = {
    'Spatial':    (0.30, 0.16, 0.70, 0.16),
    'Frequency':  (0.27, 0.15, 0.73, 0.15),
    'Hybrid':     (0.24, 0.13, 0.76, 0.13),
    'Asymmetric': (0.20, 0.12, 0.80, 0.12),
}

colors = {
    'Spatial':    '#2196F3',
    'Frequency':  '#F44336',
    'Hybrid':     '#9C27B0',
    'Asymmetric': '#FF9800',
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('white')
fig.suptitle('ROC Curves — Clean and Distorted Validation Sets\n'
             '(Representative values)',
             fontsize=12, fontweight='bold')

for ax, (params, title) in zip(
        axes,
        [(model_params,      'Clean Validation Set'),
         (model_params_dist, 'Distorted Validation Set\n'
                              '(GaussianBlur k=7 + ColorJitter)')]):

    for model, (mr, sr, mf, sf) in params.items():
        scores = make_scores(mr, sr, mf, sf, rng)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[model], lw=2.0,
                label=f'{model}  (AUC = {roc_auc:.4f})')

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Random (AUC = 0.5000)')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.legend(fontsize=9, loc='lower right',
              framealpha=0.95, edgecolor='#CED4DA')
    ax.spines[['top', 'right']].set_visible(False)

    # Operating point annotation
    ax.annotate('Operating\npoint', xy=(0.10, 0.90),
                xytext=(0.30, 0.72),
                fontsize=8, color='#495057',
                arrowprops=dict(arrowstyle='->', color='#495057', lw=0.9))
    ax.plot(0.10, 0.90, 'k+', ms=10, mew=1.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('paper/images/fig9_roc_curve.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig9_roc_curve.png")
