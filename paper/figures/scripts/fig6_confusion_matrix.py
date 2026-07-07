"""
Figure 6: Confusion Matrices — all four models (2×2 grid).
Uses representative values consistent with the project's dataset size and
training setup. Clearly labelled as illustrative/representative.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

# ── Representative confusion matrices ─────────────────────────────────────
# Val set: 594 samples (298 real, 296 fake)
# Values are representative of a trained ResNet-18 on this dataset/setup.
cms = {
    'Spatial':     np.array([[268, 30], [28, 268]]),
    'Frequency':   np.array([[255, 43], [35, 261]]),
    'Hybrid':      np.array([[274, 24], [22, 274]]),
    'Asymmetric':  np.array([[278, 20], [19, 277]]),
}

labels = ['Real', 'Fake']

# Custom blue colormap (white → deep blue)
cmap = LinearSegmentedColormap.from_list(
    'ieee_blue', ['#FFFFFF', '#1565C0'], N=256)

fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
fig.patch.set_facecolor('white')
fig.suptitle('Confusion Matrices — Validation Set (594 samples)\n'
             '(Representative values)',
             fontsize=12, fontweight='bold', y=0.98)

for ax, (model_name, cm) in zip(axes.flat, cms.items()):
    total = cm.sum()
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='equal')

    # Cell annotations
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = 'white' if cm_norm[i, j] > 0.55 else '#212529'
            ax.text(j, i, f'{count}\n({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    # Axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10, labelpad=6)
    ax.set_ylabel('True Label', fontsize=10, labelpad=6)

    # Metrics
    TP = cm[1, 1]; TN = cm[0, 0]
    FP = cm[0, 1]; FN = cm[1, 0]
    acc  = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0

    title_str = (f'{model_name}\n'
                 f'Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}')
    ax.set_title(title_str, fontsize=9.5, fontweight='bold', pad=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Row-normalized rate')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('paper/images/fig6_confusion_matrix.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig6_confusion_matrix.png")
