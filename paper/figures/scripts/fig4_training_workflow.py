"""
Figure 4: Training Workflow
Shows the four independent training pipelines and the asymmetric data flow.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

fig, ax = plt.subplots(figsize=(16, 6.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6.5)
ax.axis('off')
fig.patch.set_facecolor('white')

C_BORDER = '#343A40'
C_ARROW  = '#495057'

def box(ax, x, y, w, h, label, sub='', color='#FFFFFF', fs=8.5, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.06",
                          linewidth=1.1, edgecolor=C_BORDER, facecolor=color)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    yoff = 0.13 if sub else 0
    ax.text(x+w/2, y+h/2+yoff, label, ha='center', va='center',
            fontsize=fs, fontweight=weight, color='#212529')
    if sub:
        ax.text(x+w/2, y+h/2-0.22, sub, ha='center', va='center',
                fontsize=7, color='#495057', style='italic')

def arr(ax, x1, y1, x2, y2, color=C_ARROW, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=1.3, mutation_scale=13))

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(8, 6.25, 'Training Workflow — Four Independent Training Pipelines',
        ha='center', va='center', fontsize=12, fontweight='bold')

# ── Shared data source ─────────────────────────────────────────────────────
box(ax, 6.5, 5.2, 3.0, 0.75, 'FaceForensics++ C23',
    'Deepfakes + FaceSwap\n3,963 face images', '#E8F4FD', bold=True)

box(ax, 6.5, 4.2, 3.0, 0.75, 'Stratified Split',
    'Train 70% / Val 15% / Test 15%\nrandom_state=42', '#FFF3CD')

arr(ax, 8.0, 5.2, 8.0, 4.95)
arr(ax, 8.0, 4.2, 8.0, 3.95)

# ── Four pipeline columns ──────────────────────────────────────────────────
pipelines = [
    # (x_center, title, dataset_cls, transform, model, save, color)
    (2.0,  'Pipeline 1\nSpatial',
     'DeepfakeDataset',
     'spatial_train_transform\n(mild blur p=0.2)',
     'SpatialModel\nResNet-18 → 512-d → 2',
     'spatial.pth',
     '#D4EDDA'),
    (6.0,  'Pipeline 2\nFrequency',
     'DeepfakeFreqDataset',
     'frequency_train_transform\n(stronger blur p=0.3)',
     'FrequencyModel\nResNet-18 → 512-d → 2',
     'frequency.pth',
     '#F8D7DA'),
    (10.0, 'Pipeline 3\nHybrid',
     'HybridDataset',
     'spatial + freq transforms\n(FFT on clean image)',
     'HybridModel\n2×ResNet-18 → 1024-d → 2',
     'hybrid.pth',
     '#E2D9F3'),
    (14.0, 'Pipeline 4\nAsymmetric',
     'AsymmetricDataset',
     'spatial clean +\nFFT on distorted image',
     'HybridModel\n(same arch, robust training)',
     'asymmetric.pth',
     '#FFF0F0'),
]

for (xc, title, ds, tf, model, save, color) in pipelines:
    x0 = xc - 1.7
    # Header
    box(ax, x0, 3.3, 3.4, 0.6, title, '', color, fs=9, bold=True)
    # Dataset
    box(ax, x0, 2.55, 3.4, 0.6, ds, '', '#F8F9FA', fs=7.5)
    # Transform
    box(ax, x0, 1.8, 3.4, 0.6, tf, '', '#FFFDE7', fs=7.2)
    # Model
    box(ax, x0, 1.05, 3.4, 0.6, model, '', color, fs=7.5)
    # Save
    box(ax, x0, 0.3, 3.4, 0.6, f'Save: {save}', '', '#D1ECF1', fs=7.5)

    # Arrows within pipeline
    arr(ax, xc, 3.95, xc, 3.9)
    arr(ax, xc, 3.3, xc, 3.15)
    arr(ax, xc, 2.55, xc, 2.4)
    arr(ax, xc, 1.8, xc, 1.65)
    arr(ax, xc, 1.05, xc, 0.9)

    # Arrow from split to pipeline header
    arr(ax, 8.0, 3.95, xc, 3.9)

# ── Common training config box ─────────────────────────────────────────────
box(ax, 5.5, 0.0, 5.0, 0.25,
    'All pipelines: Adam lr=1e-4 | CrossEntropyLoss | 5 epochs | batch=8 | best val-acc checkpoint',
    '', '#F8F9FA', fs=7.5)

# ── Asymmetric highlight ───────────────────────────────────────────────────
ax.annotate('Key novelty:\naugment RGB → FFT\n(distorted spectrum)',
            xy=(14.0, 1.8), xytext=(14.0, 0.55),
            ha='center', fontsize=7, color='#721C24',
            arrowprops=dict(arrowstyle='->', color='#721C24', lw=0.9),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0',
                      edgecolor='#F5C6CB', linewidth=0.8))

plt.tight_layout(pad=0.3)
plt.savefig('paper/images/fig4_training_workflow.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig4_training_workflow.png")
