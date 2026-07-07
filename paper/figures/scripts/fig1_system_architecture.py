"""
Figure 1: System Architecture Diagram
IEEE-style block diagram of the full deepfake detection system.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──────────────────────────────────────────────────────────
C_INPUT   = '#E8F4FD'   # light blue
C_PREP    = '#FFF3CD'   # light yellow
C_SPATIAL = '#D4EDDA'   # light green
C_FREQ    = '#F8D7DA'   # light red/pink
C_FUSION  = '#E2D9F3'   # light purple
C_OUT     = '#D1ECF1'   # light cyan
C_BORDER  = '#343A40'   # dark grey
C_ARROW   = '#495057'

def box(ax, x, y, w, h, label, sublabel='', color='#FFFFFF', fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          linewidth=1.2,
                          edgecolor=C_BORDER,
                          facecolor=color)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
            label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='#212529')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22,
                sublabel, ha='center', va='center',
                fontsize=7.5, color='#495057', style='italic')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.4, mutation_scale=14))

def label_arrow(ax, x, y, text):
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.5, color='#6C757D', style='italic')

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(7, 6.7, 'Robust Deepfake Detection System — Architecture Overview',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#212529')

# ── Row 1: Input ───────────────────────────────────────────────────────────
box(ax, 0.3, 4.8, 2.0, 0.9, 'Input Media',
    'Image / Video', C_INPUT, bold=True)

box(ax, 0.3, 3.5, 2.0, 0.9, 'Face Detection',
    'Haar Cascade\n(OpenCV)', C_PREP)

box(ax, 0.3, 2.2, 2.0, 0.9, 'Face Crop & Resize',
    '224 × 224 px', C_PREP)

# arrows col 0
arrow(ax, 1.3, 4.8, 1.3, 4.4)
arrow(ax, 1.3, 3.5, 1.3, 3.1)

# ── Preprocessing split ────────────────────────────────────────────────────
# Spatial branch label
ax.text(5.0, 5.55, 'Spatial Branch', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#155724')
ax.text(9.2, 5.55, 'Frequency Branch', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#721C24')

# Spatial preprocessing
box(ax, 3.6, 4.8, 2.8, 0.9, 'RGB Normalization',
    'ImageNet mean/std', C_SPATIAL)

# Frequency preprocessing
box(ax, 7.8, 4.8, 2.8, 0.9, 'Per-Channel FFT',
    'log|F(ω)| → normalize', C_FREQ)

# Asymmetric augmentation note
box(ax, 7.8, 3.5, 2.8, 0.9, 'Asymmetric Augmentation',
    'Blur + ColorJitter\n(freq. branch only)', '#FFF0F0')

# arrows from face crop to both branches
arrow(ax, 2.3, 2.65, 3.6, 5.25)
arrow(ax, 2.3, 2.65, 7.8, 5.25)

# arrow freq branch
arrow(ax, 9.2, 4.8, 9.2, 4.4)

# ── Models ─────────────────────────────────────────────────────────────────
# Spatial model
box(ax, 3.6, 3.5, 2.8, 0.9, 'Spatial Model',
    'ResNet-18 → 512-d', C_SPATIAL)

# Frequency model
box(ax, 7.8, 2.2, 2.8, 0.9, 'Frequency Model',
    'ResNet-18 → 512-d', C_FREQ)

# Hybrid model
box(ax, 5.7, 2.2, 2.8, 0.9, 'Hybrid Model',
    'ResNet-18 × 2\nConcat → 1024-d', C_FUSION)

# Asymmetric model
box(ax, 3.6, 2.2, 1.9, 0.9, 'Asymmetric\nModel',
    'HybridModel\n(robust FFT)', '#F3E5F5')

# arrows to models
arrow(ax, 5.0, 4.8, 5.0, 4.4)   # spatial norm → spatial model
arrow(ax, 9.2, 3.5, 9.2, 3.1)   # aug → freq model
arrow(ax, 5.0, 3.5, 6.15, 3.1)  # spatial → hybrid
arrow(ax, 9.2, 3.5, 7.15, 3.1)  # freq aug → hybrid
arrow(ax, 5.0, 3.5, 4.55, 3.1)  # spatial → asymmetric

# ── Fusion / Ensemble ──────────────────────────────────────────────────────
box(ax, 5.0, 0.9, 4.0, 0.9, 'Ensemble Voting',
    '4-model majority vote\n(tiered threshold for Asymmetric)', C_OUT, bold=True)

# arrows to ensemble
arrow(ax, 5.0, 2.2, 5.8, 1.8)
arrow(ax, 6.15, 2.2, 6.6, 1.8)
arrow(ax, 7.15, 2.2, 7.2, 1.8)
arrow(ax, 9.2, 2.2, 8.2, 1.8)

# ── Output ─────────────────────────────────────────────────────────────────
box(ax, 5.5, 0.1, 3.0, 0.65, 'REAL / SUSPICIOUS / FAKE',
    '', '#FFFFFF', bold=True)
arrow(ax, 7.0, 0.9, 7.0, 0.75)

# ── Video path note ────────────────────────────────────────────────────────
box(ax, 11.0, 4.8, 2.7, 0.9, 'Video Mode',
    'Sample every 15th frame\nAverage fake scores', '#F8F9FA')
ax.annotate('', xy=(11.0, 5.25), xytext=(10.6, 5.25),
            arrowprops=dict(arrowstyle='<-', color='#6C757D', lw=1.2))

# ── Legend ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,   edgecolor=C_BORDER, label='Input / Output'),
    mpatches.Patch(facecolor=C_PREP,    edgecolor=C_BORDER, label='Preprocessing'),
    mpatches.Patch(facecolor=C_SPATIAL, edgecolor=C_BORDER, label='Spatial Branch'),
    mpatches.Patch(facecolor=C_FREQ,    edgecolor=C_BORDER, label='Frequency Branch'),
    mpatches.Patch(facecolor=C_FUSION,  edgecolor=C_BORDER, label='Fusion / Hybrid'),
    mpatches.Patch(facecolor=C_OUT,     edgecolor=C_BORDER, label='Ensemble / Decision'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=8,
          framealpha=0.95, edgecolor='#CED4DA',
          bbox_to_anchor=(0.0, 0.0))

plt.tight_layout(pad=0.3)
plt.savefig('paper/images/fig1_system_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig1_system_architecture.png")
