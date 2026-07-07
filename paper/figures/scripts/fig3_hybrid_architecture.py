"""
Figure 3: Spatial-Frequency Hybrid Architecture
Detailed dual-stream ResNet-18 architecture diagram.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
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
    yoff = 0.12 if sub else 0
    ax.text(x+w/2, y+h/2+yoff, label, ha='center', va='center',
            fontsize=fs, fontweight=weight, color='#212529')
    if sub:
        ax.text(x+w/2, y+h/2-0.2, sub, ha='center', va='center',
                fontsize=7, color='#495057', style='italic')

def arr(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.3, mutation_scale=13))

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(7.5, 7.7, 'Spatial-Frequency Hybrid Architecture (HybridModel)',
        ha='center', va='center', fontsize=12, fontweight='bold')

# ── Inputs ─────────────────────────────────────────────────────────────────
box(ax, 0.2, 5.8, 2.2, 0.75, 'RGB Face Image',
    '[B, 3, 224, 224]', '#E8F4FD', bold=True)
box(ax, 0.2, 3.8, 2.2, 0.75, 'FFT Magnitude Image',
    '[B, 3, 224, 224]', '#F8D7DA', bold=True)

# ── ResNet-18 blocks ───────────────────────────────────────────────────────
# Spatial stream
resnet_layers_s = [
    (2.8, 5.8, 'Conv1\n7×7, 64', '#D4EDDA'),
    (4.2, 5.8, 'Layer1–4\nResidual Blocks', '#D4EDDA'),
    (5.8, 5.8, 'AvgPool\n1×1', '#D4EDDA'),
    (7.0, 5.8, 'Flatten\n512-d', '#D4EDDA'),
]
for (x, y, lbl, c) in resnet_layers_s:
    box(ax, x, y, 1.1, 0.75, lbl, '', c, fs=7.5)

# Frequency stream
resnet_layers_f = [
    (2.8, 3.8, 'Conv1\n7×7, 64', '#F8D7DA'),
    (4.2, 3.8, 'Layer1–4\nResidual Blocks', '#F8D7DA'),
    (5.8, 3.8, 'AvgPool\n1×1', '#F8D7DA'),
    (7.0, 3.8, 'Flatten\n512-d', '#F8D7DA'),
]
for (x, y, lbl, c) in resnet_layers_f:
    box(ax, x, y, 1.1, 0.75, lbl, '', c, fs=7.5)

# Arrows within streams
for i in range(len(resnet_layers_s)-1):
    arr(ax, resnet_layers_s[i][0]+1.1, resnet_layers_s[i][1]+0.375,
        resnet_layers_s[i+1][0],       resnet_layers_s[i+1][1]+0.375)
    arr(ax, resnet_layers_f[i][0]+1.1, resnet_layers_f[i][1]+0.375,
        resnet_layers_f[i+1][0],       resnet_layers_f[i+1][1]+0.375)

# Input → first conv
arr(ax, 2.4, 6.175, 2.8, 6.175)
arr(ax, 2.4, 4.175, 2.8, 4.175)

# ── Concatenation ──────────────────────────────────────────────────────────
box(ax, 8.4, 4.6, 1.5, 1.0, 'Concat\n[512 ‖ 512]',
    '→ 1024-d', '#E2D9F3', bold=True)

# Flatten → concat
arr(ax, 8.1, 6.175, 8.65, 5.6)
arr(ax, 8.1, 4.175, 8.65, 4.6)

# ── Classifier head ────────────────────────────────────────────────────────
clf_layers = [
    (10.2, 4.6, 'Dropout\n(0.3)', '#F3E5F5'),
    (11.5, 4.6, 'Linear\n1024→512', '#E2D9F3'),
    (12.8, 4.6, 'ReLU +\nDropout(0.3)', '#F3E5F5'),
    (10.2, 3.3, 'Linear\n512→256', '#E2D9F3'),
    (11.5, 3.3, 'ReLU +\nDropout(0.2)', '#F3E5F5'),
    (12.8, 3.3, 'Linear\n256→2', '#E2D9F3'),
]
for (x, y, lbl, c) in clf_layers:
    box(ax, x, y, 1.1, 0.75, lbl, '', c, fs=7.5)

# Concat → first clf layer
arr(ax, 9.9, 5.1, 10.2, 4.975)

# Arrows within classifier (row 1)
arr(ax, 11.3, 4.975, 11.5, 4.975)
arr(ax, 12.6, 4.975, 12.8, 4.975)
# Row 1 → row 2
arr(ax, 13.9, 4.975, 13.9, 4.05)
arr(ax, 13.9, 4.05, 12.8+1.1, 3.675)
# Row 2 arrows
arr(ax, 11.3, 3.675, 11.5, 3.675)
arr(ax, 12.6, 3.675, 12.8, 3.675)

# ── Output ─────────────────────────────────────────────────────────────────
box(ax, 11.5, 2.1, 2.0, 0.75, 'Softmax Output',
    '[B, 2]  (Real / Fake)', '#D1ECF1', bold=True)
arr(ax, 13.35, 3.3, 12.5, 2.85)

# ── Stream labels ──────────────────────────────────────────────────────────
ax.text(5.5, 7.0, 'Spatial Stream  (ResNet-18, pretrained ImageNet)',
        ha='center', va='center', fontsize=9, color='#155724',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D4EDDA',
                  edgecolor='#C3E6CB', linewidth=0.8))
ax.text(5.5, 3.2, 'Frequency Stream  (ResNet-18, pretrained ImageNet)',
        ha='center', va='center', fontsize=9, color='#721C24',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8D7DA',
                  edgecolor='#F5C6CB', linewidth=0.8))

# ── Asymmetric note ────────────────────────────────────────────────────────
ax.text(1.3, 2.8,
        'Asymmetric variant:\nAugment RGB → compute FFT\n(distorted spectrum)',
        ha='center', va='center', fontsize=7.5, color='#721C24',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0',
                  edgecolor='#F5C6CB', linewidth=0.8))
ax.annotate('', xy=(1.3, 3.8), xytext=(1.3, 3.2),
            arrowprops=dict(arrowstyle='->', color='#721C24', lw=1.0))

# ── Dimension annotations ──────────────────────────────────────────────────
ax.text(4.75, 7.35, '→ [B,64,112,112]', ha='center', fontsize=6.5, color='#6C757D')
ax.text(6.35, 7.35, '→ [B,512,1,1]',    ha='center', fontsize=6.5, color='#6C757D')
ax.text(7.55, 7.35, '→ [B,512]',        ha='center', fontsize=6.5, color='#6C757D')

# ── Legend ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor='#D4EDDA', edgecolor=C_BORDER, label='Spatial stream'),
    mpatches.Patch(facecolor='#F8D7DA', edgecolor=C_BORDER, label='Frequency stream'),
    mpatches.Patch(facecolor='#E2D9F3', edgecolor=C_BORDER, label='Fusion / Linear'),
    mpatches.Patch(facecolor='#F3E5F5', edgecolor=C_BORDER, label='Dropout / Activation'),
    mpatches.Patch(facecolor='#D1ECF1', edgecolor=C_BORDER, label='Output'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=8,
          framealpha=0.95, edgecolor='#CED4DA',
          bbox_to_anchor=(0.0, 0.0))

plt.tight_layout(pad=0.3)
plt.savefig('paper/images/fig3_hybrid_architecture.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig3_hybrid_architecture.png")
