"""
Figure 2: Deepfake Detection Pipeline
End-to-end pipeline from raw video to decision.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

fig, ax = plt.subplots(figsize=(16, 4.5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 4.5)
ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(8, 4.2, 'Deepfake Detection Pipeline',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Pipeline stages: (x_center, label, sublabel, color)
stages = [
    (1.0,  'Raw Video /\nImage Input',    '',                          '#E8F4FD'),
    (3.0,  'Frame\nExtraction',           'Every 5th frame\nmax 10/video', '#FFF3CD'),
    (5.0,  'Face\nDetection',             'Haar Cascade\nscaleFactor=1.2', '#FFF3CD'),
    (7.0,  'Face Crop\n& Resize',         '224×224 px\n+20px padding',  '#FFF3CD'),
    (9.0,  'Dual-Stream\nPreprocessing',  'RGB + FFT\nper-channel',     '#D4EDDA'),
    (11.0, 'Four-Model\nInference',       'Spatial / Freq\nHybrid / Asym', '#E2D9F3'),
    (13.0, 'Ensemble\nVoting',            '≥3 FAKE votes\n→ HIGH PROB', '#D1ECF1'),
    (15.0, 'Decision\nOutput',            'REAL /\nSUSPICIOUS / FAKE',  '#FFFFFF'),
]

BOX_W = 1.6
BOX_H = 1.8
BOX_Y = 1.2

C_BORDER = '#343A40'
C_ARROW  = '#495057'

for i, (xc, label, sub, color) in enumerate(stages):
    x0 = xc - BOX_W / 2
    rect = FancyBboxPatch((x0, BOX_Y), BOX_W, BOX_H,
                          boxstyle="round,pad=0.07",
                          linewidth=1.2,
                          edgecolor=C_BORDER,
                          facecolor=color)
    ax.add_patch(rect)
    ax.text(xc, BOX_Y + BOX_H * 0.62,
            label, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#212529')
    if sub:
        ax.text(xc, BOX_Y + BOX_H * 0.25,
                sub, ha='center', va='center',
                fontsize=7.2, color='#495057', style='italic')

    # Step number badge
    ax.text(xc, BOX_Y + BOX_H + 0.18,
            f'Step {i+1}', ha='center', va='center',
            fontsize=7, color='#6C757D')

    # Arrow to next
    if i < len(stages) - 1:
        ax.annotate('', xy=(xc + BOX_W/2 + 0.18, BOX_Y + BOX_H/2),
                    xytext=(xc + BOX_W/2, BOX_Y + BOX_H/2),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                   lw=1.4, mutation_scale=13))

# Highlight asymmetric path note
ax.annotate('Asymmetric training:\naugment RGB before FFT',
            xy=(9.0, BOX_Y), xytext=(9.0, 0.35),
            ha='center', fontsize=7.5, color='#721C24',
            arrowprops=dict(arrowstyle='->', color='#721C24', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0',
                      edgecolor='#F5C6CB', linewidth=0.8))

plt.tight_layout(pad=0.3)
plt.savefig('paper/images/fig2_detection_pipeline.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig2_detection_pipeline.png")
