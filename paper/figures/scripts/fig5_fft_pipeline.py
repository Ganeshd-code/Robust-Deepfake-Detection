"""
Figure 5: Frequency-Domain Preprocessing Pipeline
Shows the per-channel FFT computation steps with synthetic example spectra.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

fig = plt.figure(figsize=(16, 6))
fig.patch.set_facecolor('white')

# ── Title ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, 'Frequency-Domain Preprocessing Pipeline (get_fft)',
         ha='center', va='top', fontsize=12, fontweight='bold')

# ── Top row: pipeline diagram ──────────────────────────────────────────────
ax_diag = fig.add_axes([0.0, 0.55, 1.0, 0.38])
ax_diag.set_xlim(0, 16)
ax_diag.set_ylim(0, 2.5)
ax_diag.axis('off')

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
        ax.text(x+w/2, y+h/2-0.22, sub, ha='center', va='center',
                fontsize=7, color='#495057', style='italic')

def arr(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.3, mutation_scale=13))

steps = [
    (0.3,  'RGB Input\nImage',         'float32\nH×W×3',          '#E8F4FD'),
    (2.5,  'Split into\nR, G, B',      'Per-channel\nprocessing',  '#FFF3CD'),
    (4.7,  'np.fft.fft2\n(channel)',   '2D DFT\ncomplex output',   '#D4EDDA'),
    (6.9,  'np.fft.fftshift',          'Center DC\ncomponent',     '#D4EDDA'),
    (9.1,  'np.abs(fft_shift)',        'Magnitude\nspectrum',      '#D4EDDA'),
    (11.3, 'np.log1p\n(magnitude)',    'Log scaling\ndynamic range','#D4EDDA'),
    (13.5, 'cv2.normalize\n[0, 255]',  'NORM_MINMAX\nuint8',       '#D4EDDA'),
]

for i, (x, lbl, sub, c) in enumerate(steps):
    box(ax_diag, x, 0.5, 1.9, 1.5, lbl, sub, c, fs=8, bold=(i==0))
    if i < len(steps)-1:
        arr(ax_diag, x+1.9, 1.25, x+2.5, 1.25)

# Stack note
box(ax_diag, 15.5, 0.5, 0.4, 1.5, '×3', '', '#E2D9F3', fs=9, bold=True)
ax_diag.text(15.7, 2.2, 'Stack R,G,B\n→ H×W×3 FFT image',
             ha='center', va='center', fontsize=7.5, color='#495057')

# ── Bottom row: synthetic spectrum visualizations ──────────────────────────
rng = np.random.default_rng(42)
N = 64

def make_face_fft(rng, N):
    """Synthetic FFT magnitude for a real face (smooth, concentrated center)."""
    img = rng.normal(0, 1, (N, N)).astype(np.float32)
    # Add low-frequency structure
    for f in [2, 4, 6]:
        img += 0.5 * np.sin(2*np.pi*f*np.arange(N)/N)[None, :]
        img += 0.5 * np.sin(2*np.pi*f*np.arange(N)/N)[:, None]
    fft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(fft))
    return mag / mag.max()

def make_fake_fft(rng, N):
    """Synthetic FFT for a deepfake (GAN grid artifacts at high freq)."""
    img = rng.normal(0, 1, (N, N)).astype(np.float32)
    # GAN upsampling artifacts: periodic high-freq spikes
    for f in [N//4, N//3, N//2-2]:
        img += 0.8 * np.sin(2*np.pi*f*np.arange(N)/N)[None, :]
        img += 0.8 * np.sin(2*np.pi*f*np.arange(N)/N)[:, None]
    fft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(fft))
    return mag / mag.max()

panels = [
    ('Real Face\nRGB Channel', make_face_fft(rng, N), 'Greens'),
    ('Real Face\nFFT Spectrum', make_face_fft(rng, N), 'hot'),
    ('Deepfake\nRGB Channel', make_fake_fft(rng, N), 'Reds'),
    ('Deepfake\nFFT Spectrum', make_fake_fft(rng, N), 'hot'),
]

axes_pos = [
    [0.04, 0.04, 0.20, 0.44],
    [0.27, 0.04, 0.20, 0.44],
    [0.54, 0.04, 0.20, 0.44],
    [0.77, 0.04, 0.20, 0.44],
]

for pos, (title, data, cmap) in zip(axes_pos, panels):
    ax = fig.add_axes(pos)
    im = ax.imshow(data, cmap=cmap, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('#343A40')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Annotation
fig.text(0.5, 0.02,
         'GAN upsampling artifacts manifest as periodic high-frequency spikes in the FFT spectrum, '
         'distinguishing deepfakes from real faces.',
         ha='center', va='bottom', fontsize=8, color='#495057', style='italic')

plt.savefig('paper/images/fig5_fft_pipeline.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig5_fft_pipeline.png")
