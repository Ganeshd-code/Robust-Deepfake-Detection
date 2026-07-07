# Architecture Summary

## Overview

The system implements four model variants, all built on ResNet-18 backbones pretrained on ImageNet. The four models form a progression from unimodal to multimodal to robustness-aware multimodal.

---

## Model 1: Spatial Model (`models/spatial_model.py`)

Processes standard RGB face images.

```
Input: RGB image tensor [B, 3, 224, 224]
  └─ ResNet-18 feature extractor (all layers except final FC)
       └─ Output: [B, 512, 1, 1]
  └─ Flatten → [B, 512]
  └─ Dropout(0.3)
  └─ Linear(512 → 256)
  └─ ReLU
  └─ Dropout(0.2)
  └─ Linear(256 → 2)
Output: logits [B, 2]  (class 0=real, class 1=fake)
```

- Backbone: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`
- Feature extractor: `nn.Sequential(*list(base.children())[:-1])` — removes original FC layer
- Parameters: ~11.2M (ResNet-18 backbone) + small classifier head

---

## Model 2: Frequency Model (`models/frequency_model.py`)

Identical architecture to SpatialModel. Processes FFT magnitude spectrum images instead of RGB.

```
Input: FFT magnitude image tensor [B, 3, 224, 224]
  └─ ResNet-18 feature extractor
       └─ Output: [B, 512, 1, 1]
  └─ Flatten → [B, 512]
  └─ Dropout(0.3)
  └─ Linear(512 → 256)
  └─ ReLU
  └─ Dropout(0.2)
  └─ Linear(256 → 2)
Output: logits [B, 2]
```

The distinction is entirely in the input: FFT images are computed per-channel and passed through the same ResNet-18 architecture.

---

## Model 3: Hybrid Model (`models/hybrid_model.py`)

Dual-stream architecture with concatenation fusion. Used for both the "Hybrid" and "Asymmetric" trained variants.

```
Input A: RGB tensor [B, 3, 224, 224]
Input B: FFT tensor [B, 3, 224, 224]

Stream A (Spatial):
  └─ ResNet-18 backbone → [B, 512, 1, 1] → Flatten → [B, 512]

Stream B (Frequency):
  └─ ResNet-18 backbone → [B, 512, 1, 1] → Flatten → [B, 512]

Fusion:
  └─ torch.cat([spatial_features, frequency_features], dim=1) → [B, 1024]

Classifier:
  └─ Dropout(0.3)
  └─ Linear(1024 → 512)
  └─ ReLU
  └─ Dropout(0.3)
  └─ Linear(512 → 256)
  └─ ReLU
  └─ Dropout(0.2)
  └─ Linear(256 → 2)

Output: logits [B, 2]
```

- Two independent ResNet-18 backbones (not weight-shared)
- Fusion mechanism: simple feature concatenation (no attention, no learned weighting)
- Classifier: 3-layer MLP with progressive dropout

---

## FFT Feature Extraction (`utils/fft.py`)

Converts an RGB image into a 3-channel FFT magnitude spectrum image.

```python
For each channel c in {R, G, B}:
    fft = np.fft.fft2(channel.astype(float32))
    fft_shift = np.fft.fftshift(fft)          # center low frequencies
    magnitude = np.abs(fft_shift)              # magnitude spectrum
    magnitude = np.log1p(magnitude)            # log scaling for dynamic range
    magnitude = cv2.normalize(magnitude, 0, 255, NORM_MINMAX).astype(uint8)

Output: np.stack([R_fft, G_fft, B_fft], axis=2)  → shape (H, W, 3)
```

The FFT image is then passed through the same normalization pipeline as RGB images.

---

## Model 4: Asymmetric Model

Uses the same `HybridModel` architecture as Model 3. The difference is in the **training data pipeline**, not the model architecture.

The asymmetric design is in `AsymmetricDataset`:
- Spatial branch receives clean RGB images (mild augmentation)
- Frequency branch receives augmented images **before** FFT computation
  - Augmentation is applied to the RGB image first
  - Then FFT is computed on the distorted image
  - This forces the frequency branch to learn robust frequency features under distortion

This is the key novelty: the frequency branch is trained on distorted FFT representations, making it more robust to real-world image degradation.

---

## Architecture Comparison

| Property | Spatial | Frequency | Hybrid | Asymmetric |
|----------|---------|-----------|--------|------------|
| Input streams | 1 (RGB) | 1 (FFT) | 2 (RGB+FFT) | 2 (RGB+distorted FFT) |
| Backbone | ResNet-18 | ResNet-18 | 2× ResNet-18 | 2× ResNet-18 |
| Feature dim | 512 | 512 | 1024 (concat) | 1024 (concat) |
| Fusion | N/A | N/A | Concatenation | Concatenation |
| Architecture class | SpatialModel | FrequencyModel | HybridModel | HybridModel |
| Robustness training | No | No | No | Yes |
| Inference threshold | argmax | argmax | argmax | 0.90 (fake), 0.70 (suspicious) |

---

## Inference Decision Logic (app.py)

Standard models (Spatial, Frequency, Hybrid): argmax of softmax probabilities.

Asymmetric model uses a tiered threshold:
- `fake_prob >= 0.90` → FAKE
- `fake_prob >= 0.70` → SUSPICIOUS
- otherwise → REAL

Final ensemble verdict: if 3+ of 4 models vote FAKE → HIGH PROBABILITY; 1+ vote FAKE/SUSPICIOUS → SUSPICIOUS; else REAL.

Video mode: samples every 15th frame, averages fake scores per model, final score = mean(hybrid_avg, asymmetric_avg).
