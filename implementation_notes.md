# Implementation Notes

## Key Implementation Details

### Dataset Classes

#### `DeepfakeDataset` (utils/dataset_spatial.py)
- Reads CSV with `path` and `label` columns
- Loads image with `cv2.imread` → converts BGR to RGB
- Applies transform → returns `(tensor, torch.long label)`

#### `DeepfakeFreqDataset` (utils/dataset_freq.py)
- Same as above but calls `get_fft(img)` after BGR→RGB conversion
- Applies transform to the FFT image, not the original RGB
- Returns `(fft_tensor, label)`

#### `HybridDataset` (utils/dataset_hybrid.py)
- Loads RGB image, applies `t_rgb` transform → `rgb` tensor
- Computes FFT on original RGB, applies `t_fft` transform → `fft` tensor
- Returns `(rgb, fft, label)`
- Note: FFT is computed on the **original** (non-augmented) image

#### `AsymmetricDataset` (utils/dataset_asymmetric.py)
- Loads RGB image, applies `t_s` (spatial transform) → `spatial` tensor
- Applies `t_f` (frequency/distortion transform) to RGB → distorted tensor
- Converts distorted tensor back to numpy: `permute(1,2,0).cpu().numpy()`
- Clips to [0,1] and scales to uint8
- Computes FFT on the distorted uint8 image
- Applies `fft_tensor_transform` (ToTensor + Normalize) → `fft` tensor
- Returns `(spatial, fft, label)`
- Has its own internal `fft_tensor_transform` — does NOT use the passed `t_f` for the final FFT tensor

### Important: Asymmetric vs Hybrid Dataset Difference

In `HybridDataset`, FFT is computed on the **original clean image** before any augmentation.
In `AsymmetricDataset`, FFT is computed on the **augmented/distorted image**.

This is the sole implementation difference between the Hybrid and Asymmetric training pipelines.

---

## FFT Implementation Details (`utils/fft.py`)

```python
image = image.astype(np.float32)
for i in range(3):
    channel = image[:, :, i]
    fft = np.fft.fft2(channel)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    magnitude = np.log1p(magnitude)          # log(1 + |F|) for dynamic range compression
    magnitude = cv2.normalize(magnitude, normalized, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channels.append(magnitude)
return np.stack(channels, axis=2)
```

- `np.fft.fftshift` centers the DC component (low frequencies at center)
- `np.log1p` compresses the wide dynamic range of FFT magnitudes
- `cv2.normalize` with NORM_MINMAX maps to [0, 255] per-channel independently
- Output is uint8 HWC numpy array, same spatial size as input

---

## Model Architecture Notes

### ResNet-18 Feature Extractor
```python
base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
self.features = nn.Sequential(*list(base.children())[:-1])
```
`list(base.children())[:-1]` removes the final `Linear(512, 1000)` layer but keeps the `AdaptiveAvgPool2d(1,1)`. Output shape after pool: `[B, 512, 1, 1]`, flattened to `[B, 512]`.

### Hybrid Model Fusion
```python
fused = torch.cat([spatial_features, frequency_features], dim=1)  # [B, 1024]
```
Simple concatenation — no attention, no gating, no learned weighting.

### Dropout Schedule
- Single-stream models: Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→2)
- Hybrid/Asymmetric: Dropout(0.3) → Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→2)

---

## Training Loop Notes

All four training scripts follow the same pattern:
1. Forward pass
2. `criterion(outputs, y)` — CrossEntropyLoss on raw logits
3. `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
4. After each epoch: full validation pass with `torch.no_grad()`
5. Save checkpoint only if `val_acc > best_acc`

No gradient clipping, no mixed precision, no learning rate warmup or decay.

---

## Inference Pipeline (`app.py`)

```python
# Image preprocessing
fft_image = get_fft(image)                          # compute FFT
rgb_tensor = spatial_transform(image).unsqueeze(0)  # [1, 3, 224, 224]
fft_tensor = frequency_transform(fft_image).unsqueeze(0)

# Per-model inference
spatial_probs = softmax(spatial_model(rgb_tensor))
frequency_probs = softmax(frequency_model(fft_tensor))
hybrid_probs = softmax(hybrid_model(rgb_tensor, fft_tensor))
asymmetric_probs = softmax(asymmetric_model(rgb_tensor, fft_tensor))
```

All four models run on the same `rgb_tensor` and `fft_tensor`. The asymmetric model at inference uses the same clean FFT as the hybrid model — the asymmetric training augmentation only applies during training.

---

## Robustness Evaluation Notes (`evaluate_distortion.py`)

The distorted transform applies:
- `GaussianBlur(kernel_size=7)` — deterministic (no randomness)
- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3)` — deterministic at eval

For hybrid/asymmetric under distortion, **both** branches receive the distorted transform. This is a worst-case scenario test — the asymmetric model was only trained with distortion on the frequency branch, not both.

DataLoader: `batch_size=8`, `shuffle=False`, `num_workers=0`.

---

## Known Implementation Observations

1. The `AsymmetricDataset` denormalization step clips to [0,1] before scaling to uint8. If the frequency transform normalizes with ImageNet stats, the denormalization is approximate (no explicit inverse normalization). This means the distorted image fed to FFT may have slightly different pixel values than expected, but the distortion pattern is preserved.

2. The `evaluate_distortion.py` uses `val.csv` for both clean and distorted evaluation, not `test.csv`. The held-out test set is never evaluated in any script.

3. `paper/generate_figures.py` is empty — no figure generation code exists.

4. The `paper_texts.json` file contains extracted text from 12 research papers, used for literature review via `analyze_papers.py`.

5. All four model checkpoints (`spatial.pth`, `frequency.pth`, `hybrid.pth`, `asymmetric.pth`) exist in `models/`, confirming training was completed.
