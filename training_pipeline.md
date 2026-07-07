# Training Pipeline

## Common Training Configuration (all four models)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss function | CrossEntropyLoss |
| Batch size | 8 |
| Epochs | 5 |
| num_workers | 0 |
| Model selection | Best validation accuracy |

All models are trained independently. There is no joint training, no learning rate scheduling, no weight decay, and no early stopping beyond the 5-epoch limit.

---

## Training Scripts

### 1. Spatial Model (`train/train_spatial.py`)

- Dataset: `DeepfakeDataset` with `get_spatial_train_transform()`
- Validation: `DeepfakeDataset` with `get_spatial_infer_transform()`
- Input: RGB face images
- Saves best model to `models/spatial.pth`

### 2. Frequency Model (`train/train_frequency.py`)

- Dataset: `DeepfakeFreqDataset` with `get_frequency_train_transform()`
- Validation: `DeepfakeFreqDataset` with `get_frequency_infer_transform()`
- Input: FFT magnitude spectrum images (computed on-the-fly from RGB)
- Saves best model to `models/frequency.pth`

### 3. Hybrid Model (`train/train_hybrid.py`)

- Dataset: `HybridDataset` with spatial + frequency train transforms
- Validation: `HybridDataset` with spatial + frequency infer transforms
- Input: (RGB tensor, FFT tensor) pairs
- Forward: `model(rgb, fft)`
- Saves best model to `models/hybrid.pth`

### 4. Asymmetric Model (`train/train_asymmetric.py`)

- Dataset: `AsymmetricDataset` with spatial + frequency train transforms
- Validation: `AsymmetricDataset` with spatial + frequency infer transforms
- Model class: `HybridModel` (same architecture as Hybrid)
- Input: (clean RGB tensor, distorted FFT tensor) pairs
- Saves best model to `models/asymmetric.pth`

---

## Augmentation Strategy

### Spatial Branch Training Transform (`get_spatial_train_transform`)

```
ToPILImage()
Resize((224, 224))
RandomApply([GaussianBlur(kernel_size=3)], p=0.2)
RandomApply([ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)], p=0.2)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Mild augmentation — preserves spatial texture integrity for the spatial branch.

### Frequency Branch Training Transform (`get_frequency_train_transform`)

```
ToPILImage()
Resize((224, 224))
RandomApply([GaussianBlur(kernel_size=5)], p=0.3)
RandomApply([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Stronger augmentation — larger blur kernel, higher distortion probability and magnitude.

### Inference/Validation Transforms (both branches)

```
ToPILImage()
Resize((224, 224))
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Deterministic — no randomness during validation or inference.

---

## Asymmetric Dataset Pipeline (Key Novelty)

The `AsymmetricDataset` implements a unique data flow for the frequency branch:

```
1. Load RGB image
2. Spatial branch: apply spatial_train_transform(RGB) → clean tensor
3. Frequency branch:
   a. Apply frequency_train_transform(RGB) → distorted tensor
   b. Convert tensor back to numpy uint8 image (denormalize + clip)
   c. Compute FFT on the distorted image
   d. Apply fft_tensor_transform (ToTensor + Normalize)
   → distorted FFT tensor
4. Return (clean_spatial_tensor, distorted_fft_tensor, label)
```

This means the frequency branch sees FFT representations of augmented/distorted images during training, forcing it to learn frequency-domain features that are robust to blur and color distortion.

---

## Validation Metrics (computed each epoch)

All training scripts compute and print the following on the validation set after each epoch:

- Accuracy (sklearn `accuracy_score`)
- Precision (sklearn `precision_score`, zero_division=0)
- Recall (sklearn `recall_score`, zero_division=0)
- F1 Score (sklearn `f1_score`, zero_division=0)
- ROC-AUC (sklearn `roc_auc_score`, with try/except fallback to 0.0)

Label mapping: 0 = REAL, 1 = FAKE. ROC-AUC uses `probs[:, 1]` (fake class probability).

---

## Robustness Evaluation (`evaluate_distortion.py`)

Evaluates all four models on both clean and distorted versions of the validation set.

### Distortion Transform (deterministic, applied at eval time)

```
ToPILImage()
Resize((224, 224))
GaussianBlur(kernel_size=7)           # stronger than training blur
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

For hybrid/asymmetric models under distortion, both branches receive the distorted transform.

### Evaluation Metrics

Accuracy, Precision, Recall, F1, ROC-AUC — same as training validation.

---

## No Training Logs Available

No log files, metric JSON files, or saved epoch-by-epoch results exist in the workspace. The only artifacts from training are the four `.pth` checkpoint files. Actual numerical results from training runs are not available on disk.
