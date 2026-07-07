# Metrics Summary

## Important Notice

No training logs, evaluation output files, metric JSON files, confusion matrices, or saved numerical results exist anywhere in this workspace. The only training artifacts present are the four model checkpoint files:

- `models/spatial.pth`
- `models/frequency.pth`
- `models/hybrid.pth`
- `models/asymmetric.pth`

All metrics below describe what is **measured and reported** by the training and evaluation scripts, not actual recorded values.

---

## Metrics Computed During Training (per epoch, on validation set)

All four training scripts (`train_spatial.py`, `train_frequency.py`, `train_hybrid.py`, `train_asymmetric.py`) compute and print the following after each epoch:

| Metric | Implementation | Notes |
|--------|---------------|-------|
| Loss | `CrossEntropyLoss` average over training batches | Training loss only |
| Accuracy | `sklearn.accuracy_score` | Primary model selection criterion |
| Precision | `sklearn.precision_score(zero_division=0)` | Fake class (positive=1) |
| Recall | `sklearn.recall_score(zero_division=0)` | Fake class (positive=1) |
| F1 Score | `sklearn.f1_score(zero_division=0)` | Fake class |
| ROC-AUC | `sklearn.roc_auc_score` | Uses `probs[:, 1]` (fake probability) |

Model checkpoints are saved only when validation accuracy improves over the previous best.

---

## Metrics Computed During Robustness Evaluation (`evaluate_distortion.py`)

Evaluated on `data/splits/val.csv` under two conditions:

### Clean Condition
Standard inference transforms (resize + normalize, no augmentation).

### Distorted Condition
Deterministic distortion applied:
- `GaussianBlur(kernel_size=7)`
- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3)`

### Models Evaluated
1. Spatial (clean + distorted)
2. Frequency (clean + distorted)
3. Hybrid (clean + distorted, both branches distorted)
4. Asymmetric (clean + distorted, both branches distorted)

### Metrics Reported
Accuracy, Precision, Recall, F1, ROC-AUC — same as training validation.

---

## Dataset Split Statistics (actual, from CSV files)

| Split | Total | Real (0) | Fake (1) | Balance |
|-------|-------|----------|----------|---------|
| Train | 2,774 | 1,389 | 1,385 | ~50/50 |
| Val   | 594   | 298   | 296   | ~50/50 |
| Test  | 595   | 298   | 297   | ~50/50 |
| Total | 3,963 | 1,985 | 1,978 | ~50/50 |

The dataset is well-balanced (near 50/50 real/fake split across all splits).

---

## How to Obtain Actual Metrics

To get real numbers, run the evaluation scripts:

```bash
# Robustness evaluation (clean vs distorted, on val set)
python evaluate_distortion.py

# Individual model training (will print per-epoch metrics)
python train/train_spatial.py
python train/train_frequency.py
python train/train_hybrid.py
python train/train_asymmetric.py
```

To evaluate on the held-out test set, modify `evaluate_distortion.py` to use `data/splits/test.csv` instead of `data/splits/val.csv`.

---

## Inference Behavior (from app.py)

The Streamlit app reports per-model confidence scores at inference time. These are softmax probabilities, not calibrated probabilities. The asymmetric model uses a higher threshold (0.90 for FAKE, 0.70 for SUSPICIOUS) compared to the other models (argmax, effectively 0.50 threshold).

Video analysis averages fake scores across sampled frames (every 15th frame) and uses the mean of hybrid and asymmetric scores as the final verdict score, with threshold 0.50.
