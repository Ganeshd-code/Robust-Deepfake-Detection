# Contribution Summary

## Core Research Question

Can combining spatial (RGB) and frequency (FFT) domain features in a dual-stream CNN improve deepfake detection accuracy and robustness compared to single-stream approaches? And can asymmetric training — where the frequency branch is trained on distorted inputs — further improve robustness to real-world image degradation?

---

## True Contributions (from actual implementation)

### 1. Dual-Stream Spatial-Frequency Architecture

The primary architectural contribution is the `HybridModel`: a dual-stream ResNet-18 network that processes both the original RGB face image and its per-channel FFT magnitude spectrum simultaneously. The two 512-dimensional feature vectors are concatenated into a 1024-dimensional representation before classification.

This is grounded in the observation that deepfake generation artifacts manifest in both the spatial domain (blending boundaries, texture inconsistencies) and the frequency domain (unnatural spectral patterns introduced by GAN upsampling and compression).

### 2. Per-Channel FFT Magnitude Spectrum as Input Representation

The `get_fft()` function computes a 3-channel FFT image by applying 2D FFT independently to each RGB channel, centering the spectrum, computing log-scaled magnitude, and normalizing to [0, 255]. This produces a visually interpretable frequency representation that can be fed directly into a standard CNN without architectural modification.

This allows the frequency branch to reuse the same ResNet-18 backbone as the spatial branch, keeping the system simple and reproducible.

### 3. Asymmetric Training Strategy

The `AsymmetricDataset` implements a novel training regime where:
- The spatial branch receives clean, mildly augmented RGB images
- The frequency branch receives FFT computed from **distorted** versions of the same image

The distortion (stronger blur + color jitter) is applied to the RGB image before FFT computation. This forces the frequency branch to learn spectral features that remain discriminative even when the input has been degraded, without corrupting the spatial branch's learning signal.

The asymmetric model uses the same `HybridModel` architecture as the standard hybrid — the contribution is entirely in the training data pipeline.

### 4. Tiered Confidence Thresholding for Asymmetric Model

The asymmetric model uses a three-tier decision rule at inference:
- Fake probability ≥ 0.90 → FAKE
- Fake probability ≥ 0.70 → SUSPICIOUS
- Otherwise → REAL

This reflects the model's higher confidence calibration from robustness training and provides a more nuanced output than binary classification.

### 5. Four-Model Ensemble Forensic System

The Streamlit application (`app.py`) runs all four models in parallel and aggregates their votes:
- 3+ FAKE votes → HIGH PROBABILITY OF MANIPULATION
- 1+ FAKE or SUSPICIOUS → SUSPICIOUS
- 0 → LIKELY AUTHENTIC

For video, it samples every 15th frame, averages per-model fake scores, and uses the mean of hybrid and asymmetric scores as the final verdict score.

---

## What This Work Does NOT Claim

- No attention mechanisms or learned fusion weights (fusion is simple concatenation)
- No cross-manipulation generalization testing (only Deepfakes + FaceSwap used for training)
- No comparison against published state-of-the-art methods (no benchmark numbers)
- No novel backbone architecture (standard ResNet-18 throughout)
- No temporal modeling for video (frame-level predictions averaged independently)
- The asymmetric model uses the identical architecture as the hybrid model — the novelty is the dataset pipeline only

---

## Positioning Against Literature

The project draws on the following established ideas:
- Frequency-domain analysis for deepfake detection (spectral artifacts from GAN generation)
- Multi-stream CNN fusion (spatial + frequency complementarity)
- Data augmentation for robustness

The specific contribution is the **asymmetric augmentation strategy** applied to the frequency branch input, which is a practical and lightweight approach to improving frequency-domain robustness without architectural complexity.

---

## Limitations (from implementation)

- Small dataset: ~3,963 face images total (limited by frame sampling and face detection yield)
- Only 2 of 6 FF++ manipulation types used for training
- 5 epochs of training — likely underfitting for a research-grade result
- Batch size of 8 — small, may cause noisy gradient estimates
- No learning rate scheduling or early stopping
- Face detection uses Haar Cascade (lower accuracy than MTCNN or RetinaFace)
- No cross-dataset evaluation
- No ablation study results saved to disk
