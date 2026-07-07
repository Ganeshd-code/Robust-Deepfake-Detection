# Related Work Summary and Project Positioning

## 1. Overview of the Literature Landscape

The 12 papers span four categories: detection methods (P2, P3, P6, P7, P8, P12), comprehensive surveys (P1, P4, P11), benchmarks (P5, P9), and an adjacent domain paper (P10 — text fake news). Together they establish the state of deepfake detection research from 2021 to 2024.

The dominant finding across surveys (Wang et al., 2024; Heidari et al., 2024; Patel et al., 2023) is that three challenges remain unsolved: **transferability** (cross-dataset generalization), **interpretability** (explaining decisions), and **robustness** (performance under real-world degradation). As Wang et al. (2024) state: "the general reliability of a detection model has been barely considered, leading to the lack of reliable evidence in real-life usages."

---

## 2. What the Literature Establishes

### 2.1 Spatial-Only CNN Approaches Are the Baseline

The majority of detection papers (P2, P3, P6, P8) use single-stream CNN architectures operating on RGB images. Raza et al. (2022) achieve 94% accuracy with a VGG16+CNN hybrid on a small photoshopped face dataset. Abir et al. (2023) achieve 99.87% with InceptionResNetV2 on a StyleGAN dataset. Taeb & Chi (2022) find VGG19 achieves 95% on an augmented Kaggle dataset.

These results are high but achieved on clean, controlled datasets (GAN-generated faces, photoshopped images) rather than video-extracted deepfakes from FF++. As DeepfakeBench (Yan et al., 2023) demonstrates, performance on such controlled datasets does not transfer to real-world benchmarks.

### 2.2 Frequency Domain Analysis Is Recognized but Underexplored in Practice

DeepfakeBench (Yan et al., 2023) categorizes detectors into naive, spatial, and frequency types. Frequency detectors (F3Net, SPSL, SRM) are included but represent a minority. The benchmark notes that "frequency detector addresses this limitation by focusing on the frequency domain for forgery detection." Zhao et al. (2021) acknowledge that "[26] uses a two-branch representation extractor to combine information from the color domain and the frequency domain" as related work, but their own method (multi-attentional) remains purely spatial.

Taeb & Chi (2022) explicitly note that "there needs to be frequency-specific anomaly analysis" as a future direction but do not implement it.

### 2.3 Multi-Stream Fusion Is Emerging

Zhao et al. (2021) reference a two-branch spatial+frequency approach (their reference [26]) that achieves 73.41% AUC on Celeb-DF — the best cross-dataset result in their comparison table — despite lower in-dataset performance. This suggests that frequency information provides complementary generalization benefits that spatial-only methods lack.

### 2.4 Robustness Under Distortion Is an Open Problem

Wang et al. (2024) identify robustness as one of three core reliability challenges. Zhao et al. (2021) acknowledge their method is "sensitive to high compression rate which blurs most of the useful information in spatial domain." No paper in the reviewed set implements or evaluates a training strategy specifically designed to make the frequency branch robust to image distortion before computing the FFT.

### 2.5 Benchmarks Reveal Generalization Failures

DeepfakeBench (Yan et al., 2023) shows that even sophisticated detectors fail cross-domain. DF40 (Yan et al., 2024) demonstrates that models trained on FF++ "fail to generalize to modern deepfake techniques." The standard protocol of training on FF++ and testing on others is described as insufficient.

### 2.6 Attention Mechanisms Improve Local Artifact Detection

Zhao et al. (2021) reformulate deepfake detection as fine-grained classification and propose multi-attention heads with regional independence loss and AGDA. This achieves 99.80% on FF++ (HQ) and 67.44% AUC on Celeb-DF. The attention-guided data augmentation (deliberately blurring high-response regions) is conceptually related to the asymmetric training idea — both force the model to learn from non-obvious features.

---

## 3. What My Project Improves Over the Literature

### 3.1 Over Single-Stream Spatial Methods (P2, P3, P6, P8)

All four single-stream papers (Raza et al., Suganthi et al., Abir et al., Taeb & Chi) operate exclusively on RGB images. None incorporate frequency domain information. My project adds a parallel frequency branch that processes per-channel FFT magnitude spectra, providing complementary artifact signals that spatial features alone cannot capture.

As noted by Taeb & Chi (2022): "it is becoming harder for machine-learning techniques to identify convolutional traces of deepfake generation algorithms, as there needs to be frequency-specific anomaly analysis." My project directly implements this.

### 3.2 Over Purely Spatial Multi-Stream Methods (P12)

Zhao et al. (2021) use multiple attention heads but remain in the spatial domain. Their method is acknowledged to be "sensitive to high compression rate which blurs most of the useful information in spatial domain." My hybrid model combines spatial and frequency streams, so when spatial features are degraded by compression or blur, the frequency branch can still provide discriminative signal.

### 3.3 Over Standard Hybrid Approaches

The two-branch spatial+frequency approach referenced in Zhao et al. (2021) as reference [26] applies the same augmentation to both branches. My asymmetric training strategy applies distortion specifically to the frequency branch input (before FFT computation), forcing the frequency branch to learn spectral features that remain discriminative under degradation — without corrupting the spatial branch's learning signal.

### 3.4 Over Benchmark-Only Papers (P5, P9)

DeepfakeBench and DF40 identify the problem of generalization failure but do not propose solutions. My asymmetric training strategy is a concrete, implementable response to the robustness challenge identified by Wang et al. (2024) and the generalization failures documented by Yan et al. (2023, 2024).

---

## 4. Research Gap Addressed

The literature establishes the following gap: **no existing method in the reviewed papers explicitly trains the frequency branch of a dual-stream detector on distorted inputs to improve robustness to real-world image degradation.**

Specifically:
- Spatial-only methods (P2, P3, P6, P8) have no frequency branch at all
- Multi-attention spatial methods (P12) are explicitly sensitive to compression
- Benchmark papers (P5, P9) document generalization failures but propose no training-level solution
- Survey papers (P1, P4, P11) identify robustness as an open challenge but do not propose solutions
- The two-branch spatial+frequency approach referenced in P12 applies symmetric augmentation to both branches

My project fills this gap with the `AsymmetricDataset` pipeline: augmentation is applied to the RGB image before FFT computation on the frequency branch, while the spatial branch receives clean images. This is a training-time intervention that requires no architectural change and no additional parameters.

---

## 5. Robustness Advantages of My Method

### 5.1 Frequency Branch Trained on Distorted Spectra

In standard hybrid training (HybridDataset), FFT is computed on the original clean image. In asymmetric training (AsymmetricDataset), FFT is computed on an augmented/distorted version of the image. The distortions applied (GaussianBlur kernel=5, ColorJitter brightness/contrast/saturation=0.2, p=0.3) simulate real-world degradation.

This means the frequency branch learns to detect deepfake artifacts in spectral representations that have been corrupted by blur and color distortion — conditions that commonly occur in social media compression, video transcoding, and screenshot artifacts.

### 5.2 Spatial Branch Preserved

Because the spatial branch receives clean images during asymmetric training, it retains full spatial feature quality. The asymmetric design prevents the spatial branch from being degraded by the frequency branch's augmentation strategy.

### 5.3 Robustness Evaluation

The `evaluate_distortion.py` script evaluates all four models under a deterministic distortion (GaussianBlur kernel=7, ColorJitter brightness=0.4, contrast=0.4, saturation=0.3) — stronger than training augmentation. This directly tests whether the asymmetric training strategy improves robustness compared to the standard hybrid.

### 5.4 Tiered Confidence Thresholding

The asymmetric model uses a three-tier decision rule (≥0.90 → FAKE, ≥0.70 → SUSPICIOUS, else REAL) rather than argmax. This reflects the model's calibration from robustness training and provides a more conservative, forensically appropriate output — aligned with the reliability framework proposed by Wang et al. (2024).

---

## 6. What Is Genuinely Novel

Based strictly on what is implemented and what the literature contains:

### 6.1 Asymmetric Frequency Branch Training (Primary Novelty)

No paper in the reviewed set applies distortion to the input image **before** FFT computation as a training strategy for the frequency branch of a dual-stream detector. The `AsymmetricDataset` pipeline — augment RGB → compute FFT on distorted image → train frequency branch on distorted spectra — is not described in any of the 12 reviewed papers.

This is distinct from:
- Standard data augmentation (applied after FFT, or to RGB only)
- AGDA in Zhao et al. (2021) (blurs attention regions during training, not the FFT input)
- Symmetric augmentation in hybrid methods (same distortion to both branches)

### 6.2 Four-Model Ensemble with Asymmetric Voting

The inference system runs four models (Spatial, Frequency, Hybrid, Asymmetric) in parallel and aggregates votes with a tiered threshold for the asymmetric model. No reviewed paper implements a four-model ensemble combining unimodal and multimodal detectors with asymmetric confidence thresholding.

### 6.3 Per-Channel FFT Magnitude Spectrum as Direct CNN Input

The `get_fft()` function computes log-scaled, normalized FFT magnitude spectra independently per RGB channel and stacks them into a 3-channel image that can be fed directly into a standard ResNet-18 without architectural modification. This is a practical implementation choice that enables frequency analysis without custom layers or learnable filters (unlike F3Net's learnable frequency filters referenced in DeepfakeBench).

---

## 7. Honest Limitations Relative to the Literature

- **Dataset scale:** 3,963 face images is small compared to FF++ (4,000+ videos), DFDC (100K+ clips), and the 140K images used by Abir et al. (2023). DeepfakeBench and DF40 use orders of magnitude more data.
- **No cross-dataset evaluation:** Unlike Zhao et al. (2021) who test on Celeb-DF after training on FF++, this project has no cross-dataset evaluation.
- **Only 2 of 6 FF++ manipulation types used:** Face2Face, FaceShifter, NeuralTextures, and DeepFakeDetection are not included in training.
- **No attention mechanism:** Zhao et al. (2021) demonstrate that multi-attention heads improve both accuracy and transferability. This project uses simple concatenation fusion.
- **5 epochs of training:** Insufficient for convergence compared to standard practice in the literature.
- **No comparison against published baselines:** Unlike all method papers reviewed, this project does not report results against Xception, EfficientNet-B4, or other standard baselines on the same data.
