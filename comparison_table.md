# Comparison Table

## Table 1: Paper Overview

| # | Paper | Year | Venue | Type | Task |
|---|-------|------|-------|------|------|
| P1 | Wang et al., "Deepfake Detection: A Comprehensive Survey from the Reliability Perspective" | 2024 | ACM Computing Surveys | Survey | Deepfake detection (reliability) |
| P2 | Raza et al., "A Novel Deep Learning Approach for Deepfake Image Detection" | 2022 | Applied Sciences | Method | Deepfake image detection |
| P3 | Suganthi ST et al., "Deep learning model for deep fake face recognition and detection" | 2022 | PeerJ Computer Science | Method | Deepfake face detection |
| P4 | Patel et al., "Deepfake Generation and Detection: Case Study and Challenges" | 2023 | IEEE Access | Survey | Deepfake generation + detection |
| P5 | Yan et al., "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection" | 2023 | NeurIPS | Benchmark | Deepfake detection benchmark |
| P6 | Abir et al., "Detecting Deepfake Images Using Deep Learning Techniques and XAI" | 2023 | IASC | Method | Deepfake image detection + XAI |
| P7 | Latif et al., "Face Recognition from Video Using Deep Learning Models" | 2024 | VAWKUM Trans. | Method | Video face recognition |
| P8 | Taeb & Chi, "Comparison of Deepfake Detection Techniques through Deep Learning" | 2022 | J. Cybersecurity & Privacy | Comparison | Deepfake detection comparison |
| P9 | Yan et al., "DF40: Toward Next-Generation Deepfake Detection" | 2024 | NeurIPS | Benchmark | Next-gen deepfake benchmark |
| P10 | Jaybhaye et al., "Fake News Detection using LSTM" | 2023 | ITM Web of Conf. | Method | Fake news (text) detection |
| P11 | Heidari et al., "Deepfake detection using deep learning methods: A systematic review" | 2024 | WIREs DMKD | Survey | Deepfake detection (all modalities) |
| P12 | Zhao et al., "Multi-attentional Deepfake Detection" | 2021 | CVPR | Method | Deepfake detection (fine-grained) |
| **Ours** | **Spatial-Frequency Hybrid with Asymmetric Training** | **2025** | — | **Method** | **Deepfake image/video detection** |

---

## Table 2: Dataset Comparison

| Paper | Dataset Used | Size | Type | Standard Benchmark? |
|-------|-------------|------|------|---------------------|
| P2 (Raza et al.) | Kaggle CIPLAB (Yonsei) | 2,041 images | Photoshopped faces | No |
| P3 (Suganthi et al.) | FFHQ, 100K-Faces, DFFD, CASIA-WebFace | 70K–500K | GAN-generated | Partial |
| P5 (DeepfakeBench) | FF++, CelebDF-v1/v2, DFD, DFDC-P, DFDC, UADFV, FaceShifter, DF-1.0 | Large-scale | Video-based | Yes |
| P6 (Abir et al.) | Kaggle (Flickr + StyleGAN) | 140,000 images | GAN-generated | No |
| P8 (Taeb & Chi) | Augmented Kaggle dataset | Not specified | Mixed | No |
| P9 (DF40) | DF40 (40 techniques) | Large-scale | Video + image | Yes (new) |
| P12 (Zhao et al.) | FF++ (HQ/LQ), DFDC, Celeb-DF | Large-scale | Video-based | Yes |
| **Ours** | **FF++ C23 (Deepfakes + FaceSwap only)** | **3,963 face images** | **Video-extracted faces** | **Partial** |

---

## Table 3: Model Architecture Comparison

| Paper | Backbone | Input Modality | Fusion | Attention | Frequency Domain |
|-------|----------|---------------|--------|-----------|-----------------|
| P2 (Raza et al.) | VGG16 + custom CNN | RGB | Sequential | No | No |
| P3 (Suganthi et al.) | Fisherface + DBN/RBM | RGB | Sequential | No | No |
| P5 (DeepfakeBench) | Multiple (ResNet, EfficientNet, Xception) | RGB | Various | Various | F3Net, SPSL, SRM |
| P6 (Abir et al.) | InceptionResNetV2, DenseNet201, InceptionV3, ResNet152V2 | RGB | None | No | No |
| P8 (Taeb & Chi) | VGG19, DenseNet-121, Custom CNN | RGB | None | No | No |
| P12 (Zhao et al.) | EfficientNet-B4 | RGB | Bilinear attention pooling | Yes (multi-head) | No |
| **Ours (Spatial)** | **ResNet-18** | **RGB** | **None** | **No** | **No** |
| **Ours (Frequency)** | **ResNet-18** | **FFT magnitude** | **None** | **No** | **Yes (FFT)** |
| **Ours (Hybrid)** | **2× ResNet-18** | **RGB + FFT** | **Concatenation** | **No** | **Yes (FFT)** |
| **Ours (Asymmetric)** | **2× ResNet-18** | **RGB + distorted FFT** | **Concatenation** | **No** | **Yes (FFT + augmentation)** |

---

## Table 4: Training Strategy Comparison

| Paper | Optimizer | Epochs | Batch Size | Augmentation | Robustness Training |
|-------|-----------|--------|------------|--------------|---------------------|
| P2 (Raza et al.) | Adam | 20 | 100 | Standard | No |
| P3 (Suganthi et al.) | N/A (DBN/RBM) | N/A | N/A | Kalman filter noise removal | No |
| P6 (Abir et al.) | Adam (TF/Keras) | Not specified | Not specified | Standard | No |
| P8 (Taeb & Chi) | Not specified | Not specified | Not specified | Data augmentation | No |
| P12 (Zhao et al.) | Not specified | Not specified | Not specified | AGDA (attention-guided) | No |
| **Ours (Spatial/Freq)** | **Adam (lr=1e-4)** | **5** | **8** | **Mild blur + color jitter** | **No** |
| **Ours (Hybrid)** | **Adam (lr=1e-4)** | **5** | **8** | **Mild + stronger augmentation** | **No** |
| **Ours (Asymmetric)** | **Adam (lr=1e-4)** | **5** | **8** | **Asymmetric: clean spatial, distorted FFT** | **Yes** |

---

## Table 5: Evaluation Metrics Reported

| Paper | Accuracy | Precision | Recall | F1 | AUC | Specificity | Other |
|-------|----------|-----------|--------|-----|-----|-------------|-------|
| P2 (Raza et al.) | ✓ (94%) | ✓ (95%) | — | ✓ (94%) | — | ✓ (94%) | Geometric mean, loss |
| P3 (Suganthi et al.) | ✓ (up to 98.82%) | — | ✓ | ✓ | — | ✓ | RMSE, SNR, PSNR, MAE |
| P5 (DeepfakeBench) | ✓ | ✓ | ✓ | — | ✓ | — | AP, EER |
| P6 (Abir et al.) | ✓ (up to 99.87%) | — | — | — | — | — | LIME explanations |
| P8 (Taeb & Chi) | ✓ (VGG19: 95%) | — | — | — | — | — | — |
| P12 (Zhao et al.) | ✓ (FF++ HQ: 99.80%) | — | — | — | ✓ (Celeb-DF: 67.44%) | — | DFDC logloss: 0.1679 |
| **Ours** | **✓** | **✓** | **✓** | **✓** | **✓** | **—** | **Clean + distorted eval** |

---

## Table 6: Robustness and Generalization

| Paper | Robustness Testing | Cross-Dataset Eval | Distortion Testing | Compression Testing |
|-------|-------------------|-------------------|-------------------|---------------------|
| P1 (Wang et al.) | ✓ (discussed) | ✓ (reliability study) | — | — |
| P2 (Raza et al.) | No | No | No | No |
| P3 (Suganthi et al.) | No | Partial (4 datasets) | No | No |
| P5 (DeepfakeBench) | Partial | ✓ (9 datasets) | No | No |
| P6 (Abir et al.) | No | No | No | No |
| P8 (Taeb & Chi) | No | No | No | No |
| P9 (DF40) | ✓ (diverse techniques) | ✓ (40 techniques) | No | No |
| P12 (Zhao et al.) | Partial | ✓ (Celeb-DF) | No | ✓ (HQ vs LQ) |
| **Ours** | **✓ (GaussianBlur + ColorJitter)** | **No** | **✓ (evaluate_distortion.py)** | **No** |

---

## Table 7: Key Contributions Summary

| Paper | Key Contribution |
|-------|-----------------|
| P1 (Wang et al., 2024) | Reliability framework with statistical confidence intervals for forensic use |
| P2 (Raza et al., 2022) | Hybrid VGG16+CNN (DFP) for deepfake image detection |
| P3 (Suganthi et al., 2022) | FF-LBPH-DBN combining face recognition and deepfake detection |
| P4 (Patel et al., 2023) | Comprehensive survey of generation + detection; IBMM multi-modal case study |
| P5 (Yan et al., 2023) | First unified benchmark (15 detectors, 9 datasets, standardized protocols) |
| P6 (Abir et al., 2023) | XAI (LIME) for interpretable deepfake detection |
| P7 (Latif et al., 2024) | Video-based face recognition with cosine similarity matching |
| P8 (Taeb & Chi, 2022) | Comparative study of CNN architectures for deepfake detection |
| P9 (Yan et al., 2024) | DF40: 40-technique dataset for next-generation deepfake detection |
| P10 (Jaybhaye et al., 2023) | LSTM-based fake news (text) detection |
| P11 (Heidari et al., 2024) | Systematic review across video, image, audio, hybrid deepfake detection |
| P12 (Zhao et al., 2021) | Multi-attentional fine-grained detection with AGDA and regional independence loss |
| **Ours** | **Dual-stream spatial+FFT fusion with asymmetric frequency branch training for robustness** |
