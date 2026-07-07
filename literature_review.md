# Literature Review

## Paper 1: Deepfake Detection: A Comprehensive Survey from the Reliability Perspective

**Citation:** Tianyi Wang, Xin Liao, Kam Pui Chow, Xiaodong Lin, and Yinglong Wang. "Deepfake Detection: A Comprehensive Survey from the Reliability Perspective." ACM Computing Surveys, Vol. 57, No. 3, Article 58, November 2024. https://doi.org/10.1145/3699710

**Problem Statement:**
Existing deepfake detection models achieve high benchmark accuracy but lack real-world reliability. Three core challenges are identified: transferability (cross-dataset generalization), interpretability (explaining detection decisions), and robustness (performance under real-world conditions). No authenticated scheme exists to nominate detection models as reliable forensic evidence for court use.

**Methodology:**
Comprehensive survey of deepfake detection approaches from a reliability perspective. Introduces a model reliability study metric using statistical random sampling. Proposes a standardized data preprocessing workflow (frame extraction, face detection, cropping). Evaluates selected state-of-the-art models under a unified protocol and applies them to real-life deepfake case studies at 90% and 95% confidence levels.

**Datasets:**
Multiple benchmark datasets reviewed: UADFV, DeepfakeTIMIT, FaceForensics++ (FF++), Deepfake Detection Dataset (DFD), DFDC Preview, Celeb-DF, DeeperForensics-1.0 (DF1.0), DFDC. Categorized into three generations based on quality and diversity.

**Preprocessing:**
Standardized workflow: image frame selection and extraction from videos, face detection and cropping. Noted as "barely mentioned with concrete details in past work."

**Model Architecture:**
Survey covers: autoencoder-based face-swap detection, GAN-based methods, CNN binary classifiers, multi-stream networks, frequency-domain detectors, attention-based methods.

**Strengths:**
- First survey to formally address model reliability for forensic use
- Introduces statistical confidence interval framework for detection reliability
- Covers three generations of datasets and detection methods comprehensively
- Provides real-life case studies

**Weaknesses:**
- Survey paper — does not propose a new detection model
- Reliability framework is theoretical; practical adoption requires institutional authentication

**Limitations:**
- Reliability study limited to publicly available benchmark datasets
- Does not address audio deepfakes in depth

**Reported Metrics:**
Detection model reliability evaluated at 90% and 95% confidence intervals. Per-model accuracy, AUC reported across benchmark datasets. Specific numerical values depend on the models evaluated under the unified protocol.

---

## Paper 2: A Novel Deep Learning Approach for Deepfake Image Detection

**Citation:** Ali Raza, Kashif Munir, and Mubarak Almutairi. "A Novel Deep Learning Approach for Deepfake Image Detection." Applied Sciences, 2022, 12, 9820. https://doi.org/10.3390/app12199820

**Problem Statement:**
Deepfake image detection is a critical cybersecurity challenge. Existing transfer learning approaches (Xception, NAS-Net, MobileNet, VGG16) achieve limited accuracy. A more efficient hybrid architecture is needed.

**Methodology:**
Proposes a novel Deepfake Predictor (DFP) — a hybrid of VGG16 and CNN layers. VGG16 features are passed through additional convolutional layers, max pooling, dropout, flatten, and dense layers. Binary cross-entropy loss, Adam optimizer, 20 epochs. Hyperparameter tuning applied. K-fold cross-validation (4 folds) used for validation.

**Dataset:**
Kaggle deepfake dataset from CIPLAB, Yonsei University. Contains 1,081 real and 960 fake face images (expert-generated photoshopped faces). 90% train / 10% test split.

**Preprocessing:**
Images resized to 256×256. Standard normalization. No explicit face detection step described.

**Model Architecture:**
- Input: (100, 256, 256, 3)
- VGG16 backbone (ImageNet pretrained, 14.7M params) → (None, 8, 8, 512)
- Conv layer (1025 filters, ReLU) → (100, 6, 6, 1025)
- MaxPool (2×2) → (100, 3, 3, 1025)
- Dropout (0.02)
- Flatten → (100, 9225)
- Dense (1025, ReLU)
- Output Dense (1, Sigmoid)
- Loss: Binary Cross-Entropy

**Strengths:**
- Simple, reproducible architecture
- Outperforms all compared baselines on the same dataset
- K-fold cross-validation provides generalization evidence

**Weaknesses:**
- Very small dataset (2,041 images total)
- Dataset is photoshopped faces, not GAN-generated deepfakes — limited realism
- No frequency domain analysis
- No robustness testing under distortion
- Single dataset evaluation

**Limitations:**
- Results not validated on standard benchmarks (FF++, Celeb-DF, DFDC)
- Binary cross-entropy with sigmoid output — not standard for multi-class deepfake detection
- 90/10 split with no validation set

**Reported Metrics:**
| Model | Accuracy | Precision | F1 | Specificity |
|-------|----------|-----------|-----|-------------|
| NAS-Net | 83% | 80% | 86% | 73% |
| Xception | 84% | 82% | 86% | 75% |
| MobileNet | 88% | 86% | 89% | 84% |
| VGG16 | 90% | 88% | 92% | 84% |
| Proposed DFP | 94% | 95% | 94% | 94% |
K-fold average accuracy: 95% ± 1%

---

## Paper 3: Deep Learning Model for Deep Fake Face Recognition and Detection

**Citation:** Suganthi ST, Mohamed Uvaze Ahamed Ayoobkhan, Krishna Kumar V, Nebojsa Bacanin, Venkatachalam K, Hubálovský Štěpán, and Trojovský Pavel. "Deep learning model for deep fake face recognition and detection." PeerJ Computer Science, 8:e881, 2022. https://doi.org/10.7717/peerj-cs.881

**Problem Statement:**
Existing deepfake detection methods suffer from inaccuracy and high computation time. Face recognition and deepfake detection need to be combined in a unified pipeline.

**Methodology:**
Proposes FF-LBPH-DBN: Fisherface algorithm with Local Binary Pattern Histogram (LBPH) for dimensionality reduction and feature extraction, followed by Deep Belief Network (DBN) with Restricted Boltzmann Machine (RBM) for classification. Four phases: preprocessing, dimensionality reduction, feature extraction, classification.

**Datasets:**
FFHQ (70,000 GAN-generated faces), 100K-Faces (StyleGAN), DFFD (ProGAN + StyleGAN, ~300K images), CASIA-WebFace (500,000 images, 10,575 subjects).

**Preprocessing:**
- Resize to 256×256
- Noise removal using Kalman filter (prediction + update process)
- Normalization

**Model Architecture:**
- Fisherface (LDA-based) for dimensionality reduction in face space
- LBPH for texture feature extraction
- DBN with RBM layers for classification
- Not a CNN-based approach

**Strengths:**
- Evaluated on multiple large-scale datasets
- Achieves high accuracy on CASIA-WebFace (98.82%)
- Combines face recognition and deepfake detection
- Lower error rate than SVM, LDA, KNN, CNN baselines

**Weaknesses:**
- Traditional feature engineering (LBPH) rather than end-to-end deep learning
- No frequency domain analysis
- No robustness testing
- Datasets are GAN-generated synthetic faces, not video-based deepfakes

**Limitations:**
- DBN/RBM approach is computationally expensive and less scalable than CNNs
- Not evaluated on standard video deepfake benchmarks (FF++, DFDC)
- No cross-dataset generalization study

**Reported Metrics:**
| Method | FFHQ | 100K-Faces | DFFD | CASIA-WebFace |
|--------|------|-----------|------|---------------|
| SVM | 82.5% | 70.12% | 84.43% | 85.25% |
| CNN | 89.23% | 82.45% | 88.55% | 86.12% |
| FF-LBPH-DBN | 94.92% | 95.55% | 97.82% | 98.82% |

---

## Paper 4: Deepfake Generation and Detection: Case Study and Challenges

**Citation:** Yogesh Patel, Sudeep Tanwar, Rajesh Gupta, Pronaya Bhattacharya, Innocent Ewean Davidson, Royi Nyameko, Srinivas Aluvala, and Vrince Vimal. "Deepfake Generation and Detection: Case Study and Challenges." IEEE Access, Vol. 11, 2023. https://doi.org/10.1109/ACCESS.2023.3342107

**Problem Statement:**
Existing surveys focus on detection but inadequately cover the generation process. A comprehensive review of both deepfake generation and detection is needed, along with a case study (IBMM — Inconsistencies and Incompatibilities Between different types of input data and Models) for multi-modal deepfake detection.

**Methodology:**
Comprehensive survey of deepfake generation (autoencoders, GANs) and detection (CNN-based, LSTM-based, multi-modal). Presents IBMM case study: a multi-modal overview examining inconsistencies between different input modalities and detection models. Reviews ML/DL approaches for image, video, and audio deepfake detection.

**Datasets:**
Reviews: UADFV, DeepfakeTIMIT, FF++, DFD, DFDC Preview, Celeb-DF, DeeperForensics-1.0, DFDC, ForgeryNet. Discusses dataset limitations in terms of diversity and scale.

**Preprocessing:**
Reviews various preprocessing approaches used in the literature. No single unified preprocessing pipeline proposed.

**Model Architecture:**
Survey covers: autoencoder-based (FakeApp, FaceSwap), GAN-based (FaceShifter, SimSwap, MegaFS, HifiFace), CNN classifiers, LSTM temporal models, multi-modal networks.

**Strengths:**
- Covers both generation and detection — fills a survey gap
- IBMM case study provides practical multi-modal perspective
- Discusses implementation challenges and future directions
- IEEE Access open access

**Weaknesses:**
- Survey paper — no new model proposed
- IBMM case study is descriptive, not quantitatively evaluated
- Does not address robustness to image distortion

**Limitations:**
- Rapidly outdated given the pace of deepfake generation advances
- Limited coverage of frequency-domain detection methods

**Reported Metrics:**
Survey paper — reports metrics from reviewed works. No original experimental results.

---

## Paper 5: DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection

**Citation:** Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection." NeurIPS 2023, Track on Datasets and Benchmarks. arXiv:2307.01426v2.

**Problem Statement:**
The deepfake detection field lacks a standardized, unified, comprehensive benchmark. Inconsistent data processing pipelines, experimental settings, and evaluation metrics make fair comparison impossible and results potentially misleading.

**Methodology:**
Builds DeepfakeBench: a unified platform with (1) a unified data management system, (2) an integrated framework implementing 15 state-of-the-art detectors, (3) standardized evaluation metrics and protocols. Evaluates 15 detectors on 9 datasets under within-domain and cross-domain protocols. Analyzes effects of data augmentation, backbone architecture, pre-training, and number of training frames.

**Datasets:**
9 datasets: FF++ (c23), CelebDF-v1, CelebDF-v2, DFD, DFDC-Preview, DFDC, UADFV, FaceShifter, DeeperForensics-1.0.

**Preprocessing:**
Unified data management module: consistent face detection, cropping, and resizing across all detectors.

**Model Architecture:**
15 detectors across three categories:
- Naive: MesoNet, MesoInception, CNN-Aug, EfficientNet-B4, Xception
- Spatial: Capsule, DSP-FWA, Face X-ray, FFD, CORE, RECCE, UCF
- Frequency: F3Net, SPSL, SRM

**Strengths:**
- First truly unified benchmark for fair comparison
- Covers naive, spatial, and frequency detector categories
- Reveals that naive detectors (Xception, EfficientB4) perform comparably to sophisticated methods under consistent settings
- Open-source codebase

**Weaknesses:**
- Does not include robustness evaluation under image distortion
- Cross-domain performance remains low for most detectors
- Does not cover audio or multi-modal deepfakes

**Limitations:**
- Benchmark frozen at time of publication — new methods require integration
- Cross-domain generalization remains an unsolved problem

**Reported Metrics (AUC, within-domain, trained on FF++ c23):**
- UCF: 95.37% avg
- Xception: 94.50% avg
- EfficientB4: 93.89% avg
- F3Net: 94.49% avg
- Cross-domain performance significantly lower for all methods

---

## Paper 6: Detecting Deepfake Images Using Deep Learning Techniques and Explainable AI Methods

**Citation:** Wahidul Hasan Abir, Faria Rahman Khanam, Kazi Nabiul Alam, Myriam Hadjouni, Hela Elmannai, Sami Bourouis, Rajesh Dey, and Mohammad Monirujjaman Khan. "Detecting Deepfake Images Using Deep Learning Techniques and Explainable AI Methods." Intelligent Automation & Soft Computing, 2023. https://doi.org/10.32604/iasc.2023.029653

**Problem Statement:**
Deep learning models for deepfake detection are "black boxes" — they cannot explain their decisions. Explainability is needed for trustworthy deployment. The paper combines deepfake detection with XAI (Explainable AI) using LIME.

**Methodology:**
Evaluates four CNN models (InceptionResNetV2, DenseNet201, InceptionV3, ResNet152V2) on a large-scale GAN face dataset. Best-performing model (InceptionResNetV2) analyzed with LIME (Local Interpretable Model-Agnostic Explanations) to identify which image regions drive classification decisions.

**Dataset:**
Kaggle dataset: 70,000 real faces from Flickr (Nvidia) + 70,000 fake faces generated by StyleGAN, 256px. Total: 140,000 images.

**Preprocessing:**
Standard CNN preprocessing. Images at 256px. TensorFlow/Keras pipeline.

**Model Architecture:**
- InceptionResNetV2 (best): 99.87% accuracy
- DenseNet201: 99.81%
- InceptionV3: 99.68%
- ResNet152V2: 99.19%
All models use ImageNet pretrained weights with fine-tuning.

**Strengths:**
- Large dataset (140K images)
- Introduces XAI (LIME) for interpretability — addresses the black-box problem
- Very high accuracy on the test set
- Practical explainability for forensic use

**Weaknesses:**
- Dataset is StyleGAN-generated faces, not video-based deepfakes
- No frequency domain analysis
- No robustness testing under distortion or compression
- LIME explanations are approximate and not always reliable

**Limitations:**
- High accuracy likely due to dataset simplicity (clean StyleGAN vs. Flickr real faces)
- Not evaluated on FF++, DFDC, or Celeb-DF
- No cross-dataset generalization

**Reported Metrics:**
| Model | Accuracy |
|-------|----------|
| InceptionResNetV2 | 99.87% |
| DenseNet201 | 99.81% |
| InceptionV3 | 99.68% |
| ResNet152V2 | 99.19% |

---

## Paper 7: Face Recognition from Video Using Deep Learning Models

**Citation:** Muhammad Latif, Mansoor Ebrahim, Abdul Salam Abro, Maaz Ahmed, Muhammad Daud Abbasi, and Imran Aziz Tunio. "Face Recognition from Video by Matching Images Using Deep Learning-Based Models." VAWKUM Transactions on Computer Sciences, Vol. 12, Issue 2, 2024. https://doi.org/10.21015/vtcs.v12i2.1916

**Problem Statement:**
Traditional face recognition systems are limited to static images and fail under dynamic conditions (movement, lighting changes, expression variation). Video-based face recognition requires integration of detection, tracking, and recognition.

**Methodology:**
Systematic approach for video-based face recognition: face detection from video frames, feature extraction (1,000 face feature vectors of size 128), cosine similarity matching with threshold 0.7. Segments 100 human faces from video frames (150×150 pixels average).

**Dataset:**
Custom video dataset. 100 human faces segmented from video frames.

**Preprocessing:**
Face segmentation from video frames, resizing to 150×150 pixels.

**Model Architecture:**
Deep learning-based face detection + feature extraction pipeline. Cosine similarity (threshold 0.7) for matching. Specific backbone not detailed in extracted text.

**Strengths:**
- Addresses video-based recognition (dynamic conditions)
- Discusses ethical considerations (privacy, consent, transparency)
- Practical system design

**Weaknesses:**
- Very small dataset (100 faces)
- Not a deepfake detection paper — face recognition focus
- 85% accuracy is modest
- No frequency domain analysis

**Limitations:**
- Limited to face recognition, not deepfake detection
- Small-scale evaluation
- No comparison with deepfake-specific methods

**Reported Metrics:**
- Recognition accuracy: 85%
- Feature vector size: 128
- Cosine similarity threshold: 0.7

---

## Paper 8: Comparison of Deepfake Detection Techniques through Deep Learning

**Citation:** Maryam Taeb and Hongmei Chi. "Comparison of Deepfake Detection Techniques through Deep Learning." Journal of Cybersecurity and Privacy, 2022, 2, 89–106. https://doi.org/10.3390/jcp2010007

**Problem Statement:**
Deepfake media poses a critical threat to digital forensics and legal evidence. Identifying the most reliable CNN-based classifier for face-image forgery detection is needed to guide future development.

**Methodology:**
Compares three CNN architectures — Custom CNN, VGG19, and DenseNet-121 — on an augmented real and fake face detection dataset. Data augmentation used to boost performance and reduce computational cost. Binary classification task.

**Dataset:**
Augmented real and fake face detection dataset (Kaggle-based). Specific size not stated in extracted text.

**Preprocessing:**
Data augmentation applied. Standard CNN preprocessing.

**Model Architecture:**
- Custom CNN: baseline convolutional network
- VGG19: 19-layer VGG with ImageNet pretraining
- DenseNet-121: dense block architecture with skip connections

**Strengths:**
- Direct comparison of three architectures under identical conditions
- Data augmentation improves generalization
- Discusses frequency-specific anomaly analysis as future direction
- Forensic application context

**Weaknesses:**
- No frequency domain analysis implemented
- Small dataset
- No robustness testing under distortion
- Not evaluated on standard benchmarks

**Limitations:**
- VGG19 best at 95% but not validated on FF++, DFDC, or Celeb-DF
- No cross-dataset evaluation
- Augmentation strategy not fully detailed

**Reported Metrics:**
- VGG19: 95% accuracy (best)
- Custom CNN: lower than VGG19
- DenseNet-121: lower than VGG19

---

## Paper 9: DF40: Toward Next-Generation Deepfake Detection

**Citation:** Zhiyuan Yan, Taiping Yao, Shen Chen, Yandan Zhao, Xinghe Fu, Junwei Zhu, Donghao Luo, Chengjie Wang, Shouhong Ding, Yunsheng Wu, and Li Yuan. "DF40: Toward Next-Generation Deepfake Detection." NeurIPS 2024, Track on Datasets and Benchmarks.

**Problem Statement:**
Existing deepfake detection benchmarks (especially FF++) are outdated, contain limited forgery diversity, and use evaluation protocols that do not reflect real-world conditions. Models trained on FF++ fail to generalize to modern deepfake techniques. The "train on FF++, test on others" protocol is insufficient.

**Methodology:**
Constructs DF40: a large-scale, highly diverse deepfake dataset with 40 distinct deepfake techniques (10× more than FF++). Covers face-swapping (10 methods), face-reenactment (13 methods), entire face synthesis (12 methods), and face editing (5 methods). Includes state-of-the-art methods: DiT, PixArt-α, DeepFaceLab, HeyGen. Evaluates 8 representative detection methods under 4 standard protocols, producing 2,000+ evaluations.

**Datasets:**
DF40 dataset: 40 deepfake techniques. Known domains: FF++ and CDF real data. Unknown domains: other real datasets. Comparison datasets: DF-TIMIT, UADFV, FF++, DFD, CDF, DFFD, DeeperForensics-1.0, DFDC, ForgeryNet.

**Preprocessing:**
Unified preprocessing pipeline. Face detection and cropping standardized.

**Model Architecture:**
Evaluates 8 representative detectors. Specific architectures not detailed in extracted text.

**Strengths:**
- Most diverse deepfake dataset to date (40 techniques)
- Includes latest AIGC methods (DiT, HeyGen, DeepFaceLab)
- 4 evaluation protocols for comprehensive assessment
- Reveals critical generalization failures of current detectors
- Open-source dataset and code

**Weaknesses:**
- Dataset construction paper — does not propose a new detection method
- Computational cost of evaluating 40 techniques is high

**Limitations:**
- Even with 40 techniques, real-world deepfakes continue to evolve
- Detection methods still struggle with cross-domain generalization

**Reported Metrics:**
Over 2,000 evaluations across 8 detectors and 4 protocols. Key finding: models trained on FF++ fail to generalize to modern deepfake techniques in DF40. Specific AUC values reported per detector-protocol combination in the paper.

---

## Paper 10: Fake News Detection Using LSTM-Based Deep Learning Approach

**Citation:** Sangita M. Jaybhaye, Vivek Badade, Aryan Dodke, Apoorva Holkar, and Priyanka Lokhande. "Fake News Detection using LSTM based deep learning approach." ITM Web of Conferences 56, 03005, ICDSAC 2023. https://doi.org/10.1051/itmconf/20235603005

**Problem Statement:**
The spread of fake news through social media is a critical concern. ML algorithms cannot effectively distinguish real from fake news articles with high accuracy. A deep learning approach using LSTM is needed.

**Methodology:**
Reviews ML and DL approaches for fake news detection. Proposes LSTM-based model trained on Kaggle news dataset. Takes textual content as input, uses LSTM to capture temporal dependencies of text. Deployed via Streamlit.

**Dataset:**
Kaggle fake news dataset (news articles, real and fake labels).

**Preprocessing:**
Text preprocessing: cleaning, filtering. Standard NLP pipeline.

**Model Architecture:**
LSTM neural network for sequence modeling of text. Not a vision/image model.

**Strengths:**
- Practical deployment via Streamlit
- Addresses temporal dependencies in text
- 94% accuracy on the Kaggle dataset

**Weaknesses:**
- Not a deepfake image/video detection paper — fake news text detection
- Limited relevance to visual deepfake detection
- Single dataset evaluation

**Limitations:**
- LSTM approach for text is not applicable to visual deepfake detection
- No cross-domain evaluation

**Reported Metrics:**
- LSTM accuracy: 94% on Kaggle fake news dataset

**Note:** This paper is about text-based fake news detection, not visual deepfake detection. Its relevance to the project is limited to the broader context of misinformation detection.

---

## Paper 11: Deepfake Detection Using Deep Learning Methods: A Systematic and Comprehensive Review

**Citation:** Arash Heidari, Nima Jafari Navimipour, Hasan Dag, and Mehmet Unal. "Deepfake detection using deep learning methods: A systematic and comprehensive review." WIREs Data Mining and Knowledge Discovery, 2024, 14:e1520. https://doi.org/10.1002/widm.1520

**Problem Statement:**
Deepfake technology threatens democracy, national security, and individual privacy. A systematic review of deepfake detection strategies using DL-based algorithms is needed, covering video, image, audio, and hybrid multimedia detection.

**Methodology:**
Systematic literature review. Categorizes deepfake detection methods by application domain: video detection, image detection, audio detection, hybrid multimedia detection. Reviews CNN, RNN, LSTM, GAN-based detection approaches. Analyzes datasets, preprocessing methods, and evaluation metrics used across the literature.

**Datasets:**
Reviews all major datasets: FF++, DFDC, Celeb-DF, DeeperForensics, UADFV, DeepfakeTIMIT, and others.

**Preprocessing:**
Reviews various preprocessing approaches. Notes that CNN is the most commonly employed DL method in publications.

**Model Architecture:**
Survey covers: CNN (most common), RNN, LSTM, GAN-based detectors, attention mechanisms, multi-stream networks.

**Strengths:**
- Comprehensive coverage of video, image, audio, and hybrid detection
- Systematic methodology for literature review
- Identifies CNN as dominant approach
- Covers domain adaptation and transfer learning

**Weaknesses:**
- Survey paper — no new model proposed
- Rapidly outdated given pace of field
- Limited coverage of frequency-domain methods

**Limitations:**
- Most reviewed articles focus on a single parameter (accuracy)
- Cross-dataset generalization rarely addressed in reviewed works

**Reported Metrics:**
Survey paper — reports metrics from reviewed works. Notes that accuracy is the most commonly reported metric, with most reviewed methods focusing on enhancing a single parameter.

---

## Paper 12: Multi-Attentional Deepfake Detection

**Citation:** Hanqing Zhao, Wenbo Zhou, Dongdong Chen, Tianyi Wei, Weiming Zhang, and Nenghai Yu. "Multi-attentional Deepfake Detection." CVPR 2021, pp. 2185–2194.

**Problem Statement:**
Most deepfake detection methods model the task as vanilla binary classification using global features. Since deepfake artifacts are subtle and local, global feature extraction is suboptimal. The problem should be reformulated as fine-grained classification.

**Methodology:**
Proposes a multi-attentional deepfake detection network with three components:
1. Multiple spatial attention heads to attend to different local face regions
2. Textural feature enhancement block to amplify subtle artifacts in shallow features
3. Bilinear attention pooling to aggregate low-level textural and high-level semantic features
Training innovations: regional independence loss (forces different attention heads to attend to different regions) and attention-guided data augmentation (AGDA — deliberately blurs high-response regions to force learning from other regions).

**Datasets:**
FaceForensics++ (FF++, HQ c23 and LQ c40), DFDC, Celeb-DF (cross-dataset evaluation).

**Preprocessing:**
EfficientNet-B4 backbone. Standard face detection and cropping. 4× augmentation for real/fake label balance on FF++.

**Model Architecture:**
- Backbone: EfficientNet-B4 (pretrained)
- Attention Module: multiple spatial attention heads (M=4 optimal)
- Textural Feature Enhancement Block: shallow layer feature amplification
- Bilinear Attention Pooling: aggregates textural + semantic features
- Regional Independence Loss + AGDA for training

**Strengths:**
- Reformulates deepfake detection as fine-grained classification — novel perspective
- Multiple attention heads capture diverse local artifacts
- State-of-the-art on FF++ (HQ) and DFDC at time of publication
- Cross-dataset evaluation on Celeb-DF
- Ablation study validates each component

**Weaknesses:**
- Sensitive to high compression (LQ version performance drops vs. F3-Net)
- Textural features degraded by heavy compression
- No frequency domain analysis
- Attention mechanism adds architectural complexity

**Limitations:**
- Performance on LQ (c40) version is 1.5% below F3-Net due to compression sensitivity
- Cross-dataset AUC on Celeb-DF (67.44%) shows limited generalization

**Reported Metrics:**
| Method | FF++ (HQ) Acc | Celeb-DF AUC |
|--------|--------------|--------------|
| Xception-c23 | 99.70% | 65.30% |
| Two Branch | 93.18% | 73.41% |
| F3-Net | 98.10% | 65.17% |
| EfficientNet-B4 | 99.70% | 64.29% |
| Ours (Multi-Attn) | 99.80% | 67.44% |

DFDC logloss: 0.1679 (best among compared methods including competition winners).
