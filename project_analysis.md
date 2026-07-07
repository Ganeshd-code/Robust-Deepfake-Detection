# Project Analysis: Robust Deepfake Detection System

## Overview

This is a deepfake detection research project built on the FaceForensics++ (C23 compression) dataset. It implements and compares four model variants — Spatial, Frequency, Hybrid, and Asymmetric Hybrid — using a dual-stream CNN architecture based on ResNet-18 backbones. A Streamlit web application serves as the inference frontend.

---

## Project Structure

```
.
├── app.py                        # Streamlit inference UI
├── extract_frames.py             # Video → frame extraction
├── extract_faces.py              # Frame → face crop extraction
├── create_splits.py              # Train/val/test CSV split creation
├── evaluate_distortion.py        # Clean vs distorted robustness evaluation
├── analyze_papers.py             # Literature paper text analysis helper
├── extract_papers.py             # PDF text extraction utility
├── paper_texts.json              # Extracted paper text cache
│
├── models/
│   ├── spatial_model.py          # ResNet-18 spatial branch model
│   ├── frequency_model.py        # ResNet-18 frequency branch model
│   ├── hybrid_model.py           # Dual-stream fusion model (used for both Hybrid and Asymmetric)
│   ├── spatial.pth               # Saved spatial model checkpoint
│   ├── frequency.pth             # Saved frequency model checkpoint
│   ├── hybrid.pth                # Saved hybrid model checkpoint
│   └── asymmetric.pth            # Saved asymmetric model checkpoint
│
├── train/
│   ├── train_spatial.py          # Spatial model training script
│   ├── train_frequency.py        # Frequency model training script
│   ├── train_hybrid.py           # Hybrid model training script
│   └── train_asymmetric.py       # Asymmetric model training script
│
├── utils/
│   ├── transforms.py             # Augmentation and normalization transforms
│   ├── fft.py                    # Per-channel FFT magnitude spectrum computation
│   ├── dataset_spatial.py        # Dataset for spatial branch (RGB images)
│   ├── dataset_freq.py           # Dataset for frequency branch (FFT images)
│   ├── dataset_hybrid.py         # Dataset for hybrid model (RGB + FFT pairs)
│   └── dataset_asymmetric.py     # Dataset for asymmetric model (augmented FFT)
│
├── data/
│   ├── frames/real|fake/         # Extracted video frames
│   ├── faces/real|fake/          # Cropped face images (224×224)
│   └── splits/
│       ├── train.csv             # 2,774 samples (1389 real, 1385 fake)
│       ├── val.csv               # 594 samples (298 real, 296 fake)
│       └── test.csv              # 595 samples (298 real, 297 fake)
│
├── FaceForensics++_C23/
│   ├── original/                 # Real videos
│   ├── Deepfakes/                # 1000 Deepfakes manipulation videos
│   ├── FaceSwap/                 # FaceSwap manipulation videos
│   ├── Face2Face/                # (metadata only, not used in training)
│   ├── FaceShifter/              # (metadata only, not used in training)
│   ├── NeuralTextures/           # (metadata only, not used in training)
│   ├── DeepFakeDetection/        # (metadata only, not used in training)
│   └── csv/                      # Per-manipulation metadata CSVs
│
└── paper/
    ├── deepfake_ieee_paper.tex   # IEEE paper LaTeX source
    └── generate_figures.py       # Empty (not yet implemented)
```

---

## Dataset

### Source
FaceForensics++ at C23 (light) compression quality.

### Manipulations Used for Training
- Real: `FaceForensics++_C23/original/` (up to 200 videos processed)
- Fake: `FaceForensics++_C23/Deepfakes/` + `FaceForensics++_C23/FaceSwap/` (up to 200 videos each)

### Manipulations Present but NOT Used in Training
Face2Face, FaceShifter, NeuralTextures, DeepFakeDetection (metadata CSVs exist but paths not in `extract_frames.py`)

### Dataset Statistics (from actual split CSVs)
| Split | Total | Real | Fake |
|-------|-------|------|------|
| Train | 2,774 | 1,389 | 1,385 |
| Val   | 594   | 298  | 296  |
| Test  | 595   | 298  | 297  |
| **Total** | **3,963** | **1,985** | **1,978** |

Split ratio: 70% train / 15% val / 15% test (stratified, random_state=42)

### Video Metadata (from Mean_Data.csv)
| Subset | Avg Frames | Resolution | Avg Size |
|--------|-----------|------------|----------|
| Deepfakes | 509 | 1036×637 | 1.90 MB |
| FaceSwap | 406 | 1036×637 | 1.56 MB |
| Original | 509 | 1036×637 | 1.85 MB |

---

## Preprocessing Pipeline

1. **Frame Extraction** (`extract_frames.py`)
   - Reads MP4 videos from `original/`, `Deepfakes/`, `FaceSwap/`
   - Samples every 5th frame, saves up to 10 frames per video
   - Processes up to 200 videos per class
   - Saves as JPG to `data/frames/real/` and `data/frames/fake/`

2. **Face Detection & Cropping** (`extract_faces.py`)
   - Uses OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
   - Resizes large images to max 1000px before detection
   - Detects faces with `scaleFactor=1.2`, `minNeighbors=5`, `minSize=(60,60)`
   - Keeps only the largest detected face
   - Adds 20px padding around face bounding box
   - Resizes final crop to **224×224 pixels**
   - Skips frames with no detected face

3. **Dataset Splitting** (`create_splits.py`)
   - Loads all face image paths with labels (0=real, 1=fake)
   - Stratified 70/15/15 split using sklearn
   - Saves `train.csv`, `val.csv`, `test.csv` with columns: `path`, `label`

---

## No Training Logs Found

No training log files, metric JSON files, confusion matrices, or evaluation output files exist in the workspace. Only the four model checkpoint `.pth` files are present. Actual training metrics are not recoverable from disk.
