# System Architecture - Container ID Extraction Pipeline

**Project:** Container ID Extraction Research  
**Version:** 1.0  
**Last Updated:** 2024-12-04

---

## Table of Contents

1. [Overview](#overview)
2. [System Design](#system-design)
3. [Module Architecture](#module-architecture)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Deployment Architecture](#deployment-architecture)

---

## Overview

The Container ID Extraction system is a multi-stage computer vision pipeline designed to automatically extract container identification numbers from images of container doors.

### Design Principles

1. **Modularity**: Each stage is independent and can be developed/tested separately
2. **Scalability**: Pipeline supports both real-time and batch processing
3. **Robustness**: Handles various edge cases (poor lighting, angles, occlusion)
4. **Reproducibility**: All experiments tracked and data versioned

---

## System Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input Image                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: Container Door Detection (YOLOv11)                    │
│  - Detects container door in image                               │
│  - Outputs: Bounding box coordinates                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 2: Image Quality Assessment                              │
│  - Evaluates image quality (blur, lighting, size)                │
│  - Outputs: Quality score, pass/fail decision                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 3: Container ID Localization (YOLOv11-Pose)              │
│  - Predicts 4-point polygon around ID region                     │
│  - Outputs: Keypoint coordinates (TL, TR, BR, BL)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 4: Perspective Correction (OpenCV)                       │
│  - Applies perspective transform to straighten ID                │
│  - Outputs: Rectified ID region image                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 5: OCR Extraction                                        │
│  - Extracts text from rectified image                            │
│  - Validates format (4 letters + 7 digits)                       │
│  - Outputs: Container ID string                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Container ID Output                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### Module 1: Container Door Detection

**Purpose**: Detect and localize container door in image

**Model**: YOLOv11-Nano/Small  
**Input**: RGB image (variable size)  
**Output**: Bounding box [x, y, w, h], confidence score

**Key Features**:
- Fast inference (real-time capable)
- Handles multiple container orientations
- Robust to partial occlusions

### Module 2: Image Quality Assessment

**Purpose**: Filter out low-quality images that would fail OCR

**Approach**: Rule-based + ML classifier  
**Input**: Cropped container door image  
**Output**: Quality score (0-1), quality gate decision

**Quality Metrics**:
- Blur detection (Laplacian variance)
- Brightness/contrast analysis
- Size adequacy check
- Overall quality score

### Module 3: Container ID Localization

**Purpose**: Precisely locate the 4 corners of container ID region

**Model**: YOLOv11-Pose  
**Input**: Container door image  
**Output**: 4 keypoints with confidence scores

**Keypoint Order** (clockwise from top-left):
1. Top-Left (TL)
2. Top-Right (TR)
3. Bottom-Right (BR)
4. Bottom-Left (BL)

### Module 4: Perspective Correction

**Purpose**: Rectify the ID region to frontal view

**Technique**: Perspective transformation (homography)  
**Input**: 4 keypoints + original image  
**Output**: Rectified ID image (standardized size)

**Process**:
1. Compute homography matrix from 4 keypoints
2. Apply perspective warp
3. Resize to standard dimensions (e.g., 400x100)
4. Quality check on warped image

### Module 5: OCR Extraction

**Purpose**: Extract and validate container ID text

**Engine**: PaddleOCR / EasyOCR / Tesseract  
**Input**: Rectified ID image  
**Output**: Container ID string

**Post-Processing**:
- Format validation (regex)
- Character corrections (common OCR errors)
- Check digit validation

---

## Data Flow

### Training Data Flow

```
Raw Images (831) + COCO Annotations
           ↓
    Data Stratification (Label Powerset)
           ↓
    ┌──────┴──────┬──────────┐
    ↓             ↓          ↓
  Train (70%)   Val (15%)  Test (15%)
    ↓             ↓          ↓
COCO → YOLO Format Conversion
    ↓
Module-Specific Datasets
    ↓
Model Training (YOLOv11)
    ↓
Trained Weights (.pt files)
```

### Inference Data Flow

```
User Image
    ↓
Detection → Quality Check → Localization
    ↓           ↓              ↓
  bbox      pass/fail      keypoints
                ↓              ↓
           Perspective Correction
                ↓
          Rectified Image
                ↓
              OCR
                ↓
         Container ID
```

---

## Technology Stack

### Core Framework
- **Python**: 3.13
- **Deep Learning**: PyTorch, Ultralytics YOLOv11
- **Computer Vision**: OpenCV, Albumentations

### Data Management
- **Version Control**: Git
- **Data Versioning**: DVC + Google Drive
- **Dependency Management**: Poetry

### Experiment Tracking
- **Tracking**: Weights & Biases (wandb)
- **Metrics**: mAP, OKS, inference time

### Deployment (Future)
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Serving**: ONNX Runtime / TorchServe

---

## Deployment Architecture

### Research Environment (Current)

```
┌─────────────────────────────────────────────┐
│  Development Machine                        │
│  ┌─────────────────────────────────────┐   │
│  │  Jupyter Notebooks (EDA)            │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Training Scripts                   │   │
│  │  - src/detection/train.py           │   │
│  │  - src/localization/train.py        │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  DVC (Data Management)              │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    ↓
            Google Drive (DVC Remote)
                    ↓
            Wandb (Experiment Tracking)
```

### Production Environment (Future)

```
┌────────────────────────────────────────────────────┐
│  Backend Service (FastAPI)                         │
│  ┌──────────────────────────────────────────────┐ │
│  │  API Endpoints                               │ │
│  │  - POST /api/v1/extract                      │ │
│  │  - GET  /api/v1/health                       │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │  Pipeline Orchestrator                       │ │
│  │  - Loads all 5 modules                       │ │
│  │  - Manages inference queue                   │ │
│  └──────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────┐ │
│  │  Model Serving                               │ │
│  │  - YOLOv11 Detection                         │ │
│  │  - YOLOv11 Pose                              │ │
│  │  - OCR Engine                                │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
                    ↓
            Database (Results Storage)
```

---

## Performance Considerations

### Latency Targets

| Module | Target Latency | Model Size |
|--------|---------------|------------|
| Detection | < 50ms | ~6 MB |
| Quality | < 10ms | Rule-based |
| Localization | < 50ms | ~8 MB |
| Alignment | < 20ms | OpenCV |
| OCR | < 100ms | ~10 MB |
| **Total** | **< 250ms** | **~24 MB** |

### Throughput

- **Single Image**: ~4 FPS
- **Batch Processing**: ~20 FPS (batch size 8)

### Hardware Requirements

**Development**:
- GPU: NVIDIA RTX 3060+ (6GB VRAM)
- RAM: 16 GB
- Storage: 50 GB

**Production** (per instance):
- GPU: NVIDIA T4 or better
- RAM: 8 GB
- Storage: 20 GB

---

## Future Enhancements

1. **Real-time Video Processing**: Process video streams
2. **Multi-Container Detection**: Handle multiple containers in one image
3. **Mobile Deployment**: TensorFlow Lite / ONNX for mobile
4. **Active Learning**: Continuous model improvement pipeline
5. **Multi-Language Support**: Support non-English container IDs

---

**Document Maintainer**: duyhxm  
**Organization**: SOWATCO

