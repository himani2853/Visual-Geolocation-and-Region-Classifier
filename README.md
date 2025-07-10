# Visual-Geolocation-and-Region-Classifier

This project implements a comprehensive computer vision system for analyzing geospatial imagery, tackling three interconnected tasks: **region classification**, **coordinate prediction**, and **orientation detection**. The system uses state-of-the-art deep learning architectures to extract meaningful geographic information from satellite and aerial imagery.

## 📋 Project Overview

The project is divided into three main components:

1. **Region Classification** (`regionID/`) - Classifies images into 15 distinct geographical regions
2. **Coordinate Prediction** (`latlong/`) - Predicts precise latitude and longitude coordinates
3. **Orientation Detection** (`direction/`) - Determines compass direction (0-360 degrees)

## 🏗️ Architecture Overview

```
Input Image → Region Classification → Coordinate Prediction → Orientation Detection
     ↓                    ↓                      ↓                    ↓
Vision Transformer    ResNet-101 +           ResNet-50 +
   (ViT-Base)        Region Embedding      Circular Regression
     ↓                    ↓                      ↓
  Region ID          Lat/Long Coords        Angle (degrees)
```

## 📁 Project Structure

```
2022102032/
├── README.md                    # This file
├── direction/                   # Orientation detection module
│   ├── angle.py                # Main training/inference script
│   ├── README.md               # Detailed technical documentation
│   └── submission.csv          # Angle predictions
├── latlong/                    # Coordinate prediction module
│   ├── lat_long.py            # Main training/inference script
│   ├── README.md              # Detailed technical documentation
│   └── submission.csv         # Coordinate predictions
└── regionID/                   # Region classification module
    ├── region.py              # Main training/inference script
    ├── README.md              # Detailed technical documentation
    └── submission.csv         # Region predictions
```

## 🔧 Technical Implementation

### 1. Region Classification (regionID/)

**Model**: Vision Transformer (ViT-Base-Patch16-224)
- **Architecture**: Pre-trained ViT with custom classification head
- **Input**: 224×224 RGB images
- **Output**: 15 region classes (Region_ID 1-15)
- **Key Features**:
  - Transformer-based attention mechanism
  - Class-balanced loss with label smoothing
  - Cosine annealing with warm restarts
  - Comprehensive data augmentation

**Performance**: 95.66% test accuracy (353/369 correct)

### 2. Coordinate Prediction (latlong/)

**Model**: ResNet-101 with Region-Aware Attention
- **Architecture**: Pre-trained ResNet-101 + Region embeddings + Attention mechanism
- **Input**: 224×224 RGB images + Region_ID embedding
- **Output**: Latitude and longitude coordinates
- **Key Features**:
  - Hierarchical prediction using region context
  - Custom attention mechanism for feature weighting
  - Normalized coordinate space for better convergence
  - Region-specific visual cue learning

**Performance**: Validation Loss: 56,270.82

### 3. Orientation Detection (direction/)

**Model**: ResNet-50 with Circular Regression
- **Architecture**: Pre-trained ResNet-50 with custom regression head
- **Input**: 224×224 RGB images
- **Output**: Compass direction (0-360 degrees)
- **Key Features**:
  - Circular regression using unit vector representation (sin θ, cos θ)
  - Multi-component angular loss function
  - Mixed precision training with gradient scaling
  - Sophisticated geospatial augmentation pipeline

**Performance**: Mean Absolute Angular Error (MAAE): 24.16°

## 🚀 Innovation Highlights

### 1. Circular Regression for Angles
- **Problem**: Direct angle prediction creates discontinuities at 0°/360°
- **Solution**: Predict unit vector coordinates (sin θ, cos θ) instead
- **Benefit**: Eliminates discontinuity, ensures smooth learning

### 2. Region-Aware Coordinate Prediction
- **Problem**: Global coordinate prediction is challenging
- **Solution**: Two-stage approach using region context
- **Benefit**: Leverages geographic hierarchy for better accuracy

### 3. Multi-Component Loss Functions
- **Angle Detection**: Combines cosine loss, L1 angular loss, and circumference loss
- **Coordinate Prediction**: Balanced MSE for latitude and longitude
- **Region Classification**: Class-weighted cross-entropy with label smoothing

### 4. Advanced Training Techniques
- **Mixed Precision Training**: Faster training with maintained accuracy
- **Gradient Clipping**: Prevents exploding gradients
- **Sophisticated Schedulers**: Cosine annealing and adaptive learning rates
- **Early Stopping**: Prevents overfitting with patience mechanisms

## 📊 Performance Metrics

| Task | Model | Metric | Performance |
|------|--------|--------|-------------|
| Region Classification | ViT-Base | Accuracy | 95.66% |
| Coordinate Prediction | ResNet-101 | Validation Loss | 56,270.82 |
| Orientation Detection | ResNet-50 | MAAE | 24.16° |

## 🔬 Data Processing Pipeline

### Image Preprocessing
- **Resizing**: 224×224 (ViT, ResNet-101) or 256×256→224×224 (ResNet-50)
- **Normalization**: ImageNet mean and standard deviation
- **Augmentation**: Task-specific augmentation strategies

### Data Augmentation Strategies
- **Region Classification**: Random crops, flips, rotations, color jitter
- **Coordinate Prediction**: Horizontal flips, rotations, color jitter
- **Orientation Detection**: Geospatial-specific augmentations (blur, sharpness, affine)

### Data Cleaning
- **Coordinate Prediction**: Outlier removal (1st-99th percentile filtering)
- **Orientation Detection**: Angle validation (0-360 degree range)
- **Region Classification**: Balanced sampling strategies

## 🛠️ Setup and Installation

### Prerequisites
```bash
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.15.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.3.0
tqdm>=4.62.0
matplotlib>=3.4.0
```

### Installation
```bash
# Clone or download the project
cd 2022102032/

# Install dependencies
pip install torch torchvision transformers numpy pandas Pillow tqdm matplotlib

# Ensure CUDA is available for GPU acceleration (recommended)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🏃‍♂️ Usage

### 1. Region Classification
```bash
cd regionID/
python region.py
```

### 2. Coordinate Prediction
```bash
cd latlong/
python lat_long.py
```

### 3. Orientation Detection
```bash
cd direction/
python angle.py
```

## 📝 Model Outputs

Each module generates a `submission.csv` file containing:
- **Region Classification**: `filename`, `pred_Region_ID`
- **Coordinate Prediction**: `filename`, `latitude`, `longitude`
- **Orientation Detection**: `filename`, `angle`

## 🎯 Use Cases

- **Satellite Image Analysis**: Automated processing of satellite imagery
- **Geographic Information Systems (GIS)**: Enhanced spatial data analysis
- **Navigation Systems**: Orientation detection for autonomous vehicles
- **Environmental Monitoring**: Region-based analysis of geographic changes
- **Urban Planning**: Spatial analysis and coordinate mapping

## 🔗 Pre-trained Models

Pre-trained models are available at:
[SharePoint Models Directory](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/himani_sharma_students_iiit_ac_in/EqaWbJvCExpBtsWDtuRqY1gBK70GtG3y4nWoodplS5_r3A?e=zuc7tv)

## 📚 Technical Documentation

For detailed technical documentation, please refer to the README.md files in each module:
- [`regionID/README.md`](regionID/README.md) - Vision Transformer implementation details
- [`latlong/README.md`](latlong/README.md) - Region-aware coordinate prediction
- [`direction/README.md`](direction/README.md) - Circular regression for orientation

## 🤝 Contributing

This project represents a comprehensive approach to geospatial computer vision. Each module can be used independently or as part of the integrated pipeline for complete geospatial analysis.

## 📄 License

This project is part of academic research and is provided for educational and research purposes.

---

*This project demonstrates state-of-the-art techniques in computer vision, deep learning, and geospatial analysis, combining multiple architectures and innovative approaches to solve complex geographic information extraction tasks.*
