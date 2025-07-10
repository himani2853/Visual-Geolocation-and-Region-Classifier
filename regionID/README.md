# Analysis of Geospatial Region Classification System

This code implements a sophisticated computer vision system for classifying geospatial images into specific regions. 

## Task Overview

The task involves classifying geospatial imagery into 15 distinct geographical regions (labeled as Region_ID 1-15). This is formulated as a multi-class classification problem where the model must identify the region from visual patterns in the satellite or aerial imagery.

## Model Architecture

The system employs a **Vision Transformer (ViT)** architecture, specifically using the pre-trained `google/vit-base-patch16-224-in21k` model as its foundation. This represents a departure from traditional CNN-based approaches to geospatial imagery analysis.

The Vision Transformer works by:
1. Dividing the input image into fixed-size patches (16×16 pixels)
2. Linearly embedding these patches
3. Adding position embeddings
4. Processing them through a series of transformer blocks with self-attention mechanisms

The implementation includes several notable architectural decisions:

- **Fine-tuning approach**: The entire pre-trained ViT is unfrozen (`for p in self.vit.parameters(): p.requires_grad = True`), allowing the model to adapt its learned features to the geospatial domain
- **Custom classification head**: Rather than using a simple linear layer, the model employs a two-layer classification head with:
  - A dimensionality reduction layer (768 → 512)
  - GELU activation for non-linearity
  - Dropout for regularization
  - Final projection to 15 output classes

## Data Processing & Augmentation

The preprocessing pipeline incorporates several techniques to enhance model generalization:

1. **Training augmentations**:
   - Random resized cropping (from 256px to 224px)
   - Random horizontal flips
   - Random rotations (±15°)
   - Color jitter (brightness, contrast, saturation)

2. **Validation/Testing transforms**:
   - Center cropping to ensure consistent evaluation
   - Standard normalization using ImageNet statistics

The data loading approach uses a custom `GeoImageDataset` class that handles the integration of image data with corresponding region labels from the CSV files.

## Training Methodology

The training process incorporates several advanced techniques:

### 1. Class-Balanced Loss

The implementation addresses potential class imbalance by computing class weights inversely proportional to class frequency:
```python
class_counts = np.bincount([label for _, label in train_ds])
class_weights = 1. / torch.Tensor(class_counts)
```

### 2. Label Smoothing

The CrossEntropyLoss incorporates label smoothing (0.1), which helps:
- Prevent the model from becoming overconfident
- Improve generalization by avoiding hard probability assignments
- Create more calibrated prediction confidences

### 3. Optimization Strategy

The training process uses:
- **AdamW optimizer**: Combining adaptive learning rates with proper weight decay
- **Gradient clipping**: Preventing exploding gradients by clamping to a maximum norm of 1.0
- **Cosine annealing with warm restarts**: A sophisticated learning rate schedule that periodically resets the learning rate to allow for better exploration of the loss landscape

### 4. Early Stopping

The implementation includes a patience-based early stopping mechanism that monitors validation accuracy and terminates training if performance decreases consistently over a specified number of epochs.

## Evaluation and Prediction

The model achieved impressive performance with a test accuracy of 95.66% (353/369 correct), demonstrating strong generalization capabilities.

The prediction process includes:
1. Loading the best saved model weights
2. Processing test images through the same validation transforms
3. Converting model outputs (logits) to class predictions via argmax
4. Adjusting predictions from 0-14 back to the original 1-15 Region_ID range

## Innovative Technical Aspects

Several innovative aspects stand out in this implementation:

1. **Transformer-based architecture**: Using a Vision Transformer instead of a CNN represents a modern approach to image classification, leveraging the power of attention mechanisms for geospatial analysis

2. **Transfer learning effectiveness**: Successfully adapting a model pre-trained on general images (ImageNet-21k) to the specialized domain of geospatial imagery demonstrates the versatility of transformer architectures

3. **Sophisticated training regimen**: The combination of class weighting, label smoothing, and advanced learning rate scheduling creates a robust training approach

4. **Comprehensive evaluation**: The code includes detailed evaluation on training, validation, and test sets with proper metrics tracking

The model's high accuracy (95.66%) suggests that Vision Transformers can be highly effective for geospatial region classification tasks when properly trained and optimized.
Final MAE: 24.16° | Score: 0.0398

## Models
https://iiithydstudents-my.sharepoint.com/:f:/g/personal/himani_sharma_students_iiit_ac_in/EqaWbJvCExpBtsWDtuRqY1gBK70GtG3y4nWoodplS5_r3A?e=zuc7tv => Models
