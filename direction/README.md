# Analysis of Geospatial Orientation Detection System

This code implements a computer vision system designed to predict the orientation angle of geospatial images. 

## Task Overview

The task involves predicting the compass direction (angle between 0-360 degrees) of geospatial images. This is formulated as a regression problem where the model must predict a continuous angle value representing the image orientation.

## Model Architecture

The system uses a deep learning approach with a **fine-tuned ResNet50 CNN** as its backbone. This is a strategic choice because:

1. The ResNet50 is pre-trained on ImageNet, providing robust feature extraction capabilities for visual data
2. The deep residual network architecture allows effective learning of complex patterns without degradation problems

The model architecture includes several innovative elements:

- The original classification head is replaced with a custom **regression head** that outputs orientation angles
- The regression head uses **residual blocks** (similar to the main ResNet architecture) to maintain gradient flow
- A **2D vector output** approach is used instead of direct angle prediction - the model outputs (sin θ, cos θ) coordinates on the unit circle, which are then converted back to angles

## Data Processing & Augmentation

The preprocessing pipeline includes:

1. **Data cleaning**: Filtering to ensure all angles are within the 0-360 degree range
2. **Resizing**: Images are resized to 256×256 pixels for training and 224×224 for validation
3. **Normalization**: Using ImageNet mean and standard deviation values
4. **Augmentation**: Several sophisticated augmentation techniques specifically designed for geospatial data:
   - Color jitter (brightness, contrast, saturation)
   - Random grayscale conversion
   - Gaussian blur
   - Sharpness adjustment
   - Affine transformations (conditional)

The augmentation approach is particularly well-suited for geospatial imagery, enhancing model robustness to variations in lighting, weather conditions, and image quality.

## Innovative Technical Solutions

Several innovative approaches stand out in this implementation:

### 1. Circular Regression Representation

Instead of directly predicting angles (which creates discontinuities at 0°/360°), the model predicts **unit vector coordinates** (sin θ, cos θ). This approach:
- Eliminates the discontinuity problem
- Ensures smooth learning around the full circle
- Avoids the need to handle the periodic nature of angles separately

### 2. Multi-Component Angular Loss

The custom loss function combines three components:
- **Cosine loss**: Ensures vector similarity
- **L1 angular loss**: Minimizes absolute angle differences
- **Circumference loss**: Uses a cosine-based penalty to handle the circular nature of the problem

This weighted combination (α=0.6, β=0.3, γ=0.1) provides a balanced optimization target.

### 3. Advanced Training Techniques

Several training optimizations are implemented:
- **Mixed precision training** using GradScaler for computational efficiency
- **Gradient clipping** to prevent exploding gradients
- **AdamW optimizer** with weight decay for better generalization
- **Cosine annealing learning rate scheduler** to effectively navigate the loss landscape
- **Early stopping** with patience to prevent overfitting

### 4. Custom Evaluation Metric

The model uses **Mean Absolute Angular Error (MAAE)** as its primary evaluation metric, which correctly handles the circular nature of angles by taking the minimum of the direct difference and the complementary difference (360° - direct difference).

## Model Performance

The code includes comprehensive validation procedures:
- Model checkpointing based on validation MAAE
- A score calculation using 1/(1+MAAE) for easier interpretation
- Separate test set prediction

## Final Output

The system produces rounded angle predictions for both validation and test sets, saving them in a CSV file for submission. The results are carefully formatted to ensure compatibility with the expected output format.


## Models
https://iiithydstudents-my.sharepoint.com/:f:/g/personal/himani_sharma_students_iiit_ac_in/EqaWbJvCExpBtsWDtuRqY1gBK70GtG3y4nWoodplS5_r3A?e=zuc7tv => Models
