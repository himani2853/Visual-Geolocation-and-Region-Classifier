# Geolocation and Region Classification Using Deep Learning

This project implements an image-based geolocation system that predicts the latitude and longitude coordinates of images. This model uses Region_ID embedding as an input to the model which he;ps in predicting latitude, longitude in a better way

## Task Overview

Here we are giving coordinates predicted from region_ID [`pred_Region_ID`] model into the 'resnet-101' model as embedding.

## Model Architecture and Training Strategy

Coordinate Prediction with ResNet and Region Embeddings

After classifying the region in region_ID, we predict the precise coordinates:

- **Model Type**: Pre-trained ResNet-101 with custom attention mechanism
- **Approach**: Fine-tuning the pre-trained CNN backbone with an innovative region-aware prediction head
- **Architecture Details**:
  - ResNet-101 backbone extracts 2048-dimensional image features
  - Region information is encoded using a 256-dimensional embedding layer
  - Combines image features and region embeddings
  - Uses an attention mechanism to weight the importance of different features
  - Final head uses two fully connected layers (512 neurons in hidden layer) to predict lat/long coordinates

## Preprocessing and Data Handling

### Image Preprocessing

- **Resizing**: Images are resized to 224×224 pixels to match the input requirements of the models
- **Data Augmentation**: 
  - For ResNet (coordinate prediction): Horizontal flips, rotations (15°), and color jitter
- **Normalization**: Images are normalized using ImageNet mean and standard deviation
The model normalizes latitude and longitude values during training using the mean and standard deviation of the training set. This normalization helps the model converge faster and learn more effectively. During prediction, the outputs are denormalized back to the original coordinate space.


### Coordinate Preprocessing

- **Outlier Removal**: 
  - Filtering out images with latitude/longitude outside the 1st-99th percentile range
  - Explicit removal of problematic samples (rows 95, 145, 146, 158-161) as stated 
- **Normalization**: Coordinates are normalized using mean and standard deviation from the training set for better model convergence

## Innovative Ideas and Technical Highlights

1. **Two-Stage Hierarchical Approach**: Rather than predicting coordinates directly, the system first classifies regions, which provides useful context for the coordinate prediction stage

2. **Region-Aware Attention Mechanism**: The coordinate prediction model uses an attention mechanism that learns to weight different features based on the region information, allowing it to focus on region-specific visual cues

3. **Region Embeddings**: Converting categorical region IDs into dense embeddings allows the coordinate model to learn meaningful representations of different geographic regions

4. **Mixed Loss Function**: For coordinate prediction, you're using a specialized average MSE loss that balances latitude and longitude errors equally (0.5 * lat_mse + 0.5 * lon_mse)

5. **Regularization Techniques**:
   - Class weighting and label smoothing in the region classification
   - Gradient clipping to prevent exploding gradients
   - Early stopping with patience to prevent overfitting

6. **Learning Rate Scheduling**:
   - Cosine annealing with warm restarts for region classification
   - ReduceLROnPlateau for coordinate prediction

## Training Methodology
The model uses several techniques to ensure effective training:
- MSE loss calculated separately for latitude and longitude and then averaged
- Adam optimizer with a modest learning rate (1e-4)
- Learning rate scheduler (ReduceLROnPlateau with factor 0.5 and patience 5) that reduces the learning rate when validation performance plateaus
- Early stopping with patience of 5 epochs to prevent overfitting
- Gradient clipping to prevent explosive gradients

## Attention Mechanism
The model implements an attention mechanism that learns to weight the combined features. This allows the model to focus on the most relevant aspects of the input for predicting coordinates. The attention module consists of a two-layer neural network that outputs weights, which are then applied to the combined features before final prediction.


## Results and Evaluation

- **Coordinate Prediction**: Evaluated using MSE loss on denormalized latitude and longitude coordinates


The high accuracy in region classification (95.66%) helps the coordinate prediction stage focus on region-specific features for more accurate geo-localization.
Validation Loss: 56270.820766


## Models
https://iiithydstudents-my.sharepoint.com/:f:/g/personal/himani_sharma_students_iiit_ac_in/EqaWbJvCExpBtsWDtuRqY1gBK70GtG3y4nWoodplS5_r3A?e=zuc7tv => Models
