import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageOps
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast


class GeoAngleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        
        # Clean and filter data
        self.annotations = self.annotations[
            (self.annotations['angle'].between(0, 360))
        ].reset_index(drop=True)
        
        # Extract necessary columns
        self.annotations = self.annotations[['filename', 'angle']]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]['filename'])
        angle_deg = float(self.annotations.iloc[idx]['angle'])
        
        image = Image.open(img_path).convert('RGB')
        
        if self.augment:
            image, angle_deg = self._augment_image(image, angle_deg)
            
        if self.transform:
            image = self.transform(image)
            
        angle_rad = np.deg2rad(angle_deg)
        return image, torch.tensor([np.sin(angle_rad), np.cos(angle_rad)], dtype=torch.float32)

    def _augment_image(self, image, angle):
        """Enhanced geospatial-specific augmentations"""

        # Color transformations
        transform_list = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)) if random.random() < 0.1 else lambda x: x,
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1)
        ])
        image = transform_list(image)
        
        # Random affine transformation ## affine_removed
        if random.random() < 0.2:
            image = transforms.functional.affine(
                image, angle=0, translate=(10,10), scale=1.0, shear=10
            )
        
        return image, angle

    
class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Get sorted list of JPG files with numeric sorting
        self.image_files = sorted(
            [f for f in os.listdir(img_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(x.split('_')[1].split('.')[0])  # Extract numeric part after 'img_'
        )
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image  # Return only image tensor

class ResNet50GeoAngleRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Enhanced regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(in_features),
            ResidualBlock(in_features, 1024),
            nn.Dropout(0.3),
            ResidualBlock(1024, 512),
            nn.Dropout(0.2),
            ResidualBlock(512, 256),
            nn.Linear(256, 2)
        )
            
    def forward(self, x):
        features = self.resnet(x)
        return F.normalize(self.regressor(features), p=2, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, drop_rate=0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        self.skip = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        
    def forward(self, x):
        identity = self.skip(x)
        out = self.linear1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + identity


def angular_loss(pred, target, alpha=0.6, beta=0.3, gamma=0.1):
    """Enhanced loss function with three components"""
    cosine_loss = 1 - torch.mean(torch.sum(pred * target, dim=1))
    
    pred_angle = torch.atan2(pred[:,0], pred[:,1])
    target_angle = torch.atan2(target[:,0], target[:,1])
    angle_diff = torch.abs(pred_angle - target_angle)
    angle_diff = torch.minimum(angle_diff, 2*np.pi - angle_diff)
    
    l1_loss = torch.mean(angle_diff)
    circumference_loss = torch.mean(1 - torch.cos(2*angle_diff))
    
    return alpha*cosine_loss + beta*l1_loss + gamma*circumference_loss


def calculate_maae(pred, target):
    pred_deg = torch.rad2deg(torch.atan2(pred[:,0], pred[:,1])) % 360
    true_deg = torch.rad2deg(torch.atan2(target[:,0], target[:,1])) % 360
    diff = torch.abs(pred_deg - true_deg)
    return torch.mean(torch.minimum(diff, 360 - diff))


def train_model(train_loader, val_loader, device, epochs=50):
    model = ResNet50GeoAngleRegressor().to(device)
    print(f"no of epochs: {epochs}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) #### 24.09 #### 0.0399 ####
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=3,
    #     min_lr=1e-6,
    #     verbose=True
    # )


    scaler = GradScaler()
    best_maae = float('inf')
    best_score = 0.0
    patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        
        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                preds = model(images)
                loss = angular_loss(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(loss.item())
        
        # Validation
        model.eval()
        val_maae = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                val_maae.append(calculate_maae(preds, targets).item())
        
        current_maae = np.mean(val_maae)
        score = 1/(1+current_maae)
        scheduler.step() #### 24.09 #### 0.0399 ####
        
        if current_maae < best_maae:
            best_maae = current_maae
            best_score = score
            torch.save(model.state_dict(), '../Models/best_resnet50_geo_angle_2.pth') #### 24.09 #### 0.0399 ####
            patience = 10  # Reset patience
            print(f"* New best MAE: {best_maae:.2f}° | best score: {best_score:.4f}")
        else:
            patience -= 1
            
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {np.mean(train_loss):.4f} | "
              f"Val MAE: {current_maae:.2f}° | "
              f"Score: {score:.4f} | "
              f"Best MAE: {best_maae:.2f}° | "
              f"Best score: {best_score:.4f} | "
              f"Patience: {patience}")
              
        if patience <= 0:
            print("Early stopping triggered")
            break
    
    model.load_state_dict(torch.load('../Models/best_resnet50_geo_angle_2.pth')) #### 24.09 #### 0.0399 ####
    return model


def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            preds = model(images)
            
            # Convert predictions to degrees
            pred_deg = torch.rad2deg(torch.atan2(preds[:,0], preds[:,1])) % 360
            all_preds.extend(pred_deg.cpu().numpy())
            all_targets.append(targets.cpu())
    
    # Calculate final MAE
    targets = torch.cat(all_targets)
    pred_vecs = torch.stack([torch.sin(torch.deg2rad(torch.tensor(all_preds))),
                            torch.cos(torch.deg2rad(torch.tensor(all_preds)))], dim=1)
    final_maae = calculate_maae(pred_vecs, targets).item()
    score = 1/(1+final_maae)
    print(f"Final MAE: {final_maae:.2f}° | Score: {score:.4f}")
    
    return all_preds

def predict_test_set(model, test_dir, batch_size=16, device = 'cuda'):
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestImageDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_filenames = test_dataset.image_files
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            preds = model(images)
            pred_deg = torch.rad2deg(torch.atan2(preds[:,0], preds[:,1])) % 360
            all_preds.extend(pred_deg.cpu().numpy())
    
    # for filename, pred in zip(all_filenames, all_preds):
    #     print(f"{filename}: {pred:.2f} degrees")
    return all_preds



# Configuration
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    test_dir = '../Data/images_test'
    # Datasets
    train_ds = GeoAngleDataset(
        '../Data/labels_train.csv',
        '../Data/images_train',
        transform=train_transform,
        augment=True
    )
    
    val_ds = GeoAngleDataset(
        '../Data/labels_val.csv',
        '../Data/images_val',
        transform=val_transform
    )
    
    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Train
    # model = train_model(train_loader, val_loader, device, epochs=50)
    model = ResNet50GeoAngleRegressor().to(device)
    model.load_state_dict(torch.load('../Models/best_resnet50_geo_angle_2.pth')) #### 24.09 #### 0.0399 ####
    
    # Predict
    val_pred = predict(model, val_loader, device)
    test_pred = predict_test_set(model, test_dir, batch_size=16, device=device)
    
    val_pred = np.array(val_pred)
    test_pred = np.array(test_pred)
    
    print(len(val_pred), len(test_pred))
    submission_df = pd.DataFrame({
        'id': range(len(test_pred) + len(val_pred)),
        'angle': [0] * len(test_pred) + [0] * len(val_pred)
    })

    rounded_val_preds = np.round(val_pred % 360).astype(int)
    rounded_test_preds = np.round(test_pred % 360).astype(int)

    # Create final predictions column
    submission_df.loc[:len(rounded_val_preds) - 1, 'angle'] = rounded_val_preds
    submission_df.loc[len(rounded_val_preds):, 'angle'] = rounded_test_preds
    submission_df.to_csv('../Results/2022102032_angle.csv', index=False)
    print(f"Saved predictions to ../Results/2022102032_angle.csv ({len(submission_df)} rows)")
    # submission_df.to_csv('../Results/resnet50_predictions_2.csv', index=False)  #### 24.09 #### 0.0399 ####
    # print("Predictions saved to Results/resnet50_predictions_2.csv") ##### 24.09 #### 0.0399 ####


