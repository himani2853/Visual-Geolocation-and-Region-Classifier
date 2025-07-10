######################### # Training complete.  Test Accuracy: 0.9566 (353/369)#########################
# This is the best accuracy, ./Models/best_vit_geo_region_1.pth is enough #######################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
from transformers import ViTModel

class GeoImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Read CSV and name columns
        self.df = pd.read_csv(csv_file)
        self.df.columns = [
            'filename', 'timestamp', 'latitude', 'longitude', 'angle', 'Region_ID'
        ]
        self.img_dir   = img_dir
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1) Load image
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.filename)
        image = Image.open(img_path).convert('RGB')
        
        # 2) Apply torchvision-style transform (Resize, Crop, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)  # returns a FloatTensor [3×224×224]
        
        # 3) Convert Region_ID from [1..15] → [0..14]
        label = int(row.Region_ID) - 1
        
        return image, label
    
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


# ../Models/best_vit_geo_region_1.pth#######################
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])


class ViTGeoClassifier(nn.Module):
    def __init__(self, num_classes=15, dropout=0.3):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for p in self.vit.parameters(): p.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixels):
        # pixels: [B, 3, 224, 224]
        outputs = self.vit(pixel_values=pixels)
        cls_token = outputs.last_hidden_state[:,0]  # [B, 768]
        return self.classifier(self.dropout(cls_token))

    
# 4. Training loop
def train_and_validate(
    train_csv, train_dir,
    val_csv,   val_dir,
    num_epochs=20,
    batch_size=16,
    device='cuda',
    early_stop_patience=4
):
    # Data
    train_ds = GeoImageDataset(train_csv, train_dir, transform=train_transform)
    val_ds   = GeoImageDataset(val_csv,   val_dir,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    model = ViTGeoClassifier(num_classes=15, dropout=0.3).to(device)
    
    class_counts = np.bincount([label for _, label in train_ds])
    class_weights = 1. / torch.Tensor(class_counts)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2) # ../Models/best_vit_geo_region_1.pth#######################
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, num_epochs+1):
        # — Training —
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
        scheduler.step()

        # — Validation —
        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_ds)
        val_loss   /= len(val_ds)
        val_acc = correct / total

        # Checkpoint
        if val_acc > best_acc:
            torch.save(model.state_dict(), '../Models/best_vit_geo_region_1.pth')
            print(f"→ New best model saved (Val Acc: {val_acc:.4f})")
            best_acc = val_acc

           

        print(f"Epoch {epoch:02d} → Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} "
              f"(Best: {best_acc:.4f})")
        

    # Track validation accuracies for consecutive decrease check
        if epoch == 1:
            val_acc_history = [val_acc]
        else:
            val_acc_history.append(val_acc)
            if len(val_acc_history) >= early_stop_patience:  # Check last 5 epochs (including current)
                recent = val_acc_history[-early_stop_patience:]
                if all(recent[i] < recent[i - 1] for i in range(1, early_stop_patience)):
                    print(f"→ Early stopping triggered (validation accuracy decreased in last {early_stop_patience} epochs).")
                    break
                
    print("Training complete. Best Val Acc:", best_acc)

def test(model,
        val_ds,
        batch_size=16,
        device='cuda'):

    print("Test function executed successfully.")
    model.load_state_dict(torch.load('../Models/best_vit_geo_region_1.pth'))
    model.eval()

    # Collect predictions
    val_preds = []
    true_labels = []
    with torch.no_grad():
        for imgs, labels in DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4):
            imgs = imgs.to(device)
            logits = model(imgs)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(batch_preds)
            true_labels.extend(labels.numpy())

    correct = np.sum(np.array(val_preds) == np.array(true_labels))
    accuracy = correct / len(val_ds)
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{len(val_ds)})\n")
    return val_preds


def predict_test_set(model, test_dir, batch_size=16, device='cuda'):
    # Use the same transform as validation
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    test_ds = TestImageDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.load_state_dict(torch.load('../Models/best_vit_geo_region_1.pth'))
    model.eval()
    
    test_preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Predicting Test Set"):
            imgs = imgs.to(device)
            logits = model(imgs)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(batch_preds)
    
    # Convert predictions from 0-14 to 1-15 (original Region_ID)
    # test_preds = [p + 1 for p in test_preds]
    test_preds = [p for p in test_preds]
    
    return test_preds

# 5. Run
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_and_validate(
    #     train_csv="../Data/labels_train.csv",
    #     train_dir="../Data/images_train",
    #     val_csv="../Data/labels_val.csv",
    #     val_dir="../Data/images_val",
    #     num_epochs=30,
    #     batch_size=64,
    #     device=device
    # )
    train_csv="../Data/labels_train.csv"
    train_dir="../Data/images_train"
    val_csv="../Data/labels_val.csv"
    val_dir="../Data/images_val"
    test_dir = "../Data/images_test"

    batch_size=16
    train_ds = GeoImageDataset(train_csv, train_dir, transform=train_transform)
    val_ds   = GeoImageDataset(val_csv,   val_dir,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)



    # Model, loss, optimizer, scheduler
    model = ViTGeoClassifier(num_classes=15, dropout=0.3).to(device)
    train_pred_ids = test(model,
        train_ds,
        batch_size=16,
        device=device)
        # path='../Results/train_predict_ids.csv')

    submission_df = pd.DataFrame({
        'id': range(len(train_pred_ids)),
        'Region_ID': [0] * len(train_pred_ids)
    })
    submission_df.loc[:len(train_pred_ids) - 1, 'Region_ID'] = train_pred_ids
    submission_df['Region_ID'] += 1  # Adjust Region_ID to match original range
    submission_df.to_csv('../Results/train_predictions.csv', index=False)
    print(f"Saved predictions to ../Results/train_predictions.csv ({len(submission_df)} rows)")

    val_pred_ids = test(model,
        val_ds,
        batch_size=16,
        device=device)
        # path='../Results/val_predict_ids.csv')

    submission_df = pd.DataFrame({
        'id': range(len(val_pred_ids)),
        'Region_ID': [0] * len(val_pred_ids)
    })
    submission_df.loc[:len(val_pred_ids) - 1, 'Region_ID'] = val_pred_ids
    submission_df['Region_ID'] += 1  # Adjust Region_ID to match original range
    submission_df.to_csv('../Results/val_predictions.csv', index=False)
    print(f"Saved predictions to ../Results/val_predictions.csv ({len(submission_df)} rows)")

    test_pred = predict_test_set(
        model,
        test_dir="../Data/images_test",  # Path to test images
        device=device
    )

    submission_df1 = pd.DataFrame({
        'id': range(len(test_pred)),
        'Region_ID': [0] * len(test_pred)
    })
    submission_df1.loc[:len(test_pred) - 1, 'Region_ID'] = test_pred
    submission_df1['Region_ID'] += 1  # Adjust Region_ID to match original range
    submission_df1.to_csv('../Results/test_predictions.csv', index=False)
    print(f"Saved predictions to ../Results/test_predictions.csv ({len(submission_df)} rows)")
    print(len(test_pred), len(val_pred_ids))
    submission_df = pd.DataFrame({
        'id': range(len(test_pred) + len(val_pred_ids)),
        'Region_ID': [0] * len(test_pred) + [0] * len(val_pred_ids)
    })
    submission_df.loc[:len(val_pred_ids) - 1, 'Region_ID'] = val_pred_ids
    submission_df.loc[len(val_pred_ids):, 'Region_ID'] = test_pred
    submission_df['Region_ID'] += 1  # Adjust Region_ID to match original range
    submission_df.to_csv('../Results/2022102032_region.csv', index=False)
    print(f"Saved predictions to ../Results/2022102032_region.csv ({len(submission_df)} rows)")

    train_df = pd.read_csv(train_csv)
    train_df.columns = ['filename', 'timestamp', 'latitude', 'longitude', 'angle', 'Region_ID']
    print(f"Loaded {len(train_df)} training samples from {train_csv}")
    pred_df = pd.read_csv('../Results/train_predictions.csv')
    pred_df.columns = ['id', 'Region_ID']
    print(f"Loaded {len(pred_df)} predictions from ../Results/train_predictions.csv")
    assert len(train_df) == len(pred_df), f"Mismatch: train_df has {len(train_df)} rows, pred_df has {len(pred_df)} rows"
    train_df['pred_Region_ID'] = pred_df['Region_ID']
    train_df.to_csv('../Results/labels_train_with_predictions.csv', index=False)
    print(f"Saved training predictions to ../Results/labels_train_with_predictions.csv")

    val_df = pd.read_csv(val_csv)
    val_df.columns = ['filename', 'timestamp', 'latitude', 'longitude', 'angle', 'Region_ID']
    print(f"Loaded {len(val_df)} Validation samples from {val_csv}")
    pred_val_df = pd.read_csv('../Results/val_predictions.csv')
    pred_val_df.columns = ['id', 'Region_ID']
    print(f"Loaded {len(pred_val_df)} predictions from ../Results/val_predictions.csv")
    assert len(val_df) == len(pred_val_df), f"Mismatch: val_df has {len(val_df)} rows, pred_val_df has {len(pred_val_df)} rows"
    val_df['pred_Region_ID'] = pred_val_df['Region_ID']
    val_df.to_csv('../Results/labels_val_with_predictions.csv', index=False)
    print(f"Saved training predictions to ../Results/labels_val_with_predictions.csv")

# Test Accuracy: 0.9566 (353/369)
