import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class GeoDataset(Dataset):
    def __init__(self, df, img_dir, task, transform,
                 lat_mean=None, lat_std=None, lon_mean=None, lon_std=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.task = task
        self.transform = transform
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.lon_mean = lon_mean
        self.lon_std = lon_std
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        region_id = torch.tensor(row['pred_Region_ID']-1, dtype=torch.long)
        lat = (row['latitude'] - self.lat_mean) / self.lat_std
        lon = (row['longitude'] - self.lon_mean) / self.lon_std
        target = torch.tensor([lat, lon], dtype=torch.float32)
        return image, region_id, target
    


def average_mse(pred, target, lat_mean, lat_std, lon_mean, lon_std):
    latitude_predicted = pred[:, 0] * lat_std + lat_mean
    longitude_predicted = pred[:, 1] * lon_std + lon_mean
    latitude_target = target[:, 0] * lat_std + lat_mean
    longitude_target = target[:, 1] * lon_std + lon_mean
    latitude_mse = nn.functional.mse_loss(latitude_predicted, latitude_target)
    longitude_mse = nn.functional.mse_loss(longitude_predicted, longitude_target)
    total_mse = latitude_mse + longitude_mse
    return 0.5 * (total_mse)


class GeoLatLong(nn.Module):
    def __init__(self, num_regions):
        super().__init__()
        resnet_model = models.resnet101(weights='IMAGENET1K_V1')
        self.resnet_feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1]) 
        embedding_dim = 256
        input_layer_dim = 2048
        hidden_layer_dim = 512
        output_layer_dim = 2
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(input_layer_dim + embedding_dim, hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(hidden_layer_dim, output_layer_dim),
            nn.Softmax(dim=1)
        )
        
        self.head = nn.Sequential(
            nn.Linear(input_layer_dim + embedding_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, output_layer_dim)
        )
    
    def forward(self, x, region_id):
        features = self.resnet_feature_extractor(x).flatten(1)
        region_emb = self.region_embedding(region_id)
        combined = torch.cat([features, region_emb], 1)
        attn_weights = self.attention(combined)
        weighted = combined * attn_weights[:,0].unsqueeze(1)  
        return self.head(weighted)




def train_model(model, train_loader, val_loader, num_epochs, lr, lat_mean, lat_std, lon_mean, lon_std):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_mse = float('inf')
    patience = 0
    patience_limit = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for images, region_ids, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, region_ids, targets = images.to(device), region_ids.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images, region_ids)
            o1 = criterion(outputs[:, 0], targets[:, 0])
            o2 = criterion(outputs[:, 1], targets[:, 1])
            loss = 0.5 * (o1 + o2)
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, region_ids, targets in val_loader:
                images, region_ids, targets = images.to(device), region_ids.to(device), targets.to(device)
                outputs = model(images, region_ids)
                val_loss = average_mse(outputs, targets, lat_mean, lat_std, lon_mean, lon_std)
                val_losses.append(val_loss.item())

        train_loss /= len(train_loader)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)


        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"patience: {patience}")

        if val_loss < best_mse:
            best_mse = val_loss
            torch.save(model.state_dict(), f"../Models/best_lat_long_model_saved_ever.pt")
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete.")
    print(f"Best Model saved at epoch {epoch+1}")
    print(f"Best Validation Loss: {best_mse:.6f}")

    # load the best model
    model.load_state_dict(torch.load(f"../Models/best_lat_long_model_saved_ever.pt"))
    return model

def predict_validation_set(model, val_loader, device, lat_mean, lat_std, lon_mean, lon_std):
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []
    val_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, region_ids, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            region_ids = region_ids.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images, region_ids)
            
            # Calculate loss
            loss = average_mse(outputs, targets, lat_mean, lat_std, lon_mean, lon_std)
            val_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Denormalize predictions and targets
            pred_lats = outputs[:, 0].cpu().numpy() * lat_std + lat_mean
            pred_lons = outputs[:, 1].cpu().numpy() * lon_std + lon_mean
            true_lats = targets[:, 0].cpu().numpy() * lat_std + lat_mean
            true_lons = targets[:, 1].cpu().numpy() * lon_std + lon_mean
            
            # Store results
            all_preds.extend(zip(pred_lats, pred_lons))
            all_targets.extend(zip(true_lats, true_lons))

    # Calculate final loss
    val_loss /= total_samples
    # round and convert to integer
    all_preds = [(round(p[0]), round(p[1])) for p in all_preds]
    
    return all_preds, val_loss


class TestGeoDataset(Dataset):
    def __init__(self, img_dir, region_df, transform, lat_mean, lat_std, lon_mean, lon_std):
        self.img_dir = img_dir
        self.transform = transform
        self.lat_mean = lat_mean
        self.lat_std = lat_std
        self.lon_mean = lon_mean
        self.lon_std = lon_std
        
        # Load and sort test images
        self.image_files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # Create ID to Region_ID mapping
        self.region_mapping = dict(zip(region_df['id'], region_df['Region_ID']))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        
        # Extract numeric ID from filename (e.g., "img_0045.jpg" → 45)
        img_id = int(filename.split('_')[1].split('.')[0])
        
        # Get Region_ID from mapping (convert to 0-based)
        region_id = torch.tensor(self.region_mapping[img_id] - 1, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, region_id, img_id

def predict_test_set(model, test_dataset, device, lat_mean, lat_std, lon_mean, lon_std):
    model.eval()
    predictions = []
    ids = []
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    with torch.no_grad():
        for images, region_ids, img_ids in tqdm(test_loader, desc="Predicting Test Set"):
            images = images.to(device)
            region_ids = region_ids.to(device)
            
            # Get predictions
            outputs = model(images, region_ids)
            
            # Denormalize
            pred_lats = outputs[:, 0].cpu().numpy() * lat_std + lat_mean
            pred_lons = outputs[:, 1].cpu().numpy() * lon_std + lon_mean
            
            predictions.extend(zip(pred_lats, pred_lons))
            ids.extend(img_ids.cpu().numpy())
    
    # round and convert to integer
    predictions = [(round(p[0]), round(p[1])) for p in predictions]
    return predictions, ids


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    train_df = pd.read_csv("../Results/labels_train_with_predictions.csv")
    val_df = pd.read_csv("../Results/labels_val_with_predictions.csv")
    test_region_df = pd.read_csv("../Results/test_predictions.csv")
    test_img_dir = "../Data/images_test"

    lat_lower = train_df['latitude'].quantile(0.01)
    lat_upper = train_df['latitude'].quantile(0.99)
    lon_lower = train_df['longitude'].quantile(0.01)
    lon_upper = train_df['longitude'].quantile(0.99)

    train_df = train_df[
        (train_df['latitude'].between(lat_lower, lat_upper)) &
        (train_df['longitude'].between(lon_lower, lon_upper))
    ]

    rows_to_remove = [95, 145, 146, 158, 159, 160, 161]
    print("Rows to remove:", val_df.iloc[rows_to_remove])
    val_df = val_df.drop(index=rows_to_remove).reset_index(drop=True)

    # Normalize based on train set
    lat_mean = train_df['latitude'].mean()
    lat_std = train_df['latitude'].std()
    lon_mean = train_df['longitude'].mean()
    lon_std = train_df['longitude'].std()

    batch_size = 16
    num_epochs = 30
    lr = 1e-4
    num_workers = 4
    num_regions = 15

    train_dataset = GeoDataset(
        train_df, 
        "../Data/images_train", 
        task='latlong', 
        transform=train_transform,
        lat_mean=lat_mean, 
        lat_std=lat_std, 
        lon_mean=lon_mean, 
        lon_std=lon_std
    )

    val_dataset = GeoDataset(
        val_df, 
        "../Data/images_val", 
        task='latlong', 
        transform=val_transform,
        lat_mean=lat_mean, 
        lat_std=lat_std, 
        lon_mean=lon_mean, 
        lon_std=lon_std
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    model = GeoLatLong(num_regions=num_regions).to(device)
    # best_model = train_model(model, train_dataloader, val_dataloader, num_epochs, lr, lat_mean, lat_std, lon_mean, lon_std)

    test_dataset = TestGeoDataset(
        img_dir=test_img_dir,
        region_df=test_region_df,
        transform=val_transform,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std
    )
    
    # Load best model
    # model.load_state_dict(torch.load("../Models/best_lat_long_model_saved_ever.pth"))
    model.load_state_dict(torch.load(f"../Models/best_lat_long_model_saved_ever.pt"))

    val_results, val_loss = predict_validation_set(model, val_dataloader, device, lat_mean, lat_std, lon_mean, lon_std)
    print(f"Validation Loss: {val_loss:.6f}")
    # Make predictions
    test_results, test_ids = predict_test_set(model, test_dataset, device, lat_mean, lat_std, lon_mean, lon_std)

    # Save submission
    submission_df = pd.DataFrame({
        'id' : range(len(val_results)+ len(test_results)),
        'latitude': [0]*(len(val_results)) + [0]*(len(test_results)),
        'longitude': [0]*(len(val_results)) + [0]*(len(test_results))
    })

    # Assign validation results
    submission_df.loc[:len(val_results)-1, 'latitude'] = [p[0] for p in val_results]
    submission_df.loc[:len(val_results)-1, 'longitude'] = [p[1] for p in val_results]

    # Assign test results
    submission_df.loc[len(val_results):, 'latitude'] = [p[0] for p in test_results]
    submission_df.loc[len(val_results):, 'longitude'] = [p[1] for p in test_results]


    submission_df.to_csv("../Results/2022102032_latlong_final.csv", index=False)
    print(f"Saved ../Results/2022102032_latlong_final.csv with {len(submission_df)} entries")

    # Save the best model
