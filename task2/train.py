"""
Dataset was downloaded form kaggle in the directory 'data/' contains multiple subfolders, each corresponding to a Sentinel-2 
acquisition, for example:
    data/
      S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510/
        S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE/
          GRANULE/
            L1C_T36UYA_A003350_20160212T084510/
              IMG_DATA/
                T36UYA_20160212T084052_B01.jp2
                T36UYA_20160212T084052_B02.jp2
                ...
      S2A_MSIL1C_2020xxxxTxxxxxx_.../
        ...
and so on.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.windows import Window


# Define the Dataset

class Sentinel2MatchingDataset(Dataset):
   
    def __init__(self,
                 data_dir,
                 transform=None,
                 patch_size=256,
                 season_pairs=True):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.patch_size = patch_size
        self.season_pairs = season_pairs
        
        # Gather JP2 image paths
        self.image_paths = self._collect_image_paths()
        
    def _collect_image_paths(self):
        # Recursively collect all .jp2 files in the data directory including IMG_DATA subfolders

        all_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(".jp2"):
                    all_files.append(os.path.join(root, file))
        return all_files

    def _read_patch(self, image_path, row_off, col_off, patch_size):
        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
            if row_off + patch_size > height:
                row_off = max(0, height - patch_size)
            if col_off + patch_size > width:
                col_off = max(0, width - patch_size)
            
            img = src.read(
                indexes=1,
                window=Window(col_off, row_off, patch_size, patch_size)
            )
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Read image dimensions so we can sample a random patch location
        with rasterio.open(image_path) as src:
            height, width = src.height, src.width
        
        row_off = np.random.randint(0, height - self.patch_size)
        col_off = np.random.randint(0, width - self.patch_size)
        
        patch1 = self._read_patch(image_path, row_off, col_off, self.patch_size)

        if self.season_pairs:
            idx2 = np.random.randint(0, len(self.image_paths))
            image_path2 = self.image_paths[idx2]
            patch2 = self._read_patch(image_path2, row_off, col_off, self.patch_size)
            
            if self.transform:
                patch1 = self.transform(patch1)
                patch2 = self.transform(patch2)

            return patch1, patch2
        
        else:
            label = 0
            if self.transform:
                patch1 = self.transform(patch1)
            return patch1, label

# Define a simple CNN Feature Extractor

class SimpleFeatureExtractor(nn.Module):

    # A simple CNN-based feature extractor for demonstration.

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 64 * 64, 128)

    def forward(self, x):
        # x shape: (batch_size, 3, 256, 256)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# Define a Siamese Model (optional)

class SiameseNetwork(nn.Module):

    # A basic Siamese network that uses the feature extractor on two inputs and compares them (e.g., via absolute difference).

    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

        self.fc_out = nn.Linear(128, 64)

    def forward(self, x1, x2):
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        diff = torch.abs(feat1 - feat2)
        out = self.fc_out(diff)
        return out

# Training Function

def train_siamese(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (patch1, patch2) in enumerate(dataloader):
        patch1, patch2 = patch1.to(device), patch2.to(device)
        
        optimizer.zero_grad()
        output = model(patch1, patch2)
        

        label = torch.ones_like(output)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (batch_idx + 1)


# Main function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the top-level data directory, e.g. data/')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size to crop from images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (cuda or cpu)')
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    dataset = Sentinel2MatchingDataset(
        data_dir=args.data_dir,
        transform=None,
        patch_size=args.patch_size,
        season_pairs=True
    )
    
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4)
    
    feature_extractor = SimpleFeatureExtractor()
    siamese_model = SiameseNetwork(feature_extractor).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(siamese_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = train_siamese(siamese_model, dataloader, criterion, optimizer, device)
        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f}")
    
    model_save_path = "siamese_model.pth"
    torch.save(siamese_model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")


if __name__ == "__main__":
    main()
