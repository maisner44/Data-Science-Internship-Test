import argparse
import torch
import torch.nn as nn
import numpy as np
import rasterio
from rasterio.windows import Window

# Model Definitions

class SimpleFeatureExtractor(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 64 * 64, 128)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):

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

# Helper Functions

def read_patch(
    image_path: str,
    row_off: int,
    col_off: int,
    patch_size: int,
    in_channels: int = 1,
    normalize: bool = True
) -> np.ndarray:

    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        
        if row_off + patch_size > height:
            row_off = max(0, height - patch_size)
        if col_off + patch_size > width:
            col_off = max(0, width - patch_size)
        
        if in_channels == 1:
            img = src.read(
                indexes=1,
                window=Window(col_off, row_off, patch_size, patch_size)
            )
            img = img.astype(np.float32)
            if normalize:
                img /= 10000.0

            img = np.expand_dims(img, axis=0)
            return img
        
        else:
            img = src.read(
                indexes=[1,2,3],
                window=Window(col_off, row_off, patch_size, patch_size)
            )
            img = img.astype(np.float32)
            if normalize:
                img /= 10000.0

            return img


def inference_siamese(model: nn.Module, patch1: np.ndarray, patch2: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Runs a forward pass on the Siamese model given two patches (numpy arrays).
    Returns the raw output (e.g., a 64-dimensional embedding or similarity score).
    """
    t1 = torch.from_numpy(patch1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(patch2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(t1, t2)
    
    return output.squeeze(0)


# Main Inference

def main():
    parser = argparse.ArgumentParser(description="Siamese Network Inference for Sentinel-2 JP2 images.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved siamese model weights (e.g. siamese_model.pth).")
    parser.add_argument("--img1", type=str, required=True,
                        help="Path to the first .jp2 image.")
    parser.add_argument("--img2", type=str, required=True,
                        help="Path to the second .jp2 image.")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Patch size to read from each image.")
    parser.add_argument("--row_off", type=int, default=0,
                        help="Row offset for the top-left corner of the patch.")
    parser.add_argument("--col_off", type=int, default=0,
                        help="Column offset for the top-left corner of the patch.")
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Number of input channels the model expects (1 for single-band).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu).")
    args = parser.parse_args()


    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    feature_extractor = SimpleFeatureExtractor(in_channels=args.in_channels)
    siamese_model = SiameseNetwork(feature_extractor).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()

    patch1 = read_patch(
        image_path=args.img1,
        row_off=args.row_off,
        col_off=args.col_off,
        patch_size=args.patch_size,
        in_channels=args.in_channels
    )
    patch2 = read_patch(
        image_path=args.img2,
        row_off=args.row_off,
        col_off=args.col_off,
        patch_size=args.patch_size,
        in_channels=args.in_channels
    )

    output = inference_siamese(siamese_model, patch1, patch2, device)

    print("Siamese output:", output.cpu().numpy())

    distance = torch.norm(output, p=2).item()
    print(f"Distance measure (L2 norm): {distance:.4f}")


if __name__ == "__main__":
    main()
