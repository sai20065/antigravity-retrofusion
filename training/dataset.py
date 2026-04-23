# training/dataset.py
# RADataset class — standalone module

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class RADataset(Dataset):
    """
    Dataset for retroreflectivity estimation from road sign/marking images.

    CSV format: img_path, ra_value, asset_type
    Images: road sign/marking crops, resized to 224×224

    Usage:
        dataset = RADataset("data/annotations.csv", "data/images/", augment=True)
        img_tensor, ra_value = dataset[0]
    """
    AUGMENT = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    EVAL = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, csv_path: str, img_dir: str, augment: bool = True):
        """
        Args:
            csv_path: Path to CSV with columns: img_path, ra_value, [asset_type]
            img_dir:  Root directory containing the images
            augment:  Whether to apply data augmentation
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tf = self.AUGMENT if augment else self.EVAL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(self.img_dir, row["img_path"])
        img = Image.open(path).convert("RGB")
        ra = float(row["ra_value"])

        import torch
        return self.tf(img), torch.tensor(ra, dtype=torch.float32)
