import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, max_samples=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(images_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        if max_samples:
            self.image_files = self.image_files[:max_samples]
            self.label_files = self.label_files[:max_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        image = np.array(image)
        label = np.array(label)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Normalize label to [0, 1]
        if label.max()==255:
            label = label / 255.0

        # Add channel dimension to label
        label = np.expand_dims(label, axis=0)

        # Adjust the dimensions of the image to [channels, height, width]
        image = np.transpose(image, (2, 0, 1))

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label