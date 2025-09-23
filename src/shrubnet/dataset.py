import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
import logging

logging.basicConfig(level=logging.INFO)


class RSDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        transform=None,
        augment=False,
        repeat_augmentations=0,
    ):
        self.images_dir = Path(images_dir)
        self.images = os.listdir(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels = os.listdir(labels_dir)
        self.transform = transform

        # Add a default transform
        if not self.transform:
            self.transform = transforms.ToTensor()

        self.augment = augment
        self.repeat_augmentations = repeat_augmentations

        self.image_files = [
            f for f in os.listdir(images_dir) if f.lower().endswith((".tif"))
        ]
        self.image_files.sort()

        # Define albumentations augmentation pipeline
        if self.augment:
            self.aug = A.OneOf(
                [
                    A.Rotate(limit=5, p=0.5),  # Small rotation, no flip
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                    A.GaussNoise(var_limit=(1.0, 10.0), p=0.2),
                    # Add more augmentations if needed
                ]
            )
        else:
            self.aug = None

    def __len__(self):
        return len(self.image_files) * (1 + self.repeat_augmentations)

    def __getitem__(self, idx):
        logging.debug(f"index is {idx}")

        if self.aug:
            base_image_idx = idx // (1 + self.repeat_augmentations)
            is_augmented = (idx % (1 + self.repeat_augmentations)) > 0
        else:
            is_augmented = False
            base_image_idx = idx

        image_path = str(self.images_dir / self.image_files[base_image_idx])
        logging.debug(f"path is {image_path}")
        label_path = image_path.replace("images", "labels")

        image = (
            np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        )
        label = np.array(Image.open(label_path).convert("L")) / 255.0

        if is_augmented and self.aug:
            augmented = self.aug(image=image, mask=label)
            assert "image" in augmented
            image = augmented["image"]
            label = augmented["mask"]

        image = self.transform(image)
        # Add channel dimension to label
        label = np.expand_dims(label, axis=0)

        label = torch.tensor(label, dtype=torch.float32)
        return image, label
