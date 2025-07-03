import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A


class RSDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        transform=None,
        max_samples=None,
        augment=False,
        repeat_augmentations=1,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.augment = augment
        self.repeat_augmentations = repeat_augmentations
        self.image_files = sorted(os.listdir(images_dir))

        if max_samples:
            self.image_files = self.image_files[:max_samples]

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
        return len(self.image_files) * self.repeat_augmentations

    def __getitem__(self, idx):
        base_idx = idx % len(self.image_files)
        image_path = os.path.join(self.images_dir, self.image_files[base_idx])

        # Handle negative images by checking the filename (fragile)
        # Avoid having hundreds of static blank labels, just one that we reuse
        if "negative" in image_path:
            label_path = os.path.join(self.labels_dir, "negative_0.tif")
        else:
            label_image = self.image_files[base_idx].replace('images','labels')
            label_path = os.path.join(self.labels_dir, label_image)

        image = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("L"))

        # Normalize image to [0, 1] and ensure float32 type
        image = (image / 255.0).astype(np.float32)

        # Normalize label to [0, 1] and ensure float32 type
        if label.max() == 255:
            label = (label / 255.0).astype(np.float32)
        else:
            label = label.astype(np.float32)

        # Add channel dimension to label
        label = np.expand_dims(label, axis=0)

        # Adjust the dimensions of the image to [channels, height, width]
        image = np.transpose(image, (2, 0, 1))

        # Albumentations expects [H, W, C], so transpose back for augmentation
        if self.aug:
            image_aug = np.transpose(image, (1, 2, 0))
            label_aug = np.transpose(label, (1, 2, 0))
            augmented = self.aug(image=image_aug, mask=label_aug)
            image = np.transpose(augmented["image"], (2, 0, 1))
            label = np.transpose(augmented["mask"], (2, 0, 1))

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
