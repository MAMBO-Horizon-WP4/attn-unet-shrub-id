import os
from shrubnet.dataset import RSDataset


def test_dataset_length_and_augmentation(images_dir, labels_dir):
    size = 256
    ds = RSDataset(
        str(images_dir), str(labels_dir), augment=True, repeat_augmentations=3
    )
    images = list(filter(lambda x: 'tif' in x, os.listdir(images_dir)))
    assert len(ds) == len(images) * 3  # 9 images * 3 augmentations

    # Check that you can index all elements
    for i in range(len(ds)):
        img, lbl = ds[i]
        assert img.shape[1:] == (size, size)
        assert lbl.shape[1:] == (size, size)
