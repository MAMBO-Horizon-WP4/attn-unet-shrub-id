"""Use to generate metadata useful in training and inference
This is mainly mean and standard deviation values per band
used to normalise data - enhance difference where values are similar"""

from torch.utils.data import DataLoader
from shrubnet.dataset import RSDataset


def means_and_stds(images_dir: str, labels_dir: str):
    """Load the whole dataset up as one image batch.
    Return per-band mean and standard deviation pixel values.
    https://www.codegenes.net/blog/dataloader-pytorch-mean-std-norm/ - friendly overview"""
    dataset = RSDataset(images_dir=images_dir, labels_dir=labels_dir)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # Get the images (index 0) from the giant batch
    data = next(iter(dataloader))[0]

    # They come out as tensors or np.float32 - convert to be json serialisable
    stds = data.std(dim=(0, 2, 3)).numpy().tolist()
    means = data.mean(dim=(0, 2, 3)).numpy().tolist()

    return {"std": stds, "mean": means}
