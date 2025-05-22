import torch
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from torchvision.transforms.functional import to_tensor
import cv2
import logging


def sliding_window(image, step_size, window_size):
    """
    Given an input image (numpy ndarray) return a generator of windows in it

    Args:
        image: np.ndarray of raster data
        step_size (tuple): Step size for sliding (dy, dx).
        window_size (tuple): Size of the sliding window (h, w).
    """
    for y in range(0, image.shape[1] - window_size[1] + 1, step_size[1]):
        for x in range(0, image.shape[2] - window_size[0] + 1, step_size[0]):
            yield (x, y, image[:, y : y + window_size[1], x : x + window_size[0]])


def run_inference(
    model,
    input_image_path,
    output_image_path,
    window_size=(512, 512),
    step_size=(256, 256),
    threshold=0.5,
    device="cpu",
):
    """
    Perform inference on a large image using a sliding window approach.

    Args:
        model (torch.nn.Module): Trained PyTorch model for segmentation.
        input_image_path (str): Path to the input GeoTIFF image.
        output_image_path (str): Path to save the output prediction GeoTIFF.
        window_size (tuple): Size of the sliding window (h, w).
        step_size (tuple): Step size for sliding (dy, dx).
        threshold (float): Threshold for binary segmentation.
        device (str): Device for inference ("cpu" or "cuda").

    Returns:
        None: Saves the prediction as a GeoTIFF file.
    """
    model.eval()
    model.to(device)

    try:
        with rasterio.open(input_image_path) as src:
            output_meta = src.meta
            output_meta["count"] = 1
            images = src.read()
    except RasterioIOError as err:
        logging.error(err)
        raise

    # TODO check what normalisation / standardisation was used for training!
    images = images / 255.0  # Normalize

    stitched_image = np.zeros((images.shape[1], images.shape[2]), dtype=np.uint8)

    # Calculate the total number of tiles for progress tracking
    total_tiles = ((images.shape[1] - window_size[1]) // step_size[1] + 1) * (
        (images.shape[2] - window_size[0]) // step_size[0] + 1
    )
    processed_tiles = 0

    # Sliding window inference
    for x, y, window in sliding_window(images, step_size, window_size):
        # Resize the window to the model's input size
        resized_window = cv2.resize(window.transpose(1, 2, 0), (256, 256))
        window_tensor = to_tensor(resized_window).unsqueeze(0).float().to(device)

        # Run the model and process predictions
        with torch.no_grad():
            output = model(window_tensor).float().cpu().numpy().squeeze()
            output = cv2.resize(output, window_size)
            prediction = (output > threshold).astype(np.uint8)

        # Stitch the prediction into the output image
        stitched_image[y : y + window_size[1], x : x + window_size[0]] = np.maximum(
            stitched_image[y : y + window_size[1], x : x + window_size[0]], prediction
        )

        # Update progress
        processed_tiles += 1
        progress = (processed_tiles / total_tiles) * 100
        print(
            f"Progress: {processed_tiles}/{total_tiles} tiles processed ({progress:.2f}%)",
            end="\r",
        )

    try:
        with rasterio.open(output_image_path, "w", **output_meta) as dst:
            dst.write(stitched_image, 1)
            print(f"Prediction saved to {output_image_path}")
    except RasterioIOError as err:
        logging.error(err)
        raise
